from nnunet.network_architecture.generic_UNet import Generic_UNet
from data_convert import load_pickle, resize, get_bbox_from_mask_b, resample_data
import SimpleITK as sitk
from utils.predict import Valid_utils
import os
from scipy.ndimage import binary_fill_holes
import numpy as np
import threading
import time
import queue
import concurrent.futures
import torch.nn as nn
import torch
from collections import OrderedDict
from scipy.ndimage import binary_dilation
from skimage.measure import label
import argparse
import math





def max_compoment(predict_array):
    predict_post = np.zeros_like(predict_array, dtype=predict_array.dtype)
    for organ in np.unique(predict_array):
        if organ == 0:
            continue
        copy = predict_array.copy()
        copy[predict_array != organ] = 0
        selem = np.ones((3, 3, 3), dtype=bool)
        # 一次膨胀 11个连通分量
        labels = binary_dilation(copy, selem)
        # 两次膨胀 10个连通分量
        # copy = morphology.dilation(copy, selem)

        labels = label(label_image=labels, connectivity=2)
        # print(np.unique(labels))
        connected_components = dict()
        for i in np.unique(labels):
            connected_components[i] = (np.sum(labels == i))

        retv = dict(sorted(connected_components.items(), key=lambda k: -k[1]))
        keep_labels = list(retv.keys())
        if organ == 14:
            # fix the bug label do not appear!
            keep_labels = keep_labels[:]
        else:
            keep_labels = keep_labels[:2]
        for i in retv.keys():
            if i not in keep_labels:
                labels[labels == i] = 0
        labels[labels != 0] = 1
        labels *= copy
        labels = labels.astype(predict_post.dtype)
        # predict_array[predict_array == organ] = 0
        predict_post += labels
    return predict_post





class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
            module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class my_Trainner():
    def __init__(self):
        self.base_num_features = self.num_classes = self.net_num_pool_op_kernel_sizes = self.conv_per_stage \
            = self.net_conv_kernel_sizes = None
        self.stage = 1
        self.network = None
        self.initial_par()

    def initial_par(self):
        #plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16 = \
        #    load_pickle(os.path.join(os.path.dirname(__file__), 'checkpoints/model_final_checkpoint.model.pkl')
        #                )[
        #        'init']
        plans = load_pickle(os.path.join(os.path.join(os.path.dirname(__file__), "checkpoints"), 'nnUNetPlansv2.1_plans_3D.pkl'))
        stage_plans = plans['plans_per_stage'][1]
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.base_num_features = plans['base_num_features']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            print(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

        if 'conv_kernel_sizes' not in stage_plans.keys():
            print(
                "WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

    def network_initial(self):
        conv_op = nn.Conv3d
        #conv_op = SeparableConv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        '''
        self.conv_per_stage = 1
        self.stage_num = 4
        self.base_num_features = 16
        self.max_num_features = 256

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]
        '''
        '''
        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 16
        self.max_num_features = 256
        self.max_num_epochs = 500

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]
        '''
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.base_num_features = 16
        self.net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        self.net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 2, 2]]

        self.network = Generic_UNet(1, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        print(self.network)
    def load_best_checkpoint(self, fname):
        checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        curr_state_dict_keys = list(self.network.state_dict().keys())
        new_state_dict = OrderedDict()
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)  # todo jichao params not match

    def get_network(self, fname):
        self.network_initial()
        self.load_best_checkpoint(
            fname=fname)
        return self.network


class MuiltThreadDataGenerator(object):
    def __init__(self, iter, produce_queue_number,save_path) -> None:
        self.iter = iter
        self.produce_queue_number = produce_queue_number
        self.output_queue = queue.Queue(2)
        self.save_path = save_path

    def _ini(self):
        process_thread = threading.Thread(target=self.process_data_thread)
        process_thread.start()
        self.predict_model()
        process_thread.join()
        return

    # 多线程处理数据的函数
    def process_data_thread(self):
        data_list = self.iter
        with concurrent.futures.ThreadPoolExecutor(self.produce_queue_number) as executor:
            # 多线程处理数据
            for data in data_list:
                executor.submit(self.preprocess, data)
        # 添加结束标志到队列
        for _ in range(self.produce_queue_number):
            self.output_queue.put('end', block=True)
        print("process end.")
        return

    def preprocess(self, nii_path):
        identity = nii_path.split('/')[-1].split('_0000.nii.gz')[0]   #pre
        image = sitk.ReadImage(nii_path)
        ori_spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        ori_size = np.array(image.GetSize())
        image_array = sitk.GetArrayFromImage(image)
        image_dtype = image_array.dtype
        nonzero_mask = np.zeros(image_array.shape, dtype=bool)
        nonzero_mask = nonzero_mask | (image_array > 0)
        nonzero_mask = binary_fill_holes(nonzero_mask)
        predict_final = np.zeros_like(nonzero_mask).astype(np.float32)
        # crop array
        bbox = get_bbox_from_mask_b(nonzero_mask, 0)

        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        cropped_data = np.stack((image_array[resizer], np.zeros_like(image_array[resizer])), 0)
        cropped_shape = cropped_data.shape[1:]
        # resample array
        #target_spacing = [2.5, 0.859375, 0.859375]
        #target_spacing = [2.13021975, 1.66839451, 1.66839451]
        target_spacing = [2.2838839, 1.8709113, 1.8709113]
        #target_spacing = [3.0, 2.34960903, 2.34960903]
        #target_spacing = [1., 0.78320301, 0.78320301]

        resampled_data, _ = resample_data(cropped_data, np.array(ori_spacing), target_spacing=target_spacing,
                                          only_z=False)
        ct_array = resampled_data[0].copy()

        # norm to one
        if np.max(ct_array) < 1:
            percentile_95 = np.percentile(ct_array, 95)
            percentile_5 = np.percentile(ct_array, 5)
            std = np.std(ct_array)
            mn = np.mean(ct_array)
            ct_array = np.clip(ct_array, a_min=percentile_5, a_max=percentile_95).astype(np.float32)
            ct_array = (ct_array - mn) / std
        else:
            ct_array = np.clip(ct_array, a_min=-160., a_max=240.).astype(np.float32)
            ct_array = (ct_array + 160.) / 400.
        resampled_data[0] = ct_array
        ori = torch.from_numpy(resampled_data[0:1])
        print('identity: %s, preprocessing done.' % identity)
        data = [ori, identity, origin, direction, ori_size, image_dtype, predict_final, resizer, cropped_shape,
                ori_spacing]
        self.output_queue.put(data, block=True)



    # 第二部分的模型预测函数
    def predict_model(self):
        end_count = 0.
        while True:
            try:
                start = time.time()
                data = self.output_queue.get()
                if data == 'end':
                    end_count += 1
                    print('processes end this thread')
                    if end_count == self.produce_queue_number:
                        break
                    continue
                ori, identity, origin, direction, ori_size, image_dtype, predict_final, resizer, cropped_shape, ori_spacing = data
                torch.cuda.empty_cache()
                ori = ori.cuda().float()
                # predict
                s = Valid_utils(model, 14, patch_size)

                predict = s.predict_3D(ori, do_mirror=False, mirror_axes=(0, 1, 2))
                predict = predict.softmax(0).argmax(0).detach().cpu().squeeze().numpy()

                predict_resample = resize(predict, cropped_shape, order=0, preserve_range=True, anti_aliasing=False)
                predict_final[resizer] = predict_resample
                after_itk_label = sitk.GetImageFromArray(predict_final.astype(np.uint8))
                after_itk_label.SetSpacing(ori_spacing)
                after_itk_label.SetOrigin(origin)
                after_itk_label.SetDirection(direction)
                if not os.path.exists(save_path):
                    os.makedirs(os.path.join(save_path))
                #dd = self.save_path
                save_name = os.path.join(self.save_path, '%s.nii.gz' % identity)
                sitk.WriteImage(after_itk_label, save_name)


                print('%s : cost %d s' % (identity, (time.time() - start)))
            except queue.Empty:
                time.sleep(1)
                pass
        return





def inference():
    file_list = os.listdir(base_folder_name)
    file_list = [os.path.join(base_folder_name, file_) for file_ in file_list]

    DG = MuiltThreadDataGenerator(file_list, 2, save_path)

    with torch.no_grad():
        DG._ini()


if __name__ == '__main__':
    p_start = time.time()
    argps = argparse.ArgumentParser()
    # -PSEUDO_LABEL_PATH --NNUNET_npy_PATH
    argps.add_argument('input_folder', type=str, help='input folder')
    argps.add_argument('output_folder', type=str,
                       help='output folder')

    arg_s = argps.parse_args()
    base_folder_name = arg_s.input_folder
    save_path = arg_s.output_folder
    # init model and predict utils

    patch_size = patch_size = (80, 160, 160)
    #patch_size = [80, 160, 160]
    #patch_size = (80, 160, 160)
    #patch_size = (40, 224, 192)
    #patch_size = (80, 160, 160)
    model = my_Trainner().get_network(
        os.path.join(os.path.join(os.path.dirname(__file__), 'checkpoints'), 'model_final_checkpoint.model'))
    model.cuda()
    model.eval()

    inference()
    print("cost_all---------------------", time.time() - p_start)
    pass

#可分离：222.37063002586365
# 正常：249.01559329032898

#G:\FLARE2022\Testing E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\a_imagesTest E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\a_outTest E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#G:\FLARE2022\Training\FLARE22_LabeledCase50\images E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#F:\ljc\all_organ_labeled\images F:\ljc\all_organ_labeled\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2

#F:\test\FLARE23TestImg400\FLARE23TestImg400 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs3 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#F:\test\FLARE23TestImg400\FLARE23TestImg400 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs1
#E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task099_FLARE2023\imagesTs E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs1
#E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task099_FLARE2023\imagesTs E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs3 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#644.6536412239075   部分保留最大连通域
#703.2188038825989   都保留最大连通域


#200Test
#G:\FLARE2022\Testing E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task101_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_1

#G:\FLARE2022\Tuning\aaaa G:\FLARE2022\Tuning\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task101_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_1

#G:\FLARE2022\Tuning\images G:\FLARE2022\Tuning\outout E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task101_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_1