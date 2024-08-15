import gc
import math
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from skimage.measure import label

from predict import Valid_utils
#import os
#from scipy.ndimage import binary_fill_holes
import numpy as np
#import threading
import time
#import queue
#import concurrent.futures
#import torch.nn as nn
import torch
#from collections import OrderedDict
#from scipy.ndimage import binary_dilation
#from skimage.measure import label
import argparse
import pickle
from skimage.transform import resize
#import scipy.ndimage
#from typing import Union, Tuple, List
from scipy import ndimage

def keep_largest_connected_component(binary_image):
    # 1. 计算所有连通域的标签
    labeled_array, num_features = ndimage.label(binary_image)

    if num_features == 0:
        print("没有找到连通域")
        return binary_image

    # 2. 计算每个连通域的体素数量
    sizes = ndimage.sum(binary_image, labeled_array, range(num_features + 1))

    # 3. 找到最大连通域的标签
    max_label = np.argmax(sizes)

    # 4. 创建一个新的二值图像，只保留最大连通域
    largest_component = (labeled_array == max_label)

    return largest_component


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

'''
def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    # assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    # target_step_sizes_in_voxels = [i * 0.5 for i in patch_size]
    target_step_sizes_in_voxels = [i * j for i, j in zip(step_size, patch_size)]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                 zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps
'''



def load_pickle(path):
    return pickle.load(open(path, 'rb+'))

def get_bbox_from_mask_b(nonzero_mask, outside_value):
    #nonzero_mask = nonzero_mask.astype(np.int8)
    if nonzero_mask.shape[0] > 1000:
        chunk = nonzero_mask[:math.floor(nonzero_mask.shape[0] / 2), :, :]
        chunk1 = nonzero_mask[math.floor(nonzero_mask.shape[0] / 2):, :, :]
        del nonzero_mask
        mask_voxel_coords = np.where(chunk[:, :, :] != outside_value[0])
        minzidx1 = int(np.min(mask_voxel_coords[0]))
        maxzidx1 = int(np.max(mask_voxel_coords[0])) + 1
        mask_voxel_coords = np.where(chunk[:, :, :] != outside_value[1])
        minxidx1 = int(np.min(mask_voxel_coords[0]))
        maxxidx1 = int(np.max(mask_voxel_coords[0])) + 1
        mask_voxel_coords = np.where(chunk[:, :, :] != outside_value[2])
        minyidx1 = int(np.min(mask_voxel_coords[0]))
        maxyidx1 = int(np.max(mask_voxel_coords[0])) + 1
        del mask_voxel_coords
        mask_voxel_coords = np.where(chunk1[:, :, :] != outside_value[0])
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        mask_voxel_coords = np.where(chunk1[:, :, :] != outside_value[1])
        minxidx = int(np.min(mask_voxel_coords[0]))
        maxxidx = int(np.max(mask_voxel_coords[0])) + 1
        mask_voxel_coords = np.where(chunk1[:, :, :] != outside_value[2])
        minyidx = int(np.min(mask_voxel_coords[0]))
        maxyidx = int(np.max(mask_voxel_coords[0])) + 1
        del mask_voxel_coords
        bbox = [[min(minzidx, minzidx1), max(maxzidx, maxzidx1)], [min(minxidx, minxidx1), max(maxxidx, maxxidx1)], [min(minyidx, minyidx1), max(maxyidx, maxyidx1)]]
    else:
        mask_voxel_coords = np.where(nonzero_mask != outside_value)
        del nonzero_mask
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        del mask_voxel_coords
        bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    return bbox

def resample_data(ori_array, ori_spacing,
                  target_spacing=None, only_z=False):
    # shape c w h d
    # spacing_nnunet = [1.8532123022052305, 1.512973664256994, 1.512973664256994]
    print("进入rasample")
    if target_spacing is None:
        target_spacing = [2.2838839, 1.8709113, 1.8709113]
    if only_z:
        target_spacing = [target_spacing[0], ori_spacing[0], ori_spacing[1]]
    ori_shape = ori_array[0].shape
    #target_shape = [ori_spacing[i] * ori_shape[i] / target_spacing[i] // 1 for i in range(len(ori_shape))]    #原始

    reshaped_data = []
    data = []
    ori_array = ori_array[0]

    if ori_shape[0] > 1000:
        print("ddddddddddddddddddddddddddd")
        chunk = ori_array[:math.floor(ori_shape[0] / 2), :, :]
        chunk1 = ori_array[math.floor(ori_shape[0] / 2):, :, :]
        del ori_array
        chunk = chunk.astype(np.float32)
        #chunk1 = chunk1.astype(np.float32)
        chunk_shape = chunk.shape
        target_shape = [int(ori_spacing[len(ori_shape)-i- 1] * chunk_shape[i] / target_spacing[i] // 1) for i in range(len(ori_shape))]
        ##################3

        data_torch = torch.tensor(chunk).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        del chunk
        resized_data = F.interpolate(data_torch, size=target_shape, mode='trilinear', align_corners=False)
        del data_torch

        resized_data = resized_data.squeeze(0).cpu().numpy()
        reshaped_data.append(resized_data)
        del resized_data

        #reshaped_data.append((resize(chunk, target_shape, order=3, preserve_range=True)[None]).astype(np.float32))

        #chunk = ori_array[math.floor(ori_shape[0] / 2):, :, :]
        chunk_shape = chunk1.shape
        chunk1 = chunk1.astype(np.float32)
        target_shape = [int(ori_spacing[len(ori_shape) - i - 1] * chunk_shape[i] / target_spacing[i] // 1) for i in
                        range(len(ori_shape))]
        ##########################

        data_torch = torch.tensor(chunk1).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        del chunk1
        resized_data = F.interpolate(data_torch, size=target_shape, mode='trilinear', align_corners=False)
        del data_torch
        resized_data = resized_data.squeeze(0).cpu().numpy()
        reshaped_data.append(resized_data)
        del resized_data

        #reshaped_data.append((resize(chunk1, target_shape, order=3, preserve_range=True)[None]).astype(np.float32))

        reshaped_data = np.concatenate((reshaped_data[0], reshaped_data[1]), axis=1)
        data.append(reshaped_data)
        reshaped_data = data
        del data
    else:
        print("ffffffffffffffffff")
        target_shape = [int(ori_spacing[len(ori_shape) - i - 1] * ori_shape[i] / target_spacing[i] // 1) for i in
                        range(len(ori_shape))]  # lyy  师兄修改后
        ########################
        ori_array = ori_array.astype(np.float32)
        data_torch = torch.tensor(ori_array).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        del ori_array
        resized_data = F.interpolate(data_torch, size=target_shape, mode='trilinear', align_corners=False)
        del data_torch
        resized_data = resized_data.squeeze(0).cpu().numpy().astype(np.float32)
        reshaped_data.append(resized_data)
        del resized_data


        #reshaped_data.append((resize(ori_array, target_shape, order=3, preserve_range=True)[None]).astype(np.float32))
        #del ori_array
    #reshaped_data.append(resize(ori_array[0], target_shape, order=3, preserve_range=True)[None])
    #reshaped_data.append(resize(ori_array[1], target_shape, order=0, preserve_range=True,
    #                            anti_aliasing=False)[None])
    #reshaped_data.append(resize(ori_array[-1], target_shape, order=0, preserve_range=True,
    #                            anti_aliasing=False)[None])
    print("出rasample")
    return np.vstack(reshaped_data), target_spacing




'''
class MuiltThreadDataGenerator(object):
    def __init__(self, iter, produce_queue_number,save_path) -> None:
        self.iter = iter
        self.produce_queue_number = produce_queue_number
        self.output_queue = queue.Queue(1)
        self.save_path = save_path

    def _ini(self):
        process_thread = threading.Thread(target=self.process_data_thread)
        process_thread.start()
        ###########################
        import psutil
        import os
        pid = os.getpid()
        process = psutil.Process(pid)
        # 获取虚拟内存使用量（单位：字节）
        vms = process.memory_info().vms
        # 获取物理内存使用量（单位：字节）
        rss = process.memory_info().rss
        print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
        print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
        ###############################

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
        identity = nii_path.split('/')[-1].split('_0000.nii.gz')[0]  # pre

        image = sitk.ReadImage(nii_path)
        ori_spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        ori_size = np.array(image.GetSize())
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)

        del image
        image_dtype = image_array.dtype
        # nonzero_mask = np.zeros(image_array.shape, dtype=bool)
        # nonzero_mask = nonzero_mask | (image_array > 0)
        # nonzero_mask = binary_fill_holes(nonzero_mask)
        predict_final_shape = image_array.shape
        print(predict_final_shape)
        bbox = get_bbox_from_mask_b(image_array > 0, 0)
        if predict_final_shape[0] > 1000:
            bbox[0][0] = 500
        # del nonzero_mask
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        # cropped_data = np.stack((image_array[resizer], np.zeros_like(image_array[resizer])), 0)

        cropped_data = np.stack((image_array[resizer]), 0)
        cropped_data = cropped_data.reshape(1, cropped_data.shape[0], cropped_data.shape[1], cropped_data.shape[2])
        del image_array
        cropped_shape = cropped_data.shape[1:]
        # resample array
        # target_spacing = [2.5, 0.859375, 0.859375]
        # target_spacing = [2.13021975, 1.66839451, 1.66839451]
        target_spacing = [2.2838839, 1.8709113, 1.8709113]
        # target_spacing = [5, 4, 4]
        # target_spacing = [3.0, 2.34960903, 2.34960903]
        # target_spacing = [1., 0.78320301, 0.78320301]

        resampled_data, _ = resample_data(cropped_data, np.array(ori_spacing), target_spacing=target_spacing,
                                          only_z=False)
        print("resampled_data")
        del cropped_data
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
        del ct_array
        ori = torch.from_numpy(resampled_data[0:1])
        del resampled_data
        print('identity: %s, preprocessing done.' % identity)
        data = [ori, identity, origin, direction, ori_size, image_dtype, predict_final_shape, resizer, cropped_shape,
                ori_spacing]

        self.output_queue.put(data, block=True)


    # 第二部分的模型预测函数
    def predict_model(self):
        end_count = 0.
        import psutil

        # 获取当前进程的内存使用情况
        mem = psutil.Process().memory_info()
        print(f"predict_model Memory Usage: {mem.rss / 1024 ** 2:.2f} MB")  # 以MB为单位显示
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
                # nii_path = '/home/ljc/code/FLARE2023/data/Task098_FLARE2023/imagesTs/FLARE23Ts_0001.nii.gz'
                ori, identity, origin, direction, ori_size, image_dtype, predict_final_shape, resizer, cropped_shape, ori_spacing = data
                #torch.cuda.empty_cache()
                del data
                ori = ori.float()
                # predict
                #s = Valid_utils(14, patch_size)

                predict, shape, small_resizer, do_crop, slice_xyz = s.predict_3D(ori, do_mirror=True,
                                                                                 mirror_axes=(0, 1, 2))
                ################################
                import psutil
                import os
                pid = os.getpid()
                process = psutil.Process(pid)
                # 获取虚拟内存使用量（单位：字节）
                vms = process.memory_info().vms
                # 获取物理内存使用量（单位：字节）
                rss = process.memory_info().rss
                print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
                print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
                #####################################

                del ori
                predict = predict.softmax(0).argmax(0).detach().cpu().squeeze().numpy()
                if do_crop:
                    predict_small = np.zeros(shape, dtype=np.uint8)
                    small_resizer = small_resizer[1:]
                    predict_small[small_resizer] = predict
                    predict = predict_small
                    del predict_small
                    slice_xyz = slice_xyz[1:]

                    predict = predict[slice_xyz]

                    # predict = max_compoment(predict)
                    # re resample
                predict_final = np.zeros(predict_final_shape, dtype=np.uint8)
                predict_resample = resize(predict, cropped_shape, order=0, preserve_range=True, anti_aliasing=False)
                del predict
                predict_final[resizer] = predict_resample
                del predict_resample
                after_itk_label = sitk.GetImageFromArray(predict_final.astype(np.uint8))
                del predict_final
                after_itk_label.SetSpacing(ori_spacing)
                after_itk_label.SetOrigin(origin)
                after_itk_label.SetDirection(direction)
                if not os.path.exists(save_path):
                    os.makedirs(os.path.join(save_path))
                #dd = self.save_path
                save_name = os.path.join(self.save_path, '%s.nii.gz' % identity)
                sitk.WriteImage(after_itk_label, save_name)
                del after_itk_label

                print('%s : cost %d s' % (identity, (time.time() - start)))
            except queue.Empty:
                time.sleep(1)
                pass
        return
'''



def preprocess(nii_path):
    identity = nii_path.split('/')[-1].split('_0000.nii.gz')[0]  # pre
    #ss = nii_path.replace('\\', '/')
    #identity = ss.split('/')[-1].split('_0000.nii.gz')[0]
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    ################################
    import psutil
    import os
    pid = os.getpid()
    process = psutil.Process(pid)
    # 获取虚拟内存使用量（单位：字节）
    vms = process.memory_info().vms
    # 获取物理内存使用量（单位：字节）
    rss = process.memory_info().rss
    print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
    print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
    #####################################

    image = sitk.ReadImage(nii_path)
    ori_spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", direction)
    ori_size = np.array(image.GetSize())
    #return

    print("00000000000000000000000000000000000000000000")
    ################################
    import psutil
    import os
    pid = os.getpid()
    process = psutil.Process(pid)
    # 获取虚拟内存使用量（单位：字节）
    vms = process.memory_info().vms
    # 获取物理内存使用量（单位：字节）
    rss = process.memory_info().rss
    print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
    print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
    #####################################
    if ori_size[2] > 1000:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++大于1000的++++++++++++++++++++++++++++++++++++++++")
        del image
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))

    else:
        image_array = sitk.GetArrayFromImage(image)
        del image
    ################################
    import psutil
    import os
    pid = os.getpid()
    process = psutil.Process(pid)
    # 获取虚拟内存使用量（单位：字节）
    vms = process.memory_info().vms
    # 获取物理内存使用量（单位：字节）
    rss = process.memory_info().rss
    print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
    print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
    #####################################


    image_dtype = image_array.dtype
    # nonzero_mask = np.zeros(image_array.shape, dtype=bool)
    # nonzero_mask = nonzero_mask | (image_array > 0)
    # nonzero_mask = binary_fill_holes(nonzero_mask)
    predict_final_shape = image_array.shape
    print(predict_final_shape)
    if predict_final_shape[0] > 1000:
        resizer = (slice(200, predict_final_shape[0]), slice(10, predict_final_shape[1] - 10), slice(0, predict_final_shape[2]))
    else:

        #pp = image_array > 50
        #pp = keep_largest_connected_component(pp)
        bbox = get_bbox_from_mask_b(image_array > 0, outside_value=0)
        #if predict_final_shape[0] > 1000:
        #    bbox[0][0] = 400
        # del nonzero_mask
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        #cropped_data = np.stack((image_array[resizer], np.zeros_like(image_array[resizer])), 0)

    cropped_data = np.stack((image_array[resizer]), 0)
    cropped_data = cropped_data.reshape(1,cropped_data.shape[0], cropped_data.shape[1], cropped_data.shape[2])
    del image_array
    cropped_shape = cropped_data.shape[1:]
    # resample array
    # target_spacing = [2.5, 0.859375, 0.859375]
    #target_spacing = [2.13021975, 1.66839451, 1.66839451]
    target_spacing = [2.2838839, 1.8709113, 1.8709113]
    # target_spacing = [5, 4, 4]
    # target_spacing = [3.0, 2.34960903, 2.34960903]
    # target_spacing = [1., 0.78320301, 0.78320301]

    resampled_data, _ = resample_data(cropped_data, np.array(ori_spacing), target_spacing=target_spacing,
                                      only_z=False)
    print("resampled_data")
    del cropped_data
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
    del ct_array
    ori = torch.from_numpy(resampled_data[0:1])
    del resampled_data
    print('identity: %s, preprocessing done.' % identity)
    data = [ori, identity, origin, direction, ori_size, image_dtype, predict_final_shape, resizer, cropped_shape,
            ori_spacing]
    del ori
    return data

'''
def inference():
    file_list = os.listdir(base_folder_name)
    file_list = [os.path.join(base_folder_name, file_) for file_ in file_list]

    DG = MuiltThreadDataGenerator(file_list, 1, save_path)

    with torch.no_grad():
        DG._ini()
'''



if __name__ == '__main__':
    ################################
    import psutil
    import os

    pid = os.getpid()
    process = psutil.Process(pid)
    # 获取虚拟内存使用量（单位：字节）
    vms = process.memory_info().vms
    # 获取物理内存使用量（单位：字节）
    rss = process.memory_info().rss
    print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
    print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
    #####################################
    argps = argparse.ArgumentParser()
    # -PSEUDO_LABEL_PATH --NNUNET_npy_PATH
    argps.add_argument('input_folder', type=str, help='input folder')
    argps.add_argument('output_folder', type=str,
                       help='output folder')
    argps.add_argument('model_path', type=str,
                       help='model_path')


    arg_s = argps.parse_args()
    base_folder_name = arg_s.input_folder
    save_path = arg_s.output_folder
    model_path = arg_s.model_path
    #patch_size = [96, 128, 160]
    patch_size = (80, 160, 160)
    #patch_size = (40, 224, 192)
    #patch_size = (80, 160, 160)
    #model = my_Trainner().get_network(
    #    os.path.join(checkpoint, 'model_final_checkpoint.model'))
    #model.cuda()
    #model.to('cpu')
    #model.eval()
    time_start = time.time()
    #s = Valid_utils(14, patch_size)


    #inference()

    ################################
    import psutil
    import os

    pid = os.getpid()
    process = psutil.Process(pid)
    # 获取虚拟内存使用量（单位：字节）
    vms = process.memory_info().vms
    # 获取物理内存使用量（单位：字节）
    rss = process.memory_info().rss
    print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
    print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
    #####################################

    file_list = os.listdir(base_folder_name)
    file_list = [os.path.join(base_folder_name, file_) for file_ in file_list]
    for nii_path in file_list:
        gc.collect()
        print("for start!!!!!!!!!!!!!!!!!!!!!!!")
        ################################
        import psutil
        import os

        pid = os.getpid()
        process = psutil.Process(pid)
        # 获取虚拟内存使用量（单位：字节）
        vms = process.memory_info().vms
        # 获取物理内存使用量（单位：字节）
        rss = process.memory_info().rss
        print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
        print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
        #####################################

        a_time = time.time()
        data = preprocess(nii_path)
        #continue
        print("数据预处理",time.time() - a_time)
        ori, identity, origin, direction, ori_size, image_dtype, predict_final_shape, resizer, cropped_shape, ori_spacing = data
        del data
        ori = ori.float()

        ################################
        import psutil
        import os

        pid = os.getpid()
        process = psutil.Process(pid)
        # 获取虚拟内存使用量（单位：字节）
        vms = process.memory_info().vms
        # 获取物理内存使用量（单位：字节）
        rss = process.memory_info().rss
        print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
        print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
        #####################################
        # predict
        tt = time.time()
        s = Valid_utils(14, patch_size, model_path)
        print("load model", time.time() - tt)
        print("add model &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        ################################
        import psutil
        import os

        pid = os.getpid()
        process = psutil.Process(pid)
        # 获取虚拟内存使用量（单位：字节）
        vms = process.memory_info().vms
        # 获取物理内存使用量（单位：字节）
        rss = process.memory_info().rss
        print(f"虚拟内存使用量: {vms / (1024 ** 2)} MB")
        print(f"物理内存使用量: {rss / (1024 ** 2)} MB")
        #####################################

        predict, shape, small_resizer, do_crop, slice_xyz = s.predict_3D(ori, do_mirror=True,
                                                                         mirror_axes=(0, 1, 2))
        del s
        del ori
        predict = predict.to(torch.float32)
        predict = predict.softmax(0).argmax(0).detach().cpu().squeeze().numpy()
        s_start = time.time()
        #predict = max_compoment(predict)
        print("保留最大连通域", time.time() - s_start)
        #if do_crop:
        predict_small = np.zeros(shape, dtype=np.uint8)
        small_resizer = small_resizer[1:]
        predict_small[small_resizer] = predict
        predict = predict_small
        del predict_small
        slice_xyz = slice_xyz[1:]

        predict = predict[slice_xyz]

            # predict = max_compoment(predict)
            # re resample
        predict_final = np.zeros(predict_final_shape, dtype=np.uint8)
        predict_resample = resize(predict, cropped_shape, order=0, preserve_range=True, anti_aliasing=False)
        del predict
        predict_final[resizer] = predict_resample
        del predict_resample
        after_itk_label = sitk.GetImageFromArray(predict_final.astype(np.uint8))
        del predict_final
        after_itk_label.SetSpacing(ori_spacing)
        after_itk_label.SetOrigin(origin)
        after_itk_label.SetDirection(direction)
        if not os.path.exists(save_path):
            os.makedirs(os.path.join(save_path))
        # dd = self.save_path
        save_name = os.path.join(save_path, '%s.nii.gz' % identity)
        sitk.WriteImage(after_itk_label, save_name)
        del after_itk_label

        gc.collect()
        print("cost",time.time() - a_time)




    print("finally cost", time.time() - time_start)
    pass


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
