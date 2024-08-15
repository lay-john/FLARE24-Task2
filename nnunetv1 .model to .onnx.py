import torch
from torch import nn
import pickle

from nnunet.network_architecture.mednextv1.MedNextV1 import MedNeXt_2, MedNeXt_3
from nnunet.training.model_restore import load_model_and_checkpoint_files
import os
import numpy as np

from concurrent import futures

from nnunet.network_architecture.generic_UNet import Generic_UNet, SeparableConv3d
from data_convert import load_pickle, resize, get_bbox_from_mask_b, resample_data
import SimpleITK as sitk

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
        plans = load_pickle(os.path.join(checkpoint, 'nnUNetPlansv2.1_plans_3D.pkl'))
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
        self.conv_per_stage = 2
        self.stage_num = 4
        self.base_num_features = 16
        self.max_num_features = 256
        self.max_num_epochs = 500

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]
        '''

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
        #self.conv_per_stage = 3
        #self.stage_num = 5
        #if len(self.net_conv_kernel_sizes) > self.stage_num:
        #  self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[len(self.net_conv_kernel_sizes) - self.stage_num:]
        #  self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[len(self.net_num_pool_op_kernel_sizes) - (self.stage_num - 1):]

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        #self.conv_per_stage = 1
        self.base_num_features = 16
        self.net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        self.net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 2, 2]]
        self.network = Generic_UNet(1, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)


        '''
        self.network = MedNeXt_3(
            in_channels=1,
            n_channels=16,
            n_classes=14,
            exp_r=[2, 2, 2, 2, 2, 2, 2, 2, 2],  # Expansion ratio as in Swin Transformers
            # exp_r = 2,
            kernel_size=3,  # Can test kernel_size
            deep_supervision=True,  # Can be used to test deep supervision
            do_res=True,  # Can be used to individually test residual connection
            do_res_up_down=False,
            # block_counts = [2,2,2,2,2,2,2,2,2],
            block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            checkpoint_style=None,
            dim='3d',
            grn=True

        )
        '''

        '''
        self.network = MedNeXt_2(
            in_channels = 1,
            n_channels = 32,
            n_classes = 14,
            exp_r=[2,2,2,2,2,2,2,2,2],         # Expansion ratio as in Swin Transformers
            # exp_r = 2,
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = False,
            # block_counts = [2,2,2,2,2,2,2,2,2],
            block_counts = [1,1,1,1,1,1,1,1,1],
            checkpoint_style = None,
            dim = '3d',
            grn=True

        )
        '''


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


class my_Trainner_pre():
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
        plans = load_pickle(os.path.join(checkpoint, 'nnUNetPlansv2.1_plans_3D.pkl'))
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

        #self.conv_per_stage = 1
        #self.stage_num = 4
        #self.base_num_features = 16
        #self.max_num_features = 256

        #if len(self.net_conv_kernel_sizes) > self.stage_num:
        #    self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
        #    self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        #self.conv_per_stage = 1

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


if __name__ == "__main__":
    # convert .model to .onnx
    argps = argparse.ArgumentParser()
    # -PSEUDO_LABEL_PATH --NNUNET_npy_PATH
    argps.add_argument('save_path', type=str, help='save path')


    arg_s = argps.parse_args()
    save_path = arg_s.save_path



    checkpoint = os.path.join(os.path.dirname(__file__), "checkpoints")
    patch_size = [80, 160, 160]
    model = my_Trainner().get_network(
        os.path.join(checkpoint, 'model_final_checkpoint.model'))


    #net = trainer.network
    #net.to('cpu')




    #checkpoint = torch.load(os.path.join(Model_path, folds, checkpoint_name + ".model"))

    #net.load_state_dict(checkpoint['state_dict'])
    net = model
    net.to('cpu')
    net.eval()
    for module in net.modules():
        if hasattr(module, 'training'):
            module.training = False
    # (1,1,10,20,30)是我i任意写的，你要按照自己的输入数组维度更换
    dummy_input = torch.randn(1, 1, 80, 160, 160, requires_grad=True)
    output = net(dummy_input)
    torch.onnx.export(net,  # 模型的名称
                      dummy_input,  # 一组实例化输入
                      save_path,  # 文件保存路径/名称
                      export_params=True,  # 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                      opset_version=10,  # ONNX 算子集的版本，当前已更新到15
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=['input'],  # 输入模型的张量的名称
                      output_names=['output'],  # 输出模型的张量的名称
                      # dynamic_axes将batch_size的维度指定为动态，
                      # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})