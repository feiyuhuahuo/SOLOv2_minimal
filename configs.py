#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb
from data_loader.dataset import CocoIns, CustomIns_separate
from data_loader.augmentations import TrainAug, ValAug
from utils.utils import COCO_CLASSES
from utils.onnx_trans import *

TrainBatchSize = 10


class Solov2_res50:
    def __init__(self, mode):
        self.mode = mode
        self.dataset = CocoIns
        self.data_root = '/home/feiyu/Data/coco2017/'
        self.train_imgs = self.data_root + 'train2017/'
        self.train_ann = self.data_root + 'annotations/instances_train2017.json'
        self.val_imgs = self.data_root + 'val2017/'
        self.val_ann = self.data_root + 'annotations/instances_val2017.json'
        self.pretrained = 'weights/backbone_resnet50.pth'
        self.break_weight = ''
        self.resnet_depth = 50
        self.fpn_in_c = [256, 512, 1024, 2048]
        self.class_names = COCO_CLASSES
        self.num_classes = len(self.class_names) + 1
        self.head_stacked_convs = 4
        self.head_seg_feat_c = 512
        self.head_scale_ranges = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
        self.head_ins_out_c = 256
        self.mask_feat_num_classes = 256
        self.epochs = 36
        self.train_bs = TrainBatchSize
        self.lr = 0.01 * (self.train_bs / 16)
        self.warm_up_init = self.lr * 0.01
        self.warm_up_iters = int(500 * (16 / self.train_bs))
        self.lr_decay_steps = (27, 33)
        self.train_aug = TrainAug(img_scale=[(1333, 800), (1333, 768), (1333, 736),
                                             (1333, 704), (1333, 672), (1333, 640)])
        self.train_workers = 8
        self.val_interval = 1
        self.start_save = 0

        self.val_weight = 'weights/Solov2_res50_36.pth'
        self.val_bs = 1
        self.val_aug = ValAug(img_scale=[(1333, 800)])
        self.val_num = -1
        self.postprocess_para = {'nms_pre': 500, 'score_thr': 0.1, 'mask_thr': 0.5, 'update_thr': 0.05}

        if self.mode in ('detect', 'onnx'):
            self.postprocess_para['update_thr'] = 0.3  # for detect score threshold
        self.detect_images = 'detect_imgs'
        self.detect_mode = 'contour'

        self.onnx_trans = None
        self.onnx_shape = None

    def print_cfg(self):
        print()
        title = '-' * 30 + self.__class__.__name__ + '-' * 30
        print(f'\033[0;35m{title}\033[0m')

        for k, v in vars(self).items():
            line = f'{k}: {v}'
            if self.mode == 'train':
                if k in ('break_weight', 'num_classes', 'epochs', 'train_bs', 'lr', 'val_interval', 'start_save'):
                    line = f'\033[0;35m{line}\033[0m'
            elif self.mode == 'val':
                if k in ('num_classes', 'val_weight'):
                    line = f'\033[0;35m{line}\033[0m'
            elif self.mode == 'detect':
                if k in ('num_classes', 'detect_images', 'detect_score_thr', 'detect_mode'):
                    line = f'\033[0;35m{line}\033[0m'

            print(line)
        print()

    def name(self):
        return self.__class__.__name__

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'val'


class Solov2_light_res50(Solov2_res50):
    def __init__(self, mode):
        super().__init__(mode)
        self.head_stacked_convs = 2
        self.head_seg_feat_c = 256
        self.head_ins_out_c = 128
        self.head_scale_ranges = ((1, 56), (28, 112), (56, 224), (112, 448), (224, 896))
        self.mask_feat_num_classes = 128
        self.train_aug = TrainAug(img_scale=[(768, 512), (768, 480), (768, 448),
                                             (768, 416), (768, 384), (768, 352)])
        self.val_weight = 'weights/Solov2_light_res50_36.pth'
        self.val_aug = ValAug(img_scale=[(768, 448)])


class Solov2_light_res34(Solov2_res50):
    def __init__(self, mode):
        super().__init__(mode)
        self.resnet_depth = 34
        self.pretrained = 'weights/backbone_resnet34.pth'
        self.break_weight = 'weights/Solov2_light_res34_34.pth'
        self.fpn_in_c = [64, 128, 256, 512]
        self.head_stacked_convs = 2
        self.head_seg_feat_c = 256
        self.head_ins_out_c = 128
        self.head_scale_ranges = ((1, 56), (28, 112), (56, 224), (112, 448), (224, 896))
        self.mask_feat_num_classes = 128
        self.train_aug = TrainAug(img_scale=[(768, 512), (768, 480), (768, 448),
                                             (768, 416), (768, 384), (768, 352)])
        self.val_weight = 'weights/Solov2_light_res34_36.pth'
        self.val_aug = ValAug(img_scale=[(768, 448)])


class Custom_light_res50(Solov2_light_res50):
    def __init__(self, mode):
        super().__init__(mode)
        self.dataset = CustomIns_separate
        self.data_root = '/home/feiyu/Data/nanjiao/nanjiao_seg/语义分割/'
        self.train_imgs = self.data_root + 'imgs/train'
        self.train_ann = self.data_root + 'labels/train'
        self.val_imgs = self.data_root + 'imgs/val'
        self.val_ann = self.data_root + 'labels/val'
        self.class_names = ('断栅', '黑斑', '边缘黑斑', '黑边', '划伤', '气流印', '重测',
                            '吸盘印', '隐裂', '雾状发黑', '同心圆', '块状黑斑')
        # self.class_names = ('ds', 'hb', 'byhb', 'h_bian', 'hs', 'qly', 'cc',
        #                     'xpy', 'yl', 'wzfh', 'txy', 'kzhb')
        self.num_classes = len(self.class_names) + 1
        self.epochs = 400
        self.warm_up_iters = 500  # bs=16时，基本就是500左右
        self.lr_decay_steps = (300, 360)
        self.start_save = 100
        self.val_interval = 20
        self.train_aug = TrainAug(mean=[127., 127., 127], std=[60., 60., 60],
                                  img_scale=[(576, 576), (544, 544),
                                             (512, 512), (480, 480), (448, 448)],
                                  v_flip=True)
        self.break_weight = ''

        self.val_aug = ValAug(mean=[127., 127., 127], std=[60., 60., 60],
                              img_scale=[(512, 512)])
        self.val_weight = 'weights/Custom_light_res50_340.pth'
        self.detect_images = self.val_imgs  # '/home/feiyu/Data/nanjiao/nanjiao_seg/语义分割/bgs'
        if self.mode in ('detect', 'onnx'):
            self.postprocess_para['update_thr'] = 0.3
        self.postprocess_para['mask_thr'] = 0.5

        self.onnx_trans = base_input_trans
        self.onnx_shape = (512, 512)


# todo: 调整head_scale_ranges参数
# todo： 随机padding
# todo: 可视化时用类相关的score？ 用MPP决定
