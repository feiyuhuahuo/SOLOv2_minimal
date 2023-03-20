#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb
import os
import sys
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
sys.path.remove(f'{root_path}/utils')

from data_loader.dataset import CustomIns_separate
import pycocotools.mask as mask_util
from configs import *
import json

cfg = Custom_light_res50('ann')
dataset = CustomIns_separate(cfg)

gts = {'img_ids': dataset.img_ids(), 'cate_ids': dataset.category_ids(), 'class_names': cfg.class_names, 'gts': []}
gt_id = 1  # 不能是转bool为False的值
num = len(dataset)
for i, one in enumerate(dataset):
    for j in range(one[1].shape[0]):
        one_gt = {}
        one_gt['image_id'] = i
        one_gt['id'] = gt_id
        one_gt['iscrowd'] = 0
        one_gt['category_id'] = int(one[1][j])
        x1, y1, x2, y2 = list(one[2][j])
        one_gt['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        mask = one[3][:, :, j]
        rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        counts = rle['counts'].decode()
        rle['counts'] = counts
        one_gt['segmentation'] = rle
        one_gt['area'] = int(mask.sum())

        gts['gts'].append(one_gt)
        gt_id += 1

    print(f'\rparsing: {i + 1}/{num}', end='')

file_path = 'custom_annotations.json'
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(gts, f, sort_keys=False, indent=4)

print()
print(f'Dumped as {file_path}.')
