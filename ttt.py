#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob
import pdb

import cv2
import numpy as np
import shutil
import os
import json

jsons = glob.glob('/home/feiyu/Data/nanjiao/nanjiao_seg/语义分割/labels/val/*.json')
i = 0
for one in jsons:
    name = one.split('/')[-1]
    with open(one, 'r', encoding='gbk') as f:
        content = json.load(f)

    with open(f'/home/feiyu/Data/nanjiao/nanjiao_seg/语义分割/labels/val2/{name}', 'w', encoding='utf-8') as ff:
        json.dump(content, ff, indent=4)



def batch_collator_random_pad(batch):
    batch = complement_batch(batch)
    img_batch, labels_batch, bboxes_batch, masks_batch = list(zip(*batch))
    max_size = tuple(max(s) for s in zip(*[img.shape for img in img_batch]))
    assert max_size[0] % 32 == 0 and max_size[1] % 32 == 0, 'shape error in batch_collator'

    batch_shape = (len(img_batch), max_size[0], max_size[1], max_size[2])
    pad_imgs_batch = np.zeros(batch_shape, dtype=img_batch[0].dtype)

    pad_labels_batch, pad_bboxes_batch, pad_masks_batch = [], [], []
    for i, img in enumerate(img_batch):
        ori_h, ori_w = img.shape[:2]
        new_h = random.randint(0, (max_size[0] - ori_h))
        new_w = random.randint(0, (max_size[1] - ori_w))

        pad_imgs_batch[i, new_h:new_h + ori_h, new_w:new_w + ori_w, :] = img

        pad_labels_batch.append(torch.tensor(labels_batch[i], dtype=torch.int64))

        ori_bboxes = bboxes_batch[i]
        pad_bboxes = ori_bboxes.copy()
        pad_bboxes[:, [0, 2]] = pad_bboxes[:, [0, 2]] + new_w
        pad_bboxes[:, [1, 3]] = pad_bboxes[:, [1, 3]] + new_h
        pad_bboxes_batch.append(torch.tensor(pad_bboxes, dtype=torch.float32))

        ori_masks = masks_batch[i]
        pad_masks = np.zeros((batch_shape[1], batch_shape[2], ori_masks.shape[2]), dtype='uint8')
        pad_masks[new_h:new_h + ori_h, new_w:new_w + ori_w, :] = ori_masks
        # pad_masks_batch.append(pad_masks.transpose(2, 0, 1).astype('uint8'))
        pad_masks_batch.append(torch.tensor(pad_masks.transpose(2, 0, 1), dtype=torch.uint8))

    pad_imgs_batch = pad_imgs_batch.transpose(0, 3, 1, 2)
    return torch.tensor(pad_imgs_batch, dtype=torch.float32), pad_labels_batch, pad_bboxes_batch, pad_masks_batch