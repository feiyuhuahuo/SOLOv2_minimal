#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os.path as osp
import pdb
import torch.utils.data as data
import cv2
import glob
import numpy as np
import json

from pycocotools.coco import COCO
from utils.utils import CocoIdMap, get_seg_mask


# Warning, do not use numpy random in PyTorch multiprocessing, or the random result will be the same.


class CocoIns(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.mode

        if cfg.mode in ('train', 'val'):
            self.image_folder = cfg.train_imgs if cfg.mode == 'train' else cfg.val_imgs
            self.coco = COCO(cfg.train_ann if cfg.mode == 'train' else cfg.val_ann)
            self.ids = list(self.coco.imgToAnns.keys())
        elif cfg.mode == 'detect':
            self.image_folder = glob.glob(f'{cfg.detect_images}/*')
            self.image_folder = [one for one in self.image_folder if one[-3:] in ('bmp', 'jpg', 'png')]
            self.image_folder.sort()

        self.cate_ids = {v: k for k, v in CocoIdMap.items()}

    def __getitem__(self, index):
        if self.mode == 'detect':
            img_path = self.image_folder[index]
            img_origin = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            img, resize_shape = self.cfg.val_aug(img_origin)
            img_show = cv2.resize(img_origin, (resize_shape[1], resize_shape[0]))
            return img, resize_shape, img_path.split('/')[-1], img_show
        else:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
            target = [aa for aa in target if not aa['iscrowd']]

            file_name = self.coco.loadImgs(img_id)[0]['file_name']

            img_path = osp.join(self.image_folder, file_name)
            assert osp.exists(img_path), f'Image path does not exist: {img_path}'

            img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            height, width, _ = img.shape

            assert len(target) > 0, 'No annotation in this image!'
            box_list, mask_list, label_list = [], [], []

            for aa in target:
                bbox = aa['bbox']

                # When training, some boxes are wrong, ignore them.
                if self.mode == 'train':
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 4 or bbox[3] < 4:
                        continue

                category = CocoIdMap[aa['category_id']]
                label_list.append(category)
                x1y1x2y2_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                box_list.append(x1y1x2y2_box)
                mask_list.append(self.coco.annToMask(aa))

            if len(label_list) > 0:
                labels = np.array(label_list)
                bboxes = np.array(box_list)
                masks = np.stack(mask_list, axis=2)  # (h, w, num)

                assert masks.shape == (height, width, labels.shape[0]), 'Unmatched annotations.'

                if self.mode == 'train':
                    img, bboxes, masks = self.cfg.train_aug(img, bboxes, masks)
                    return img, labels, bboxes, masks
                elif self.mode == 'val':
                    img, resize_shape = self.cfg.val_aug(img)
                    return img, (height, width), resize_shape, file_name
            else:
                if self.mode == 'val':
                    raise RuntimeError('Error, no valid object in this image.')
                else:
                    print(f'No valid object in image "img_id: {img_id}". Use a repeated image in this batch.')
                    return None, None, None, None

    def get_aspect_ratios(self):
        aspect_ratios = []
        for one in self.ids:
            img_info = self.coco.loadImgs(one)[0]
            aspect_ratios.append(float(img_info['height']) / float(img_info['width']))
        return aspect_ratios

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        elif self.mode == 'val':
            return len(self.ids) if self.cfg.val_num == -1 else min(self.cfg.val_num, len(self.ids))
        elif self.mode == 'detect':
            return len(self.image_folder)


class CustomIns_separate(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.mode

        if cfg.mode in ('train', 'val', 'ann'):
            self.image_folder = cfg.train_imgs if cfg.mode == 'train' else cfg.val_imgs
            self.label_folder = cfg.train_ann if cfg.mode == 'train' else cfg.val_ann
            self.imgs = glob.glob(f'{self.image_folder}/*')
        elif cfg.mode == 'detect':
            self.imgs = glob.glob(f'{cfg.detect_images}/*')

        self.imgs = [one for one in self.imgs if one[-3:] in ('bmp', 'jpg', 'png')]
        self.imgs.sort()
        self.ids = list(range(len(self.imgs)))

    def __getitem__(self, index):
        if self.mode == 'detect':
            img_path = self.imgs[index]
            img_origin = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            img, resize_shape = self.cfg.val_aug(img_origin)
            img_show = cv2.resize(img_origin, (resize_shape[1], resize_shape[0]))
            return img, resize_shape, img_path.split('/')[-1], img_show
        else:
            img_path = self.imgs[self.ids[index]]
            img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            img_name = img_path.split('/')[-1]
            label = f'{self.label_folder}/{img_name[:-4]}.json'
            with open(label, 'r', encoding='gbk') as f:
                content = json.load(f)

            polygons, height, width = content['polygons'], content['img_height'], content['img_width']
            assert img.shape[:2] == (height, width), 'image shape error!'

            category = [self.cfg.class_names.index(one['category']) + 1 for one in polygons]
            mask_list = get_seg_mask(self.cfg.class_names, polygons, height, width)

            bboxes = []
            for one in mask_list:
                hs, ws = np.where(one > 0)
                x1, x2 = float(ws.min()), float(ws.max())
                y1, y2 = float(hs.min()), float(hs.max())
                bboxes.append([x1, y1, x2, y2])

            if len(category) > 0:
                labels = np.array(category)
                bboxes = np.array(bboxes)
                masks = np.stack(mask_list, axis=2)  # (h, w, num)

                assert masks.shape == (height, width, labels.shape[0]), 'Unmatched annotations.'

                if self.mode == 'ann':
                    return img, labels, bboxes, masks
                elif self.mode == 'train':
                    img, bboxes, masks = self.cfg.train_aug(img, bboxes, masks)
                    return img, labels, bboxes, masks
                elif self.mode == 'val':
                    img, resize_shape = self.cfg.val_aug(img)
                    return img, (height, width), resize_shape, img_name
            else:
                if self.mode == 'val':
                    raise RuntimeError('Error, no valid object in this image.')
                else:
                    print(f'No valid object in image: "{img_name}". Use a repeated image in this batch.')
                    return None, None, None, None

    def get_aspect_ratios(self):
        aspect_ratios = []
        for one in self.ids:
            img_path = self.imgs[one]
            img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            aspect_ratios.append(float(h) / float(w))
        return aspect_ratios

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        elif self.mode in ('val', 'ann'):
            return len(self.ids) if self.cfg.val_num == -1 else min(self.cfg.val_num, len(self.ids))
        elif self.mode == 'detect':
            return len(self.imgs)

    def img_ids(self):
        return self.ids

    def category_ids(self):
        return list(range(1, self.cfg.num_classes))


class CustomIns_onefile(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.mode

        if cfg.mode in ('train', 'val', 'ann'):
            self.image_folder = cfg.train_imgs if cfg.mode == 'train' else cfg.val_imgs
            self.label_folder = cfg.train_ann if cfg.mode == 'train' else cfg.val_ann
            self.imgs = glob.glob(f'{self.image_folder}/*')
        elif cfg.mode == 'detect':
            self.imgs = glob.glob(f'{cfg.detect_images}/*')

        self.imgs = [one for one in self.imgs if one[-3:] in ('bmp', 'jpg', 'png')]
        self.imgs.sort()
        self.ids = list(range(len(self.imgs)))

    def __getitem__(self, index):
        if self.mode == 'detect':
            img_path = self.imgs[index]
            img_origin = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            img, resize_shape = self.cfg.val_aug(img_origin)
            img_show = cv2.resize(img_origin, (resize_shape[1], resize_shape[0]))
            return img, resize_shape, img_path.split('/')[-1], img_show
        else:
            img_path = self.imgs[self.ids[index]]
            img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            img_name = img_path.split('/')[-1]
            label = f'{self.label_folder}/{img_name[:-4]}.json'
            with open(label, 'r', encoding='utf-8') as f:
                content = json.load(f)

            polygons, height, width = content['polygons'], content['img_height'], content['img_width']
            assert img.shape[:2] == (height, width), 'image shape error!'

            category = [self.cfg.class_names.index(one['category']) + 1 for one in polygons]
            mask_list = get_seg_mask(self.cfg.class_names, polygons, height, width)

            bboxes = []
            for one in mask_list:
                hs, ws = np.where(one > 0)
                x1, x2 = float(ws.min()), float(ws.max())
                y1, y2 = float(hs.min()), float(hs.max())
                bboxes.append([x1, y1, x2, y2])

            if len(category) > 0:
                labels = np.array(category)
                bboxes = np.array(bboxes)
                masks = np.stack(mask_list, axis=2)  # (h, w, num)

                assert masks.shape == (height, width, labels.shape[0]), 'Unmatched annotations.'

                if self.mode == 'ann':
                    return img, labels, bboxes, masks
                elif self.mode == 'train':
                    img, bboxes, masks = self.cfg.train_aug(img, bboxes, masks)
                    return img, labels, bboxes, masks
                elif self.mode == 'val':
                    img, resize_shape = self.cfg.val_aug(img)
                    return img, (height, width), resize_shape, img_name
            else:
                if self.mode == 'val':
                    raise RuntimeError('Error, no valid object in this image.')
                else:
                    print(f'No valid object in image: "{img_name}". Use a repeated image in this batch.')
                    return None, None, None, None

    def get_aspect_ratios(self):
        aspect_ratios = []
        for one in self.ids:
            img_path = self.imgs[one]
            img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            aspect_ratios.append(float(h) / float(w))
        return aspect_ratios

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        elif self.mode in ('val', 'ann'):
            return len(self.ids) if self.cfg.val_num == -1 else min(self.cfg.val_num, len(self.ids))
        elif self.mode == 'detect':
            return len(self.imgs)

    def img_ids(self):
        return self.ids

    def category_ids(self):
        return list(range(1, self.cfg.num_classes))
