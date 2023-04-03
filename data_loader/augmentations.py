#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb

import cv2
import random
import numpy as np


def random_flip(img, bboxes, masks, v_flip=False):
    height, width = img.shape[:2]

    # horizontal flip
    if random.randint(0, 1):
        img = cv2.flip(img, 1)  # Don't use such 'image[:, ::-1]' code, may occur bugs.
        bboxes[:, 0::2] = width - bboxes[:, 2::-2]
        masks = cv2.flip(masks, 1)

    # vertical flip
    if v_flip and random.randint(0, 1):
        img = cv2.flip(img, 0)
        bboxes[:, 1::2] = height - bboxes[:, 3::-2]
        masks = cv2.flip(masks, 0)

    if masks.ndim == 2:
        masks = masks[:, :, None]

    return img, bboxes, masks


def random_resize(img, bboxes=None, masks=None, img_scale=None):
    max_size, min_size = random.choice(img_scale)
    h, w = img.shape[:2]

    short_side, long_side = min(h, w), max(h, w)
    if min_size / short_side * long_side > max_size:
        scale = max_size / long_side
    else:
        scale = min_size / short_side

    new_h, new_w = int(scale * h), int(scale * w)
    assert (min(new_h, new_w)) <= min_size and (max(new_h, new_w) <= max_size), 'Scale error when resizing.'

    img = cv2.resize(img, (new_w, new_h))

    if bboxes is not None:
        bboxes *= scale
    if masks is not None:
        masks = cv2.resize(masks, (new_w, new_h), cv2.INTER_NEAREST)
        if masks.ndim == 2:
            masks = masks[:, :, None]

    return img, bboxes, masks


def pad_to_size_divisor(img, masks=None, size_divisor=32, pad_value=None):
    h, w = img.shape[:2]
    pad_h, pad_w = h, w
    if h % size_divisor != 0:
        pad_h = (h // size_divisor + 1) * size_divisor
    if w % size_divisor != 0:
        pad_w = (w // size_divisor + 1) * size_divisor

    if pad_h == h and pad_w == w:
        return img, masks
    else:
        if pad_value is None:
            pad_value = random.randint(0, 255)
        pad_img = np.zeros((pad_h, pad_w, img.shape[2]), dtype='uint8') + pad_value
        pad_img[0:h, 0:w, ...] = img

        pad_masks = None
        if masks is not None:
            pad_masks = np.zeros((pad_h, pad_w, masks.shape[2]), dtype='int64')
            pad_masks[0:h, 0:w, ...] = masks

        return pad_img, pad_masks


# pad after normalize
def pad_to_size_divisor_float(img, masks=None, size_divisor=32):
    h, w = img.shape[:2]
    pad_h, pad_w = h, w
    if h % size_divisor != 0:
        pad_h = (h // size_divisor + 1) * size_divisor
    if w % size_divisor != 0:
        pad_w = (w // size_divisor + 1) * size_divisor

    if pad_h == h and pad_w == w:
        return img, masks
    else:
        pad_img = np.zeros((pad_h, pad_w, img.shape[2]), dtype='float32')
        pad_img[0:h, 0:w, ...] = img

        pad_masks = None
        if masks is not None:
            pad_masks = np.zeros((pad_h, pad_w, masks.shape[2]), dtype='int64')
            pad_masks[0:h, 0:w, ...] = masks

        return pad_img, pad_masks


def normalize(img, mean: np.ndarray, std: np.ndarray):
    img = img[:, :, (2, 1, 0)]  # to RGB first
    img = (img - mean) / std
    return img  # h, w, c  in rgb mode


def clip_box(hw, boxes):
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], a_min=0, a_max=hw[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], a_min=0, a_max=hw[0] - 1)
    return boxes


class TrainAug:
    def __init__(self, img_scale, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), v_flip=False):
        self.img_scale = img_scale
        self.norm_mean = np.array(mean, dtype='float32')
        self.norm_std = np.array(std, dtype='float32')
        self.v_flip = v_flip

    def __call__(self, img, bboxes, masks):
        assert img.shape[:2] == masks.shape[:2], 'img shape != mask shape before doing train aug!'
        img, bboxes, masks = random_flip(img, bboxes, masks, v_flip=self.v_flip)
        img, bboxes, masks = random_resize(img, bboxes, masks, img_scale=self.img_scale)
        bboxes = clip_box(img.shape[:2], bboxes)
        # show_ann(img, bboxes, masks)
        img = normalize(img, self.norm_mean, self.norm_std)
        img, masks = pad_to_size_divisor_float(img, masks)
        return img, bboxes, masks

    def __repr__(self):
        return f'img_scale: {self.img_scale}\n           mean: {self.norm_mean}, std: {self.norm_std}\n' \
               f'           v_flip: {self.v_flip}'


class ValAug:
    def __init__(self, img_scale, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        self.img_scale = img_scale
        self.norm_mean = np.array(mean, dtype='float32')
        self.norm_std = np.array(std, dtype='float32')

    def __call__(self, img):
        img, _, _ = random_resize(img, img_scale=self.img_scale)
        resize_shape = img.shape[:2]
        img = normalize(img, self.norm_mean, self.norm_std)
        img, _ = pad_to_size_divisor_float(img)
        return img, resize_shape

    def __repr__(self):
        return f'img_scale: {self.img_scale}\n         mean: {self.norm_mean}, std: {self.norm_std}'


def show_ann(img, boxes, masks):
    img_u8 = img.astype('uint8')

    for i in range(boxes.shape[0]):
        cv2.rectangle(img_u8, (int(boxes[i, 0]), int(boxes[i, 1])),
                      (int(boxes[i, 2]), int(boxes[i, 3])), (0, 255, 0), 1)

    print(f'\nimg shape: {img.shape}')
    print('----------------boxes----------------')
    print(boxes)

    cv2.imshow('aa', img_u8)
    cv2.waitKey()
    print(masks.shape)
    masks = masks.transpose(2, 0, 1)
    for i in range(masks.shape[0]):
        one_mask = masks[i].astype('uint8') * 200
        cv2.imshow('bb', one_mask)
        cv2.waitKey()
