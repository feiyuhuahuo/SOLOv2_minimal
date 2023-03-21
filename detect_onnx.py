#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import pdb
import shutil
import datetime
import cv2
import glob
import numpy as np
import onnxruntime as ort

from configs import *
from utils import timer
from scipy import ndimage

PALLETE = [[0, 0, 0], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
           [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
           [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
           [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
           [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
           [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
           [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
           [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
           [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
           [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
           [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
           [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
           [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
           [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
           [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
           [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]]

if __name__ == '__main__':
    cfg = Custom_light_res50(mode='onnx')
    cfg.print_cfg()

    onnx_path = 'onnx_files/Custom_light_res50.onnx'
    print(f'Using model: {onnx_path}\n')
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    name1 = sess.get_inputs()[0].name

    save_path = 'results/detect/onnx'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    imgs = glob.glob(f'{cfg.detect_images}/*.bmp')
    imgs.sort()
    val_num = len(imgs)
    print(f'Image count: {val_num}.')
    timer.reset(reset_at=0)

    segm_json_results = []
    for i, img_path in enumerate(imgs):
        timer.start(i)

        input_img = cv2.imdecode(np.fromfile(img_path, dtype='uint8'), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(input_img, cfg.onnx_shape)
        with torch.no_grad(), timer.counter('forward'):
            try:
                seg_result = sess.run(None, {name1: img_resized})
            except:
                seg_result = None

        with timer.counter('draw'):
            if seg_result is not None:
                seg_pred, cate_label, cate_score = seg_result[0], seg_result[1], seg_result[2]

                seg_show = img_resized.copy()
                for j in range(seg_pred.shape[0]):
                    cur_mask = seg_pred[j, :, :]

                    color = PALLETE[j]
                    if cfg.detect_mode == 'overlap':
                        mask_bool = cur_mask.astype('bool')
                        seg_show[mask_bool] = img_resized[mask_bool] * 0.7 + np.array(color, dtype='uint8') * 0.3
                    elif cfg.detect_mode == 'contour':
                        _, img_thre = cv2.threshold(cur_mask, 0, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(img_thre, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(seg_show, contours, contourIdx=-1, color=color, thickness=1)

                    label_text = f'{cfg.class_names[cate_label[j]]} {cate_score[j]:.02f}'
                    center_y, center_x = ndimage.center_of_mass(cur_mask)
                    vis_pos = (max(int(center_x) - 10, 0), int(center_y))
                    cv2.putText(seg_show, label_text, vis_pos, cv2.FONT_HERSHEY_COMPLEX, 0.4, tuple(color))

                # print(seg_result)
                # cv2.imshow('aa', seg_show)
                # cv2.waitKey()
                cv2.imwrite(f'{save_path}/{img_path.split("/")[-1]}', seg_show)

        timer.add_batch_time()

        if timer.started:
            t_t, t_d, t_f, t_draw = timer.get_times(['batch', 'data', 'forward', 'draw'])
            seconds = (val_num - i) * t_t
            eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

            print(f'\rDetecting: {i + 1}/{val_num} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f}'
                  f' | t_draw: {t_draw:.3f} | ETA: {eta}', end='')

    print()
    print(f'Done, results saved in "{save_path}".')
