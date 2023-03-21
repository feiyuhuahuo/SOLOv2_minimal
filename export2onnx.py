#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import pdb
from model.solov2 import SOLOv2
from configs import *
import cv2

if __name__ == '__main__':
    cfg = Custom_light_res50(mode='onnx')
    cfg.print_cfg()

    model = SOLOv2(cfg).cuda()
    state_dict = torch.load(cfg.val_weight)
    print(f'Detecting with "{cfg.val_weight}.')

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.makedirs('onnx_files', exist_ok=True)
    file_path = f'onnx_files/{cfg.__class__.__name__}.onnx'

    input_size = cfg.onnx_shape
    class_suffix = '-'.join(list(cfg.class_names))
    inp_names = [f'seg_{input_size[0]}_{input_size[1]}-{class_suffix}']
    # input_img = torch.randint(0, 255, (input_size[0], input_size[1], 3), device='cuda', dtype=torch.uint8)
    input_img = cv2.imread('detect_imgs/test1.bmp', cv2.IMREAD_GRAYSCALE)
    input_img = cv2.resize(input_img, input_size)
    input_img = torch.from_numpy(input_img).cuda()

    torch.onnx.export(model,
                      input_img,
                      file_path,
                      input_names=inp_names,
                      output_names=['output'],
                      # dynamic_axes={inp_names[0]: {0: 'bs'}},
                      verbose=False,
                      opset_version=14)

    print(f'\nSaved as {file_path}.\n')
