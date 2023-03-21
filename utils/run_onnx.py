#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb
import numpy as np
import cv2
import onnxruntime as ort

if __name__ == '__main__':
    input_size = (512, 512)
    input_img = cv2.imread('detect_imgs/test4.bmp', cv2.IMREAD_COLOR)
    input_img = cv2.resize(input_img, input_size)
    # input_img = np.random.randint(0, 255, (512, 512, 3)).astype('uint8')

    sess = ort.InferenceSession('onnx_files/Custom_light_res50.onnx', providers=['CUDAExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    results = sess.run(None, {input_name: input_img,
                              'detect': np.array(0.3, dtype='float64'), 'mask': np.array(0.5, dtype='float64')})
    print(results)
