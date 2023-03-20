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


# jsons = glob.glob('/home/feiyu/Data/nanjiao/nanjiao_seg/语义分割/labels/train2/*.json')
# i = 0
# for one in jsons:
#     name = one.split('/')[-1]
#     with open(one, 'r', encoding='utf-8') as f:
#         content = json.load(f)
#
#     pdb.set_trace()
