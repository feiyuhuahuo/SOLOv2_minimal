#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import pdb
import datetime
import json
import numpy as np
from model.solov2 import SOLOv2
from configs import *
from utils import timer
from utils.cocoeval import SelfEval
from data_loader.build_loader import make_data_loader
import pycocotools.mask as mask_util


def val(cfg, model=None):
    if model is None:
        model = SOLOv2(cfg).cuda()
        state_dict = torch.load(cfg.val_weight)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=True)
        print(f'Evaluating "{cfg.val_weight}.')

    cfg.eval()
    model.eval()
    data_loader = make_data_loader(cfg)
    dataset = data_loader.dataset

    val_num = len(data_loader)
    print(f'Length of dataloader: {val_num}.')
    timer.reset(reset_at=0)

    dt_id = 1  # 不能是转bool为False的值
    json_results = []
    for i, (img, ori_shape, resize_shape, _) in enumerate(data_loader):
        timer.start(i)

        img = img.cuda().detach()

        with torch.no_grad(), timer.counter('forward'):
            seg_result = model(img, ori_shape=ori_shape, resize_shape=resize_shape, post_mode='val')[0]

        with timer.counter('metric'):
            if seg_result is not None:
                seg_pred = seg_result[0].cpu().numpy()
                cate_label = seg_result[1].cpu().numpy()
                cate_score = seg_result[2].cpu().numpy()

                for j in range(seg_pred.shape[0]):
                    data = dict()
                    cur_mask = seg_pred[j, ...]
                    data['image_id'] = dataset.ids[i]
                    data['score'] = float(cate_score[j])
                    rle = mask_util.encode(np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
                    rle['counts'] = rle['counts'].decode()
                    data['segmentation'] = rle

                    if 'Coco' in dataset.__class__.__name__:
                        data['category_id'] = dataset.cate_ids[cate_label[j] + 1]
                    else:
                        data['category_id'] = int(cate_label[j] + 1)  # 覆盖
                        data['id'] = dt_id
                        data['iscrowd'] = 0
                        data['area'] = int(cur_mask.sum())

                        hs, ws = np.where(cur_mask > 0)
                        x1, x2 = float(ws.min()), float(ws.max())
                        y1, y2 = float(hs.min()), float(hs.max())
                        data['bbox'] = [x1, y1, x2 - x1, y2 - y1]

                    dt_id += 1
                    json_results.append(data)

        timer.add_batch_time()

        if timer.started:
            t_t, t_d, t_f, t_gm = timer.get_times(['batch', 'data', 'forward', 'metric'])
            seconds = (val_num - i) * t_t
            eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

            print(f'\rstep: {i}/{val_num} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_metric: {t_gm:.3f} | '
                  f'ETA: {eta}', end='')

    print('\n\n')

    file_folder = 'results/val'
    os.makedirs(file_folder, exist_ok=True)
    file_path = f'{file_folder}/{cfg.name()}.json'
    with open(file_path, "w") as f:
        json.dump(json_results, f)
    print(f'val result dumped: {file_path}.\n')

    if 'Coco' in dataset.__class__.__name__:
        coco_dt = dataset.coco.loadRes(file_path)
        segm_eval = SelfEval(dataset.coco, coco_dt, all_points=True, iou_type='segmentation')
    else:
        with open(f'{file_folder}/custom_annotations.json', 'r', encoding='utf-8') as f:
            GT = json.load(f)
        with open(file_path, 'r', encoding='utf-8') as f:
            DT = json.load(f)

        segm_eval = SelfEval(GT, DT, all_points=True, iou_type='segmentation')

    segm_eval.evaluate()
    segm_eval.accumulate()
    segm_eval.summarize()


if __name__ == '__main__':
    cfg = Solov2_light_res34(mode='val')
    cfg.print_cfg()
    val(cfg)
