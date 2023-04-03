#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb
import torch
import cv2
import datetime

from torch.nn.utils import clip_grad
from model.solov2 import SOLOv2
from configs import *
from utils import timer
from data_loader.build_loader import make_data_loader
from val import val


def show_ann(img, boxes, masks):
    img = img.cpu().numpy().astype('uint8')
    for i in range(img.shape[0]):
        img_np = img[i].transpose(1, 2, 0)
        one_box = boxes[i].cpu().numpy()
        one_mask = masks[i]

        for k in range(one_box.shape[0]):
            cv2.rectangle(img_np, (int(one_box[k, 0]), int(one_box[k, 1])),
                          (int(one_box[k, 2]), int(one_box[k, 3])), (0, 255, 0), 1)

        print(f'\nimg shape: {img_np.shape}')
        # print('----------------boxes----------------')
        # print(boxes)

        cv2.imshow('aa', img_np)
        cv2.waitKey()

        print('masks: ', one_mask.shape)
        for k in range(one_mask.shape[0]):
            one = one_mask[k].astype('uint8') * 200
            cv2.imshow('bb', one)
            cv2.waitKey()


if __name__ == '__main__':
    cfg = Solov2_light_res50(mode='train')
    cfg.print_cfg()

    model = SOLOv2(cfg).cuda()
    model.train()

    data_loader = make_data_loader(cfg)
    len_loader = len(data_loader)
    max_iter = len_loader * cfg.epochs
    print(f'Length of dataloader: {len_loader}, total iterations: {max_iter}.')

    start_epoch = 1
    step = 1
    start_lr = cfg.lr
    if cfg.break_weight:
        start_epoch = int(cfg.break_weight.split('_')[-1][:-4]) + 1
        step = (start_epoch - 1) * len_loader + 1
        model.load_state_dict(torch.load(cfg.break_weight), strict=True)
        print(f'\033[0;35mContinue training with "{cfg.break_weight}", epoch: {start_epoch}, step: {step}.\033[0m')

        for one in cfg.lr_decay_steps:
            if start_epoch > one:
                start_lr *= 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=0.0001)

    timer.reset(reset_at=step)
    for epoch in range(start_epoch, cfg.epochs + 1):
        for img, gt_labels, gt_bboxes, gt_masks in data_loader:
            timer.start(step)

            img = img.cuda().detach()
            gt_labels = [one.cuda().detach() for one in gt_labels]
            gt_bboxes = [one.cuda().detach() for one in gt_bboxes]
            # show_ann(img, gt_bboxes, gt_masks)

            if cfg.warm_up_iters > 0 and step <= cfg.warm_up_iters:  # warm up learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (cfg.lr - cfg.warm_up_init) * (step / cfg.warm_up_iters) + cfg.warm_up_init

            if epoch in cfg.lr_decay_steps:  # learning rate decay.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * 0.1 ** (cfg.lr_decay_steps.index(epoch) + 1)

            with timer.counter('for+loss'):
                loss_cate, loss_ins = model(img, gt_labels, gt_bboxes, gt_masks)
                loss_total = loss_cate + loss_ins

            with timer.counter('backward'):
                optimizer.zero_grad()
                loss_total.backward()
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                          max_norm=35, norm_type=2)

            with timer.counter('update'):
                optimizer.step()

            timer.add_batch_time()

            if step % 50 == 0 and step > timer.reset_step:
                cur_lr = optimizer.param_groups[0]['lr']
                time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
                t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
                seconds = (max_iter - step) * t_t
                eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

                l_c, l_ins = loss_cate.item(), loss_ins.item()
                print(f'epoch: {epoch}, step: {step} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_ins: {l_ins:.3f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_fl: {t_fl:.3f} | t_b: {t_b:.3f} | '
                      f't_u: {t_u:.3f} | ETA: {eta}')

            step += 1

        if epoch % cfg.val_interval == 0:
            if epoch > cfg.start_save:
                print(f'weights/{cfg.name()}_{epoch}.pth saved.')
                torch.save(model.state_dict(), f'weights/{cfg.name()}_{epoch}.pth')

            val(cfg, model)
            cfg.train()
            model.train()
            timer.reset(reset_at=step)
