#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb

import torch.jit

from configs import *
import torch.nn.functional as F
from utils.utils import matrix_nms


@torch.jit.script
def get_seg_scripted(cate_list: list[torch.Tensor], kernel_list: list[torch.Tensor], seg_pred: torch.Tensor):
    num_grids = (40, 36, 24, 16, 12)
    strides_set = (8, 8, 16, 32, 32)
    score_thre = 0.1
    mask_thre = 0.5
    nms_pre = 500
    update_thre = 0.05
    max_ins = 100

    cate_num = 12
    head_ins_out_c = 128
    resize_h = 512
    resize_w = 512

    num_levels = len(cate_list)
    featmap_size = seg_pred.size()[-2:]
    feat_size = [int(featmap_size[0]), int(featmap_size[1])]
    non_tensor = [torch.tensor([0.], device='cuda'), torch.tensor([0.], device='cuda'),
                  torch.tensor([0.], device='cuda')]
    result_batch_list = [[] for _ in range(seg_pred.shape[0])]
    for j in range(seg_pred.shape[0]):
        cate_list = [cate_list[i][j].view(-1, cate_num).detach() for i in range(num_levels)]
        seg_preds = seg_pred[j, ...].unsqueeze(0)
        kernel_list = [kernel_list[i][j].permute(1, 2, 0).view(-1, head_ins_out_c).detach()
                       for i in range(num_levels)]

        cate_preds = torch.cat(cate_list, dim=0)
        kernel_preds = torch.cat(kernel_list, dim=0)

        # process.
        inds = (cate_preds > score_thre)

        cate_scores = cate_preds[inds]
        if cate_scores.shape[0] == 0:
            result_batch_list[j] = non_tensor
            continue

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = torch.tensor(num_grids, device=cate_labels.device, dtype=cate_labels.dtype).pow(2).cumsum(0)
        # size_trans = cate_labels.new_tensor(num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        strides[:size_trans[0]] *= strides_set[0]
        for ind_ in range(1, len(num_grids)):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= strides_set[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks = seg_preds > mask_thre
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            result_batch_list[j] = non_tensor
            continue

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if sort_inds.shape[0] > nms_pre:
            sort_inds = sort_inds[:nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= update_thre
        if keep.sum() == 0:
            result_batch_list[j] = non_tensor
            continue

        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if sort_inds.shape[0] > max_ins:
            sort_inds = sort_inds[:max_ins]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        s1 = int(feat_size[0] * 4)
        s2 = int(feat_size[1] * 4)
        seg_masks = F.interpolate(seg_preds.unsqueeze(0), size=[s1, s2], mode='bilinear')
        seg_masks = seg_masks[:, :, :resize_h, :resize_w].squeeze(0)
        seg_masks = seg_masks > mask_thre

        one_result = [seg_masks, cate_labels, cate_scores]
        result_batch_list[j] = one_result

    return result_batch_list


if __name__ == '__main__':
    ss = torch.jit.script(get_seg_scripted)
    print(ss.code)
