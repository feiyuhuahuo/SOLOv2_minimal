#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb

import torch


def base_input_trans(x: torch.Tensor):
    x = torch.stack([x, x, x], dim=2)
    x = x[:, :, (2, 1, 0)]  # to RGB first
    x = (x - torch.tensor([127., 127., 127.], device='cuda')) / torch.tensor([60., 60., 60.], device='cuda')
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x
