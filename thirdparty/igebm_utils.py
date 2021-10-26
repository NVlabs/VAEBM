# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from an implementation for IGEBM.
#
# Source:
# https://github.com/rosinality/igebm-pytorch
#
# The license for the original version of this file can be
# found in this directory (https://github.com/rosinality/igebm-pytorch/blob/master/LICENSE). 
# The modifications
# to this file are subject to the NVIDIA Source Code License for
# VAEBM located at the root directory.
# ---------------------------------------------------------------


import torch
import numpy as np

def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))
