# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the PyTorch library.
#
# Source:
# https://github.com/pytorch/pytorch/blob/2a54533c64c409b626b6c209ed78258f67aec194/torch/nn/modules/_functions.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_PyTorch). The modifications
# to this file are subject to the NVIDIA Source Code License for
# VAEBM located at the root directory.
# ---------------------------------------------------------------


import torch
from torch.autograd.function import Function


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        count = torch.Tensor([size]).to(input.device)

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        count_all = torch.empty(world_size, 1, dtype=count.dtype, device=count.device)
        mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
        invstd_all = torch.empty(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)

        count_l = list(count_all.unbind(0))
        mean_l = list(mean_all.unbind(0))
        invstd_l = list(invstd_all.unbind(0))

        # using all_gather instead of all reduce so we can calculate count/mean/var in one go
        count_all_reduce = torch.distributed.all_gather(count_l, count, process_group, async_op=True)
        mean_all_reduce = torch.distributed.all_gather(mean_l, mean, process_group, async_op=True)
        invstd_all_reduce = torch.distributed.all_gather(invstd_l, invstd, process_group, async_op=True)

        # wait on the async communication to finish
        count_all_reduce.wait()
        mean_all_reduce.wait()
        invstd_all_reduce.wait()

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1).long().tolist()
        )

        self.save_for_backward(input, weight, mean, invstd, bias)
        self.process_group = process_group
        self.world_size = world_size

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

        # arash: apply swish
        assert eps == 1e-5, "I assumed below that eps is 1e-5"
        out = out * torch.sigmoid(out)
        # end arash

        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, bias = self.saved_tensors

        # arash: re-compute batch normalized out
        eps = 1e-5
        out = torch.batch_norm_elemt(saved_input, weight, bias, mean, invstd, eps)
        sigmoid_out = torch.sigmoid(out)
        grad_output *= (sigmoid_out * (1 + out * (1 - sigmoid_out)))
        # end arash

        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        world_size = self.world_size

        # calculate local stats as well as grad_weight / grad_bias
        mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            mean_dy_all_reduce = torch.distributed.all_reduce(
                mean_dy, torch.distributed.ReduceOp.SUM, process_group, async_op=True)
            mean_dy_xmu_all_reduce = torch.distributed.all_reduce(
                mean_dy_xmu, torch.distributed.ReduceOp.SUM, process_group, async_op=True)

            # wait on the async communication to finish
            mean_dy_all_reduce.wait()
            mean_dy_xmu_all_reduce.wait()

            mean_dy.div_(world_size)
            mean_dy_xmu.div_(world_size)
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class SyncMeanBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        count = torch.Tensor([size]).to(input.device)

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)
        # mean = torch.mean(input, dim=[0, 2, 3])
        invstd = torch.ones_like(mean)

        count_all = torch.empty(world_size, 1, dtype=count.dtype, device=count.device)
        mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
        invstd_all = torch.empty(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)

        count_l = list(count_all.unbind(0))
        mean_l = list(mean_all.unbind(0))
        invstd_l = list(invstd_all.unbind(0))

        # using all_gather instead of all reduce so we can calculate count/mean/var in one go
        count_all_reduce = torch.distributed.all_gather(count_l, count, process_group, async_op=True)
        mean_all_reduce = torch.distributed.all_gather(mean_l, mean, process_group, async_op=True)
        invstd_all_reduce = torch.distributed.all_gather(invstd_l, invstd, process_group, async_op=True)

        # wait on the async communication to finish
        count_all_reduce.wait()
        mean_all_reduce.wait()
        invstd_all_reduce.wait()

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1).long().tolist()
        )

        # apply element-wise normalization
        # out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        b = mean.view(1, -1, 1, 1) - bias.view(1, -1, 1, 1)
        input -= b

        self.save_for_backward(input, weight, mean, invstd, bias)
        self.process_group = process_group
        self.world_size = world_size

        out = input * torch.sigmoid(input)

        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, bias = self.saved_tensors

        sigmoid_out = torch.sigmoid(saved_input)
        grad_output *= (sigmoid_out * (1 + saved_input * (1 - sigmoid_out)))

        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        world_size = self.world_size

        grad_bias = torch.sum(grad_output, dim=[0, 2, 3])
        mean_dy = grad_bias / (grad_output.size(0) * grad_output.size(2) * grad_output.size(3))

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            mean_dy_all_reduce = torch.distributed.all_reduce(
                mean_dy, torch.distributed.ReduceOp.SUM, process_group, async_op=True)

            # wait on the async communication to finish
            mean_dy_all_reduce.wait()

            mean_dy.div_(world_size)
            grad_input = grad_output - mean_dy.view(1, -1, 1, 1)

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

