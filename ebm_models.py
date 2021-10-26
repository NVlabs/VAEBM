# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for VAEBM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
'''Energy networks'''


import torch

from torch import nn
from torch.nn import functional as F
from neural_operations import Conv2D

def Lip_swish(x):
    return (x * torch.sigmoid(x))/1.1



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False, data_init = True):
        super().__init__()

        self.conv1 = Conv2D(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=True,
                data_init=data_init
            )
        

        self.conv2 = Conv2D(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias= True,
                data_init=data_init
            )
        
        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(
                Conv2D(in_channel, out_channel, 1, bias=False,data_init=data_init))
            

        self.downsample = downsample

    def forward(self, input):
        out = input

        out = self.conv1(out)


        out = Lip_swish(out)

        out = self.conv2(out)


        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = Lip_swish(out)

        return out


class EBM_CelebA64(nn.Module):
    def __init__(self, nc=3, mid_channel=64, data_init = True):
        super().__init__()

        self.conv1 = Conv2D(nc, mid_channel, 3, padding=1,bias = True, data_init=data_init)

        self.blocks = nn.ModuleList(
            [
                ResBlock(mid_channel, mid_channel,  downsample=True, data_init = data_init),
                ResBlock(mid_channel, mid_channel, data_init = data_init),
                ResBlock(mid_channel, 2*mid_channel, downsample=True, data_init = data_init),
                ResBlock(2*mid_channel, 2*mid_channel, data_init = data_init),
                ResBlock(2*mid_channel, 2*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(2*mid_channel, 4*mid_channel, data_init = data_init),
                ResBlock(4*mid_channel, 4*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(4*mid_channel, 4*mid_channel, data_init = data_init),
            ]
        )

        self.linear = nn.Linear(4*mid_channel, 1)
        
        self.all_conv_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, Conv2D):
                self.all_conv_layers.append(layer)
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4


    def forward(self, input):
        out = self.conv1(input)

        out = Lip_swish(out)

        for block in self.blocks:
            out = block(out)

#        out = Lip_swish(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out.squeeze(1)
    
    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss
    

class EBM_LSUN64(nn.Module):
    def __init__(self, nc=3, mid_channel=64, data_init = True):
        super().__init__()

        self.conv1 = Conv2D(nc, mid_channel, 3, padding=1,bias = True, data_init=data_init)

        self.blocks = nn.ModuleList(
            [
                ResBlock(mid_channel, mid_channel,  downsample=True, data_init = data_init),
                ResBlock(mid_channel, mid_channel, data_init = data_init),
                ResBlock(mid_channel, 2*mid_channel, downsample=True, data_init = data_init),
                ResBlock(2*mid_channel, 2*mid_channel, data_init = data_init),
                ResBlock(2*mid_channel, 2*mid_channel, data_init = data_init),

                ResBlock(2*mid_channel, 2*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(2*mid_channel, 4*mid_channel, data_init = data_init),
                ResBlock(4*mid_channel, 4*mid_channel, data_init = data_init),

                ResBlock(4*mid_channel, 4*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(4*mid_channel, 4*mid_channel, data_init = data_init),
            ]
        )

        self.linear = nn.Linear(4*mid_channel, 1)
        
        self.all_conv_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, Conv2D):
                self.all_conv_layers.append(layer)
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4


    def forward(self, input):
        out = self.conv1(input)

        out = Lip_swish(out)

        for block in self.blocks:
            out = block(out)

#        out = Lip_swish(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out.squeeze(1)
    
    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss


class EBM_CIFAR32(nn.Module):
    def __init__(self, nc=3, mid_channel=128, data_init = True):
        super().__init__()
        
        self.conv1 = Conv2D(nc, mid_channel, 3, padding=1,bias = True, data_init=data_init)

        self.blocks = nn.ModuleList(
            [
                ResBlock(mid_channel, mid_channel, downsample=True, data_init = data_init),
                ResBlock(mid_channel, mid_channel, data_init = data_init),
                ResBlock(mid_channel, mid_channel*2, downsample=True, data_init = data_init),
                ResBlock(mid_channel*2, mid_channel*2, data_init = data_init),
                ResBlock(mid_channel*2, mid_channel*2, downsample=True, data_init = data_init),
                ResBlock(mid_channel*2, mid_channel*2, data_init = data_init),
            ]
        )


        self.linear = nn.Linear(2*mid_channel, 1)
        
        self.all_conv_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, Conv2D):
                self.all_conv_layers.append(layer)
                
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4



    def forward(self, input):
        out = self.conv1(input)

        out = Lip_swish(out)

        for block in self.blocks:
            out = block(out)

#        out = Lip_swish(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out.squeeze(1)
    
    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss



class EBM_MNIST(nn.Module):
    def __init__(self, nc=3, mid_channel=64, data_init = True):
        super().__init__()
        
        self.conv1 = Conv2D(nc, mid_channel, 3, padding=1,bias = True, data_init=data_init)

        self.blocks = nn.ModuleList(
            [
                ResBlock(mid_channel, mid_channel, downsample=True, data_init = data_init),
                ResBlock(mid_channel, mid_channel, data_init = data_init),
                ResBlock(mid_channel, mid_channel*2, downsample=True, data_init = data_init),
                ResBlock(mid_channel*2, mid_channel*2, data_init = data_init)            ]
        )


        self.linear = nn.Linear(2*mid_channel, 1)
        
        self.all_conv_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, Conv2D):
                self.all_conv_layers.append(layer)
                
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4



    def forward(self, input):
        out = self.conv1(input)

        out = Lip_swish(out)

        for block in self.blocks:
            out = block(out)

#        out = Lip_swish(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out.squeeze(1)
    
    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss
    

class EBM_CelebA256(nn.Module):
    def __init__(self, nc=3, mid_channel=64, data_init = True):
        super().__init__()

        self.conv1 = Conv2D(nc, mid_channel, 3, padding=1, bias = True, data_init=data_init)

        self.blocks = nn.ModuleList(
            [
                ResBlock(mid_channel, mid_channel,  downsample=True, data_init = data_init),
                ResBlock(mid_channel, mid_channel, data_init = data_init),
                ResBlock(mid_channel, 2*mid_channel, downsample=True, data_init = data_init),
                ResBlock(2*mid_channel, 2*mid_channel, data_init = data_init),
                ResBlock(2*mid_channel, 2*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(2*mid_channel, 2*mid_channel, data_init = data_init),
                ResBlock(2*mid_channel, 4*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(4*mid_channel, 4*mid_channel, data_init = data_init),
                ResBlock(4*mid_channel, 4*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(4*mid_channel, 4*mid_channel, data_init = data_init),
                ResBlock(4*mid_channel, 8*mid_channel,  downsample=True, data_init = data_init),
                ResBlock(8*mid_channel, 8*mid_channel, data_init = data_init),
            ]
        )

        self.linear = nn.Linear(8*mid_channel, 1)
        
        self.all_conv_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, Conv2D):
                self.all_conv_layers.append(layer)
                
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4


    def forward(self, input):
        out = self.conv1(input)

        out = Lip_swish(out)

        for block in self.blocks:
            out = block(out)

#        out = Lip_swish(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out.squeeze(1)
    
    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss