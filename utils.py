# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for VAEBM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.distributed as dist

import logging
import os
import shutil
import time
from datetime import timedelta
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

import torch.nn.functional as F
from tensorboardX import SummaryWriter


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAvgrageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class DummyDDP(nn.Module):
    def __init__(self, model):
        super(DummyDDP, self).__init__()
        self.module = model

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = x / keep_prob
        x = x * mask
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class ClassErrorMeter(object):
    def __init__(self):
        super(ClassErrorMeter, self).__init__()
        self.class_counter = {}

    def add(self, output, target):
        _, pred = output.max(dim=1)

        target = list(target.cpu().numpy())
        pred = list(pred.cpu().numpy())

        for t, p in zip(target, pred):
            if t not in self.class_counter:
                self.class_counter[t] = {'num': 0, 'correct': 0}
            self.class_counter[t]['num'] += 1
            if t == p:
                self.class_counter[t]['correct'] += 1

    def value(self, method):
        print('Error type: ', method)
        if method == 'per_class':
            mean_accuracy = 0
            for t in self.class_counter:
                class_accuracy = float(self.class_counter[t]['correct']) / \
                    self.class_counter[t]['num']
                mean_accuracy += class_accuracy
            mean_accuracy /= len(self.class_counter)
            output = mean_accuracy * 100
        elif method == 'overall':
            num_total, num_correct = 0, 0
            for t in self.class_counter:
                num_total += self.class_counter[t]['num']
                num_correct += self.class_counter[t]['correct']
            output = float(num_correct) / num_total * 100
        return [100 - output]


def sample_gumbel(shape, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0, 1).cuda()
    sample = -(torch.log(-torch.log(U + eps) + eps))
    return sample


def gumbel_softmax_sample_original(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def logsumexp(logits, dim):
    mx = torch.max(logits, dim, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(logits - mx), dim=dim, keepdim=True)) + mx


def gumbel_softmax_sample_improved(logits, temperature):
    def gsm(rho, q):
        return F.softmax((-torch.log(rho + 1e-20) + torch.log(q + 1e-20)) / temperature, dim=-1)
    q = F.softmax(logits, dim=-1)
    U = torch.Tensor(q.size()).uniform_(0, 1).cuda()
    U = torch.clamp(U, 1e-15, 1. - 1e-15)
    log_U = torch.log(U)
    rho = log_U / (torch.sum(log_U, dim=-1, keepdim=True))
    return gsm(rho.detach() - q + q.detach(), q.detach())


def gumbel_softmax_sample_rebar(logits, temperature):
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    q = F.softmax(logits, dim=-1)
    u = torch.Tensor(q.size()).uniform_(0, 1).cuda()
    u = torch.clamp(u, 1e-3, 1.-1e-3)

    # draw gsm samples
    z = logits - torch.log(- torch.log(u))
    gsm = F.softmax(z / temperature, dim=-1)

    # compute the correction term for conditional samples
    # see REBAR: https://arxiv.org/pdf/1703.07370.pdf
    k = torch.argmax(z, dim=-1, keepdim=True)
    # get v from u
    u_k = u.gather(-1, k)
    q_k = q.gather(-1, k)
    # This can cause numerical problems, better to work with log(v_k) = u_k / q_k
    # v_k = torch.pow(u_k, 1. / q_k)
    # v.scatter_(-1, k, v_k)
    log_vk = torch.log(u_k) / q_k
    log_v = torch.log(u) - q * log_vk

    # assume k and v are constant
    k = k.detach()
    log_vk = log_vk.detach()
    log_v = log_v.detach()
    g_hat = - torch.log(-log_v/q - log_vk)
    g_hat.scatter(-1, k, -torch.log(- log_vk))
    gsm1 = F.softmax(g_hat / temperature, dim=-1)

    return gsm - gsm1 + gsm1.detach()


def gumbel_softmax_sample(logits, temperature, gsm_type='improved'):
    if gsm_type == 'improved':
        return gumbel_softmax_sample_improved(logits, temperature)
    elif gsm_type == 'original':
        return gumbel_softmax_sample_original(logits, temperature)
    elif gsm_type == 'rebar':
        return gumbel_softmax_sample_rebar(logits, temperature)


def plot_alphas(alpha, cell_type, display=True):
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(7)

    alpha = alpha.data.cpu().numpy()

    num_edges = alpha.shape[0]
    ops = get_primitives_for_cell_type(cell_type)

    ax.xaxis.tick_top()
    plt.imshow(alpha, vmin=0, vmax=1)
    plt.xticks(range(len(ops)), ops)
    plt.xticks(rotation=30)
    plt.yticks(range(num_edges), range(1, num_edges+1))
    for i in range(num_edges):
        for j in range(len(ops)):
            val = alpha[i][j]
            val = '%.4f' % (val)
            ax.text(j, i, val, va='center',
                    ha='center', color='white', fontsize=8)

    plt.colorbar()
    plt.tight_layout()

    if display:
        plt.show()
    else:
        return fig


def get_alpha_for_storage(alphas):
    all_cells = dict()
    for k in alphas.keys():
        alpha = alphas[k]
        primitives = get_primitives_for_cell_type(k)
        ops = []
        for i in range(alpha.size(0)):
            if torch.max(alpha[i]) == 0:
                ops.append('none')
            else:
                indices = np.nonzero(alpha[i].cpu().numpy())
                ops.append(primitives[indices[0][0]])

        all_cells[k] = ops
        
    return all_cells


def generate_paired_indices(step):
    indices = []
    for i in range(step + 2):
        indices.append((i, i))

    for i in range(step + 2):
        for j in range(i + 1, step + 2):
            indices.append((i, j))

    return indices


class Logger(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:   # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)

    return stride


def get_cout(cin, stride):
    if stride == 1:
        cout = cin
    elif stride == -1:
        cout = cin // 2
    elif stride == 2:
        cout = 2 * cin

    return cout


def get_primitives_for_cell_type(cell_type):
    if cell_type == 'ar_nn':
        # primitives = ['skip_connect', 'sep_conv_3x3', 'conv_3x3', 'fdil_conv_3x3']
        raise NotImplementedError
    else:
        # primitives = ['conv_3x3', 'fdil_conv_3x3', 'sep_conv_3x3', 'dil_conv_3x3']
        primitives = ['mconv_e6d1k3', 'mconv_e6d1k5', 'mconv_e6d2k3', 'mconv_e6d2k5',
                      'mconv_e3d1k3', 'mconv_e3d1k5', 'mconv_e3d2k3', 'mconv_e3d2k5']

    return primitives


def get_cell_per_stage(stage):
    if stage.startswith('vanilla_vae'):
        fixed_cells = []
        trainable_cells = ['down_enc', 'normal_enc', 'up_dec', 'normal_dec']
    elif stage.startswith('bi_vae_ss'):  # single scale bidirectional VAE
        # fixed_cells = ['down_pre', 'normal_pre', 'up_post', 'normal_post']
        # trainable_cells = ['normal_cond_enc', 'normal_cond_dec']
        fixed_cells = []
        trainable_cells = ['down_enc', 'normal_enc', 'up_dec', 'normal_dec']
    elif stage.startswith('bi_vae'):
        fixed_cells = ['down_pre', 'normal_pre', 'up_post', 'normal_post']
        trainable_cells = ['normal_cond_enc', 'down_cond_enc',
                           'normal_cond_dec', 'up_cond_dec']
    else:
        raise NotImplementedError('%s is unknown' % stage)

    if stage.endswith('_nf'):
        trainable_cells.append('ar_nn')

    return fixed_cells, trainable_cells


def kl_balancer_coeff(num_scales, groups_per_scale, fun):
    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat([np.sqrt(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'square':
        coeff = torch.cat([np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_coeff(step, total_step, constant_step, min_kl_coeff):
    return max(min((step - constant_step) / total_step, 1.0), min_kl_coeff)


def log_iw(decoder, x, log_q, log_p, crop=False):
    recon = reconstruction_loss(decoder, x, crop)
    return - recon - log_q + log_p


def reconstruction_loss(decoder, x, crop=False):
    from distributions import Normal, DiscMixLogistic

    recon = decoder.log_prob(x)
    if crop:
        recon = recon[:, :, 2:30, 2:30]
    
    if isinstance(decoder, DiscMixLogistic):
        return - torch.sum(recon, dim=[1, 2])    # summation over RGB is done.
    else:
        return - torch.sum(recon, dim=[1, 2, 3])


def tile_image(batch_image, n):
    assert n * n == batch_image.size(0)
    channels, height, width = batch_image.size(1), batch_image.size(2), batch_image.size(3)
    batch_image = batch_image.view(n, n, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)                              # n, height, n, width, c
    batch_image = batch_image.contiguous().view(channels, n * height, n * width)
    return batch_image


def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size


def average_params(params):
    """ parameter averaging. """
    size = float(dist.get_world_size())
    for param in params:
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot


def num_output(dataset):
    if dataset == 'mnist':
        return 28 * 28
    elif dataset == 'stacked_mnist':
        return 28 * 28
    elif dataset == 'cifar10':
        return 3 * 32 * 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return 3 * size * size
    elif dataset == 'ffhq':
        return 3 * 256 * 256
    else:
        raise NotImplementedError


def get_input_size(dataset):
    if dataset == 'mnist':
        return 32
    elif dataset == 'stacked_mnist':
        return 32
    elif dataset == 'cifar10':
        return 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return size
    elif dataset == 'ffhq':
        return 256
    else:
        raise NotImplementedError


def pre_process(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x


def get_arch_cells(arch_type):
    if arch_type == 'res_elu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_elu', 'res_elu']
        arch_cells['down_enc'] = ['res_elu', 'res_elu']
        arch_cells['normal_dec'] = ['res_elu', 'res_elu']
        arch_cells['up_dec'] = ['res_elu', 'res_elu']
        arch_cells['normal_pre'] = ['res_elu', 'res_elu']
        arch_cells['down_pre'] = ['res_elu', 'res_elu']
        arch_cells['normal_post'] = ['res_elu', 'res_elu']
        arch_cells['up_post'] = ['res_elu', 'res_elu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnelu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep11':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k11g0']
        arch_cells['down_enc'] = ['mconv_e6k11g0']
        arch_cells['normal_dec'] = ['mconv_e6k11g0']
        arch_cells['up_dec'] = ['mconv_e6k11g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res53_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res35_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res55_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv9':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k9g0']
        arch_cells['up_dec'] = ['mconv_e6k9g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k9g0']
        arch_cells['up_post'] = ['mconv_e3k9g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_res':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_den':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g8']
        arch_cells['down_pre'] = ['mconv_e3k5g8']
        arch_cells['normal_post'] = ['mconv_e3k5g8']
        arch_cells['up_post'] = ['mconv_e3k5g8']
        arch_cells['ar_nn'] = ['']
    else:
        raise NotImplementedError

    return arch_cells


def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g


