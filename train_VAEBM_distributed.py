# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for VAEBM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
''' Code for training VAEBM distrubutedly'''

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import random

import torch.distributed as dist
from torch.multiprocessing import Process

from torch.autograd import Variable

from nvae_model import AutoEncoder
import utils
import datasets

from ebm_models import EBM_CelebA64, EBM_LSUN64, EBM_CIFAR32, EBM_CelebA256

import torchvision
from thirdparty.igebm_utils import sample_data, clip_grad


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

class SampleBuffer:
    def __init__(self, num_block, max_samples):
        self.max_samples = max_samples
        self.num_block = num_block
        self.buffer = [[] for _ in range(num_block)]  #each group of latent variable is a list

    def __len__(self):
        return len(self.buffer[0]) #len of the buffer should be the length of list for each group of latent

    def push(self, z_list): #samples is a list of torch tensor
        for i in range(self.num_block):
            zi = z_list[i]
            zi = zi.detach().data.to('cpu').clone()
            for sample in zip(zi):
                self.buffer[i].append(sample[0])
                if len(self.buffer[i]) > self.max_samples:
                    self.buffer[i].pop(0)

    def get(self, n_samples):
        sample_idx = random.sample(range(len(self.buffer[0])), n_samples)
        z_list = []
        for i in range(self.num_block):
            samples = [self.buffer[i][j] for j in sample_idx]
            samples = torch.stack(samples, 0)
            samples = samples.cuda()
            z_list.append(samples)

        return z_list
    def save(self,fname):
        torch.save(self.buffer,fname)




def sample_buffer(buffer, z_list_exampler, batch_size=64, t = 1, p=0.95):
    if len(buffer) < 1:       
        
        eps_z = [torch.Tensor(batch_size, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).cuda() \
                 for zi in z_list_exampler]

        return eps_z
    

    n_replay = (np.random.rand(batch_size) < p).sum()
    
    if n_replay > 0:
    
        eps_z_replay = buffer.get(n_replay)
        eps_z_prior = [torch.Tensor(batch_size - n_replay, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).cuda()\
                for zi in z_list_exampler]

        eps_z_combine = [torch.cat([z1,z2], dim = 0) for z1,z2 in zip(eps_z_replay, eps_z_prior)]
        
        return eps_z_combine
    else:
        eps_z = [torch.Tensor(batch_size - n_replay, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).cuda() \
                for zi in z_list_exampler]

    
        return eps_z




def main(eval_args):
    # ensures that weight initializations are all the same
    torch.manual_seed(eval_args.seed)
    np.random.seed(eval_args.seed)
    torch.cuda.manual_seed(eval_args.seed)
    torch.cuda.manual_seed_all(eval_args.seed)
    
    model_path = './saved_models/{}/{}/'.format(eval_args.dataset, eval_args.experiment)
    if eval_args.global_rank == 0:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            os.makedirs(model_path + '/images/')


    logging = utils.Logger(eval_args.global_rank, model_path)

    # Get data loaders.
    train_queue, _, _ = datasets.get_loaders(eval_args)


    # load a checkpoint
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    
    args = checkpoint['args']

    logging.info('loaded model at epoch %d', checkpoint['epoch'])
    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    model = model.cuda()
    print('num conv layers:', len(model.all_conv_layers))

    model.load_state_dict(checkpoint['state_dict'], strict = False)
    model = model.cuda()

    
    if eval_args.dataset == 'cifar10':
        EBM_model = EBM_CIFAR32(3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    
    elif eval_args.dataset == 'celeba_64':
        
        EBM_model = EBM_CelebA64(3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    
    elif eval_args.dataset == 'lsun_church':
        EBM_model = EBM_LSUN64(3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    elif eval_args.dataset == 'celeba_256':
        EBM_model = EBM_CelebA256(3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    else:
        raise Exception("choose dataset in ['cifar10', 'celeba_64', 'lsun_church', 'celeba_256']")
    
    
    init_image = []
    for idx, (image) in enumerate(train_queue):
        init_image.append(image[0])
        if idx == 4:
            break
    init_image = torch.stack(init_image)
    init_image = init_image.view(-1,3,eval_args.im_size,eval_args.im_size).cuda()
    
        
    with torch.no_grad():
        EBM_model(init_image) #for initialization

    
    t = eval_args.temperature
    
    optimizer = torch.optim.Adam(EBM_model.parameters(), lr=eval_args.lr, betas = (0.99,0.999), weight_decay = eval_args.wd)

    global_step = 0
    
    d_s_t = []
    
    with torch.no_grad():
        _, z_list, _ = model.sample(eval_args.batch_size, t) 
        num_block = len(z_list)
    
    if eval_args.use_buffer:
        buffer = SampleBuffer(num_block = num_block, max_samples=eval_args.buffer_size)
    else:
        buffer  = None


    for epoch in range(eval_args.epochs):

        if eval_args.distributed:
            train_queue.sampler.set_epoch(global_step + eval_args.seed)

        d_s_t, global_step, output = train(EBM_model, model, optimizer, buffer, t, train_queue, z_list, d_s_t, global_step, eval_args, model_path)
        
        if global_step > eval_args.total_iter:
            break


def train(model,VAE, optimizer, buffer, t, loader, z_list, d_s_t, global_step, opt, model_path):
    step_size = opt.step_size
    sample_step = opt.num_steps
    
    noise_list = [torch.randn(zi.size()).cuda() for zi in z_list]
    
    for idx, image in enumerate(loader):
        image = image[0] if len(image) > 1 else image
        
        if opt.renormalize:
            image = 2. * image - 1.
        
        image = image.cuda()
        requires_grad(model.parameters(), False)
        requires_grad(VAE.parameters(), False)

        model.eval()
        VAE.eval()
        
        noise_x = torch.randn(image.size()).cuda()
        if buffer is not None:       
            buffer_prob = min(opt.max_p, opt.max_p*global_step/opt.anneal_step)
            eps_z_nograd = sample_buffer(buffer, z_list, batch_size = image.size(0), p=buffer_prob)
            eps_z = [Variable(eps_zi, requires_grad=True) for eps_zi in eps_z_nograd]
        else:
            eps_z = [Variable(torch.Tensor(zi.size()).normal_(0, 1.).cuda() , requires_grad=True) for zi in z_list]

        eps_x = torch.Tensor(image.size()).normal_(0, 1.).cuda()    
        eps_x = Variable(eps_x, requires_grad = True)
        
        for k in range(sample_step):            
            logits, _, log_p_total = VAE.sample(opt.batch_size, t, eps_z)
            output = VAE.decoder_output(logits)
            neg_x = output.sample(eps_x) 
            log_pxgz = output.log_prob(neg_x).sum(dim = [1,2,3])
            
            if opt.renormalize:
                neg_x_renorm = 2. * neg_x - 1.
            else:
                neg_x_renorm = neg_x
                
                
            dvalue = model(neg_x_renorm) - log_p_total - log_pxgz 
        
            
            dvalue = dvalue.mean()
            dvalue.backward()

            for i in range(len(eps_z)):               
                noise_list[i].normal_(0, 1)
                eps_z[i].data.add_(-0.5*step_size, eps_z[i].grad.data * opt.batch_size)

                eps_z[i].data.add_(np.sqrt(step_size), noise_list[i].data)    
                eps_z[i].grad.detach_()
                eps_z[i].grad.zero_()
            
            noise_x.normal_(0, 1)
            eps_x.data.add_(-0.5*step_size, eps_x.grad.data * opt.batch_size)
            eps_x.data.add_(np.sqrt(step_size), noise_x.data)
            eps_x.grad.detach_()
            eps_x.grad.zero_()

        
        eps_z = [eps_zi.detach() for eps_zi in eps_z]
        eps_x = eps_x.detach()

        requires_grad(model.parameters(), True)
   
        model.zero_grad()
        logits, _, _ = VAE.sample(opt.batch_size, t, eps_z)
        output = VAE.decoder_output(logits)
        
        if opt.use_mu_cd:
            neg_x = output.dist.mu
            if not opt.renormalize:
                neg_x = 0.5 * neg_x + 0.5
        else: 
            neg_x = output.sample(eps_x)
            if opt.renormalize:
                neg_x = 2. * neg_x - 1.
        


        pos_out = model(image)
        neg_out = model(neg_x)
                    
        
        norm_loss = model.spectral_norm_parallel()
        loss_reg_s = opt.alpha_s * norm_loss
        
        loss = pos_out.mean() - neg_out.mean()
        loss_total = loss + loss_reg_s 
        


        loss_total.backward()
        utils.average_gradients(model.parameters(), eval_args.distributed)

        
        if opt.grad_clip:
            clip_grad(model.parameters(), optimizer)


        optimizer.step()
        
        if buffer is not None:
            buffer.push(eps_z)

        d_s_t.append(loss.mean().item())
        
        
        
        if eval_args.global_rank == 0:
            print('step {}, energy diff {}'.format(global_step,loss.mean().item()))

            if global_step % 100 == 0:
                
                neg_img = output.dist.mu
                torchvision.utils.save_image(
                    neg_img,
                    model_path + '/images/sample_iter_{}.png'.format(global_step),
                    nrow=16,
                    normalize=True
                )
                torch.save(d_s_t, model_path + 'd_s_t')

    
            
            if global_step >= eval_args.not_save_before and global_step % 500 == 0 or (global_step >2500 and global_step % 100 == 0):
                state_dict = {}
                state_dict['model'] = model.state_dict()
                state_dict['optimizer'] = optimizer.state_dict()
                torch.save(state_dict, model_path + 'EBM_{}.pth'.format(global_step))
                

            
        global_step += 1
        
    return d_s_t, global_step, output




def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training VAEBM distributed')    
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/VAE_checkpoint.pt',
                        help='location of the nvae checkpoint')
    parser.add_argument('--experiment', default='EBM', help='experiment name, model chekcpoint and samples will be saved here')

    # data
    parser.add_argument('--dataset', type=str, default='celeba_256',
                        help='which dataset to use')
    parser.add_argument('--im_size', type=int, default=256, help='size of image')

    parser.add_argument('--data', type=str, default='./data/celeba_256/',
                        help='location of the data file')
    # optimization
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=4e-5,
                        help='init learning rate')
    parser.add_argument('--wd', type=float, default=3e-5,
                        help='weight decay')

    parser.add_argument('--epochs', type=int, default=400,
                        help='num of training epochs')

    parser.add_argument('--grad_clip', dest='grad_clip', action='store_false', help='clip grad as done in Du et al.')

    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    # EBM training
    parser.add_argument('--n_channel', type=int, default = 64, help='initial number of channels of EBM')
    parser.add_argument('--alpha_s', type=float, default=0.3, help='spectral reg coef')

    parser.add_argument('--step_size', type=float, default=3e-6, help='step size for LD')
    parser.add_argument('--num_steps', type=int, default=6, help='number of LD steps')
    parser.add_argument('--total_iter', type=int, default=12000, help='number of training iteration')
    parser.add_argument('--data_init', dest='data_init', action='store_true', help='data depedent init for weight norm')
    parser.add_argument('--not_save_before', type=int, default=0, help='not save model before certain number of iterations')
    parser.add_argument('--use_mu_cd', dest='use_mu_cd', action='store_true', help='use mean or sample from the decoder to compute CD loss')
    parser.add_argument('--renormalize', dest='renormalize', action='store_true', help = 'renormalize [0,1] to [-1,-1]')
    parser.add_argument('--temperature', type=float, default=1., help='temperature of sampling NVAE prior')

    #buffer
    parser.add_argument('--use_buffer', dest='use_buffer', action='store_true', help='use persistent training, default is false')
    parser.add_argument('--max_p', type=float, default=0.6, help='maximum p of sampling from buffer')
    parser.add_argument('--anneal_step', type=float, default=3000., help='p annealing step')
    parser.add_argument('--buffer_size', type=int, default = 2000, help='size of buffer')


    eval_args = parser.parse_args()
    if eval_args.dataset in {'cifar10', 'mnist'}:
        eval_args.data = os.path.join(eval_args.data, eval_args.dataset)


    size = eval_args.num_process_per_node #number of GPUs

    if size > 1:
        eval_args.distributed = True
        processes = []
        for rank in range(size):
            eval_args.local_rank = rank
            global_rank = rank + eval_args.node_rank * eval_args.num_process_per_node
            global_size = eval_args.num_proc_node * eval_args.num_process_per_node
            eval_args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (eval_args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, eval_args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        eval_args.distributed = True
        init_processes(0, size, main, eval_args)


