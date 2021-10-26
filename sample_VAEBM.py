# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for VAEBM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

'''
Code for generating samples from VAEBM
'''

import argparse
import torch
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.autograd import Variable
from nvae_model import AutoEncoder
import utils
from train_VAEBM_distributed import init_processes
import torchvision
from tqdm import tqdm
from ebm_models import EBM_CelebA64, EBM_LSUN64, EBM_CIFAR32, EBM_CelebA256



def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        for i in range(iter):
            if i % 10 == 0:
                print('setting BN statistics iter %d out of %d' % (i+1, iter))
            model.train()
            model.sample(num_samples, t)
        model.eval()

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


#%%
def sample_from_EBM(model, VAE, t, opt):
    parameters = model.parameters()

    requires_grad(VAE.parameters(), False)
    requires_grad(parameters, False)

    step_size = opt.step_size
    sample_step = opt.num_steps
    with torch.no_grad():
        _, z_list, _ = VAE.sample(opt.batch_size, t) 
        image = torch.zeros(opt.batch_size,3,opt.im_size,opt.im_size) #placeholder just to get the size of image

    
    model.eval()
    VAE.eval()
    
    noise_x = torch.randn(image.size()).cuda()
    noise_list = [torch.randn(zi.size()).cuda() for zi in z_list]
    
    eps_z = [Variable(torch.Tensor(zi.size()).normal_(0, 1.).cuda() , requires_grad=True) for zi in z_list]

        
    eps_x = torch.Tensor(image.size()).normal_(0, 1.).cuda()    
    eps_x = Variable(eps_x, requires_grad = True)

    
    for k in tqdm(range(sample_step)):
        
        
        logits, _, log_p_total = VAE.sample(opt.batch_size, t, eps_z)
        output = VAE.decoder_output(logits)
        neg_x = output.sample(eps_x = eps_x)   
        if opt.renormalize:
            neg_x_renorm = 2. * neg_x - 1.
        else:
            neg_x_renorm = neg_x
        


        log_pxgz = output.log_prob(neg_x_renorm).sum(dim = [1,2,3])
        

        dvalue = model(neg_x_renorm) - log_p_total - log_pxgz 

        dvalue = dvalue.mean()
        dvalue.backward()
        for i in range(len(eps_z)):               
            noise_list[i].normal_(0, 1)

            eps_z[i].data.add_(-0.5*step_size, eps_z[i].grad.data * opt.batch_size )

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
    logits, _, _ = VAE.sample(opt.batch_size, t, eps_z)
    output = VAE.decoder_output(logits)
    final_sample = output.dist.mu

        
    return final_sample

    


def main(eval_args):
    # ensures that weight initializations are all the same
    logging = utils.Logger(eval_args.local_rank, eval_args.save)

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

    incompatible_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))

    t = 1.
    
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

        
    with torch.no_grad():
        EBM_model(torch.rand(10,3,eval_args.im_size,eval_args.im_size).cuda()) #for weight norm data dependent init

    state_EBM = torch.load(eval_args.ebm_checkpoint)
    EBM_model.load_state_dict(state_EBM['model'])
        
    iter_needed = eval_args.num_samples // eval_args.batch_size 
    model.eval()
    for i in range(iter_needed):
        i = i 
        sample = sample_from_EBM(EBM_model, model, t, eval_args)

        for j in range(sample.size(0)):
                   torchvision.utils.save_image(sample[j],(eval_args.savedir+'/EBM_sample_50k/{}.png').format(j+i*eval_args.batch_size),
                                                normalize=True)
        print(i)
    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample from VAEBM')
    # experimental results
    parser.add_argument('--checkpoint', type=str, default='/tmp/nvae/checkpoint.pth',
                        help='location of the nvae checkpoint')
    parser.add_argument('--ebm_checkpoint', type=str, default='/tmp/ebm/checkpoint.pth',
                        help='location of the EBM checkpoint')
    parser.add_argument('--save', type=str, default='/tmp/nasvae/expr',
                        help='location of the NVAE logging')

    
    parser.add_argument('--dataset', type=str, default='celeba_64',
                        help='dataset name')
    parser.add_argument('--im_size', type=int, default=64, help='size of image')


    parser.add_argument('--is_local', action='store_true', default=False,
                        help='Settings this to true will load data on the local machine.')
 
    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    
    
    parser.add_argument('--savedir', default='./samples/', type=str, help='path to save samples for eval')
    parser.add_argument('--num_samples', type=int, default = 10000, help='number of samples to generate')

    parser.add_argument('--batch_size', type=int, default = 40, help='batch size for generating samples from EBM')
    parser.add_argument('--n_channel', type=int, default = 64, help='initial number of channels of EBM')
    parser.add_argument('--data_init', dest='data_init', action='store_false', help='data depedent init for weight norm')
    parser.add_argument('--renormalize', dest='renormalize', action='store_true', help = 'renormalize [0,1] to [-1,-1]')

    parser.add_argument('--step_size', type=float, default=5e-6, help='step size for LD')
    parser.add_argument('--num_steps', type=int, default=20, help='number of LD steps')

    args = parser.parse_args()


    args.distributed = False
    init_processes(0, 1, main, args)