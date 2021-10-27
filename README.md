# Official PyTorch implementation of "VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models" [(ICLR 2021 Spotlight Paper)](https://arxiv.org/abs/2010.00654) #

<div align="center">
  <a href="https://xavierxiao.github.io/" target="_blank">Zhisheng&nbsp;Xiao</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://scholar.google.com/citations?hl=en&user=rFd-DiAAAAAJ&view_op=list_works&sortby=pubdate" target="_blank">Karsten&nbsp;Kreis</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://jankautz.com/" target="_blank">Jan&nbsp;Kautz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="http://latentspace.cc/arash_vahdat/" target="_blank">Arash&nbsp;Vahdat</a>
</div>
<br>
<br>


VAEBM trains an energy network to refine the data distribution learned by an [NVAE](https://arxiv.org/abs/2007.03898), where the enery network and the VAE jointly define an Energy-based model.
The NVAE is pretrained before training the energy network, and please refer to [NVAE's implementation](https://github.com/NVlabs/NVAE) for more details about constructing and training NVAE.

## Set up datasets ##
We trained on several datasets, including CIFAR10, CelebA64, LSUN Church 64 and CelebA HQ 256. 
For large datasets, we store the data in LMDB datasets for I/O efficiency. Check [here](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) for information regarding dataset preparation.

## Training NVAE ##
We use the following commands on each dataset for training the NVAE backbone. To train NVAEs, please use its original [codebase](https://github.com/NVlabs/NVAE) with commands given here.
#### CIFAR-10 (8x 16-GB GPUs) ####
```
python train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
      --num_channels_enc 128 --num_channels_dec 128 --epochs 400 --num_postprocess_cells 2 --num_preprocess_cells 2 \
      --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
      --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 30 --batch_size 32 \
      --weight_decay_norm 1e-1 --num_nf 1 --num_mixture_dec 1 --fast_adamax  --arch_instance res_mbconv \
      --num_process_per_node 8 --use_se --res_dist
```
#### CelebA-64 (8x 16-GB GPUs) ####
```
python train.py --data  $DATA_DIR/celeba64_lmdb --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 \
      --num_channels_enc 48 --num_channels_dec 48 --epochs 50 --num_postprocess_cells 2 --num_preprocess_cells 2 \
      --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
      --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 5 \
      --batch_size 32 --num_nf 1 --num_mixture_dec 1 --fast_adamax  --warmup_epochs 1 --arch_instance res_mbconv \
      --num_process_per_node 8 --use_se --res_dist
```
#### CelebA-HQ-256 (8x 32-GB GPUs) ####
```
python train.py -data  $DATA_DIR/celeba/celeba-lmdb --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_256 \
      --num_channels_enc 32 --num_channels_dec 32 --epochs 200 --num_postprocess_cells 2 --num_preprocess_cells 2 \
      --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_preprocess_blocks 1 \
      --num_postprocess_blocks 1 --weight_decay_norm 1e-2 --num_x_bits 5 --num_latent_scales 5 --num_groups_per_scale 4 \
      --num_nf 2 --batch_size 8 --fast_adamax  --num_mixture_dec 1 \
      --weight_decay_norm_anneal  --weight_decay_norm_init 1e1 --learning_rate 6e-3 --arch_instance res_mbconv \
      --num_process_per_node 8 --use_se --res_dist
```
#### LSUN Churches Outdoor 64 (8x 16-GB GPUs) ####
```
python train.py --data $DATA_DIR/LSUN/ --root $CHECKPOINT_DIR --save $EXPR_ID --dataset lsun_church_64 \
      --num_channels_enc 48 --num_channels_dec 48 --epochs 60 --num_postprocess_cells 2 --num_preprocess_cells 2 \
      --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
      --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 5 \
      --batch_size 32 --num_nf 1 --num_mixture_dec 1 --fast_adamax  --warmup_epochs 1 --arch_instance res_mbconv \
      --num_process_per_node 8 --use_se --res_dist
```
## Training VAEBM ##
We use the following commands on each dataset for training VAEBM. Note that you need to train the NVAE on corresponding dataset before running the training command here.
After training the NVAE, pass the path of the checkpoint to the `--checkpoint` argument.

Note that the training of VAEBM will eventually explode (See Appendix E of our paper), and therefore it is important to save checkpoint regularly. After the training explodes, stop running the code and use the last few saved checkpoints for testing.
#### CIFAR-10 ####

We train VAEBM on CIFAR-10 using one 32-GB V100 GPU. 
```
python train_VAEBM.py  --checkpoint ./checkpoints/cifar10/checkpoint.pt --experiment cifar10_exp1
--dataset cifar10 --im_size 32 --data ./data/cifar10 --num_steps 10 
--wd 3e-5 --step_size 8e-5 --total_iter 30000 --alpha_s 0.2 --lr 4e-5 --max_p 0.6 
--anneal_step 5000. --batch_size 32 --n_channel 128
```

#### CelebA 64 ####

We train VAEBM on CelebA 64 using one 32-GB V100 GPU. 
```
python train_VAEBM.py --checkpoint ./checkpoints/celeba_64/checkpoint.pt --experiment celeba64_exp1 --dataset celeba_64 
--im_size 64 --lr 5e-5 --batch_size 32 --n_channel 64 --num_steps 10 --use_mu_cd --wd 3e-5 --step_size 5e-6 --total_iter 30000 
--alpha_s 0.2 
```

#### LSUN Church 64 ####

We train VAEBM on LSUN Church 64 using one 32-GB V100 GPU. 
```
python train_VAEBM.py --checkpoint ./checkpoints/lsun_church/checkpoint.pt --experiment lsunchurch_exp1 --dataset lsun_church 
--im_size 64 --batch_size 32 --n_channel 64 --num_steps 10 --use_mu_cd --wd 3e-5 --step_size 4e-6 --total_iter 30000 --alpha_s 0.2 --lr 4e-5 
--use_buffer --max_p 0.6 --anneal_step 5000

```

#### CelebA HQ 256 ####

We train VAEBM on CelebA HQ 256 using four 32-GB V100 GPUs. 
```
python train_VAEBM_distributed.py --checkpoint ./checkpoints/celeba_256/checkpoint.pt --experiment celeba256_exp1 --dataset celeba_256
--num_process_per_node 4 --im_size 256 --batch_size 4 --n_channel 64 --num_steps 6 --use_mu_cd --wd 3e-5 --step_size 3e-6 
--total_iter 9000 --alpha_s 0.3 --lr 4e-5 --use_buffer --max_p 0.6 --anneal_step 3000 --buffer_size 2000
```

## Sampling from VAEBM ##
To generate samples from VAEBM after training, run ```sample_VAEBM.py```, and it will generate 50000 test images in your given path. When sampling, we typically use 
longer Langvin dynamics than training for better sample quality, see Appendix E of the [paper](https://arxiv.org/abs/2010.00654) for the step sizes and number of steps we use to obtain test samples
for each dataset. Other parameters that ensure successfully loading the VAE and energy network are the same as in the training codes. 

For example, the script used to sample CIFAR-10 is
```
python sample_VAEBM.py --checkpoint ./checkpoints/cifar_10/checkpoint.pt --ebm_checkpoint ./saved_models/cifar_10/cifar_exp1/EBM.pth 
--dataset cifar10 --im_size 32 --batch_size 40 --n_channel 128 --num_steps 16 --step_size 8e-5 
```

For CelebA 64, 
```
python sample_VAEBM.py --checkpoint ./checkpoints/celeba_64/checkpoint.pt --ebm_checkpoint ./saved_models/celeba_64/celeba64_exp1/EBM.pth 
--dataset celeba_64 --im_size 64 --batch_size 40 --n_channel 64 --num_steps 20 --step_size 5e-6 
```
For LSUN Church 64, 
```
python sample_VAEBM.py --checkpoint ./checkpoints/lsun_church/checkpoint.pt --ebm_checkpoint ./saved_models/lsun_chruch/lsunchurch_exp1/EBM.pth 
--dataset lsun_church --im_size 64 --batch_size 40 --n_channel 64 --num_steps 20 --step_size 4e-6 
```

For CelebA HQ 256, 
```
python sample_VAEBM.py --checkpoint ./checkpoints/celeba_256/checkpoint.pt --ebm_checkpoint ./saved_models/celeba_256/celeba256_exp1/EBM.pth 
--dataset celeba_256 --im_size 256 --batch_size 10 --n_channel 64 --num_steps 24 --step_size 3e-6 
```


## Evaluation ##
After sampling, use the [Tensorflow](https://github.com/bioinf-jku/TTUR) or [PyTorch](https://github.com/mseitzer/pytorch-fid) 
implementation to compute the FID scores. For example, when using the Tensorflow implementation, you can obtain the FID score by saving the training images in ```/path/to/training_images``` and running the script:
```
python fid.py /path/to/training_images /path/to/sampled_images
```

For CIFAR-10, the training statistics can be downloaded from [here](https://github.com/bioinf-jku/TTUR#precalculated-statistics-for-fid-calculation), and the FID score can be computed by running
```
python fid.py /path/to/sampled_images /path/to/precalculated_stats.npz
```

For the Inception Score, save samples in a single numpy array with pixel values in range [0, 255] and simply run 
```
python ./thirdparty/inception_score.py --sample_dir /path/to/sampled_images
```
where the code for computing Inception Score is adapted from [here](https://github.com/tsc2017/Inception-Score).

## License ##
Please check the LICENSE file. VAEBM may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).

## Bibtex ##
Cite our paper using the following bibtex item:

```
@inproceedings{
xiao2021vaebm,
title={VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models},
author={Zhisheng Xiao and Karsten Kreis and Jan Kautz and Arash Vahdat},
booktitle={International Conference on Learning Representations},
year={2021}
}
```