#!/usr/bin/env bash

## training the autoencoder network
python train.py \
--dataroot data/mcbb_tiny \
--name fix_tiny_aec7_aug9 \
--mode autoencoder \
--aug_method 9 \
--niter_decay 100 \
--niter 100 \
--input_nc 7 \
--batch_size 2 \
--num_threads 4 \
--gpu_ids 0 \
--arch mesh_aec \
--loss chamfer \
--neigbs 11 \