#!/usr/bin/env bash

## training the autoencoder network
python train.py \
--dataroot data/mcbb_tiny \
--name fix_tiny_aept7_aug11 \
--mode autoencoder_pt \
--aug_method 11 \
--niter_decay 100 \
--niter 100 \
--input_nc 7 \
--batch_size 2 \
--num_threads 4 \
--gpu_ids 0 \
--arch mesh_aept \
--loss chamfer \
--neigbs 11 \