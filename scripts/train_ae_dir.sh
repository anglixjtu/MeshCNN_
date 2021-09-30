#!/usr/bin/env bash

## training the autoencoder network
python train.py \
--dataroot data/mcbb_tiny \
--name fix_tiny_ae5_aug4 \
--mode autoencoder_pt \
--niter_decay 100 \
--niter 100 \
--input_nc 5 \
--batch_size 2 \
--num_threads 4 \
--gpu_ids -1 \
--arch mesh_aept \
--loss chamfer \