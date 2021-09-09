#!/usr/bin/env bash

## training the autoencoder network
python train.py \
--dataroot data/mcbb_tiny \
--name fix_tiny_ae8_aug1 \
--ncf 32 64 64 128 512 \
--mode autoencoder \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--input_nc 8 \
--neigbs 11 \
--batch_size 2 \
--num_threads 4 \
--gpu_ids -1 \
--arch mesh_aec \
--loss chamfer \