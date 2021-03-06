#!/usr/bin/env bash

## training the autoencoder network
python train.py \
--dataroot G:/dataset/MCB_B/ \
--namelist_file G:/dataset/MCB_B/namelist/mcbb_5c1000s.json \
--name fix_tiny_ae_9 \
--ncf 32 64 64 128 512 \
--mode autoencoder \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--niter_decay 100 \
--input_nc 9 \
--neigbs 11 \
--batch_size 2 \
--num_threads 4 \
--gpu_ids -1 \
--arch mesh_aec \
--loss chamfer \
--len_feature True \