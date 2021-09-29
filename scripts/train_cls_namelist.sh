#!/usr/bin/env bash

## training the classification network
python train.py \
--dataroot G:/dataset/MCB_B/ \
--namelist_file G:/dataset/MCB_B/namelist/mcbb_5c1000s.json \
--name fix_cls_8 \
--ncf 64 64 128 256 512 \
--mode classification \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--input_nc 8 \
--neigbs 11 \
--batch_size 8 \
--num_threads 1 \
--gpu_ids -1 \
--arch mesh_cls \
--loss ce