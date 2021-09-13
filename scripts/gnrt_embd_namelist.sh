#!/usr/bin/env bash

## generate embeddings for database
## set pooling = global_mean_pool if which_layer = encoderx
## else pooling = None if which_layer = gb_pool
python generate_embedding_db.py \
--dataroot G:/dataset/MCB_B/ \
--namelist_file G:/dataset/MCB_B/namelist/mcbb_5c1000s.json \
--name fix_aec8_aug4_bd_na \
--input_nc 8 \
--set test \
--which_epoch latest \
--which_layer encoder1 \
--pooling global_mean_pool \
--normalize 2 \
--batch_size 16 \
--num_threads 4 \
--gpu_ids -1 \
--save_dir G:/dataset/MCB_B/database \