#!/usr/bin/env bash

## generate embeddings for database
## set pooling = global_mean_pool if which_layer = encoderx
## else pooling = None if which_layer = gb_pool
python generate_embedding_db.py \
--dataroot empty \
--name fix_aept7_aug11 \
--set train test \
--which_epoch latest \
--which_layer gb_pool \
--pooling None \
--normalize 2 \
--batch_size 16 \
--num_threads 4 \
--gpu_ids -1 \
--save_dir checkpoints \