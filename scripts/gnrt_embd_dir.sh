#!/usr/bin/env bash

## generate embeddings for database
## set pooling = global_mean_pool if which_layer = encoderx
## else pooling = None if which_layer = gb_pool
python generate_embedding_db.py \
--dataroot data/mcbb_tiny/ \
--name fix_ae_8 \
--set test \
--which_epoch latest \
--which_layer encoder0 \
--pooling global_mean_pool \
--normalize 2 \
--batch_size 16 \
--num_threads 4 \
--gpu_ids -1 \
--arch mesh_aec \
--save_dir checkpoints \