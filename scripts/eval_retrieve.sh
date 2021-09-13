#!/usr/bin/env bash

## evaluate retrieval results
python eval_retrieve.py \
--dataroot G:\\dataset\\MCB_B \
--database_path G:/dataset/MCB_B/database/fix_aec8_aug4_bd_na/encoder1/traintest.pt \
--query_path G:/dataset/MCB_B/database/fix_aec8_aug4_bd_na/encoder1/test.pt \
--search_methods IndexFlatL2 \
--num_neigb 6 \
--show_examples True \
--save_examples True \
--gpu_ids -1 \