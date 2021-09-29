#!/usr/bin/env bash

## evaluate retrieval results
python retrieve.py \
--dataroot G:\\dataset\\MCB_B \
--database_path G:/dataset/MCB_B/database/fix_ae_8/encoder0/traintest.pt \
--query_path test/motor/00064280.obj \
--search_methods IndexFlatL2 \
--num_neigb 6 \
--show_examples True \
--save_examples True \
--gpu_ids -1 \