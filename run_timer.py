from options.timer_options import TimerOptions
from data import CreateDataset
from torch_geometric.data import DataLoader
from models import CreateModel
from util.retriever import Retriever
import trimesh as tm
import time
import os
import csv
import numpy as np
from util.timer import *







def run_timer():
    print('Start timer')
    opt = TimerOptions().parse() 

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    log_name = os.path.join(save_dir, 'runtime.txt')
    feature_path = opt.dataroot + opt.feature_dir#'database/mini.csv'
    test_set = opt.train_namelist.split('/')[-1][:-4]
    k = 5 # K nearest neighbors for searching
    n = 1 # iters for test loading features

    # time for load mesh
    record_load_mesh(opt.train_namelist, opt.dataroot, log_name)

    # time for inference
    fea_db = record_inference(opt, test_set, log_name)
    
    with open(feature_path,'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(fea_db)


    # TEST1: directly load data to memory
    # stage 1: load features into RAM
    start_t0 = time.time()
    fea_db = record_load_fea(n, feature_path, test_set, log_name)
    fea_db = fea_db.astype(np.float32)

    # stage 2: search by using FLAT 
    start_t = time.time()
    len_dataset = record_flat(opt, fea_db, k, test_set, log_name)
    end_t = time.time()
    t_search_total = 1000 * (end_t - start_t) / len_dataset
    t_total = 1000 * (end_t - start_t0) - t_search_total * (len_dataset-1)
    message = '======= Time for search in RAM (%s) ==========\n'\
              'Search in RAM (include initialization, data, inference):      [%.2f] ms. \n'\
              'TOTAL TIME:                                                   [%.2f] ms. \n'\
     %(test_set, t_search_total, t_total)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)



if __name__ == '__main__':
    run_timer()

