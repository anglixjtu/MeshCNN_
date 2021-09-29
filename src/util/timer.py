import os
import sys
sys.path.append("..")
import time
import numpy as np
import trimesh as tm
from data import CreateDataset
from torch_geometric.data import DataLoader
from models import CreateModel
from util.retriever import Retriever
import csv
from faiss import IndexFlatL2

def read_fea(file):
    with open(file,'r', newline='') as f:
        f_csv = csv.reader(f)
        for a in f_csv:
            x = np.array(a[0].strip('[]').replace('\n', '').split(' '))
            invalid = np.arange(len(x))[x=='']
            x = np.delete(x, invalid)
            x = x.astype(float)

            fea = np.zeros((len(a), len(x)))
            fea[0, :] = x
            for i, row in enumerate(a[1:]):
                x = np.array(row.strip('[]').replace('\n', '').split(' '))
                invalid = np.arange(len(x))[x=='']
                x = np.delete(x, invalid)
                x = x.astype(float)

                fea[i+1, :] = x
    
    return fea

def record_load_mesh(namelist, dataroot, log_name):
    
    names = open(namelist, "r").readlines()
    start_t = time.time()
    for model_name in open(namelist, "r"):
        model_name = dataroot + model_name.strip('\n')
        model = tm.load(model_name)
    end_t = time.time()
    period = 1000. * (end_t - start_t) / len(names)
    now = time.strftime("%c")

    message = '================ Running time (%s) ================\n'\
    'Load mesh: [%.2f] ms. \n'\
     %(now, period)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)


def record_inference(opt, test_set, log_name):
    
    start_t = time.time()

    retriever = Retriever(opt)
    dataset = CreateDataset(opt)
    model = CreateModel(opt)
    
    fea_db = retriever.extract_database_features(model, dataset)
    end_t = time.time()
    t_enc = 1000 * (end_t - start_t) / len(dataset)

    

    message = '========= Time for extract embeddings (%s) ========\n'\
              'Preprocess:      [%.2f] ms. \n'\
              'Extract feature: [%.2f] ms. \n'\
              'Inference:       [%.2f] ms. \n'\
              'Total embedding: [%.2f] ms. \n'\
     %(test_set, 1000 * opt.t_pp / len(dataset) , 1000 * opt.t_ef / len(dataset),
       1000 * opt.t_infr / len(dataset), t_enc)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

    return fea_db

def record_load_fea(n, feature_path, test_set, log_name):
    start_t = time.time()
    for i in range(n):
        fea_db = read_fea(feature_path)
    end_t = time.time()
    t_load = 1000 * (end_t - start_t) / n
    message = '======= Time for load embeddings from disc (%s) ==========\n'\
              'Load embeddings: [%.2f] ms. \n'\
     %(test_set, t_load)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)
    return fea_db


def record_flat(opt, fea_db, k, test_set, log_name):
    """
    Record the running time for compare all features in RAM
    """
    len_data, dim = fea_db.shape
    np.random.shuffle(fea_db)  
    index = np.random.randint(len_data,size=len_data)
    start_t = time.time()
    retriever = Retriever(opt)
    model = CreateModel(opt)
    

    for i in index: 
        fea_q = fea_db[i, :].reshape(1,-1)

        dist, ranked_list, dissm = retriever.search_indexflat(fea_q, fea_db, dim, k)
        '''index = IndexFlatL2(dim)
        index.add(fea_db) 


        D, I = index.search(fea_q, k)
        # compute dis-similarity score
        max_dist = D.max()
        dissm = D / max_dist'''

    end_t = time.time()
    t_search = 1000 * (end_t - start_t) / len_data


    message = '======= Time for search and embedding in RAM (%s) ==========\n'\
              'Search in RAM (include data, inference):      [%.2f] ms. \n'\
     %(test_set, t_search)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

    return len_data
