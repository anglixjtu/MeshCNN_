from src import data
from src.options.retrieval_options import RetrievalOptions
from src.data import create_dataloader
from torch_geometric.data import DataLoader
from src.models import Model
from src.util.retriever import Retriever
import numpy as np
import torch


def run_retrieve():
    print('Running Retrieval')
    opt = RetrievalOptions().parse()
    retriever = Retriever(opt)
    dataloader, dataset = create_dataloader(opt, 'retrieval')

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))\
        if (opt.gpu_ids and torch.cuda.is_available())\
        else torch.device('cpu')

    model = Model(opt, device, phase='retrieval')

    fea_db = retriever.extract_database_features(model, dataset)

    '''query_namelist =  ['test/gear/00070120.obj', 'test/gear/00000219.obj', 'test/motor/00053644.obj', 'test/motor/00064756.obj', 'test/pin/00052497.obj', 'test/pin/00052388.obj', 'test/rotor/00067691.obj', 'test/rotor/00067746.obj', 'test/washer/00040300.obj', 'test/washer/00041575.obj']

    query_idx = []
    for name in query_namelist:
        query_idx += [retriever.database_namelist.index(name)]

    query_set = dataset[query_idx]
    fea_q = fea_db[query_idx, :]'''

    query_set = dataset
    fea_q = fea_db

    #np.random.shuffle(fea_db) 

    dist, ranked_list, dissm = retriever.retrieve(model, query_set, dataset, fea_db, fea_q)
    idx_query = list(range(len(dataset)))

    # retriever.show_embedding(fea_db, idx_query)

    retriever.evaluate_results(idx_query, ranked_list)

    # retriever.show_results(query_idx, ranked_list, dissm)

    debug = 295


if __name__ == '__main__':
    run_retrieve()

