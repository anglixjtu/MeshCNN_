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
    dataloader_d, database = create_dataloader(opt, 'database')
    dataloader_q, query_set = create_dataloader(opt, 'query')

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))\
        if (opt.gpu_ids and torch.cuda.is_available())\
        else torch.device('cpu')

    model = Model(opt, device, phase='retrieval')

    fea_db = retriever.extract_database_features(model, database)

    '''query_namelist =  ['test/gear/00070120.obj', 'test/gear/00000219.obj', 'test/motor/00053644.obj', 'test/motor/00064756.obj', 'test/pin/00052497.obj', 'test/pin/00052388.obj', 'test/rotor/00067691.obj', 'test/rotor/00067746.obj', 'test/washer/00040300.obj', 'test/washer/00041575.obj']

    query_idx = []
    for name in query_namelist:
        query_idx += [database.paths.index(opt.dataroot + name)]

    query_set = database[query_idx]
    fea_q = fea_db[query_idx, :]'''

    fea_q = retriever.extract_database_features(model, query_set)

    #np.random.shuffle(fea_db) 

    dist, ranked_list, dissm = retriever.retrieve(model, query_set, database, fea_db, fea_q)
    idx_query = list(range(len(query_set)))

    # retriever.show_embedding(fea_db, idx_query)

    retriever.evaluate_results(idx_query,  ranked_list, query_set.paths, database.paths)
    '''ranked_list1, dissm1 = {}, {}
    ranked_list2, dissm2 = {}, {}
    ranked_list1['IndexFlatL2'] = ranked_list['IndexFlatL2'][:5, 1:].copy()
    dissm1['IndexFlatL2'] = dissm['IndexFlatL2'][:5, 1:].copy()
    retriever.show_results(query_idx[:5], ranked_list1, database.paths,
                           database.paths, dissm1, (1000, 1200))
    ranked_list2['IndexFlatL2'] = ranked_list['IndexFlatL2'][5:, 1:].copy()
    dissm2['IndexFlatL2'] = dissm['IndexFlatL2'][5:, 1:].copy()
    retriever.show_results(query_idx[5:], ranked_list2, database.paths,
                           database.paths, dissm2, (1000, 1200))'''

    debug = 295


if __name__ == '__main__':
    run_retrieve()

