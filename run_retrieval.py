from options.retrieval_options import RetrievalOptions
from data import CreateDataset
from torch_geometric.data import DataLoader
from models import CreateModel
from util.retriever import Retriever
import numpy as np


def run_retrieve():
    print('Running Retrieval')
    opt = RetrievalOptions().parse() 
    retriever = Retriever(opt)
    dataset = CreateDataset(opt)
    model = CreateModel(opt)

    fea_db = retriever.extract_database_features(model, dataset)

    query_namelist =  ['test/gear/00070120.obj', 'test/gear/00000219.obj', 'test/motor/00053644.obj', 'test/motor/00064756.obj', 'test/pin/00052497.obj', 'test/pin/00052388.obj', 'test/rotor/00067691.obj', 'test/rotor/00067746.obj', 'test/washer/00040300.obj', 'test/washer/00041575.obj']

    query_idx = []
    for name in query_namelist:
        query_idx += [retriever.database_namelist.index(name)]

    query_set = dataset[query_idx]
    fea_q = fea_db[query_idx, :]

    #np.random.shuffle(fea_db) 

    dist, ranked_list, dissm = retriever.retrieve(model, query_set, dataset, fea_db, fea_q)
    idx_query = list(range(len(dataset)))

    retriever.show_embedding(fea_db, idx_query)

    #retriever.evaluate_results(idx_query, ranked_list)

    i=0
    retriever.show_results(query_idx, ranked_list, dissm)

    debug = 295


if __name__ == '__main__':
    run_retrieve()

