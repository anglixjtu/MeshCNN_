from options.retrieval_options import RetrievalOptions
from data import CreateDataset
from torch_geometric.data import DataLoader
from models import CreateModel
from util.retriever import Retriever


def run_retrieve():
    print('Running Retrieval')
    opt = RetrievalOptions().parse() 
    retriever = Retriever(opt)
    dataset = CreateDataset(opt)
    model = CreateModel(opt)

    fea_db = retriever.extract_database_features(model, dataset)

    query_set = dataset
    fea_q = fea_db

    dist, ranked_list, dissm = retriever.retrieve(model, query_set, dataset, fea_db, fea_q)
    idx_query = list(range(len(dataset)))

    retriever.show_embedding(fea_db, idx_query)

    retriever.evaluate_results(idx_query, ranked_list)

    i=0
    #query = dataset[i]
    #dist, ranked_list = retriever.retrive_one_example(model, query, dataloader, fea_db)

    retriever.show_results(i, ranked_list, dissm)

    debug = 295


if __name__ == '__main__':
    run_retrieve()

