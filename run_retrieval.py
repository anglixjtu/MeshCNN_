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
    dataloader = DataLoader( dataset, batch_size=1,#opt.batch_size
            shuffle=False,
            num_workers=1)
    model = CreateModel(opt)

    fea_db = retriever.extract_database_features(model, dataset)

    for i, query in enumerate(dataset):
        dist, ranked_list = retriever.retrive_one_example(model, query, dataloader, fea_db)

        retriever.show_results(i, ranked_list)

        debug = 295


if __name__ == '__main__':
    run_retrieve()

