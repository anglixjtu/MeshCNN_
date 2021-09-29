import time
import os
import torch
import numpy as np
import torch

from src.options.retrieval_options import RetrievalOptions
from src.util.retriever import Retriever
from src.util.evaluation import Evaluator
from src.util.logger import Logger
from src.util.util import get_labels_from_path
from src.util.visualization import visualize_retrieval

import time
import os

from torch.nn.functional import embedding
from src import data

from src.data import create_dataloader
from src.models import Model
from src.options.extractor_options import ExtractorOptions
from src.options.load_options import load_model_opt
from src.util.logger import Logger
from src.util.util import MetricCounter, mkdir
from src.util.embd_extractor import EmbdExtractor


def retrieve(opt):
    logger = Logger(opt)
    logger.loggers['runs'].info('Loading database...')

    # load database
    if os.path.isdir(opt.database_path) or\
       not os.path.exists(opt.database_path):
        raise FileNotFoundError('opt.database_path is not found or not a file')
    database = torch.load(opt.database_path)
    fea_db, paths_db = database['embeddings'], database['paths']

    logger.loggers['runs'].info('Generating embeddings for query data...')
    phase = 'query'
    dataloader, dataset = create_dataloader(opt, phase, namelist=opt.query_path)

    model = Model(opt, phase=phase)

    extractor = EmbdExtractor(database['which_layer'],
                              dataset.raw_file_paths,
                              database['pooling'],
                              normalize=database['normalize'])

    querybase = extractor(model, dataloader)
    fea_q, paths_q = querybase['embeddings'], querybase['paths']

    logger.loggers['runs'].info('Retrieving from database of '
                                'size %d. \n' % len(paths_db))

    retriever = Retriever(opt.num_neigb, opt.search_methods)
    evaluator = Evaluator(opt.evaluation_metrics)

    _, ranked_list, dissm = retriever.get_similar(fea_q, fea_db, paths_db)

    logger.loggers['runs'].info('Done!\n')

    # Evaluation
    logger.loggers['runs'].info('Start evaluation ...\n')
    for i, path_q in enumerate(paths_q):
        gt_label = get_labels_from_path(path_q)
        pred_labels = get_labels_from_path(ranked_list[i])
        evaluator.update(gt_label, pred_labels, len(pred_labels))

    logger.record_retracc(opt.num_neigb,
                          database['which_layer'],
                          database['pooling'],
                          database['feature_size'],
                          database['normalize'],
                          opt.search_methods,
                          evaluator.metrics)

    # visualization
    if opt.show_examples:
        logger.loggers['runs'].info('Start visualization ...\n')
        paths_retr = []
        dissm_retr = np.zeros((len(paths_q), opt.num_neigb))
        out_path = None
        show_names = paths_q
        for idx, show_name in enumerate(show_names):
            paths_retr.append(ranked_list[idx])
            dissm_retr[i, :] = dissm[idx, :]
            if opt.save_examples:
                gt_label = get_labels_from_path(show_name)
                out_path = os.path.join(opt.checkpoints_dir, opt.name,
                                        gt_label + '_%d.png' % (i % 2 + 1))
            visualize_retrieval([show_name], [ranked_list[idx]],
                                dissm[idx:idx+1, :],
                                out_path=out_path)


if __name__ == '__main__':
    opt = RetrievalOptions().parse()
    # TODO: clean options
    retrieve(opt)
