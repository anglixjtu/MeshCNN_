import time
import os
import torch
import numpy as np
import torch

from src.options.retrieval_options import RetrievalOptions
from src.util.retriever import Retriever
from src.util.evaluation import Evaluator
from src.util.logger import Logger
from src.util.util import get_labels_from_path, mkdir
from src.util.visualization import visualize_retrieval


def eval_retrieve(opt):
    logger = Logger(opt)

    # load database and query features
    if os.path.isdir(opt.database_path) or\
       not os.path.exists(opt.database_path):
        raise FileNotFoundError('opt.database_path is not found or not a file')
    if os.path.isdir(opt.query_path) or\
       not os.path.exists(opt.query_path):
        raise FileNotFoundError('opt.query_path is not found or not a file')
    database = torch.load(opt.database_path)
    querybase = torch.load(opt.query_path)
    fea_db, paths_db = database['embeddings'], database['paths']
    fea_q, paths_q = querybase['embeddings'], querybase['paths']

    logger.loggers['runs'].info('Retrieving from database of '
                                'size %d. \n' % len(paths_db))

    retriever = Retriever(opt.num_neigb, opt.search_methods)
    evaluator = Evaluator(opt.evaluation_metrics)

    _, ranked_list, dissm = retriever.get_similar(fea_q, fea_db, paths_db)

    logger.loggers['runs'].info('Done!\n')

    # Evaluation
    logger.loggers['runs'].info('Start evaluating ...\n')
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
        from tests.show_namelist import show_namelist
        paths_retr = []
        dissm_retr = np.zeros((len(show_namelist()), opt.num_neigb))
        out_path = None
        show_names = [os.path.join(opt.dataroot, 'raw', x)
                      for x in show_namelist()]
        for i, show_name in enumerate(show_names):
            idx = paths_q.index(show_name)
            paths_retr.append(ranked_list[idx])
            dissm_retr[i, :] = dissm[idx, :]
            if opt.save_examples:
                gt_label = get_labels_from_path(show_name)
                save_dir = os.path.join(opt.checkpoints_dir, opt.name,
                                        database['which_layer'])
                mkdir(save_dir)
                out_path = os.path.join(save_dir,
                                        gt_label + '_%d.png' % (i % 2 + 1))
            visualize_retrieval([show_name], [ranked_list[idx]],
                                dissm[idx:idx+1, :],
                                out_path=out_path, show_self=True)
        if opt.save_examples:
            save_dir = os.path.join(opt.checkpoints_dir, opt.name,
                                    database['which_layer'])
            mkdir(save_dir)
            out_path = os.path.join(save_dir, 'all.png')
        visualize_retrieval(show_names, paths_retr, dissm_retr,
                            out_path=out_path, show_self=True)


if __name__ == '__main__':
    opt = RetrievalOptions().parse()
    # TODO: clean options
    eval_retrieve(opt)
