import time
import os
import torch
import numpy as np

from src.options.retrieval_options import RetrievalOptions
from src.util.retriever import Retriever
from src.util.evaluation import Evaluator
from src.util.logger import Logger
from src.util.util import get_labels_from_path
from src.util.visualization import visualize_retrieval

show_namelist = [['test/gear/00070120.obj', 'test/gear/00000219.obj',
                  'test/motor/00053644.obj', 'test/motor/00064756.obj',
                  'test/pin/00052497.obj', 'test/pin/00052388.obj',
                  'test/rotor/00067691.obj', 'test/rotor/00067746.obj',
                  'test/washer/00040300.obj', 'test/washer/00041575.obj']]


def eval_retrieve(opt):
    logger = Logger(opt)
    logger.loggers['runs'].info('Evaluating retrieval results...')

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

    retriever = Retriever(opt.num_neigb, opt.search_methods)
    evaluator = Evaluator(opt.evaluation_metrics)

    paths_retr, paths_q_selected = [], []
    paths_q_show = ['G:\\dataset\\MCB_B\\MCB_B\\' +
                    x for x in show_namelist[0]]
    dissm_retr = np.zeros((10, opt.num_neigb))
    ii = 0
    out_path = None

    for i, path_q in enumerate(paths_q):
        _, ranked_list, dissm = retriever.get_similar(fea_q[i:i+1, :],
                                                      fea_db, paths_db)
        gt_label = get_labels_from_path(path_q)
        pred_labels = get_labels_from_path(ranked_list)
        evaluator.update(gt_label, pred_labels, len(pred_labels))

        # for visualization
        if (path_q in paths_q_show) and opt.show_examples:
            paths_q_selected.append(path_q)
            paths_retr.append(ranked_list)
            dissm_retr[ii, :] = dissm
            ii += 1
            if opt.save_examples:
                out_path = os.path.join(opt.checkpoints_dir, opt.name,
                                       gt_label+'_%d.png'%ii)
            visualize_retrieval([path_q], [ranked_list], dissm,
                                out_path=out_path)

    # log
    logger.record_retracc(opt.num_neigb,
                          database['which_layer'],
                          database['pooling'],
                          database['feature_size'],
                          database['normalize'],
                          opt.search_methods,
                          evaluator.metrics)

    logger.loggers['runs'].info('Done!\n')

    if opt.show_examples:
        if opt.save_examples:
            out_path = os.path.join(opt.checkpoints_dir,
                                    opt.name, 'all.png')
        visualize_retrieval(paths_q_selected, paths_retr, dissm_retr,
                            out_path=out_path)


if __name__ == '__main__':
    opt = RetrievalOptions().parse()
    # TODO: clean options
    eval_retrieve(opt)
