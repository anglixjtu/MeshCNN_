import os
import argparse
from src.util import util
from .extractor_options import ExtractorOptions
import torch


class RetrievalOptions(ExtractorOptions):

    def initialize(self):
        ExtractorOptions.initialize(self)
        # for retrieve
        self.parser.add_argument('--database_path', type=str,
                                 help='Path to database(.pt) file.')
        self.parser.add_argument('--query_path', type=str,
                                 help='Path to query(.pt) file or '
                                 'load mesh(.obj).')
        self.parser.add_argument('--search_methods', type=str,
                                 default='IndexFlatL2',
                                 help='IndexFlatL2, etc')
        self.parser.add_argument('--evaluation_metrics', nargs='+', type=str,
                                 default=['patk', 'map', 'ndcg'],
                                 help='IndexFlatL2, etc')
        self.parser.add_argument('--num_neigb', type=int, default=6,
                                 help='# of returned items')
        self.parser.add_argument('--show_examples', action='store_true',
                                 help='whether to visualize example retrieval'
                                 'results or not')
        self.parser.add_argument('--save_examples', action='store_true',
                                 help='whether to save example retrieval'
                                 'results or not')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()

        if self.opt.name is None:
            self.opt.name = self.opt.database_path.split('/')[-3]

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        return self.opt
