import os
import argparse
from src.util import util


class RetrievalOptions(object):
    # TODO: inherit from BaseOption
    def __init__(self):
        self.parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
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
        self.parser.add_argument('--name', type=str,
                                 help='name of the experiment. It decides '
                                 'where to store models and log files')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
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

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdir(expr_dir)

        return self.opt
