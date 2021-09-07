from .base_options import BaseOptions
import torch


class ExtractorOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # data params
        self.parser.add_argument('--save_dir',
                                 required=True,
                                 help='path to save embeddings')
        self.parser.add_argument('--set',
                                 type=str,
                                 nargs='+',
                                 default=['train', 'test'],
                                 help='sets for extracting: train, test, etc')
        # extractor params
        self.parser.add_argument('--which_layer', type=str,
                                 default='gb_pool',
                                 help='which layer used to extract embeddings')
        self.parser.add_argument('--pooling',
                                 type=str, default=None,
                                 choices={'global_mean_pool',
                                          'global_add_pool',
                                          'global_max_pool',
                                          'global_sort_pool',
                                          'None'},
                                 help='which layer used for'
                                      'extracing embeddings',)
        self.parser.add_argument('--normalize',
                                 type=int, default=2,
                                 choices={0, 1, 2},
                                 help='normalization method: None, L1, or L2')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()

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
