from .base_options import BaseOptions


class RetrievalOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--feature_dir', type=str, default='./features/', help='saves features here.')
        self.parser.add_argument('--phase', type=str, default='retrieval', help='train, val, test, retrieval, etc') #todo delete.
        self.parser.add_argument('--which_layer', type=str, default='gb_pool', help='which layer to extract features?') 
        self.parser.add_argument('--search_methods', nargs='+', type=str, default='IndexFlatL2', help='IndexFlatL2, etc') 
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--num_aug', type=int, default=1, help='# of augmentation files')
        self.parser.add_argument('--num_neigb', type=int, default=4, help='# of augmentation files')
        self.parser.add_argument('--query_index', type=int, default=0, help='test the results of an example')
        self.is_train = False