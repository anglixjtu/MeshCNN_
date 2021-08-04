from .base_options import BaseOptions


class ExtractorOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc') #todo delete.
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--num_aug', type=int, default=1, help='# of augmentation files')
        self.parser.add_argument('--save_path', '-sp', default=None, type=str, help='save path for features')
        self.parser.add_argument('--config_file', '-cfg', default=None, metavar='FILE', type=str, help='path to config file')
        self.parser.add_argument('--save_interval', '-si', default=5000, type=int, help='number of features saved in one part file')
        self.is_train = False