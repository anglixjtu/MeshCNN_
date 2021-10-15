import argparse
import os
from src.util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot',
                                 required=True,
                                 help='path to meshes')
        self.parser.add_argument('--namelist_file',
                                 type=str,
                                 default=None,
                                 help='path to namelist file')
        self.parser.add_argument('--sample_mesh',
                                 type=str,
                                 default='trimesh',
                                 help='method for downsample mesh')
        self.parser.add_argument('--mode', choices={'classification',
                                                    'autoencoder',
                                                    'autoencoder_pt',
                                                    'autoencoder_glb'},
                                 default='classification')
        self.parser.add_argument('--ninput_edges',
                                 type=int, default=750,
                                 help='# of input edges')
        self.parser.add_argument('--neigbs', type=int,
                                 default=11,
                                 help='if >0, use neigbs to compute neighbors,'
                                      'else use 1-ring neighbors')
        self.parser.add_argument('--loss',
                                 type=str,
                                 default='ce',
                                 choices={'ce', 'mse',
                                          'chamfer', 'chamfer_pt'},
                                 help='loss function for training')
        self.parser.add_argument('--dataset_mode',
                                 type=str,
                                 default='edge',
                                 choices={'edge', 'vertice'},
                                 help='loss function for training')
        self.parser.add_argument('--max_dataset_size', type=int,
                                 default=float("inf"), help='Maximum number of samples per epoch')
        self.parser.add_argument('--len_feature', action='store_true',
                                 help='will not use edge length as features')
        # network params
        self.parser.add_argument('--which_epoch',
                                 type=str,
                                 default='latest',
                                 help='which epoch to load trained model')
        self.parser.add_argument('--aug_method', type=str,
                                 default='4',
                                 help='the index for augmentation methods')
        self.parser.add_argument(
            '--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--arch', type=str, default='mconvnet',
                                 help='selects network to use')  # todo add choices
        self.parser.add_argument('--fc_n', type=int, default=100,
                                 help='# between fc and nclasses')  # todo make generic

        self.parser.add_argument(
            '--ncf', nargs='+', default=[32, 64, 64, 128, 512], type=int, help='conv filters')
        self.parser.add_argument(
            '--pool_res', nargs='+', default=[], type=int, help='pooling res')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='instance normalization or batch normalization or group normalization')
        self.parser.add_argument(
            '--num_groups', type=int, default=16, help='# of groups for groupnorm')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02,
                                 help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--input_nc',
                                 type=int, default=8,
                                 help='# of input channels')
        # general params
        self.parser.add_argument(
            '--num_threads', default=3, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument(
            '--name', type=str, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument(
            '--seed', type=int, help='if specified, uses seed')
        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='',
                                 help='exports intermediate collapses to this folder')
        #
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train   # train or test

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

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(
                self.opt.checkpoints_dir,
                self.opt.name, self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdir(expr_dir)

        return self.opt
