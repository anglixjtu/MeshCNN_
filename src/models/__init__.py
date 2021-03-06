import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch_geometric.utils import to_dense_batch

from .base_model import BaseModel


class Model(BaseModel):
    """ Class for CNN models
    :args opt: structure containing configuration params
    e.g.,
    --mode -> classification / autoencoder)
    --arch -> network type
    """

    def __init__(self, opt, phase='train'):
        self.mode = opt.mode
        self.input_nc = opt.input_nc
        self.saved_opts = ['arch', 'fc_n', 'input_nc',
                           'mode', 'ncf', 'neigbs',
                           'ninput_edges', 'nclasses',
                           'aug_method']

        super(Model, self).__init__(opt, phase)

    def set_input(self, data):
        if self.mode == 'classification' and\
           self.phase in ['train', 'test']:
            labels = data[1]
            data = data[0]
            if not hasattr(data, 'batch'):
                data.batch = torch.zeros(len(data.x), 1)
                data.batch = data.batch.long()
            self.data = data.to(self.device)
            self.labels = labels.long().to(self.device)
        else:
            data = data[0]
            if not hasattr(data, 'batch'):
                data.batch = torch.zeros(len(data.x))
                data.batch = data.batch.long()
            self.data = data.to(self.device)

    def forward(self):
        out, embeddings = self.net(self.data.x,
                                   self.data.edge_index,
                                   self.data.batch)
        return out, embeddings

    def define_loss(self):
        if self.loss_name == 'ce':
            return CrossEntropyLoss()
        elif self.loss_name == 'mse':
            return MSELoss()
        elif self.loss_name in ['chamfer']:
            try:
                from chamferdist import ChamferDistance
                chamfer_loss = ChamferDistance()
            except ImportError:
                print('ChamferDist not installed, using ChamferLoss')
                from src.util.losses import ChamferLoss
                chamfer_loss = ChamferLoss()
            return chamfer_loss
        else:
            raise NotImplementedError('Loss function [%s]'
                                      'is not recognized' % self.loss_name)

    def backward(self, out, target):
        if self.loss_name in ['chamfer']:
            self.loss = self.criterion(out, target,
                                       bidirectional=self.opt.bidirectional)
        else:
            self.loss = self.criterion(out, target)
        self.loss.backward()

    def set_output(self, out):
        out, _ = out
        if self.loss_name in ['ce']:
            target = self.labels.reshape(-1,)
        elif self.loss_name in ['chamfer']:
            target = self.data.y.reshape(self.data.batch.max()+1, -1,
                                         self.data.y.shape[1])
            out = out.view(self.data.batch.max()+1, self.data.y.shape[1], -1)
            out = out.transpose(2, 1).contiguous()
        elif self.loss_name in ['mse']:
            target = self.data.x
        return out, target

    def get_accuracy(self, pred, gt):
        """computes accuracy for classification / autoencoder """
        if self.mode == 'classification':
            pred_class = pred.data.max(1)[1]
            correct = pred_class.eq(gt).sum()
            return correct/len(gt)
        elif self.mode in ['autoencoder', 'autoencoder_pt']:
            if self.loss_name in ['chamfer']:
                return self.criterion(pred, gt, reverse=self.opt.reverse,
                                      bidirectional=self.opt.bidirectional)
            else:
                return self.criterion(pred, gt)

    ##################

    def define_net(self, opt):
        net = None

        if opt.arch == 'mesh_cls':
            from .classification_nets import MeshClsNet
            net = MeshClsNet(opt.input_nc, opt.ncf,
                             opt.nclasses, opt.ninput_edges,
                             opt.pool_res, opt.fc_n, opt.norm)
        elif opt.arch == 'mesh_unet':
            from .u_net import GraphUNet
            pool_ratios = [380, 160, 80, 16]
            net = GraphUNet(in_channels=opt.input_nc,
                            hidden_channels=32,
                            out_channels=opt.input_nc,
                            pool_ratios=pool_ratios,
                            depth=4,
                            sum_res=True, act=F.relu,
                            nclasses=opt.nclasses,
                            mode=opt.mode)
        elif opt.arch == 'mesh_aes':
            from .autoencoder_nets import ShallowCNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = ShallowCNet(in_channels=opt.input_nc,
                              hidden_channels=opt.ncf,
                              pool_ratios=pool_ratios,
                              sum_res=True, act=F.relu)
        elif opt.arch == 'mesh_aec' and opt.norm == 'batch':
            from .autoencoder_nets import BaseCNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = BaseCNet(in_channels=opt.input_nc,
                           hidden_channels=opt.ncf,
                           pool_ratios=pool_ratios,
                           sum_res=True, act=F.relu)
        elif opt.arch == 'mesh_aept' and opt.norm == 'batch':
            from .ae_nets_pt import BaseCNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = BaseCNet(in_channels=opt.input_nc,
                           hidden_channels=opt.ncf,
                           pool_ratios=pool_ratios,
                           sum_res=True, act=F.relu)
        elif opt.arch == 'mesh_aea' and opt.norm == 'batch':
            from .ae_nets_act import BaseCNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = BaseCNet(in_channels=opt.input_nc,
                           hidden_channels=opt.ncf,
                           pool_ratios=pool_ratios,
                           sum_res=True, act=F.relu)
        elif opt.arch == 'mesh_aec' and opt.norm == 'None':
            from .nonorm_ae_nets import BaseCNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = BaseCNet(in_channels=opt.input_nc,
                           hidden_channels=opt.ncf,
                           pool_ratios=pool_ratios,
                           sum_res=True, act=F.relu)
        elif opt.arch == 'mesh_aep':
            from .autoencoder_nets import PyrmCNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = PyrmCNet(in_channels=opt.input_nc,
                           hidden_channels=opt.ncf,
                           pool_ratios=pool_ratios,
                           sum_res=True, act=F.relu)
        elif opt.arch == 'mesh_aepc':
            from .autoencoder_nets import PyrmPCNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = PyrmPCNet(in_channels=opt.input_nc,
                            hidden_channels=opt.ncf,
                            pool_ratios=pool_ratios,
                            sum_res=True, act=F.relu)
        else:
            raise NotImplementedError('Encoder model name [%s]'
                                      'is not recognized' % opt.arch)
        return self.init_net(net, opt.init_type,
                             opt.init_gain, opt.gpu_ids)
