import torch
from os.path import join
from src.util.util import print_network
from torch.nn import init
from .trainer import get_scheduler
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from src.util.losses import ChamferLoss
from torch_geometric.utils import to_dense_batch
# from chamferdist import ChamferDistance


class Model:
    """ Class for CNN models

    :args opt: structure containing configuration params
    e.g.,
    --mode -> classification / autoencoder)
    --arch -> network type
    """
    def __init__(self, opt, device, phase='train'):
        self.opt = opt
        self.phase = phase
        self.gpu_ids = opt.gpu_ids
        if phase == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.device = device
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None
        self.mode = opt.mode
        self.continue_train = opt.continue_train
        # TODO: clean continue_train
        self.which_epoch = opt.which_epoch

        self.net = self.define_net(opt)
        self.criterion = opt.loss  # define_loss(self.mode).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)
            print_network(self.net)

        self.load_ckpt()

    def set_input(self, data):
        if self.mode == 'classification':
            labels = data[1]
            data = data[0]
            # set inputs

            if not hasattr(data, 'batch'):
                data.batch = torch.zeros(len(data.x), 1)
                data.batch = data.batch.long()
            self.data = data.to(self.device)
            self.labels = labels.long().to(self.device)
        elif self.mode == 'autoencoder':
            # set inputs

            if not hasattr(data, 'batch'):
                data.batch = torch.zeros(len(data.x))
                data.batch = data.batch.long()
            self.data = data.to(self.device)

    def forward(self):
        out, embeddings = self.net(self.data.x,
                                   self.data.edge_index,
                                   self.data.batch)
        return out, embeddings

    def backward(self, out):
        if self.criterion == 'ce':
            self.loss = CrossEntropyLoss()(out, self.labels.reshape(-1,))
        elif self.criterion == 'mse':
            target, _ = to_dense_batch(self.data.x, self.data.batch)
            out = out.view(self.data.batch.max()+1, 5, -1)
            out = out.transpose(2, 1).contiguous()
            self.loss = MSELoss()(out, target)
        elif self.criterion == 'chamfer':
            target, _ = to_dense_batch(self.data.x, self.data.batch)
            out = out.view(self.data.batch.max()+1, 5, -1)
            out = out.transpose(2, 1).contiguous()
            self.loss = ChamferLoss()(out, target)
            # self.loss = ChamferDistance()(out, target)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out, _ = self.forward()
        self.backward(out)
        self.optimizer.step()

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out, _ = self.forward()
            # compute number of correct
            if self.mode == 'classification':
                pred_class = out.data.max(1)[1]
                label_class = self.labels
                accuracy = self.get_accuracy(pred_class, label_class)
            elif self.mode == 'autoencoder':
                target, _ = to_dense_batch(self.data.x, self.data.batch)
                out = out.view(self.data.batch.max()+1, 5, -1)
                out = out.transpose(2, 1).contiguous()
                accuracy = ChamferLoss()(out, target)
                # accuracy = ChamferDistance()(out, target)

        return accuracy

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        correct = pred.eq(labels).sum()
        return correct/len(labels)

    def load_ckpt(self):
        """select the checkpoint file for loading"""
        if self.phase == 'test':
            if self.continue_train:
                self.load_network("latest")
            else:
                self.load_network(self.which_epoch)

        if self.phase == 'train' and self.continue_train:
            self.load_network(self.which_epoch)

        if self.phase == 'retrieval':
            self.load_network(self.which_epoch)

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module
        print('loading the model from %s' % load_path)
        # PyTorch ne wer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.net.load_state_dict(state_dict)

    def save_network(self, which_epoch='latest'):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    ##################

    def define_net(self, opt):
        net = None

        if opt.arch == 'mesh_cls':
            from .classification_nets import MeshClsNet
            net = MeshClsNet(opt.input_nc, opt.ncf,
                             opt.nclasses, opt.ninput_edges,
                             opt.pool_res, opt.fc_n, opt.norm)
        elif opt.arch == 'mesh_ae':
            from .autoencoder_nets import BaseUNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = BaseUNet(in_channels=opt.input_nc,
                           hidden_channels=opt.ncf,
                           pool_ratios=pool_ratios,
                           sum_res=True, act=F.relu)
        elif opt.arch == 'mesh_ae1d':
            from .autoencoder_nets import OneDUNet
            pool_ratios = [0.8, 0.6, 0.4, 0.24]
            net = OneDUNet(in_channels=opt.input_nc,
                           hidden_channels=opt.ncf,
                           pool_ratios=pool_ratios,
                           sum_res=True, act=F.relu)

        else:
            raise NotImplementedError('Encoder model name [%s]'
                                      'is not recognized' % opt.arch)
        return self.init_net(net, opt.init_type,
                             opt.init_gain, opt.gpu_ids)

    def init_net(self, net, init_type, init_gain, gpu_ids):
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.cuda(gpu_ids[0])
            net = net.cuda()
            net = torch.nn.DataParallel(net, gpu_ids)
        if init_type != 'none':
            self.init_weights(net, init_type, init_gain)
        return net

    @staticmethod
    def init_weights(net, init_type='normal', init_gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif hasattr(m, 'weight') and (classname.find('BatchNorm') != -1):
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
        net.apply(init_func)
