import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, ASAPooling, global_mean_pool


###############################################################################
# Helper Functions
###############################################################################


class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
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
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_net(input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain):
    net = None

    if arch == 'mconvnet':
        net = MeshConvNet(input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_loss(opt):
    loss = torch.nn.CrossEntropyLoss()
    return loss

##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), GCNConv(ki, self.k[i + 1], add_self_loops=False, normalize=True))
            setattr(self, 'norm{}'.format(i), BatchNorm(self.k[i + 1]))
            if pool_res: # define pooling layers or not, added by Ang Li
                #setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))
                setattr(self, 'pool{}'.format(i), ASAPooling(self.k[i + 1], ratio=self.res[i + 1]))

        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, edge_index)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            if hasattr(self, 'pool{}'.format(i)):
                x = getattr(self, 'pool{}'.format(i))(x, edge_index)

        x = global_mean_pool(x, batch)
        #x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

