import torch
from os.path import join
from src.util.util import print_network
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel:
    """ Base Class for CNN models
    :args opt: structure containing configuration params
          device: gpu id or cpu
          phase: 'train', 'test', etc
    """
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.gpu_ids = opt.gpu_ids
        self.loss_name = opt.loss
        self.continue_train = opt.continue_train
        self.which_epoch = opt.which_epoch

        if phase == 'train':
            self.is_train = True
        else:
            self.is_train = False

        self.device = None
        self.optimizer = None
        self.labels = None
        self.loss = None


        self.set_device(self.gpu_ids)
        self.net = self.define_net(opt)
        self.criterion = self.define_loss()
        self.set_optimizer(opt)
        self.load_ckpt()

    def set_device(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))\
            if (gpu_ids and torch.cuda.is_available())\
            else torch.device('cpu')

    # methods for network
    def set_input(self):
        raise NotImplementedError('set_input is not implemented.'
                                  'Please set data and/or labels')

    def forward(self):
        out = self.net(self.data)
        return out

    def test(self):
        """tests model
        returns: accuracy
        """
        with torch.no_grad():
            out = self.forward()
            out, target = self.set_output(out)
            accuracy = self.get_accuracy(out, target)
        return accuracy

    def get_accuracy(self, pred, labels):
        """computes accuracy """
        correct = pred.eq(labels).sum()
        return correct/len(labels)

    def define_net(self, opt):
        raise NotImplementedError('define_net is not implemented.'
                                  'Please define the networks')

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
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] '
                                              'is not implemented' % init_type)
            elif hasattr(m, 'weight') and (classname.find('BatchNorm') != -1):
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)
        net.apply(init_func)

    # methods for loading and saving checkpoints
    def load_ckpt(self):
        """select the checkpoint file for loading"""
        if self.is_train:
            if self.continue_train:
                self.load_network(self.which_epoch)
        else:
            self.load_network(self.which_epoch)

    def load_network(self, which_epoch='latest'):
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

    # methods for training
    def define_loss(self):
        raise NotImplementedError('define_loss is not implemented.'
                                  'Please define the loss functions')

    def backward(self, out, target):
        self.loss = self.criterion(out, target)
        self.loss.backward()

    def set_output(self, out):
        return out, self.labels

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        out, target = self.set_output(out)
        self.backward(out, target)
        self.optimizer.step()

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_optimizer(self, opt):
        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = self.get_scheduler(self.optimizer, opt)
            print_network(self.net)
    
    @staticmethod
    def get_scheduler(optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) /\
                    float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            step_size=opt.lr_decay_iters,
                                            gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.2,
                                                       threshold=0.01,
                                                       patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] '
                                       'is not implemented',
                                       opt.lr_policy)
        return scheduler
