import torch
from os.path import join


def load_ckpt(phase, which_epoch, continue_train='True'):
    """select the checkpoint file for loading"""
    if phase == 'test':
        if continue_train:
            load_network("latest")
        else:
            load_network(which_epoch)

    if phase == 'train' and continue_train:
        load_network(which_epoch)

    if phase == 'retrieval':
        load_network(which_epoch)


def load_network(net, save_dir, device, which_epoch='latest'):
    """load model from disk"""
    save_filename = '%s_net.pth' % which_epoch
    load_path = join(save_dir, save_filename)
    net = net
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % load_path)
    # PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on device
    state_dict = torch.load(load_path, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net.load_state_dict(state_dict)


def save_network(net, gpu_ids, save_dir, which_epoch='latest'):
    """save model to disk"""
    save_filename = '%s_net.pth' % (which_epoch)
    save_path = join(save_dir, save_filename)
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(net.module.cpu().state_dict(), save_path)
        net.cuda(gpu_ids[0])
    else:
        torch.save(net.cpu().state_dict(), save_path)
