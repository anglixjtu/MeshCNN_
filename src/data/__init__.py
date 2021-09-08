from torch_geometric.data import DataLoader
from src.util.util import parse_file_names, find_classes
from .transforms import (SampleMesh)
from .transform_recipes import set_transforms
import os
import numpy as np
import pickle


def create_dataloader(opt, phase, namelist=None):
    """create a dataloader """
    from src.data.mesh_data import MeshDataset

    namelist_file = opt.namelist_file
    root = opt.dataroot
    pre_transform = SampleMesh(opt.ninput_edges / 1.5)

    # compute mean and std (augmentation closed by setting num_aug=1)
    transform = set_transforms('compute_mean_std', opt)
    raw_file_names = parse_file_names(root, namelist,
                                      namelist_file, ['train'])
    dataset = MeshDataset(root, raw_file_names, None,
                          transform=transform,
                          pre_transform=pre_transform)
    mean, std, ninput_channels = compute_mean_std(opt.name, dataset)

    # define dataset and dataloader
    transform = set_transforms(phase, opt, mean, std, ninput_channels)
    if phase in ['train']:
        raw_file_names = parse_file_names(root, namelist,
                                          namelist_file, ['train'])
        shuffle = True
        batch_size = opt.batch_size
        num_workers = int(opt.num_threads)
    elif phase in ['test', 'query']:
        raw_file_names = parse_file_names(root, namelist,
                                          namelist_file, ['test'])
        shuffle = False
        batch_size = 1
        num_workers = 1
    elif phase in ['database']:
        raw_file_names = parse_file_names(root, namelist,
                                          namelist_file, opt.set)
        shuffle = False
        batch_size = opt.batch_size
        num_workers = int(opt.num_threads)
    else:
        raise NotImplementedError('phase [%s] is not implemented' % phase)

    if phase in ['train', 'test'] and opt.mode in ['classification']:
        classes, class_to_idx = find_classes(root, namelist_file)
        opt.nclasses = len(classes)
    else:
        class_to_idx = None
        opt.nclasses = 0

    dataset = MeshDataset(root, raw_file_names, class_to_idx,
                          transform=transform, pre_transform=pre_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

    return dataloader, dataset


def compute_mean_std(name, dataset):
    """ Computes Mean and Standard Deviation from Training Data
    If mean/std file doesn't exist, will compute one
    :returns
    mean: N-dimensional mean
    std: N-dimensional standard deviation
         ninput_channels: N
    """

    mean_std_cache = os.path.join("checkpoints",
                                  name,
                                  'mean_std_cache.p')
    if not os.path.isfile(mean_std_cache):
        print('computing mean std from train data...')
        # doesn't run augmentation during m/std computation
        mean, std = np.array(0), np.array(0)
        for i, data in enumerate(dataset):
            if i % 500 == 0:
                print('{} of {}'.format(i, len(dataset)))
            features = data[0].x
            features = features.numpy()
            mean = mean + features.mean(axis=0)
            std = std + features.std(axis=0)
        mean = mean / (i + 1)
        std = std / (i + 1)
        transform_dict = {'mean': mean[:, np.newaxis],
                          'std': std[:, np.newaxis],
                          'ninput_channels': len(mean)}
        with open(mean_std_cache, 'wb') as f:
            pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
    # open mean / std from file
    with open(mean_std_cache, 'rb') as f:
        transform_dict = pickle.load(f)
        print('loaded mean / std from cache')
        mean = transform_dict['mean']
        std = transform_dict['std']
        ninput_channels = transform_dict['ninput_channels']
    return mean, std, ninput_channels
