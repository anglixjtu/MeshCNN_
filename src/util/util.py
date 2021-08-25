from __future__ import print_function
import torch
import numpy as np
import os
import json


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


MESH_EXTENSIONS = [
    '.obj',
]


def parse_file_names(dataroot, namelist=None,
                     namelist_file=None, sets='train'):
    """Parse paths of data.

    Args:
        dataroot(str): Root directory for data. If neither 'namelist'
                      nor 'namelist_file' is defined, find files from
                      this directory.
        namelist(list, optional): List of file names. If defined, parse this.
        namelist_file(str, optional): Path to the .json file which
                                      stores the namelist. If defined,
                                      parse this.
        sets(list, optional): The split ('train', 'test', or others) to load.
    """
    paths = []
    if namelist:  # parse form namelist
        paths = namelist
    elif namelist_file:  # find from namelist_file
        if namelist_file[-5:] == '.json':
            with open(namelist_file, 'r') as f:
                namelist = json.load(f)

            for set in sets:
                dataset = namelist[set]
                classes = list(dataset.keys())
                for target in classes:
                    items = [x for x in dataset[target]]
                    paths += items
    else:  # find directly from directory
        for set in sets:
            dir = os.path.join(dataroot, set)
            # subdirs = [os.path.join(dir, x) for x in os.listdir(dir)]
            for target in os.listdir(dir):
                subdir = os.path.join(dir, target)
                items = [os.path.join(set, target, x)
                         for x in os.listdir(subdir)]
                paths += items
    # TODO: check the case without classes/subdir
    return paths


def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)


def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad,
                  mode='constant', constant_values=val)


def seg_accuracy(predicted, ssegs, meshes):
    correct = 0
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
        edge_areas = torch.from_numpy(mesh.get_edge_areas())
        correct += (correct_vec.float() * edge_areas).sum()
    return correct


def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy


class MetricCounter(object):
    """store metrics and compute average"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val += [val]
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count