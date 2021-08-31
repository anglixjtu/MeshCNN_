import os

import torch
from torch_geometric.data import Data, Dataset
import trimesh as tm
from src.data.preprocess import compute_features
from src.util.util import pad
import numpy as np
import pickle
import time
import json
from torch_geometric.nn import knn_graph
import random

class MeshDataset(Dataset):
    def __init__(self, opt, sets, phase='train', transform=None, pre_transform=None):
        super(MeshDataset, self).__init__(None, transform, pre_transform)

        self.root = opt.dataroot
        self.namelist_file = opt.namelist_file
        self.mode = opt.mode
        self.phase = phase
        self.classes, self.class_to_idx = \
            self.find_classes(self.root, self.namelist_file)
        self.paths = self.find_paths(self.root, self.namelist_file, sets)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.mean = 0
        self.std = 1
        self.ninput_channels = opt.input_nc
        opt.nclasses = self.nclasses
        # for timer
        self.time = {'preprocess': 0, 'input_feature': 0}
        # for preprocess
        self.saveroot = './data/processed/'
        self.ninput_edges = opt.ninput_edges
        self.sample_save_mesh()

        self.opt = opt
        self.get_mean_std(opt)

    def len(self):
        return self.size

    def get(self, idx):

        if self.phase in ['train']:
            crop_aug = random.random()
            if crop_aug > 0.5:
                path = self.paths[idx]
                mesh_in = tm.load(path)
                ntarget_faces = int(self.ninput_edges/1.5)
                n_faces = len(mesh_in.faces)
                if n_faces > ntarget_faces + 100:
                    start_f = random.randint(0, n_faces-ntarget_faces-100)
                    triangle_id = list(range(start_f,
                                             start_f+ntarget_faces+100))
                    mesh_in = mesh_in.submesh([triangle_id])
                    mesh_in = mesh_in[0]

            else:
                path = self.pp_paths[idx]
                mesh_in = tm.load(path)
        else:
            path = self.pp_paths[idx]
            mesh_in = tm.load(path)

        mesh_out, meshcnn_data = compute_features(mesh_in, self.opt)

        start_t = time.time()
        if meshcnn_data.features.shape[1] < self.opt.ninput_edges:
            edge_features = pad(meshcnn_data.features, self.opt.ninput_edges)
            edge_pos = pad(meshcnn_data.pos, self.opt.ninput_edges, dim=0)
            edge_len = pad(meshcnn_data.edge_lengths.reshape(-1, 1),
                           self.opt.ninput_edges, dim=0)
        else:
            edge_features = meshcnn_data.features[:, :self.opt.ninput_edges]
            edge_pos = meshcnn_data.pos[:self.opt.ninput_edges, :]
            edge_len = meshcnn_data.edge_lengths.reshape(-1, 1)[:self.opt.ninput_edges, :]

        # edge_features = self.normalize_max_min(edge_features)

        if self.opt.knn:
            edge_pos = torch.tensor(edge_pos,
                                    dtype=torch.float)
            batch = torch.zeros(len(edge_pos), dtype=torch.long)
            edge_connections = knn_graph(edge_pos, k=11,
                                         batch=batch, loop=False)
        else:
            edge_connections = self.get_edge_connect(meshcnn_data.gemm_edges)
            out = np.min(edge_connections, axis=0)
            out_i = np.arange(len(out))[out>=self.opt.ninput_edges]
            if len(out_i) > 0:
                edge_connections = np.delete(edge_connections, out_i, axis=1)
            edge_connections = torch.tensor(edge_connections,
                                        dtype=torch.long)

        edge_features = torch.tensor(edge_features.transpose(),
                                     dtype=torch.float)

        graph_data = Data(x=edge_features, edge_index=edge_connections,
                          pos=edge_pos)

        if self.mode == 'autoencoder':
            graph_data = self.transform(graph_data)
        # TODO: is the transformation necessary

        if self.ninput_channels == 8:
            graph_data.x = torch.cat((graph_data.x, graph_data.pos), 1)

        mean = torch.tensor(self.mean).reshape(1, -1)
        std = torch.tensor(self.std).reshape(1, -1)
        graph_data.x = (graph_data.x - mean) / std

        end_t = time.time()
        self.time['input_feature'] += end_t - start_t

        if self.mode == 'classification':
            label = self.paths[idx].split('/')[-2]
            label = self.class_to_idx[label]
            label = torch.Tensor(np.array([label]))
            return graph_data, label
        elif self.mode == 'autoencoder':
            return graph_data

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dataroot, namelist_file=None):
        if namelist_file:  # find classes from namelist file
            with open(namelist_file, 'r') as f:
                namelist = json.load(f)
            dataset = namelist['train']
            classes = list(dataset.keys())
        else:  # find directly from directory
            dir = os.path.join(dataroot, 'train')
            classes = [d for d in os.listdir(dir)
                       if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def find_paths(dataroot, namelist_file=None, sets=['train']):
        paths = []
        if namelist_file:  # find from namelist_file
            with open(namelist_file, 'r') as f:
                namelist = json.load(f)
            for set in sets:
                dataset = namelist[set]
                classes = list(dataset.keys())
                for target in classes:
                    items = [os.path.join(dataroot, x)
                             for x in dataset[target]]
                    paths += items
        else:  # find directly from directory
            for set in sets:
                dir = os.path.join(dataroot, set)
                subdirs = [os.path.join(dir, x) for x in os.listdir(dir)]
                for subdir in subdirs:
                    items = [os.path.join(subdir, x)
                             for x in os.listdir(subdir)]
                    paths += items
        # TODO: check the case without classes/subdir
        return paths

    @staticmethod
    def get_edge_connect(gemm_edges):
        sz = len(gemm_edges)
        edge_indices = np.arange(sz)
        edge_connection = np.zeros((2, 4*sz))
        for i in range(gemm_edges.shape[1]):
            edge_connection[0, i*sz:(i+1)*sz] = edge_indices
            edge_connection[1, i*sz:(i+1)*sz] = gemm_edges[:, i]
        valid = np.min(edge_connection, axis=0)
        valid = np.tile([valid > -1], (2, 1))
        edge_connection = edge_connection[valid].reshape(2, -1)

        return edge_connection

    def get_mean_std(self, opt):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join("checkpoints",
                                      opt.name,
                                      'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                if self.opt.mode == 'classification':
                    features = data[0].x
                elif self.opt.mode == 'autoencoder':
                    features = data.x
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
            self.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

    @staticmethod
    def normalize_max_min(edge_features):
        edge_features_out = edge_features
        edge_features_out[0:3, :] = edge_features[0:3, :] / np.pi
        max_ratios = np.max(edge_features[3:, :], axis=1)
        edge_features_out[3, :] = edge_features[3, :] / max_ratios[0]
        edge_features_out[4, :] = edge_features[4, :] / max_ratios[1]
        return edge_features_out

    def sample_save_mesh(self):
        nfaces_target = self.ninput_edges / 1.5
        self.pp_paths = []

        for path in self.paths:
            phase = path.split('/')[-3]
            target = path.split('/')[-2]
            obj_name = path.split('/')[-1]
            mesh_in = tm.load(path)
            nfaces = len(mesh_in.faces)
            mesh_out = mesh_in
            save_dir = os.path.join(self.saveroot, phase, target)
            pp_path = os.path.join(save_dir, obj_name)
            self.pp_paths.append(pp_path)
            if not os.path.exists(pp_path):
                # upsample
                if nfaces < nfaces_target:
                    nsub = max(1, round((nfaces_target/nfaces)**0.25))
                    for i in range(nsub):
                        mesh_out = mesh_out.subdivide()
                # downsample
                mesh_out = mesh_out.simplify_quadratic_decimation(nfaces_target)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                mesh_out.export(pp_path)
