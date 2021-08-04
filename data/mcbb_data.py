import os

import torch
from torch_geometric.data import Data, Dataset
import trimesh as tm
import pyvista as pv
from data.preprocess import sample_and_compute_features
from util.util import pad, is_mesh_file
import numpy as np
import pickle


class MCBBDataset(Dataset):
    def __init__(self, opt, transform=None, pre_transform=None):
        super(MCBBDataset, self).__init__(opt, transform, pre_transform)

        
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        if opt.phase == "train" or opt.phase == "retrieval":
            self.namelist_file = opt.train_namelist
        elif opt.phase == "test":
            self.namelist_file = opt.test_namelist
        self.dir = os.path.join(opt.dataroot)
        # self.classes, self.class_to_idx = self.find_classes(os.path.join(self.dir, opt.phase))
        self.paths, self.classes, self.class_to_idx = self.make_dataset_by_class_from_namelist(self.root, self.namelist_file)# added by ang li
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        opt.nclasses = self.nclasses
        self.opt = opt
        self.get_mean_std(opt) #modified by Ang Li

        # modify for network later.

    def len(self):
        return self.size

    def get(self, idx):
        path = self.paths[idx][0]
        label = self.paths[idx][1]
        
        mesh_in = tm.load(path)
        #print("Number of faces before process is %d"%(len(mesh_in.faces)))

        mesh_pv, meshcnn_data = sample_and_compute_features(mesh_in, path, self.opt)

        edge_features = pad(meshcnn_data.features, self.opt.ninput_edges)
        edge_features  = (edge_features - self.mean) / self.std
        edge_connections = self.get_edge_connection(meshcnn_data.gemm_edges)

        edge_features = torch.tensor(edge_features.transpose(), dtype=torch.float)
        edge_connections = torch.tensor(edge_connections, dtype=torch.long)

        graph_data = Data(x=edge_features, edge_index=edge_connections)
        label = torch.Tensor(np.array([label]))

        
        '''print("Number of faces after process is %d"%(len(mesh_pv.faces)//4))
        # visualization for checking the mesh
        p = pv.Plotter(shape=(2, 1))
        p.subplot(0, 0)
        p.add_mesh(mesh_in, color="tan", show_edges=True)
        p.subplot(1, 0)
        p.add_mesh(mesh_pv, color="tan", show_edges=True)
        p.show()#'''

        return graph_data, label

    
   # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx



    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes

    @staticmethod
    def make_dataset_by_class_from_namelist(dataroot, namelist_file):
        meshes = []
        classes = []
        class_to_idx = {}
        class_count = 0
        for model_name in open(namelist_file, "r"):
            model_name = model_name.strip('\n')
            name_parts = model_name.split("/")
            target = name_parts[1]
            if  target not in classes:
                classes += [target]
                class_to_idx[target] = class_count
                class_count += 1
            item = (os.path.join(dataroot, model_name), class_to_idx[target])
            meshes.append(item)
        return meshes, classes, class_to_idx

    @staticmethod
    def get_edge_connection(gemm_edges):
        if -1 in gemm_edges:
            debug=295
        sz = len(gemm_edges)
        edge_indices = np.arange(sz)
        edge_connection = np.zeros((2, 4*sz))
        for i in range(gemm_edges.shape[1]):
            edge_connection[0, i*sz:(i+1)*sz] = edge_indices
            edge_connection[1, i*sz:(i+1)*sz] = gemm_edges[:, i]
        #edge_connection[0, 4*sz:(4+1)*sz] = edge_indices
        #edge_connection[1, 4*sz:(4+1)*sz] = edge_indices
        valid = np.min(edge_connection, axis=0)
        valid = np.tile([valid>-1], (2,1))
        edge_connection = edge_connection[valid].reshape(2,-1)

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

        mean_std_cache = os.path.join("checkpoints", opt.name, 'mean_std_cache.p') #modified by Ang Li
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):           
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data[0].x
                features = features.numpy()
                mean = mean + features.mean(axis=0)
                std = std + features.std(axis=0)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
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


