import os

import torch
from torch_geometric.data import Data, Dataset
import trimesh as tm
from src.util.util import pad
import numpy as np
import pickle
import time
import json
from torch_geometric.nn import knn_graph


class MeshDataset(Dataset):
    def __init__(self, root, file_names, class_to_idx=None,
                 transform=None, pre_transform=None):

        self.raw_file_names = file_names
        self.processed_file_names = file_names
        self.class_to_idx = class_to_idx
        self.data_transform = transform

        super(MeshDataset, self).__init__(root,
                                          None,
                                          pre_transform)

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @raw_file_names.setter
    def raw_file_names(self, value):
        self._raw_file_names = value

    @property
    def processed_file_names(self):
        return self._processed_file_names

    @processed_file_names.setter
    def processed_file_names(self, value):
        self._processed_file_names = value

    def process(self):
        self.pp_paths = []
        for i, raw_file_name in enumerate(self.raw_file_names):
            save_dir = os.path.join('./data/processed/',
                                    self.processed_file_names[i])
            self.pp_paths.append(save_dir)

            if not os.path.exists(save_dir):
                raw_path = os.path.join(self.root, raw_file_name)
                mesh = tm.load(raw_path)

                if self.pre_transform is not None:
                    mesh = self.pre_transform(mesh)

                if not os.path.exists(os.path.split(save_dir)[0]):
                    os.makedirs(save_dir)
                mesh.export(save_dir)
        # TODO: do not save processed files in memory

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        path = self.pp_paths[idx]

        mesh_in = tm.load(path)

        graph_data = self.data_transform(mesh_in)

        if self.class_to_idx is not None:
            label = self.raw_file_names[idx].split('/')[-2]
            label = self.class_to_idx[label]
            label = torch.Tensor(np.array([label]))
            return graph_data, label
        else:
            return graph_data

