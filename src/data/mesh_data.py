import os
import torch
import numpy as np
from torch_geometric.data import Dataset

from src import util
from torch_geometric.io import read_obj

from .transforms import Rotate
from torch_geometric.transforms import (NormalizeRotation, RandomRotate)


class MeshDataset(Dataset):
    def __init__(self, root, file_names, phase='train',
                 class_to_idx=None,
                 transform=None, pre_transform=None):

        self.raw_file_names = file_names
        self.processed_file_names = file_names
        self.class_to_idx = class_to_idx
        self.data_transform = transform
        self.pp_paths, self.raw_file_paths = [], []
        self.phase = phase

        super(MeshDataset, self).__init__(root,
                                          None,
                                          pre_transform)

        self.generate_paths()

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
        self._processed_file_names = []
        for path in value:
            save_path = os.path.splitext(path)[0] + '.pt'
            self._processed_file_names.append(save_path)

    def generate_paths(self):
        for i, raw_file_name in enumerate(self.raw_file_names):
            save_path = os.path.join(self.processed_dir,  # './data/processed/'
                                     self.processed_file_names[i])
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            self.pp_paths.append(save_path)
            self.raw_file_paths.append(raw_path)

    def process(self):
        self.pp_paths, self.raw_file_paths = [], []
        for i, raw_file_name in enumerate(self.raw_file_names):
            save_path = os.path.join(self.processed_dir,  # './data/processed/'
                                     self.processed_file_names[i])
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            self.pp_paths.append(save_path)
            self.raw_file_paths.append(raw_path)

            if not os.path.exists(save_path):

                mesh = self.load_mesh(raw_path)

                if self.pre_transform is not None:
                    graph = self.pre_transform(mesh)

                save_dir = os.path.split(save_path)[0]
                util.util.mkdir(save_dir)
                torch.save(graph, save_path)

    def len(self):
        return len(self.processed_file_names)

    def load_mesh(self, path):
        try:
            import trimesh as tm
            mesh = tm.load(path)
        except ImportError:
            # print('Trimesh not installed, using pytorch-geometric')
            from torch_geometric.io import read_obj
            mesh = read_obj(path)
        return mesh

    def get(self, idx):

        path = self.pp_paths[idx]

        mesh_in = torch.load(path)

        #mesh_in = self.load_mesh(path)
        # mesh_in = read_obj(path)
        # print(path)

        '''mesh_in = Rotate(90, 0)(mesh_in)
        mesh_in = Rotate(90, 1)(mesh_in)
        mesh_in = Rotate(90, 2)(mesh_in)'''

        '''mesh_in = RandomRotate(180, 0)(mesh_in)
        mesh_in = RandomRotate(180, 1)(mesh_in)
        mesh_in = RandomRotate(180, 2)(mesh_in)'''

        graph_data = self.data_transform(mesh_in)

        if self.class_to_idx is not None:
            if '/' in self.raw_file_names[idx]:
                label = self.raw_file_names[idx].split('/')[-2]
            elif '\\' in self.raw_file_names[idx]:
                label = self.raw_file_names[idx].split('\\')[-2]
            label = self.class_to_idx[label]
            label = torch.Tensor(np.array([label]))
            return graph_data, label, idx
        else:
            return graph_data, idx
