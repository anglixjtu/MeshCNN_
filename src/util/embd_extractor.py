import numpy as np
import torch
import torch.nn.functional as F
import os
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import (global_mean_pool,
                                global_add_pool,
                                global_max_pool,
                                global_sort_pool)


class EmbdExtractor(object):
    def __init__(self, which_layer, file_names,
                 pooling=None, normalize=2):
        self.which_layer = which_layer
        self.file_names = file_names
        self.pooling = pooling
        self.normalize = normalize
        self.pooling_set = ['global_mean_pool',
                            'global_add_pool',
                            'global_max_pool',
                            'global_sort_pool']

    def __call__(self, model, dataloader):
        x = {'paths': [], 'embeddings': [],
             'which_layer': self.which_layer,
             'pooling': self.pooling,
             'normalize': self.normalize}
        for i, data in enumerate(dataloader):
            features = self.extract_one(model, data)
            paths = [self.file_names[i] for i in data[-1]]
            x['paths'] += paths

            if i == 0:
                x['embeddings'] = features
            else:
                x['embeddings'] = np.append(x['embeddings'], features, axis=0)
        x['feature_size'] = features.shape[1]
        return x

    def extract_one(self, model, data):
        model.net.eval()
        model.set_input(data)
        with torch.no_grad():
            _, features = model.forward()
        embeddings = features[self.which_layer]
        if self.pooling in self.pooling_set:
            batch = data[0].batch  # only for without sampling in the net
            embeddings = eval(self.pooling)(embeddings, batch)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=self.normalize, dim=1)
        self.embeddings_size = embeddings.shape[1]
        return embeddings.cpu().detach().numpy()

    @staticmethod
    def save(save_dir, embeddings):
        save_dir = save_dir + '.pt'
        torch.save(embeddings, save_dir)
