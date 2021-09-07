import numpy as np
import torch
import torch.nn.functional as F
import os
from torch_geometric.nn import (global_mean_pool,
                                global_add_pool,
                                global_max_pool,
                                global_sort_pool)


class EmbdExtractor(object):
    def __init__(self, which_layer,
                 pooling=None, normalize=2):
        self.which_layer = which_layer
        self.pooling = pooling
        self.normalize = normalize
        self.pooling_set = ['global_mean_pool',
                            'global_add_pool',
                            'global_max_pool',
                            'global_sort_pool']

    def __call__(self, model, dataset):
        for i, data in enumerate(dataset):
            features = self.extract_one(model, data)

            if i == 0:
                x = features
            else:
                x = torch.cat((x, features), axis=0)
        return x

    def extract_one(self, model, data):
        # TODO: modify for batch_size > 1
        model.net.eval()
        model.set_input(data)
        with torch.no_grad():
            _, features = model.forward()
        embeddings = features[self.which_layer]
        if self.pooling in self.pooling_set:
            batch = torch.zeros(len(embeddings))  # only for batch_size 1
            embeddings = eval(self.pooling)(embeddings, batch.long())
        if self.normalize:
            embeddings = F.normalize(embeddings, p=self.normalize, dim=1)
        self.embeddings_size = embeddings.shape[1]
        return embeddings.cpu().detach().reshape(1, -1)

    @staticmethod
    def save(save_dir, embeddings):
        save_dir = save_dir + '.pt'
        torch.save(embeddings, save_dir)
