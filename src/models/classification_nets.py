import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import SAGPooling,  global_mean_pool


class MeshClsNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, nf0, conv_res, nclasses, input_res,
                 pool_res, fc_n, norm):
        super(MeshClsNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.min_score = [0.05, 0.05, 0.05, 0.05]
        self.ratio = [0.8, 0.6, 0.4, 0.24]

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'mlp_lin{}'.format(i), Lin(ki, self.k[i + 1]))
            setattr(self, 'mlp_norm{}'.format(i), BN(self.k[i + 1]))
            setattr(self, 'conv{}'.format(i), GCNConv(self.k[i + 1],
                    self.k[i + 1], add_self_loops=True, normalize=True))
            setattr(self, 'norm{}'.format(i), BatchNorm(self.k[i + 1]))
            if pool_res:  # define pooling layers or not, added by Ang Li
                setattr(self, 'pool{}'.format(i),
                        SAGPooling(self.k[i + 1], ratio=0.8))

        self.fc1 = Lin(self.k[-1], fc_n)
        self.fc2 = Lin(fc_n, nclasses)

    def forward(self, x0, edge_index, batch):

        embeddings = {}
        x = x0
        for i in range(len(self.k) - 1):
            x = getattr(self, 'mlp_lin{}'.format(i))(x)
            x = F.relu(x)
            x = getattr(self, 'mlp_norm{}'.format(i))(x)
            embeddings['mlp{}'.format(i)] = x
            x = getattr(self, 'conv{}'.format(i))(x, edge_index)
            embeddings['conv{}'.format(i)] = x
            x = getattr(self, 'norm{}'.format(i))(x)
            embeddings['norm{}'.format(i)] = x
            x = F.relu(x)
            embeddings['conv_relu{}'.format(i)] = x
            if hasattr(self, 'pool{}'.format(i)):
                x, edge_index, _, batch, perm, score = \
                    getattr(self, 'pool{}'.format(i))(x, edge_index,
                                                      None, batch)
                embeddings['pool{}'.format(i)] = x

        x = global_mean_pool(x, batch)
        embeddings['gb_pool'] = x
        x = self.fc1(x)
        embeddings['fc1'] = x
        x = F.relu(x)
        x = self.fc2(x)
        embeddings['fc2'] = x
        return x, embeddings
