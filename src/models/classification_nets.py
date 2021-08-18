import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import TopKPooling,  global_mean_pool


##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################


class MeshMLPNRNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, nf0, conv_res, nclasses, input_res, pool_res, fc_n, norm, batch_size,
                 nresblocks=3):
        super(MeshMLPNRNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.min_score = [0.05, 0.05, 0.05, 0.05]
        self.ratio = [0.8, 0.6, 0.4, 0.24]


        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'mlp{}'.format(i), self.MLP(ki, self.k[i + 1], norm))
            setattr(self, 'conv{}'.format(i), GCNConv(self.k[i + 1], self.k[i + 1], add_self_loops=True, normalize=True))
            setattr(self, 'norm{}'.format(i), BatchNorm(self.k[i + 1]))
            if pool_res: # define pooling layers or not, added by Ang Li
                #setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))
                setattr(self, 'pool{}'.format(i),   SAGPooling(self.k[i + 1], ratio=0.8))

        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def MLP(self, channel_in, channel_out, norm='batch'):
        if norm=='batch':
            return Seq(Lin(channel_in, channel_out), ReLU(), BN(channel_out))
        else:
            return Seq(Lin(channel_in, channel_out), ReLU())
    

    def forward(self, x0, edge_index, batch):

        #x0, edge_index, batch = data.x, data.edge_index, data.batch

        embeddings = {}
        x = x0
        for i in range(len(self.k) - 1):
            x = getattr(self, 'mlp{}'.format(i))(x)
            embeddings['mlp{}'.format(i)] = x
            x = getattr(self, 'conv{}'.format(i))(x, edge_index)
            embeddings['conv{}'.format(i)] = x
            x = getattr(self, 'norm{}'.format(i))(x)
            embeddings['norm{}'.format(i)] = x
            x = F.relu(x)
            embeddings['conv_relu{}'.format(i)] = x
            #x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            if hasattr(self, 'pool{}'.format(i)):
                x, edge_index, _, batch, perm, score = getattr(self, 'pool{}'.format(i))(x, edge_index, None, batch)
                embeddings['pool{}'.format(i)] = x

        x = global_mean_pool(x, batch)
        #x = x.view(-1, self.k[-1])
        embeddings['gb_pool'] = x
        x = self.fc1(x)
        embeddings['fc1'] = x
        x = F.relu(x)
        x = self.fc2(x)
        embeddings['fc2'] = x
        return x, embeddings

class MeshPoolNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    A pooling layer is at the first, to handle large input graph
    """
    def __init__(self, nf0, conv_res, nclasses, input_res, pool_res, fc_n, norm, batch_size,
                 nresblocks=3):
        super(MeshPoolNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.min_score = [0.05, 0.05, 0.05, 0.05]
        self.ratio = [0.8, 0.6, 0.4, 0.24]

        setattr(self, 'pool00',   TopKPooling(nf0, ratio=input_res))
        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'mlp{}'.format(i), self.MLP(ki, self.k[i + 1], norm))
            setattr(self, 'conv{}'.format(i), GCNConv(self.k[i + 1], self.k[i + 1], add_self_loops=True, normalize=True))
            setattr(self, 'norm{}'.format(i), BatchNorm(self.k[i + 1]))
            if pool_res: # define pooling layers or not, added by Ang Li
                setattr(self, 'pool{}'.format(i),   SAGPooling(self.k[i + 1], ratio=0.8))

        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def MLP(self, channel_in, channel_out, norm='batch'):
        if norm=='batch':
            return Seq(Lin(channel_in, channel_out), ReLU(), BN(channel_out))
        else:
            return Seq(Lin(channel_in, channel_out), ReLU())

    def forward(self, x0, edge_index, batch):

        embeddings = {}
        x, edge_index, _, batch, perm, score = self.pool00(x0, edge_index, batch=batch)
        for i in range(len(self.k) - 1):
            x = getattr(self, 'mlp{}'.format(i))(x)
            embeddings['mlp{}'.format(i)] = x
            x = getattr(self, 'conv{}'.format(i))(x, edge_index)
            embeddings['conv{}'.format(i)] = x
            x = getattr(self, 'norm{}'.format(i))(x)
            embeddings['norm{}'.format(i)] = x
            x = F.relu(x)
            embeddings['conv_relu{}'.format(i)] = x
            if hasattr(self, 'pool{}'.format(i)):
                x, edge_index, _, batch, perm, score = getattr(self, 'pool{}'.format(i))(x, edge_index, None, batch)
                embeddings['pool{}'.format(i)] = x

        x = global_mean_pool(x, batch)
        #x = x.view(-1, self.k[-1])
        embeddings['gb_pool'] = x
        x = self.fc1(x)
        embeddings['fc1'] = x
        x = F.relu(x)
        x = self.fc2(x)
        embeddings['fc2'] = x
        return x, embeddings

class MeshMLPL2Net(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, nf0, conv_res, nclasses, input_res, pool_res, fc_n, norm, batch_size,
                 nresblocks=3):
        super(MeshMLPL2Net, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.min_score = [0.05, 0.05, 0.05, 0.05]
        self.ratio = [0.8, 0.6, 0.4, 0.24]


        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'mlp{}'.format(i), self.MLP(ki, self.k[i + 1], norm))
            setattr(self, 'conv{}'.format(i), GCNConv(self.k[i + 1], self.k[i + 1], add_self_loops=True, normalize=True))
            setattr(self, 'norm{}'.format(i), BatchNorm(self.k[i + 1]))
            if pool_res: # define pooling layers or not, added by Ang Li
                #setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))
                setattr(self, 'pool{}'.format(i),   SAGPooling(self.k[i + 1], ratio=0.8))

        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def MLP(self, channel_in, channel_out, norm='batch'):
        if norm=='batch':
            return Seq(Lin(channel_in, channel_out), ReLU(), BN(channel_out))
        else:
            return Seq(Lin(channel_in, channel_out), ReLU())
    

    def forward(self, data):

        x0, edge_index, batch = data.x, data.edge_index, data.batch


        x = x0
        for i in range(len(self.k) - 1):
            x = getattr(self, 'mlp{}'.format(i))(x)
            x = getattr(self, 'conv{}'.format(i))(x, edge_index)
            x = F.relu(x)
            x = getattr(self, 'norm{}'.format(i))(x)
            #x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            if hasattr(self, 'pool{}'.format(i)):
                x, edge_index, _, batch, perm, score = getattr(self, 'pool{}'.format(i))(x, edge_index, None, batch)

        x = global_mean_pool(x, batch)
        #x = x.view(-1, self.k[-1])
        embeddings = x
        x = self.fc1(x)
        x = self.fc2(x)
        return x, embeddings

class NoNormNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, nf0, conv_res, nclasses, input_res, pool_res, fc_n, norm, batch_size,
                 nresblocks=3):
        super(NoNormNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.min_score = [0.05, 0.05, 0.05, 0.05]
        self.ratio = [0.8, 0.6, 0.4, 0.24]


        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'mlp{}'.format(i), self.MLP(ki, self.k[i + 1], norm))
            setattr(self, 'conv{}'.format(i), GCNConv(self.k[i + 1], self.k[i + 1], add_self_loops=True, normalize=True))
            #setattr(self, 'norm{}'.format(i), BatchNorm(self.k[i + 1]))
            if pool_res: # define pooling layers or not, added by Ang Li
                #setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))
                setattr(self, 'pool{}'.format(i),   SAGPooling(self.k[i + 1], ratio=0.8))

        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def MLP(self, channel_in, channel_out, norm='batch'):
        if norm=='batch':
            return Seq(Lin(channel_in, channel_out), ReLU(), BN(channel_out))
        else:
            return Seq(Lin(channel_in, channel_out), ReLU())
    

    def forward(self, data):

        x0, edge_index, batch = data.x, data.edge_index, data.batch

        embeddings = {}
        x = x0
        for i in range(len(self.k) - 1):
            x = getattr(self, 'mlp{}'.format(i))(x)
            embeddings['mlp{}'.format(i)] = x
            x = getattr(self, 'conv{}'.format(i))(x, edge_index)
            embeddings['conv{}'.format(i)] = x
            x = getattr(self, 'norm{}'.format(i))(x)
            embeddings['norm{}'.format(i)] = x
            x = F.relu(x)
            embeddings['conv_relu{}'.format(i)] = x

            if hasattr(self, 'pool{}'.format(i)):
                x, edge_index, _, batch, perm, score = getattr(self, 'pool{}'.format(i))(x, edge_index, None, batch)
                embeddings['pool{}'.format(i)] = x

        x = global_mean_pool(x, batch)
        #x = x.view(-1, self.k[-1])
        embeddings['gb_pool'] = x
        x = self.fc1(x)
        embeddings['fc1'] = x
        x = F.relu(x)
        x = self.fc2(x)
        embeddings['fc2'] = x
        return x, embeddings



class VisualizationNet(nn.Module):
    """Network for visualization the graph
    """
    def __init__(self, nf0, conv_res, nclasses, input_res, pool_res, fc_n, norm, batch_size,
                 nresblocks=3):
        super(VisualizationNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        self.min_score = [0.05, 0.05, 0.05, 0.05]
        self.ratio = [0.8, 0.6, 0.4, 0.24]


        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'mlp{}'.format(i), self.MLP(ki, self.k[i + 1], norm))
            setattr(self, 'conv{}'.format(i), GCNConv(self.k[i + 1], self.k[i + 1], add_self_loops=True, normalize=True))
            setattr(self, 'norm{}'.format(i), BatchNorm(self.k[i + 1]))
            if pool_res: # define pooling layers or not, added by Ang Li
                #setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))
                setattr(self, 'pool{}'.format(i),   SAGPooling(self.k[i + 1], ratio=0.8))

        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def MLP(self, channel_in, channel_out, norm='batch'):
        if norm=='batch':
            return Seq(Lin(channel_in, channel_out), ReLU(), BN(channel_out))
        else:
            return Seq(Lin(channel_in, channel_out), ReLU())
    

    def forward(self, x0, edge_index, batch):
        x0 = x0.reshape(-1, 5)
        edge_index = edge_index.reshape(2, -1).long()
        batch = batch.reshape(-1,).long()

        #x0, edge_index, batch = data.x, data.edge_index, data.batch
        x = x0
        for i in range(len(self.k) - 1):
            x = getattr(self, 'mlp{}'.format(i))(x)
            x = getattr(self, 'conv{}'.format(i))(x, edge_index)
            x = getattr(self, 'norm{}'.format(i))(x)
            x = F.relu(x)
            #x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            if hasattr(self, 'pool{}'.format(i)):
                x, edge_index, _, batch, perm, score = getattr(self, 'pool{}'.format(i))(x, edge_index, None, batch)

        x = global_mean_pool(x, batch)
        #x = x.view(-1, self.k[-1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x, x
