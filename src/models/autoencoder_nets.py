import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, BatchNorm
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


class EncoderLayer(torch.nn.Module):
    def __init__(self, channel_in, channel_out, pool_ratio=0.5,
                 act=F.relu, norm='batch'):
        super(EncoderLayer, self).__init__()
        self.pool_ratio = pool_ratio
        self.act = act
        if pool_ratio < 1:
            setattr(self, 'pool', TopKPooling(channel_in, pool_ratio))
        setattr(self, 'mlp', self.MLP(channel_in, channel_out, norm))
        setattr(self, 'conv', GCNConv(channel_out, channel_out,
                                      add_self_loops=True, normalize=True,
                                      improved=True))
        setattr(self, 'norm', BatchNorm(channel_out))
        # TODO: check the hyperparameters and norm

    def MLP(self, channel_in, channel_out, norm='batch'):
        if norm == 'batch':
            return Seq(Lin(channel_in, channel_out), ReLU(), BN(channel_out))
        else:
            return Seq(Lin(channel_in, channel_out), ReLU())

    def forward(self, x, edge_index, edge_weight, batch):
        if self.pool_ratio < 1:
            x, edge_index, edge_weight,\
                batch, perm, score = self.pool(x, edge_index,
                                               edge_weight, batch)
        else:
            perm = 1
        x = self.mlp(x)
        x = self.conv(x, edge_index, edge_weight)
        x = self.norm(x)
        x = self.act(x)
        embedding = x
        return x, edge_index, edge_weight,\
            batch, perm, embedding


class DecoderLayer(torch.nn.Module):
    def __init__(self, channel_in, channel_out, act=None):
        super(DecoderLayer, self).__init__()
        self.act = act
        setattr(self, 'conv', GCNConv(channel_in, channel_out,
                                      add_self_loops=True, normalize=True,
                                      improved=True))
        # TODO: check the architecture

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        embedding = x
        if self.act:
            return self.act(x), embedding
        else:
            return x, embedding


class BaseUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(BaseUNet, self).__init__()

        self.channels = [in_channels] + hidden_channels  # [5, 64, 64, 128, 256, 512]
        self.de_channels = self.channels[::-1]  # [512, 256, 128, 64, 64, 5]
        self.pool_ratios = [1] + pool_ratios  # [1, 0.8, 0.6, 0.4, 0.24]
        self.act = act
        self.sum_res = sum_res  # change de_channels if not sum_res

        # construct encoder
        for i, pool_ratio in enumerate(self.pool_ratios):
            setattr(self, 'encoder{}'.format(i),
                    EncoderLayer(self.channels[i], self.channels[i + 1],
                                 pool_ratio=pool_ratio, act=self.act))

        # construct decoder
        for i in range(len(self.channels)-1, 1, -1):
            setattr(self, 'decoder{}'.format(i-1),
                    DecoderLayer(self.channels[i], self.channels[i - 1],
                                 act=self.act))
        setattr(self, 'decoder0',
                DecoderLayer(self.channels[1], self.channels[0],
                             act=None))

    def forward(self, x, edge_index, batch=None):

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        embeddings = {}
        xs = []
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        # encoder
        for i in range(len(self.channels) - 1):
            x, edge_index, edge_weight,\
                batch, perm, embedding = \
                getattr(self, 'encoder{}'.format(i))(x,
                                                     edge_index,
                                                     edge_weight,
                                                     batch)
            embeddings['encoder{}'.format(i)] = embedding
            xs += [x]
            edge_indices += [edge_index]
            edge_weights += [edge_weight]
            perms += [perm]

        # decoder
        for i in range(len(self.channels) - 2, 0, -1):
            res = xs[i-1]
            edge_index = edge_indices[i+1]
            edge_weight = edge_weights[i+1]
            perm = perms[i]

            x, embedding = getattr(self, 'decoder{}'.format(i))(x,
                                                                edge_index,
                                                                edge_weight)

            up = torch.zeros_like(res)
            up[perm] = (x + res[perm]) / 2.
            non_perm = [x for x in range(len(up)) if x not in perm]
            up[non_perm] = res[non_perm]
            x = up
            embeddings['encoder{}'.format(i)] = embedding

        # final layer
        edge_index = edge_indices[0]
        edge_weight = edge_weights[0]
        x, embedding = getattr(self, 'decoder0')(x, edge_index, edge_weight)
        embeddings['encoder0'] = embedding

        return x, embeddings

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
