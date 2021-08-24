import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, BatchNorm, global_mean_pool
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops, to_dense_batch)
from torch_geometric.utils.repeat import repeat
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d


class EncoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, act=None):
        super(EncoderLayer, self).__init__()
        self.act = act

        self.mlp = nn.Sequential(Linear(in_channels, out_channels),
                                 BatchNorm1d(out_channels),
                                 self.act())
        self.conv = GCNConv(out_channels, out_channels,
                            add_self_loops=True, normalize=True)
        self.bn = BatchNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = self.conv(x, edge_index)
        if self.act:
            x = self.bn(x)
            return self.act()(x)
        else:
            return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, act=None):
        super(DecoderLayer, self).__init__()
        self.act = act
        if self.act:
            self.conv = nn.Sequential(Linear(in_channels, out_channels),
                                      BatchNorm1d(out_channels),
                                      self.act())
        else:
            self.conv = nn.Sequential(Linear(in_channels, out_channels))

    def forward(self, x):
        x = self.conv(x)
        return x


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
        self.act = ReLU
        self.sum_res = sum_res  # change de_channels if not sum_res

        # construct encoder
        self.encoder0 = EncoderLayer(5, 64, self.act)
        self.encoder1 = EncoderLayer(64, 64, self.act)
        self.encoder2 = EncoderLayer(64, 128, self.act)
        self.encoder3 = EncoderLayer(128, 256, self.act)
        self.encoder4 = EncoderLayer(256, 512, self.act)

        # construct decoder
        self.decoder2 = DecoderLayer(512, 1024, self.act)
        self.decoder1 = DecoderLayer(1024, 2048, self.act)
        self.decoder0 = DecoderLayer(2048, 750 * 5)

    def forward(self, x, edge_index, batch=None):

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        batch_size = batch.max() + 1
        embeddings = {}

        x = self.encoder0(x, edge_index)
        embeddings['encoder0'] = x
        x = self.encoder1(x, edge_index)
        embeddings['encoder1'] = x
        x = self.encoder2(x, edge_index)
        embeddings['encoder2'] = x
        x = self.encoder3(x, edge_index)
        embeddings['encoder3'] = x
        x = self.encoder4(x, edge_index)
        embeddings['encoder4'] = x

        x = global_mean_pool(x, batch)
        embeddings['gb_pool'] = x

        x = self.decoder2(x)
        embeddings['decoder2'] = x
        x = self.decoder1(x)
        embeddings['decoder1'] = x
        x = self.decoder0(x)
        embeddings['decoder0'] = x

        # x = nn.Tanh()(x)

        return x, embeddings

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


class OneDUNet(torch.nn.Module):
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
        super(OneDUNet, self).__init__()

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
        setattr(self, 'decoder2',
                Lin(self.channels[-1], self.channels[-1]*2, bias=True))
        setattr(self, 'decoder1',
                Lin(self.channels[-1]*2, self.channels[-1]*4, bias=True))
        setattr(self, 'decoder0',
                Lin(self.channels[-1]*4, 750 * 5, bias=True))

    def forward(self, x, edge_index, batch=None):

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        batch_size = batch.max() + 1
        edge_weight = x.new_ones(edge_index.size(1))
        embeddings = {}

        # encoder
        for i in range(len(self.channels) - 1):
            x, edge_index, edge_weight,\
                batch, perm, embedding = \
                getattr(self, 'encoder{}'.format(i))(x,
                                                     edge_index,
                                                     edge_weight,
                                                     batch)
            embeddings['encoder{}'.format(i)] = embedding

        x = global_mean_pool(x, batch)

        # decoder
        for i in range(2, 0, -1):
            x = getattr(self, 'decoder{}'.format(i))(x)
            x = self.act(x)
            embeddings['decoder{}'.format(i)] = x

        # final layer
        x = getattr(self, 'decoder0')(x)
        embeddings['decoder0'] = x
        x = x.reshape(batch_size, -1, 5)

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