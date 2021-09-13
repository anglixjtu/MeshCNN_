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


class BaseCNet(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(BaseCNet, self).__init__()

        self.channels = [in_channels] + hidden_channels  # [5, 64, 64, 128, 256, 512]
        self.de_channels = self.channels[::-1]  # [512, 256, 128, 64, 64, 5]
        self.pool_ratios = [1] + pool_ratios  # [1, 0.8, 0.6, 0.4, 0.24]
        self.act = ReLU
        self.sum_res = sum_res  # change de_channels if not sum_res

        # construct encoder
        # construct encoder
        self.encoder0 = EncoderLayer(in_channels, 32, self.act)
        self.encoder1 = EncoderLayer(32, 64, self.act)
        self.encoder2 = EncoderLayer(64, 64, self.act)

        self.linear = DecoderLayer(32+64+64, 128, self.act)

        # construct decoder
        # self.decoder2 = DecoderLayer(128, 512, self.act)
        self.decoder1 = DecoderLayer(128, 512, self.act)
        self.decoder0 = DecoderLayer(512, 2048 * 3, None)

    def forward(self, x, edge_index, batch=None):

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        batch_size = batch.max() + 1
        embeddings = {}

        x0 = self.encoder0(x, edge_index)
        embeddings['encoder0'] = x0
        x1 = self.encoder1(x0, edge_index)
        embeddings['encoder1'] = x1
        x2 = self.encoder2(x1, edge_index)
        embeddings['encoder2'] = x2

        embeddings['encoder2.5'] = torch.cat([x0, x1, x2], dim=1)

        x3 = self.linear(torch.cat([x0, x1, x2], dim=1))
        embeddings['encoder3'] = x3

        xg = global_mean_pool(x3, batch)
        embeddings['gb_pool'] = xg

        # xg = self.decoder2(xg)
        # embeddings['decoder2'] = xg
        xg = self.decoder1(xg)
        embeddings['decoder1'] = xg
        xg = self.decoder0(xg)
        embeddings['decoder0'] = xg

        # xg = nn.Tanh()(xg)

        return xg, embeddings

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
