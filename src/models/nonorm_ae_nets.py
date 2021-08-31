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
    """encoder layer without normalization"""
    def __init__(self, in_channels, out_channels, act=None):
        super(EncoderLayer, self).__init__()
        self.act = act

        self.mlp = nn.Sequential(Linear(in_channels, out_channels),
                                 BatchNorm(out_channels),
                                 self.act())
        self.conv = GCNConv(out_channels, out_channels,
                            add_self_loops=True, normalize=True)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = self.conv(x, edge_index)
        if self.act:
            return self.act()(x)
        else:
            return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, act=None):
        super(DecoderLayer, self).__init__()
        self.act = act
        if self.act:
            self.conv = nn.Sequential(Linear(in_channels, out_channels),
                                      BatchNorm(out_channels),
                                      self.act())
        else:
            self.conv = nn.Sequential(Linear(in_channels, out_channels))

    def forward(self, x):
        x = self.conv(x)
        return x


class ConstractiveLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio, act=None):
        super(ConstractiveLayer, self).__init__()
        self.act = act

        self.mlp = nn.Sequential(Linear(in_channels, out_channels),
                                 BatchNorm1d(out_channels),
                                 self.act())
        self.conv = GCNConv(out_channels, out_channels,
                            add_self_loops=True, normalize=True)
        self.bn = BatchNorm(out_channels)
        self.pool = TopKPooling(out_channels, ratio)

    def forward(self, x, edge_index, batch):
        x = self.mlp(x)
        x = self.conv(x, edge_index)
        if self.act:
            x = self.bn(x)
            x = self.act()(x)
            x, edge_index, _, batch, perm, score = self.pool(x, edge_index,
                                                             None, batch)
            return x, edge_index, batch
        else:
            return x, edge_index, batch


class BaseUNet(torch.nn.Module):

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

        if self.in_channels == 3:
            x = nn.Tanh()(x)

        return x, embeddings

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


class BaseCNet(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(BaseCNet, self).__init__()

        self.in_channels = in_channels
        self.channels = [in_channels] + hidden_channels  # [5, 64, 64, 128, 256, 512]
        self.de_channels = self.channels[::-1]  # [512, 256, 128, 64, 64, 5]
        self.pool_ratios = [1] + pool_ratios  # [1, 0.8, 0.6, 0.4, 0.24]
        self.act = ReLU
        self.sum_res = sum_res  # change de_channels if not sum_res

        # construct encoder
        # construct encoder
        self.encoder0 = EncoderLayer(in_channels, 32, self.act)        # 750 x 32 (before: 64)
        self.encoder1 = EncoderLayer(32, 64, self.act)       # 750 x 64 (before: 64)
        self.encoder2 = EncoderLayer(64, 64, self.act)       # 750 x 64 (before: 128)

        self.linear = DecoderLayer(32+64+64, 128, self.act)  # 750 x 128 (before: 256)
                                                             # global_mean_pool
        # construct decoder                                  # 1 x 128  (before: 256)
        # self.decoder2 = DecoderLayer(128, 512, self.act)
        self.decoder1 = DecoderLayer(128, 512, self.act)     # 1 x 512 
        self.decoder0 = DecoderLayer(512, 750 * in_channels, None)      # 1 x (750 x 5) 

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

        if self.in_channels == 3:
            x = nn.Tanh()(x)

        return xg, embeddings

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)


class PyrmCNet(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(PyrmCNet, self).__init__()

        self.channels = [in_channels] + hidden_channels  # [5, 64, 64, 128, 256, 512]
        self.de_channels = self.channels[::-1]  # [512, 256, 128, 64, 64, 5]
        self.pool_ratios = [1] + pool_ratios  # [1, 0.8, 0.6, 0.4, 0.24]
        self.act = ReLU
        self.sum_res = sum_res  # change de_channels if not sum_res

        # construct encoder
        # construct encoder
        self.encoder0 = ConstractiveLayer(5, 64, 0.8, self.act)
        self.encoder1 = ConstractiveLayer(64, 64, 0.6, self.act)
        self.encoder2 = ConstractiveLayer(64, 64, 0.4, self.act)

        # self.linear = DecoderLayer(64, 128, self.act)

        # construct decoder
        # self.decoder2 = DecoderLayer(128, 512, self.act)
        self.decoder1 = DecoderLayer(64, 512, self.act)
        self.decoder0 = DecoderLayer(512, 750 * 5, None)

    def forward(self, x, edge_index, batch=None):

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        batch_size = batch.max() + 1
        embeddings = {}

        # input: 750 x 5
        # batch_size: 1
        x0, edge_index0, batch0 = self.encoder0(x, edge_index, batch)   # 600 x 64
        embeddings['encoder0'] = x0
        x1, edge_index1, batch1 = self.encoder1(x0, edge_index0, batch0)  # 360 x 64
        embeddings['encoder1'] = x1
        x2, edge_index2, batch2 = self.encoder2(x1, edge_index1, batch1)  # 144 x 64
        embeddings['encoder2'] = x2

        x3 = torch.cat([x0, x1, x2], dim=0)   # (600 + 360 + 144) x 64
        batch3 = torch.cat([batch0, batch1, batch2], dim=0)
        xg = global_mean_pool(x3, batch3)     # 1 x 64
        embeddings['gb_pool'] = xg

        xg = self.decoder1(xg)                # 1 x 512
        embeddings['decoder1'] = xg
        xg = self.decoder0(xg)
        embeddings['decoder0'] = xg           # 1 x (750 x 5)

        # x = nn.Tanh()(x)

        return xg, embeddings

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
