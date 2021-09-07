import numpy as np
from faiss import IndexFlatIP, IndexFlatL2
import pyvista as pv
import os
import time
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import json
import math
import torch
from .util import get_labels_from_path


def add_subplot(plotter, coord_y, coord_x,
                label,  mesh, font_size, dissm=None):
    plotter.subplot(coord_y, coord_x)
    plotter.add_text("{}".format(label),
                     font_size=font_size, color='black')
    if dissm is not None:
        plotter.add_text("\n\ndistance: %.3f" %
                         (dissm), font_size=font_size, color='black')
    plotter.set_background('white')
    plotter.add_mesh(mesh, color="tan", show_edges=True)


def visualize_retrieval(paths_q, paths_retr, dissm=None, show_self=False,
                        sub_size=(220, 150), font_size=10, out_path=None):
    num_query = len(paths_q)
    if show_self:
        start_ri = 0
    else:
        start_ri = 1
    num_retr = len(paths_retr[0][start_ri:])
    num_subplot = (num_query, num_retr+1)
    fig_size = ((num_retr+1)*sub_size[1], num_query*sub_size[0])

    p = pv.Plotter(shape=num_subplot,
                   window_size=fig_size, border_color='gray')
    for qi, path_q in enumerate(paths_q):
        mesh_q = pv.read(path_q)
        label = get_labels_from_path(path_q)
        label = 'Query - ' + label
        add_subplot(p, qi, 0, label, mesh_q, font_size)

        for ri, path_r in enumerate(paths_retr[qi][start_ri:]):
            mesh_r = pv.read(path_r)
            label = get_labels_from_path(path_r)
            dissm_r = dissm[qi, ri+start_ri]
            add_subplot(p, qi, ri+1, label, mesh_r, font_size, dissm_r)
    p.show(screenshot=out_path)




def show_embedding(self, features, idx_list):
    label_list = self.get_labels_from_index(idx_list)
    writer = SummaryWriter('runs/embedding')
    writer.add_embedding(features,
                             metadata=label_list)
    writer.close()
