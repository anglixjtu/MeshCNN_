import numpy as np
from faiss import IndexFlatIP, IndexFlatL2
import pyvista as pv
import os
import time
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .util import get_labels_from_path


def add_subplot(plotter, coord_y, coord_x,
                mesh, font_size, label=None,
                dissm=None, filename=None,
                show_edges=True):
    plotter.subplot(coord_y, coord_x)

    text = ''
    if label is not None:
        text += label + '\n'
    if dissm is not None:
        text += "distance: %.3f \n" % (dissm)
    if filename is not None:
        text += filename
    if label or dissm or filename:
        plotter.add_text(text, font_size=font_size, color='black')
    plotter.set_background('white')
    plotter.add_mesh(mesh, color="tan", show_edges=show_edges)


def visualize_retrieval(paths_q, paths_retr, dissm=None, show_self=False,
                        sub_size=(220, 150), font_size=10, out_path=None,
                        camera_pos=[4, 4, 4]):
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
        _, filename = os.path.split(path_q)
        label = get_labels_from_path(path_q)
        label = 'Query - ' + label
        add_subplot(p, qi, 0, mesh_q, font_size,
                    label=label, filename=filename)
        p.set_position(camera_pos)

        for ri, path_r in enumerate(paths_retr[qi][start_ri:]):
            mesh_r = pv.read(path_r)
            _, filename = os.path.split(path_r)
            label = get_labels_from_path(path_r)
            dissm_r = dissm[qi, ri+start_ri]
            add_subplot(p, qi, ri+1, mesh_r,
                        font_size, dissm=dissm_r,
                        label=label, filename=filename)
            p.set_position(camera_pos)
    p.show(screenshot=out_path)


def show_embedding(self, features, idx_list):
    label_list = self.get_labels_from_index(idx_list)
    writer = SummaryWriter('runs/embedding')
    writer.add_embedding(features,
                         metadata=label_list)
    writer.close()
