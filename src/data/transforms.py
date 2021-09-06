from .mesh_process import (init_mesh_data,
                           remove_non_manifolds,
                           remove_isolated_vertices,
                           augmentation,
                           build_gemm,
                           post_augmentation,
                           extract_features,
                           compute_edge_pos,
                           get_edge_connection)
from src.util.util import pad
import torch
from torch_geometric.nn import knn_graph
import numpy as np
from torch_geometric.data import Data


class SampleMesh(object):
    """ Sample faces of a mesh."""

    def __init__(self, ntarget_faces):
        self.ntarget_faces = ntarget_faces

    def __call__(self, mesh):
        nfaces = len(mesh.faces)
        mesh_out = mesh

        # upsample
        if nfaces < self.ntarget_faces:
            nsub = max(1, round((self.ntarget_faces/nfaces)**0.25))
            for i in range(nsub):
                mesh_out = mesh_out.subdivide()
        # downsample
        mesh_out = mesh_out.simplify_quadratic_decimation(self.ntarget_faces)

        return mesh_out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ConstructEdgeGraph(object):
    """ Construct the edge graph from a mesh.
        Data.x is the 5-channel edge features.
        num_aug: # of augmentation files
        neigbs: method for finding neighboring edges(nodes).
                if neigbs=0: 1-ring neighbors
                elif neighbs>0: knn neighbors

    Args:
        mesh_tm: input mesh data in trimesh structure
    Return:
        edge_graph: pytorch-geometric graph data for edges(mesh)

    """

    def __init__(self, ninput_edges, num_aug=1,
                 neigbs=11, 
                 scale_verts=False, flip_edges=0.2,
                 slide_verts=0.2,
                 len_feature=True):
        self.ninput_edges = ninput_edges
        self.num_aug = num_aug
        self.neigbs = neigbs
        self.scale_verts = scale_verts
        self.flip_edges = flip_edges
        self.slide_verts = slide_verts
        self.len_feature = len_feature

    def __call__(self, mesh_tm):
        mesh_data = init_mesh_data()

        faces = mesh_tm.faces
        mesh_data.vs = mesh_tm.vertices

        # remove non-manifold vertices and edges
        faces, face_areas = remove_non_manifolds(mesh_data, faces)
        faces = remove_isolated_vertices(mesh_data, faces)

        # feature augmentation
        if self.num_aug > 1:
            mesh_data, faces = augmentation(mesh_data, faces,
                                            self.scale_verts,
                                            self.flip_edges)
        build_gemm(mesh_data, faces, face_areas)

        if self.num_aug > 1:
            post_augmentation(mesh_data, self.slide_verts)

        # extract 5-/6-channel features
        mesh_data.features = extract_features(mesh_data)
        mesh_data.pos = compute_edge_pos(mesh_data.edges, mesh_data.vs)

        # resize the number of input edges
        if mesh_data.features.shape[1] < self.ninput_edges:
            edge_features = pad(mesh_data.features, self.ninput_edges)
            edge_features = edge_features.transpose()
            edge_pos = pad(mesh_data.pos, self.ninput_edges, dim=0)
            edge_len = pad(mesh_data.edge_lengths.reshape(-1, 1),
                           self.ninput_edges, dim=0)
        else:
            edge_features = mesh_data.features[:, :self.ninput_edges]
            edge_features = edge_features.transpose()
            edge_pos = mesh_data.pos[:self.ninput_edges, :]
            edge_len = mesh_data.edge_lengths.reshape(-1, 1)
            edge_len = edge_len[:self.ninput_edges, :]

        # generate connections
        if self.neigbs > 0:
            edge_pos = torch.tensor(edge_pos,
                                    dtype=torch.float)
            batch = torch.zeros(len(edge_pos), dtype=torch.long)
            edge_connections = knn_graph(edge_pos, k=self.neigbs,
                                         batch=batch, loop=False)
        else:
            edge_connections = get_edge_connection(mesh_data.gemm_edges)
            out = np.min(edge_connections, axis=0)
            out_i = np.arange(len(out))[out >= self.ninput_edges]
            if len(out_i) > 0:
                edge_connections = np.delete(edge_connections, out_i, axis=1)
            edge_connections = torch.tensor(edge_connections,
                                            dtype=torch.long)

        # convert edge features and connections to pytorch-geometric data
        if self.len_feature:
            edge_features = np.concatenate((edge_features,
                                            edge_len), 1)
        edge_features = torch.tensor(edge_features,
                                     dtype=torch.float)

        graph_data = Data(x=edge_features, edge_index=edge_connections,
                          pos=edge_pos)

        return graph_data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NormalizeFeature(object):
    """ Normalize features using mean and std computed offline."""

    def __init__(self, mean, std, ninput_channels):
        self.mean = mean
        self.std = std
        self.ninput_channels = ninput_channels

    def __call__(self, data):
        n_channels = data.x.shape[1]
        if n_channels != self.ninput_channels:
            raise ValueError("Data contains %d channels, "
                             "%d is expected for normalization"
                             % (n_channels, self.ninput_channels))
        mean = torch.tensor(self.mean).reshape(1, -1)
        std = torch.tensor(self.std).reshape(1, -1)
        data.x = (data.x - mean) / std
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class CatPos(object):
    """ Concatenate position vectors to features."""

    def __init__(self, input_nc):
        self.input_nc = input_nc

    def __call__(self, data):
        if self.input_nc > 5:
            data.x = torch.cat((data.x, data.pos), 1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
