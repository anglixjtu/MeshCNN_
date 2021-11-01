from .mesh_process import (init_mesh_data,
                           remove_non_manifolds,
                           remove_isolated_vertices,
                           augmentation,
                           build_gemm,
                           post_augmentation,
                           extract_features,
                           get_edge_connection,)
from src import util
import torch
from torch_geometric.nn import knn_graph
import numpy as np
from torch_geometric.data import Data
import math

from torch_geometric.transforms import LinearTransformation


class SampleMesh(object):
    """ Sample faces of a mesh."""

    def __init__(self, ntarget_faces):
        self.ntarget_faces = ntarget_faces

    def __call__(self, mesh):
        nfaces = len(mesh.faces)
        mesh_out = mesh

        try:
            # upsample
            if nfaces < self.ntarget_faces:
                nsub = max(1, round((self.ntarget_faces/nfaces)**0.25))
                for i in range(nsub):
                    mesh_out = mesh_out.subdivide()
            # downsample
            mesh_out = mesh_out.simplify_quadratic_decimation(
                self.ntarget_faces)
            return mesh_out
        except:
            return mesh_out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ConstructEdgeGraph(object):
    """ Construct the edge graph from a mesh.
        Data.x is the edge features.
        Data.edge_index is the connection between edges.
        Data.pos is the position of edges.
        Data.y is the postion of randomly sampled points on the mesh.
        num_aug: # of augmentation files
        neigbs: method for finding neighboring edges(nodes).
                if neigbs=0: 1-ring neighbors
                elif neighbs>0: knn neighbors

    Args:
        mesh_tm: input mesh data in trimesh structure
    Return:
        edge_graph(Data): pytorch-geometric graph data for edges(mesh)

    """

    def __init__(self, ninput_edges, num_aug=1,
                 neigbs=11,
                 scale_verts=False, flip_edges=0.2,
                 slide_verts=0.2,
                 len_feature=True,
                 input_nc=8):
        self.ninput_edges = ninput_edges
        self.num_aug = num_aug
        self.neigbs = neigbs
        self.scale_verts = scale_verts
        self.flip_edges = flip_edges
        self.slide_verts = slide_verts
        self.len_feature = len_feature
        self.input_nc = input_nc

    def sample_y(self, mesh_in, n=2048):
        y = SamplePoints(n)(mesh_in).y.numpy()
        y -= np.mean(y, 0)
        y /= np.abs(y).max()  # np.sqrt(np.std(y, 0))
        y *= 0.999999
        return y

    def __call__(self, mesh_in):
        mesh_data = init_mesh_data()

        faces = mesh_in.face.numpy().transpose()
        mesh_data.vs = mesh_in.pos.numpy()

        # remove non-manifold vertices and edges
        faces, face_areas = remove_non_manifolds(mesh_data, faces)
        faces = remove_isolated_vertices(mesh_data, faces)

        # mesh augmentation from MeshCNN including scale_verts, flip_edges
        # and slide_verts
        if self.num_aug > 1:
            faces = augmentation(mesh_data, faces,
                                 self.scale_verts,
                                 self.flip_edges)

        # Compute 4 one-ring neighbors for each edge. From MeshCNN.
        build_gemm(mesh_data, faces, face_areas)

        # mesh augmentation from MeshCNN including scale_verts, flip_edges
        # and slide_verts
        if self.num_aug > 1:
            post_augmentation(mesh_data, self.slide_verts)

        # extract edge features by Ang Li
        mesh_data.features = extract_features(mesh_data, self.input_nc)
        # features for connections

        # resize the number of input edges
        if mesh_data.features.shape[0] < self.ninput_edges:
            edge_features = util.util.pad(
                mesh_data.features, self.ninput_edges, dim=0)
            # edge_features = edge_features.transpose()
            edge_pos = util.util.pad(mesh_data.pos, self.ninput_edges, dim=0)
            '''edge_len = util.util.pad(mesh_data.edge_lengths.reshape(-1, 1),
                                     self.ninput_edges, dim=0)'''
        else:
            edge_features = mesh_data.features[:self.ninput_edges, :]
            # edge_features = edge_features.transpose()
            edge_pos = mesh_data.pos[:self.ninput_edges, :]
            '''edge_len = mesh_data.edge_lengths.reshape(-1, 1)
            edge_len = edge_len[:self.ninput_edges, :]'''

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
        edge_features = torch.tensor(edge_features,
                                     dtype=torch.float)
        y = self.sample_y(mesh_in)
        y = torch.tensor(y, dtype=torch.float)

        graph_data = Data(x=edge_features, edge_index=edge_connections,
                          pos=edge_pos, y=y)

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
        if n_channels in [8]:  # do not use mean/std normalize pos5
            mean[:, 5:] = 0
            std[:, 5:] = 1
        data.x = (data.x - mean) / std
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class SetX(object):
    """ Set data.x by concatenating position vectors to features."""

    def __init__(self, input_nc):
        self.input_nc = input_nc

    def __call__(self, data):
        if self.input_nc in [8]:
            data.x = torch.cat((data.x, data.pos), 1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class SamplePoints(object):
    r"""From torch-geometic.
    Uniformly samples :obj:`num` points on the mesh faces according to
    their face area.

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
    """

    def __init__(self, num, remove_faces=True, include_normals=False):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def __call__(self, data):
        data_out = data.clone()
        pos, face = data_out.pos, data_out.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.max()
        pos = pos / pos_max

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data_out.norm = torch.nn.functional.normalize(
                vec1.cross(vec2), p=2)

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data_out.y = pos_sampled

        if self.remove_faces:
            data_out.face = None

        return data_out

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.num)


class SetY(object):
    """ Set Y(target) in Data."""

    def __init__(self, mode, input_nc):
        self.mode = mode
        self.input_nc = input_nc

    def __call__(self, data):
        if self.mode in ['autoencoder']:
            x = data.x.clone()
            '''x_mean = torch.mean(x, 0, keepdim=True)
            x -= x_mean
            num = torch.max(x.abs(), 0, keepdim=True)[0]
            x = (x / num) * 0.99999'''

            # data.x = x

            if self.input_nc in [5, 7, 6, 9, 10, 16]:
                data.y = x
            elif self.input_nc in [8]:
                pos = data.pos.clone()
                pos_mean = torch.mean(pos, 0, keepdim=True)
                pos -= pos_mean
                data.y = (torch.cat((x, pos), 1))
        if self.mode in ['autoencoder_glb']:
            x = data.x[:, [5, 10, 11, 12, 13, 14, 15]].clone()
            '''x_mean = torch.mean(x, 0, keepdim=True)
            x -= x_mean
            num = torch.max(x.abs(), 0, keepdim=True)[0]
            x = (x / num) * 0.99999'''
            data.y = x

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class Rotate(object):
    r"""From Torch-geometric.
    Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, axis=0):
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * self.degrees / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)
