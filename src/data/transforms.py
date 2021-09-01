

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

    def __init__(self, num_aug, neigbs=11):
        self.num_aug = num_aug
        self.neigbs = neigbs

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