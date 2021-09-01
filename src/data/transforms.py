from torch_geometric.transforms import BaseTransform


class SampleMesh(BaseTransform):
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
