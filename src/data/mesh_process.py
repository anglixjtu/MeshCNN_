import numpy as np
import time


def init_mesh_data():
    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.pos = None
    return mesh_data


def compute_features(mesh_tm, opt):
    """compute 5-channel input feature from mesh"""

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.pos = None

    mesh_out = mesh_tm
    faces = mesh_out.faces
    mesh_data.vs = mesh_out.vertices
    # remove non-manifold vertices and edges
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    faces = remove_isolated_vertices(mesh_data, faces)

    if opt.num_aug > 1:
        # TODO: check the right augmentations for mechanical data
        faces = augmentation(mesh_data, opt, faces)
    build_gemm(mesh_data, faces, face_areas)

    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)

    mesh_data.features = extract_features(mesh_data)
    mesh_data.pos = compute_edge_pos(mesh_data.edges, mesh_data.vs)

    return mesh_out, mesh_data


def sample_and_compute_features(mesh_tm, opt):
    # by Ang Li

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []

    start_t = time.time()
    if opt.sample_mesh == 'trimesh':
        mesh_out, faces, face_areas = sample_mesh_tm(mesh_tm,
                                                     mesh_data,
                                                     opt.ninput_edges,
                                                     opt.sample_mesh)
    elif opt.sample_mesh == 'pyvista':
        mesh_out, faces, face_areas = sample_mesh(mesh_tm,
                                                  mesh_data,
                                                  opt.ninput_edges,
                                                  opt.sample_mesh)
    else:
        mesh_out = mesh_tm
        faces = mesh_out.faces
        mesh_data.vs = mesh_out.vertices
        # remove non-manifold vertices and edges
        faces, face_areas = remove_non_manifolds(mesh_data, faces)
        faces = remove_isolated_vertices(mesh_data, faces)
        # TODO: clean this part

    if opt.num_aug > 1:
        # TODO: check the right augmentations for mechanical data
        faces = augmentation(mesh_data, opt, faces)
    build_gemm(mesh_data, faces, face_areas)

    # remove edges until the number of edges is less than the threshold
    while (len(mesh_data.edges) > opt.ninput_edges) and\
          (opt.sample_mesh in ['trimesh', 'pyvista']):
        mesh_data.edge_areas = []
        mesh_data.edge_lengths = None
        mesh_data.edges = None
        mesh_data.gemm_edges = mesh_data.sides = None
        mesh_data.edges_count = None
        mesh_data.ve = None

        if opt.sample_mesh == 'trimesh':
            mesh_out, faces, face_areas = sample_mesh_tm(mesh_out,
                                                         mesh_data,
                                                         opt.ninput_edges,
                                                         opt.sample_mesh,
                                                         rate=0.01)
        elif opt.sample_mesh == 'pyvista':
            mesh_out, faces, face_areas = sample_mesh(mesh_out,
                                                      mesh_data,
                                                      opt.ninput_edges,
                                                      opt.sample_mesh,
                                                      rate=0.01)
        else:
            mesh_out = mesh_tm
            faces = mesh_out.faces
            mesh_data.vs = mesh_out.vertices
            # remove non-manifold vertices and edges
            faces, face_areas = remove_non_manifolds(mesh_data, faces)
            faces = remove_isolated_vertices(mesh_data, faces)

        if opt.num_aug > 1:
            faces = augmentation(mesh_data, opt, faces)
        build_gemm(mesh_data, faces, face_areas)

    end_t = time.time()
    opt.t_pp += end_t - start_t

    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)

    start_t = time.time()
    mesh_data.features = extract_features(mesh_data)
    end_t = time.time()
    opt.t_ef += end_t - start_t

    return mesh_out, mesh_data


def get_edge_connection(gemm_edges):
    sz = len(gemm_edges)
    edge_indices = np.arange(sz)
    edge_connection = np.zeros((2, 4*sz))
    for i in range(gemm_edges.shape[1]):
        edge_connection[0, i*sz:(i+1)*sz] = edge_indices
        edge_connection[1, i*sz:(i+1)*sz] = gemm_edges[:, i]
    valid = np.min(edge_connection, axis=0)
    valid = np.tile([valid > -1], (2, 1))
    edge_connection = edge_connection[valid].reshape(2, -1)

    return edge_connection


# Preprocess methods by Ang Li
def sample_mesh(mesh, mesh_data, ninput_edges,
                sample_mesh='trimesh', rate=None):
    nfaces_target = ninput_edges / 1.5
    nfaces = len(mesh.faces)

    if rate is None:
        # Subdivide the mesh if the number of faces is less than a threshold
        if (nfaces < nfaces_target) and (rate is None):
            nsub = max(1, round((nfaces_target/nfaces)**0.25))
            for i in range(nsub):
                mesh = mesh.subdivide()
            nfaces = len(mesh.faces)

        # convert from trimesh to pyvista
        if sample_mesh == 'pyvista':
            import pyvista as pv
            vertices = mesh.vertices
            faces = np.concatenate((np.ones((len(mesh.faces), 1))*3,
                                    mesh.faces), axis=1)
            faces = np.hstack(faces)
            mesh = pv.PolyData(vertices, faces.astype(np.int64))
            # Decimate the mesh if #faces is larger than a threshold
            if (nfaces > nfaces_target):
                rate = 1.0 - nfaces_target/nfaces
                mesh = mesh.decimate(rate)
    else:
        mesh = mesh.decimate(rate)

    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    mesh_data.vs = np.array(mesh.points)
    # remove non-manifold vertices and edges
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    faces = remove_isolated_vertices(mesh_data, faces)
    return mesh, faces, face_areas


def sample_mesh_tm(mesh, mesh_data, ninput_edges,
                   sample_mesh=True, rate=None):
    nfaces_target = ninput_edges / 1.5
    nfaces = len(mesh.faces)

    if rate is None:
        # Subdivide the mesh if the number of faces is less than a threshold
        if (nfaces < nfaces_target) and (rate is None) and (sample_mesh):
            nsub = max(1, round((nfaces_target/nfaces)**0.25))
            for i in range(nsub):
                mesh = mesh.subdivide()
            nfaces = len(mesh.faces)

        # Decimate the mesh if the number of faces is larger than a threshold
        if (nfaces > nfaces_target) and (sample_mesh):
            mesh = mesh.simplify_quadratic_decimation(nfaces_target)
    else:
        mesh = mesh.simplify_quadratic_decimation(round(nfaces*(1-rate)))

    faces = mesh.faces
    mesh_data.vs = mesh.vertices
    # remove non-manifold vertices and edges
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    faces = remove_isolated_vertices(mesh_data, faces)
    return mesh, faces, face_areas


def remove_non_manifolds(mesh, faces):
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    # _, face_areas = compute_face_normals_and_areas(mesh, faces)
    face_areas = compute_face_areas(mesh, faces)

    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]


def remove_isolated_vertices(mesh, faces):
    """Written by Ang Li"""
    v_in_faces = np.unique(faces)
    if len(v_in_faces) < len(mesh.vs):
        v_without_edges = set(range(len(mesh.vs))) - set(v_in_faces)
        v_without_edges = sorted(list(v_without_edges), reverse=True)
        for v in v_without_edges:
            mesh.vs = np.delete(mesh.vs, v, axis=0)
            faces[faces >= v] -= 1

    return faces


def build_gemm(mesh, faces, face_areas):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating
    where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """

    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] =\
                edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] =\
                edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] =\
                nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] =\
                nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) /\
        np.sum(face_areas)


def compute_face_normals_and_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= face_areas[:, np.newaxis]
    if np.any(face_areas[:, np.newaxis] == 0):
        debug_point = 195
    # assert (not np.any(face_areas[:, np.newaxis] == 0)),
    # 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas


def compute_face_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    # assert (not np.any(face_areas[:, np.newaxis] == 0)),
    # 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_areas


# Data augmentation methods
def augmentation(mesh, faces=None,
                scale_verts_f=False,
                flip_edges_f=0.2):
    if scale_verts_f:
        scale_verts(mesh)
    if flip_edges_f > -1:
        faces = flip_edges(mesh, flip_edges_f, faces)
    return faces


def post_augmentation(mesh, slide_verts_f=0.2):
    if slide_verts_f > -1:
        slide_verts(mesh, slide_verts_f)


def slide_verts(mesh, prct):
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze()
    thr = np.mean(dihedral) + np.std(dihedral)
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0

    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]

            if min(dihedral[edges]) > 2.65:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) *\
                    (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.1):
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] -\
            mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] -\
            mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1),
                             epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^
                             set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]],
                 [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j],
                                         cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2],
                                     edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    # print(flipped)
    return faces


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face


def check_area(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] -
                                  mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    mesh.edge_lengths = edge_lengths


def extract_features(mesh, input_nc):
    if input_nc in [6, 7]:
        extractors = [dihedral_angle_areas]
    elif input_nc in [5, 8, 9]:
        extractors = [dihedral_angle,
                      symmetric_opposite_angles,
                      symmetric_ratios]
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in extractors:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def extract_features_3(mesh, extractors):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in extractors:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def dihedral_angle_areas(mesh, edge_points):
    normals_a, areas_a = get_normals_areas(mesh, edge_points, 0)
    normals_b, areas_b = get_normals_areas(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    max_area = areas_a.max()
    areas_a /= max_area
    areas_b /= max_area
    return np.concatenate((angles, areas_a, areas_b), axis=0)


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0),
                             np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0),
                             np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are
        the two vertices which define edge_id
        each adjacent face to edge_id has another vertex,
        which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex],
            edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - \
        mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - \
        mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals


def get_normals_areas(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - \
        mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - \
        mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    areas = np.linalg.norm(normals, ord=2, axis=1)
    div = fixed_division(areas, epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals, 0.5 * areas.reshape(1, -1)


def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - \
        mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - \
        mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1),
                              epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1),
                              epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] -
                                   mesh.vs[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) /\
        fixed_division(np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] *\
        line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths


def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div


def compute_edge_pos(edges, vertices):
    pos = np.zeros((len(edges), 3))
    for i, edge in enumerate(edges):
        pos_a = vertices[edge[0], :]
        pos_b = vertices[edge[1], :]
        pos[i, :] = np.array((pos_a + pos_b) / 2.)
    return pos
