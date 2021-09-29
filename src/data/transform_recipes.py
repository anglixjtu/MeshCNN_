from .transforms import (ConstructEdgeGraph,
                         NormalizeFeature,
                         SetX,
                         SetY)
from torch_geometric.transforms import (NormalizeScale,
                                        NormalizeRotation,
                                        RandomRotate,
                                        Compose)


def set_transforms10(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [RandomRotate(180, 0),
             RandomRotate(180, 1),
             RandomRotate(180, 2),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms9(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [RandomRotate(180, 0),
             RandomRotate(180, 1),
             RandomRotate(180, 2),
             NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms8(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [RandomRotate(180, 0),
                               RandomRotate(180, 1),
                               RandomRotate(180, 2),
                               NormalizeScale(),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [NormalizeScale(),
             NormalizeRotation(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms7(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [RandomRotate(180, 0),
                               RandomRotate(180, 1),
                               RandomRotate(180, 2),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [NormalizeScale(),
             NormalizeRotation(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms6(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [RandomRotate(180, 0),
                               RandomRotate(180, 1),
                               RandomRotate(180, 2),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms5(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [RandomRotate(180, 0),
             RandomRotate(180, 1),
             RandomRotate(180, 2),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms4(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [NormalizeRotation(),
             NormalizeScale(),
             ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc),
             SetY(mode=opt.mode, input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms3(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               RandomRotate([-180, 180], 0),
                               RandomRotate([-180, 180], 1),
                               RandomRotate([-180, 180], 2),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms2(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               RandomRotate([-180, 180], 0),
                               RandomRotate([-180, 180], 1),
                               RandomRotate([-180, 180], 2),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               NormalizeRotation(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms1(phase, opt, mean=0, std=1, ninput_channels=5):

    if phase in 'train':
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]
        general_transformer = [NormalizeScale(),
                               NormalizeRotation(),
                               SetX(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               SetX(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms0(phase, opt, mean=0, std=1, ninput_channels=5):
    general_transformer = [NormalizeScale(),
                           NormalizeRotation(),
                           SetX(opt.input_nc)]
    if phase in 'train':
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature,
                                input_nc=opt.input_nc)]

    if phase in ['compute_mean_std']:
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])
