from .transforms import (ConstructEdgeGraph,
                         NormalizeFeature,
                         CatPos)
from torch_geometric.transforms import (NormalizeScale,
                                        NormalizeRotation,
                                        Compose)


def set_transforms(phase, opt, mean=0, std=1, ninput_channels=5):
    
    if phase in 'train':
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature)]
        general_transformer = [NormalizeScale(),
                               CatPos(opt.input_nc)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature)]
        general_transformer = [NormalizeScale(),
                               NormalizeRotation(),
                               CatPos(opt.input_nc)]

    if phase in ['compute_mean_std']:
        general_transformer = [NormalizeScale(),
                               CatPos(opt.input_nc)]
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])


def set_transforms0(phase, opt, mean=0, std=1, ninput_channels=5):
    general_transformer = [NormalizeScale(),
                           NormalizeRotation(),
                           CatPos(opt.input_nc)]
    if phase in 'train':
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=opt.num_aug,
                                neigbs=opt.neigbs,
                                scale_verts=opt.scale_verts,
                                flip_edges=opt.flip_edges,
                                slide_verts=opt.slide_verts,
                                len_feature=opt.len_feature)]
    else:  # no augmentation for test and retrieval
        graph_transformer = \
            [ConstructEdgeGraph(ninput_edges=opt.ninput_edges,
                                num_aug=1,
                                neigbs=opt.neigbs,
                                len_feature=opt.len_feature)]

    if phase in ['compute_mean_std']:
        return Compose(graph_transformer + general_transformer)
    else:
        return Compose(graph_transformer +
                       general_transformer +
                       [NormalizeFeature(mean, std, ninput_channels)])
