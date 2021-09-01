from torch_geometric.data import DataLoader
from src.util.util import parse_file_names
from .transforms import SampleMesh
from torch_geometric.transforms import NormalizeScale, Compose


def create_dataloader(opt, phase, namelist=None):
    """create a dataloader """
    from src.data.mesh_data import MeshDataset

    namelist_file = opt.namelist_file

    if phase in ['train']:
        raw_file_names = parse_file_names(opt.dataroot, namelist,
                                          namelist_file, ['train'])
        shuffle = True
        batch_size = opt.batch_size
        num_workers = int(opt.num_threads)
        pre_transform = SampleMesh(opt.ninput_edges / 1.5)
        transform = NormalizeScale()
    elif phase in ['test', 'query']:
        raw_file_names = parse_file_names(opt.dataroot, namelist,
                                          namelist_file, ['test'])
        shuffle = False
        batch_size = 1
        num_workers = 1
        pre_transform = SampleMesh(opt.ninput_edges / 1.5)
        transform = NormalizeScale()
    elif phase in ['database']:
        raw_file_names = parse_file_names(opt.dataroot, namelist,
                                          namelist_file, ['train'])
        shuffle = False
        batch_size = 1
        num_workers = 1
        pre_transform = SampleMesh(opt.ninput_edges / 1.5)
        transform = NormalizeScale()
        # TODO: batch this phase
    else:
        raise NotImplementedError('phase [%s] is not implemented' % phase)

    dataset = MeshDataset(opt, raw_file_names, phase,
                          transform=transform, pre_transform=pre_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

    return dataloader, dataset
