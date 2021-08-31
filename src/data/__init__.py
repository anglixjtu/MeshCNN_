from torch_geometric.data import DataLoader
from src.util.util import parse_file_names


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

    elif phase in ['test', 'query']:
        raw_file_names = parse_file_names(opt.dataroot, namelist,
                                          namelist_file, ['test'])
        shuffle = False
        batch_size = 1
        num_workers = 1
    elif phase in ['database']:
        raw_file_names = parse_file_names(opt.dataroot, namelist,
                                          namelist_file, ['train', 'test'])
        shuffle = False
        batch_size = 1
        num_workers = 1
        # TODO: batch this phase
    else:
        raise NotImplementedError('phase [%s] is not implemented' % phase)

    dataset = MeshDataset(opt.dataroot, opt, raw_file_names, phase)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)

    return dataloader, dataset
