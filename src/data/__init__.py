from torch_geometric.data import DataLoader


def create_dataset(opt):
    """loads dataset class"""

    from src.data.mesh_data import MeshDataset
    dataset = MeshDataset(opt)
    return dataset


def create_dataloader(opt, phase):
    """create a dataloader """
    from src.data.mesh_data import MeshDataset

    if phase in ['train']:
        dataset = MeshDataset(opt, ['train'])
        dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=int(opt.num_threads))
    elif phase in ['test', 'retrieval']:
        dataset = MeshDataset(opt, ['test'])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False,
                                num_workers=1)
    # TODO: add for embedding, retrieval phase ...

    return dataloader, dataset