from torch_geometric.data import DataLoader
from torch_geometric.transforms import NormalizeScale, Compose, FaceToEdge


def create_dataset(opt):
    """loads dataset class"""

    from src.data.mesh_data import MeshDataset
    dataset = MeshDataset(opt)
    return dataset


def create_dataloader(opt, phase):
    """create a dataloader """
    if opt.dataset_mode == 'edge' and opt.mode == 'autoencoder':
        from src.data.mesh_data import MeshDataset
        transform = NormalizeScale()
    elif opt.dataset_mode == 'edge' and opt.mode == 'classification':
        from src.data.mesh_data import MeshDataset
        transform = None
    elif opt.dataset_mode == 'vertice':
        from src.data.vertice_data import MeshDataset
        transform = Compose([FaceToEdge(False),
                            NormalizeScale()])

    if phase in ['train']:
        dataset = MeshDataset(opt, ['train'], phase=phase, transform=transform)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=int(opt.num_threads))
    elif phase in ['test', 'query']:
        dataset = MeshDataset(opt, ['test'], phase=phase, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False,
                                num_workers=1)
    elif phase in ['database']:
        dataset = MeshDataset(opt, ['test', 'train'], phase=phase, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False,
                                num_workers=1)
    # TODO: add for embedding, retrieval phase ...

    return dataloader, dataset