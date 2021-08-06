from options.test_options import TestOptions
from data import CreateDataset
from torch_geometric.data import DataLoader
from models import CreateModel
from util.writer import Writer
from torchsummary import summary


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    dataset = CreateDataset(opt)
    opt.serial_batches = True  # no shuffle
    dataloader = DataLoader( dataset, batch_size=1,#opt.batch_size
            shuffle=False,
            num_workers=1)
    model = CreateModel(opt)
    #summary(model.net, input_size=[(750, 5), (2, 3000), (750,1) ])

    writer = Writer(opt, False)
    # test
    writer.reset_counter()

    
    
    for i, data in enumerate(dataloader):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.plot_arch(model.net, model.data.x, model.data.edge_index, model.data.batch)    
    writer.print_acc(epoch, writer.acc)
    writer.close()
    return writer.acc


if __name__ == '__main__':
    run_test()
