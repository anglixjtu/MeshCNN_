import time
import torch

from src.data import create_dataloader
from src.models import Model
from src.options.train_options import TrainOptions
from src.util.logger import Logger
from src.util.util import MetricCounter


def run_training(opt):
    logger = Logger(opt)

    dataloaders, dataset = {}, {}
    for phase in ['train', 'test']:
        dataloaders[phase], dataset[phase] = create_dataloader(opt, phase)
        logger.record_dataset(len(dataset[phase]), opt.mode, phase)

    logger.record_opt(opt)

    model = Model(opt, phase='train')

    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # train the network
        model.net.train()
        for i, data in enumerate(dataloaders['train']):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            '''if i == 0 and epoch == 1:
                writer.plot_arch(
                    model.net, model.data.x,
                    model.data.edge_index, model.data.batch)'''

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                logger.record_losses(epoch, epoch_iter, loss, t, t_data)

            if i % opt.save_latest_freq == 0:
                model.save_network('latest')
                logger.record_saving(epoch, total_steps)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            model.save_network('latest')
            model.save_network(epoch)
            logger.record_saving(epoch, total_steps)

        logger.record_time(epoch, opt.niter + opt.niter_decay,
                           time.time() - epoch_start_time)
        model.update_learning_rate()
        '''if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)'''

        # test
        model.net.eval()
        acc_counter = MetricCounter()
        for i, data in enumerate(dataloaders['test']):
            model.set_input(data)
            accuracy = model.test()
            acc_counter.update(accuracy, n=1)
        logger.record_acc(epoch, acc_counter.avg)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    # TODO: clean options
    run_training(opt)
