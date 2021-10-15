import time
import os

from torch.nn.functional import embedding
from src import data

from src.data import create_dataloader
from src.models import Model
from src.options.extractor_options import ExtractorOptions
from src.options.load_options import load_model_opt
from src.util.logger import Logger
from src.util.util import MetricCounter, mkdir
from src.util.embd_extractor import EmbdExtractor


def run_extractor(opt):
    logger = Logger(opt)
    logger.loggers['runs'].info('Generating embeddings...')

    # load model configurations from checkpoint file (.txt)
    model_opt_file = os.path.join('checkpoints', opt.name, 'opt.log')
    opt = load_model_opt(model_opt_file, opt)

    phase = 'database'
    dataloader, dataset = create_dataloader(opt, phase)

    model = Model(opt, phase=phase)

    extractor = EmbdExtractor(opt.which_layer,
                              dataset.raw_file_paths,
                              pooling=opt.pooling,
                              normalize=opt.normalize)

    embeddings = extractor(model, dataloader)
    logger.loggers['runs'].info('Done!\n')

    save_dir = os.path.join(opt.save_dir, opt.name, opt.which_layer)
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, ''.join(opt.set))  # + '_rr'
    extractor.save(save_dir, embeddings)

    logger.loggers['runs'].info('Embeddings saved as %s' % save_dir+'.pt')


if __name__ == '__main__':
    opt = ExtractorOptions().parse()
    # TODO: clean options
    run_extractor(opt)
