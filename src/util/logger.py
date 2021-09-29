import logging
import os
import time


class Logger:
    def __init__(self, opt, level=logging.INFO):
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.modes = {'loss': 'a', 'testacc': 'a',
                      'runs': 'a', 'retracc': 'a'}
        self.logger_files = {}
        self.loggers = {}
        self.level = level

        for logger_name in self.modes.keys():
            logger_file = os.path.join(self.save_dir, logger_name+'.log')
            mode = self.modes[logger_name]
            self.logger_files[logger_name] = logger_file
            self.loggers[logger_name] = self.setup_logger(logger_name,
                                                          logger_file,
                                                          mode,
                                                          level)

        # TODO: add log for retrieval
        # TODO: add tensorboard

    @staticmethod
    def setup_logger(logger_name, log_file, mode='w', level=logging.INFO):
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(fmt='%(asctime)s[line:%(lineno)d] '
                                          '%(levelname)s %(message)s',
                                      datefmt='%a, %d-%m-%Y %H:%M:%S')
        fileHandler = logging.FileHandler(log_file, mode=mode)
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)
        return logger

    def record_dataset(self, dataset_size,
                       mode='classification', phase='train'):
        message = 'dataset mode = %s; #%s data = %d' %\
                  (mode, phase, dataset_size)
        self.loggers['runs'].info(message)

    def record_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' %\
                      (epoch, i, t, t_data, losses.item())
        self.loggers['loss'].info(message)

    def record_saving(self, epoch, i):
        message = 'saving the model (epoch %d, total_steps %d)' %\
            (epoch, i)
        self.loggers['runs'].info(message)

    def record_time(self, epoch, i, t):
        message = 'End of epoch %d / %d \t Time Taken: %d sec' %\
            (epoch, i, t)
        self.loggers['loss'].info(message)

    def record_acc(self, epoch, acc):
        message = 'epoch: {}, TEST ACC/ERR: [{:.2}]\n' \
            .format(epoch, acc)
        self.loggers['testacc'].info(message)

    def record_retracc(self, num_neigb, which_layer,
                       pooling, feature_size, normalize,
                       search_methods, metrics):
        self.loggers['retracc'].handlers[0].setFormatter(
            logging.Formatter('%(message)s'))
        self.loggers['retracc'].handlers[1].setFormatter(
            logging.Formatter('%(message)s'))

        now = time.strftime("%c")
        message = '================ Retrieval Acc (%s) ================\n'\
                  'Maximum retrieve %d nearest samples. \n'\
                  'Using the embeddings from layer [%s]. \n'\
                  'Using pooling                   [%s]. \n'\
                  'Feature length                  [%d]. \n'\
                  'Normalize                       [%d]. \n'\
                  'Searching method                [%s]. \n'\
                  % (now, num_neigb, which_layer, pooling,
                     feature_size, normalize, search_methods)
        self.loggers['retracc'].info(message)

        message = ''
        for metric in metrics.keys():
            message += metric + ': ' + str(metrics[metric])[:5] + ', '
        message += '\n'
        self.loggers['retracc'].info(message)

    def record_opt(self, opt):
        logger_file = os.path.join(self.save_dir, 'opt.log')
        mode = 'w'
        self.logger_files['opt'] = logger_file
        self.loggers['opt'] = self.setup_logger('opt',
                                                logger_file,
                                                mode,
                                                self.level)
        self.loggers['opt'].handlers[0].setFormatter(
            logging.Formatter('%(message)s'))
        self.loggers['opt'].handlers[1].setFormatter(
            logging.Formatter('%(message)s'))

        args = vars(opt)

        self.loggers['opt'].info('------------ Options -------------')
        for k, v in sorted(args.items()):
            self.loggers['opt'].info('%s: %s' % (str(k), str(v)))
        self.loggers['opt'].info('------------ End -------------')
