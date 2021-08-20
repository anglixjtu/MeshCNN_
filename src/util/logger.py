import logging
import os


class Logger:
    def __init__(self, opt, level=logging.INFO):
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.logger_names = ['loss', 'testacc', 'runs']
        self.modes = {'loss': 'a', 'testacc': 'a', 'runs': 'w'}
        self.logger_files = {}
        self.loggers = {}

        for logger_name in self.logger_names:
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
