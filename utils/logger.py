import datetime
import logging
from pathlib import Path


def init_logging(log_dir, model_name):
    # define log dir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # init tensorboard
    tensorboard_dir = experiment_dir.joinpath('tensorboard')
    tensorboard_dir.mkdir(exist_ok=True)

    # init logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # add file output
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # add console output
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(console_formatter)
    logger.addHandler(consoleHandler)
    return logger, experiment_dir


def log_training(logger, loss, accuracy):
    logger.info('Training mean loss: %f' % loss)
    logger.info('Training accuracy: %f' % accuracy)


def log_eval(logger, eval_loss, mIoU, accuracy, class_acc,
             total_correct_class, total_iou_deno_class, class_distribution, class_names):
    logger.info('eval mean loss: %f' % eval_loss)
    logger.info('eval point avg class IoU: %f' % mIoU)
    logger.info('eval point accuracy: %f' % accuracy)
    logger.info('eval point avg class acc: %f' % class_acc)
    iou_per_class_str = '------- IoU --------\n'
    for j in range(len(class_distribution)):
        iou_per_class_str += 'class %s ratio: %f, IoU: %f \n' % (
            class_names[j] + ' ' * (14 - len(class_names[j])), class_distribution[j],
            total_correct_class[j] / float(total_iou_deno_class[j]))

    logger.info(iou_per_class_str)
