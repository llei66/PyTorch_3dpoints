import logging
from pathlib import Path


def init_logging(exp_dir, model_name, to_console: bool, to_file: bool, output_file_suffix="train"):
    # define log dir
    experiment_dir = Path(exp_dir)
    experiment_dir.mkdir(exist_ok=True)

    experiment_dir = experiment_dir.joinpath(model_name)
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

    if to_file:
        # add file output
        file_formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler('%s/%s_%s.txt' % (log_dir, model_name, output_file_suffix))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if to_console:
        # add console output
        console_formatter = logging.Formatter('%(message)s')
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(console_formatter)
        logger.addHandler(consoleHandler)
    return logger, experiment_dir, checkpoints_dir


def log_training(logger, loss, accuracy):
    logger.info('Training mean loss: %f' % loss)
    logger.info('Training accuracy: %f' % accuracy)


def log_eval(logger, eval_loss, mIoU, accuracy, class_acc,
             total_correct_class, total_iou_deno_class, class_distribution, class_names):
    logger.info('mean loss: %f' % eval_loss)
    logger.info('point avg class IoU: %f' % mIoU)
    logger.info('point accuracy: %f' % accuracy)
    logger.info('point avg class acc: %f' % class_acc)
    iou_per_class_str = '------- IoU --------\n'
    for j in range(len(class_distribution)):
        iou_per_class_str += 'class %s ratio: %f, IoU: %f \n' % (
            class_names[j] + ' ' * (14 - len(class_names[j])), class_distribution[j],
            total_correct_class[j] / float(total_iou_deno_class[j]))

    logger.info(iou_per_class_str)
