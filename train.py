# TODO add more comments on what is happening

import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.denmark_dataset import get_data_loader
from utils.logger import init_logging, log_training, log_eval
from utils.model_helper import TrainModel


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--data-path', type=str, default='./data/train_test_whole_class_1km',
                        help='path to data [default: "./data/train_test_whole_class_1km"]')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=128, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log-dir', type=str, default="./log", help='Log path [default: "./log"]')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--points-per-sample', type=int, default=4096, help='points per sample [default: 4096]')
    parser.add_argument('--eval-points-per-sample', type=int, default=10000,
                        help='points per sample during eval [default: 10000]')
    parser.add_argument('--blocks-per-epoch', type=int, default=4096, help='blocks per epoch [default: 4096]')
    parser.add_argument('--block-size-x', type=float, default=0.05,
                        help='normalized block size for x coordinate [default: 0.05]')
    parser.add_argument('--block-size-y', type=float, default=0.05,
                        help='normalized block size for x coordinate [default: 0.05]')
    parser.add_argument('--step-size', type=int, default=10,
                        help='Decay step for lr and batch norm mementum decay [default: every 10 epochs]')
    parser.add_argument('--lr-decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--weighting', type=str, default="equal",
                        help='defines the loss weighting method, either "equal" or "class" [default: "equal"]')
    parser.add_argument('--n-data-worker', type=int, default=4,
                        help='data preprocessing threads [default: 4]')
    parser.add_argument('--no-rgb', action='store_true', default=False, help="ignores RBG if used")
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='internal of saving model [default: every 10 epochs]')
    parser.add_argument('--no-console-logging', type=bool, default=False, action='store_true',
                        help='deactivates most console output')
    parser.add_argument('--no-file-logging', type=bool, default=False, action='store_true',
                        help='deactivates logging into a file')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = parse_args()

    # init logger
    logger, checkpoint_dir = init_logging(
        args.log_dir, args.model, not args.no_console_logging, not args.no_file_logging
    )
    save_epoch = args.save_epoch

    # init data loader
    # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)
    block_size = (args.block_size_x, args.block_size_y)
    logger.info("start loading train data ...")
    # TODO add back augmentations
    train_loader = get_data_loader(
        batch_size=args.batch_size, blocks_per_epoch=args.blocks_per_epoch, points_per_sample=args.points_per_sample,
        block_size=block_size, data_path=args.data_path, split="train",
        use_rgb=not args.no_rgb, training=True, n_data_worker=args.n_data_worker
    )
    # save stats for external testing
    torch.save({
        "global_z": train_loader.dataset.global_z,
        "n_classes": train_loader.dataset.n_classes
    }, f"{str(checkpoint_dir)}/data_stats.pth")

    logger.info("start loading validation data ...")
    # val loader has to use batch size of 1 to allow for varying point clouds
    val_loader = get_data_loader(
        batch_size=1, points_per_sample=args.eval_points_per_sample, data_path=args.data_path, split="validate",
        use_rgb=not args.no_rgb, training=False, n_data_worker=args.n_data_worker,
        global_z=train_loader.dataset.get_global_z(),
    )

    # determine weighting method for loss function
    class_weights = None
    if args.weighting == "class":
        # load already inverted weights TODO might be cleaner to invert them here
        class_weights = torch.Tensor(train_loader.dataset.class_weights)

    ## import tensorboard to view

    writer = SummaryWriter(str(checkpoint_dir) + "/tensorboard")

    # init model and optimizer
    model = TrainModel(
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        model_name=args.model,
        optimizer_name=args.optimizer,
        n_classes=train_loader.dataset.n_classes,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        class_weights=class_weights,
        use_rgb=not args.no_rgb
    )
    logger.info('Parameters ...')
    logger.info(args)

    best_iou = 0
    for epoch in range(model.start_epoch, args.epoch):
        logger.info('**** Epoch %d (%d/%s) ****' % (epoch + 1, epoch + 1, args.epoch))

        # adjust learning rate and bn momentum TODO we do not know if we need this, so it is disabled for now
        # model.adjust_lr(epoch)
        # model.adjust_bn_momentum(epoch)

        loss, accuracy = model.train(train_loader)
        log_training(logger, loss, accuracy)

        # evaluate
        logger.info('---- Epoch %03d Evaluation ----' % (epoch + 1))
        (
            eval_loss, mIoU, accuracy, class_acc,
            total_correct_class, total_iou_deno_class, class_distribution
        ) = model.eval(val_loader)

        log_eval(logger, eval_loss, mIoU, accuracy, class_acc, total_correct_class,
                 total_iou_deno_class, class_distribution, val_loader.dataset.classes)

        ## write to tensorboard

        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('eval_Loss', eval_loss, epoch)
        writer.add_scalar('eval_mIoU', mIoU, epoch)

        # save as best model if mIoU is better
        if mIoU >= best_iou:
            best_iou = mIoU
            model.save_model(epoch, mIoU, "best_model")

        # save model every 5 epochs
        if epoch % save_epoch == 0:
            model.save_model(epoch, mIoU, str(epoch))

        logger.info('Best mIoU: %f' % best_iou)
        epoch += 1

    # load best model
    model.load_model("best_model")

    logger.info("start loading test data ...")
    # test loader has to use batch size of 1 to allow for varying point clouds
    test_loader = get_data_loader(
        batch_size=1, points_per_sample=args.eval_points_per_sample, data_path=args.data_path, split="test",
        use_rgb=not args.no_rgb, training=False, n_data_worker=args.n_data_worker,
        global_z=train_loader.dataset.get_global_z()
    )

    # evaluate on test data only once (everything else is cheating ;)
    logger.info('---- Test Evaluation ----')
    (
        eval_loss, mIoU, accuracy, class_acc,
        total_correct_class, total_iou_deno_class, class_distribution
    ) = model.eval(test_loader)

    log_eval(logger, eval_loss, mIoU, accuracy, class_acc,
             total_correct_class, total_iou_deno_class, class_distribution, test_loader.dataset.classes)
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
