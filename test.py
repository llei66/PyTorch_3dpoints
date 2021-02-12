# TODO add more comments on what is happening

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from utils.denmark_dataset import get_data_loader
from utils.logger import init_logging, log_eval
from utils.model_helper import TestModel


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--data-path', type=str, default='./data/train_test_whole_class_1km',
                        help='path to data [default: "./data/train_test_whole_class_1km"]')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch Size during testing (only 1 supported atm) [default: 1]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log-dir', type=str, default="./log", help='Log path [default: "./log"]')
    parser.add_argument('--pred-dir', type=str, default="./pred", help='where to save predictions [default: "./pred"]')
    parser.add_argument('--eval-points-per-sample', type=int, default=10000,
                        help='points per sample during eval [default: 10000]')
    parser.add_argument('--n-data-worker', type=int, default=4,
                        help='data preprocessing threads [default: 4]')
    parser.add_argument('--no-rgb', action='store_true', default=False, help="ignores RBG if used")
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
        args.log_dir, args.model, not args.no_console_logging, not args.no_file_logging, "test"
    )
    # init data loader
    # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)
    block_size = (1, 1)
    logger.info("start loading train data stats to acquire global_z scaling and n_classes...")
    try:
        stats = torch.load(f"{str(checkpoint_dir)}/data_stats.pth")
        global_z = stats["global_z"]
        n_classes = stats["n_classes"]
    except FileNotFoundError:
        logger.info("stats not found: loading training data instead...")
        train_loader = get_data_loader(
            batch_size=1, points_per_sample=1, data_path=args.data_path,
            split="train", use_rgb=not args.no_rgb, training=True, n_data_worker=1
        )
        global_z = train_loader.dataset.get_global_z()
        n_classes = train_loader.dataset.n_classes

        # save stats for later test runs
        torch.save({
            "global_z": train_loader.dataset.global_z,
            "n_classes": train_loader.dataset.n_classes
        }, f"{str(checkpoint_dir)}/data_stats.pth")
        del train_loader

    # init model and optimizer
    model = TestModel(
        logger=logger,
        model_name=args.model,
        n_classes=n_classes,
        use_rgb=not args.no_rgb,
        checkpoint_dir=checkpoint_dir,
    )

    # load best model
    model.load_model("best_model")

    logger.info("start loading test data ...")
    # test loader has to use batch size of 1 to allow for varying point clouds
    test_loader = get_data_loader(
        batch_size=1, points_per_sample=args.eval_points_per_sample, data_path=args.data_path, split="test",
        use_rgb=not args.no_rgb, n_data_worker=args.n_data_worker, global_z=global_z, training=False,
        return_idx=True
    )

    # evaluate on test data only once (everything else is cheating ;)
    logger.info('---- Test Evaluation ----')
    (
        eval_loss, mIoU, accuracy, class_acc,
        total_correct_class, total_iou_deno_class, class_distribution,
        predictions
    ) = model.eval(test_loader)

    # save predictions for each room
    logger.info('saving the predictions now ...')
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(exist_ok=True)
    pred_dir = pred_dir.joinpath(args.model)
    pred_dir.mkdir(exist_ok=True)
    for room_pred, room_path in zip(predictions, test_loader.dataset.room_names):
        room_pred_path = pred_dir.joinpath(room_path)
        np.savetxt(room_pred_path, room_pred.astype(int))

    log_eval(logger, eval_loss, mIoU, accuracy, class_acc,
             total_correct_class, total_iou_deno_class, class_distribution, test_loader.dataset.classes)


if __name__ == '__main__':
    args = parse_args()
    main(args)
