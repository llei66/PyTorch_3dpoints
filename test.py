import argparse
import os

from sklearn import metrics
from torch.utils.data import DataLoader

import torch
import datetime
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

from data_utils.biomass_dataset import BiomassDataset
from models.regression_model import get_model


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_reg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--data_dir', type=str, default="./data", help='data path [default: ./data]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('reg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = args.batch_size
    data_dir = Path(args.data_dir)

    print("start loading test data ...")
    test_dataset = BiomassDataset(data_dir, "val_split.csv")
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                  drop_last=False)

    log_string("The number of test data is: %d" % len(test_data_loader))

    '''MODEL LOADING'''
    model = get_model(n_targets=1, extra_channel=0)
    model.to(device)

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')

    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        num_batches = len(test_data_loader)

        print("this is samples num:", num_batches)

        # for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

        preds = []
        ys = []
        pbar = tqdm(test_data_loader, total=len(test_data_loader), smoothing=0.9)
        model = model.eval()
        for data in pbar:
            points, target, features, machines, year_diff, files = data
            points, target = points.to(device), target.float().to(device)
            pred, trans_feat = model(points)
            pred = pred.squeeze()

            ys.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, 0)
        ys = np.concatenate(ys, 0)

        test_r2_score = metrics.r2_score(preds, ys)
        test_rmse_score = metrics.mean_squared_error(preds, ys, squared=False)
        test_mae_score = metrics.mean_absolute_error(preds, ys)

        log_string('===================  Test mean RMSE: %f  ==============' % (test_rmse_score))
        log_string('===================  Test mean r2 score: %f  ==============' % (test_r2_score))
        log_string('===================  Test mean MAE: %f   ==============' % (test_mae_score))


if __name__ == '__main__':
    args = parse_args()
    main(args)
