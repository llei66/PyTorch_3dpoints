import argparse
import os

# from models.pointnet_reg import
from torch.utils.tensorboard import SummaryWriter

import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import pdb

from sklearn.metrics import r2_score

from data_utils.biomass_dataset import BiomassDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))



def parse_args():

    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_reg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=300, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--data_dir', type=str, default="./data", help='data path [default: ./data]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,  default=80, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')

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

    # root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/20_july_2020/train'
    # test_root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/20_july_2020/test'

    root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test/train'
    test_root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/train_test_split_v2/test_set_txt'
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    # Block_size = 1.0
    Block_size = 2

    # pdb.set_trace()
    # NUM_POINT

    print("start loading training data ...")
    train_dataset = BiomassDataset(args.data_dir, "train_split.csv")
    print("start loading test data ...")
    val_dataset = BiomassDataset(args.data_dir, "val_split.csv")
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # weights = torch.Tensor([1,1,1,1]).cude()

    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(val_dataset))

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    # classifier = MODEL.get_model(NUM_CLASSES).cuda()
    Block_num = 1
    classifier = model.get_model(Block_num).cuda()

    criterion = model.get_loss().cuda()
    # criterion = r2_loss().cuda()



    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/50model.pth')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')

    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    r2_score_sum = 0
    target_all = list()
    reg_all = list()
    with torch.no_grad():
        num_batches = len(val_data_loader)

        # num_batches = len(trainDataLoader)
        # total_correct = 0
        # total_seen = 0
        loss_sum = 0
        print("this is samples num:", num_batches)

        # for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

        for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = torch.Tensor(points)
            # target to tensor
            target = target.data.numpy()
            target = torch.Tensor(target)
            points, target = points.float().cuda(), target.float().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            seg_pred, trans_feat = classifier(points)
            # pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1)
            batch_label = target.cpu().data.numpy()
            # target = target.view(-1, 1)[:, 0]

            target = target[:, 0]

            loss = criterion(seg_pred, target.float(), trans_feat)
            loss_sum += loss

            ## computer r2 scores
            # pdb.set_trace()
            # seg_pred = 2000. - seg_pred* 2000.
            # r2_score_each = r2_score(target.cpu(), seg_pred.cpu())
            target_all.append(target.cpu())
            reg_all.append(seg_pred.cpu())
            # r2_score_each = r2_score(seg_pred.cpu(), target.cpu())

            # r2_score_sum += r2_score_each
        print(target_all)
        print(reg_all)
        r2_score_all = r2_score(target_all, reg_all)

        # avl_r2_score = r2_score_sum/num_batches
        avl_loss = loss_sum/num_batches
        log_string('===================  Test mean L2 loss: %f  ==============' % (avl_loss))
        log_string('===================  Test mean r2 score: %f  ==============' % (r2_score_all))
        log_string('===================  Test mean L1 loss: %f   ==============' % (avl_loss))





if __name__ == '__main__':
    args = parse_args()
    main(args)
