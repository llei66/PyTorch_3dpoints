import argparse
import os

import numpy as np
from sklearn import metrics

from data_utils.biomass_dataset import BiomassDataset
from models.regression_model import get_model

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import datetime
import logging
from pathlib import Path
import sys
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']

# classes = ['ground','low_vegetatation','medium_vegetation','high_vegetataion','building','water']
# classes = ['unclassified','ground','low_vegetatation','medium_vegetation','high_vegetataion','building','noise']

classes = ['ground', 'vegetatation', 'building']

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_reg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=300, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--data_dir', type=str, default=None, help='data path [default: ./data]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

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

    print("start loading training data ...")
    train_dataset = BiomassDataset(data_dir, "train_split.csv")
    print("start loading test data ...")
    val_dataset = BiomassDataset(data_dir, "val_split.csv")

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True
    )

    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(val_dataset))

    '''MODEL LOADING'''
    model = get_model(n_targets=1, extra_channel=0)
    model.to(device)

    criterion = torch.nn.SmoothL1Loss()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    model = model.apply(weights_init)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        model = model.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    global_epoch = 0
    best_loss = np.inf

    timestamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())

    writer = SummaryWriter(str(experiment_dir) + "/tensorboard/" + timestamp)

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))

        num_batches = len(train_data_loader)
        loss_sum = 0

        model = model.train()
        pbar = tqdm(train_data_loader, total=len(train_data_loader), smoothing=0.9)
        for data in pbar:
            points, target, features, machines, year_diff, files = data

            points, target = points.to(device), target.float().to(device)
            optimizer.zero_grad()
            reg_pred, trans_feat = model(points)

            loss = criterion(reg_pred.squeeze(1), target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"loss: {loss.item():.3f}")
            loss_sum += loss
        avl_loss = loss_sum / num_batches

        writer.add_scalar('Train_Loss', avl_loss, epoch)

        log_string('==========Training mean loss: %f ================' % (avl_loss))

        if epoch % 50 == 0:
            logger.info('Save model...')
            # pdb.set_trace()
            savepath = str(checkpoints_dir) + '/' + str(epoch) + 'model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate'''
        with torch.no_grad():
            num_batches = len(val_data_loader)

            print("this is samples num:", num_batches)

            # for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            preds = []
            ys = []
            pbar = tqdm(val_data_loader, total=len(val_data_loader), smoothing=0.9)
            model = model.eval()
            for data in pbar:
                points, target, features, machines, year_diff, files = data
                points, target = points.to(device), target.float().to(device)
                pred, trans_feat = model(points)
                pred = pred.squeeze(1)
                ys.append(target.cpu().numpy())
                preds.append(pred.cpu().numpy())
            preds = np.concatenate(preds, 0)
            ys = np.concatenate(ys, 0)

            test_rmse_score = metrics.mean_squared_error(preds, ys, squared=False)
            log_string('===================Test RMSE: %f==============' % (test_rmse_score))

            writer.add_scalar('Test_Loss', test_rmse_score, epoch)

            if test_rmse_score <= best_loss:
                best_loss = test_rmse_score
                best_epoch = epoch

                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': loss_sum,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best loss: %f' % best_loss)
            log_string('Best loss epoch: %f' % best_epoch)

        global_epoch += 1
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
