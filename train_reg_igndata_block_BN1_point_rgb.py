import argparse
import os
# from data_utils.S3DISDataLoader import S3DISDataset
# from data_utils.DeDataLoader_600 import S3DISDataset

# from data_utils.ign_dataset_BN1 import GeoData_crop_1
#from data_utils.ign_dataset_BN1_point import GeoData_crop_1
from data_utils.ign_dataset_BN1_point_rgb import GeoData_crop_1

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
import provider
import numpy as np
import time
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


# classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']

# classes = ['ground','low_vegetatation','medium_vegetation','high_vegetataion','building','water']
# classes = ['unclassified','ground','low_vegetatation','medium_vegetation','high_vegetataion','building','noise']

classes = ['ground', 'vegetatation','building']


class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():

    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_reg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=300, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
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

    # root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test/train'
    # test_root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test/test'
    # root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test_validate/train'
    # test_root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test_validate/validate'

    root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/train_test_split_v2/train_set_txt'
    test_root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/train_test_split_v2/val_set_txt'
#    root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test_merge/train'
#    test_root = '/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test_merge/test'
    NUM_POINT = args.npoint
    # BATCH_SIZE = args.batch_size
    # Block_size = 1.0
    Block_size = 2
    BATCH_SIZE = 1

    # pdb.set_trace()
    # NUM_POINT

    print("start loading training data ...")
    TRAIN_DATASET = GeoData_crop_1(split='train', data_root=root, num_point=NUM_POINT,  block_size=Block_size, transform=None)
    print("start loading test data ...")
    TEST_DATASET = GeoData_crop_1(split='test', data_root=test_root, num_point=NUM_POINT,  block_size=Block_size, transform=None)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # weights = torch.Tensor([1,1,1,1]).cude()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    # classifier = MODEL.get_model(NUM_CLASSES).cuda()
    Block_num = 1
    classifier = MODEL.get_model(Block_num).cuda()

    criterion = MODEL.get_loss().cuda()
    # criterion = r2_loss().cuda()
#     checkpoint = torch.load('log/reg/rgn_point_rgb_old/checkpoints/best_model.pth')
#     # start_epoch = checkpoint['epoch']
#     classifier.load_state_dict(checkpoint['model_state_dict'])
#     log_string('Use pretrain model')
#     start_epoch = 0

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    classifier = classifier.apply(weights_init)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_loss = 9999999

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())

    writer = SummaryWriter(str(experiment_dir) + "/tensorboard/" + TIMESTAMP)


    for epoch in range(start_epoch,args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        # loss_func = torch.nn.MSELoss()

        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            # points[:,:, :3] = provider.rotate_point_cloud_z(points[:,:, :3])

            target = target.data.numpy()
            target = torch.Tensor(target)

            points = torch.Tensor(points)
            points, target = points.float().cuda(),target.float().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            reg_pred, trans_feat = classifier(points)
            reg_pred = reg_pred.contiguous().view(-1)

            # batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            # target = target.view(-1, 1)[:, 0]
            target = target[:, 0]
            # pdb.set_trace()

            loss = criterion(reg_pred, target.float(), trans_feat)
            loss.backward()
            optimizer.step()
            loss_sum += loss
        avl_loss = loss_sum/num_batches

        writer.add_scalar('Train_Loss', avl_loss, epoch)

        log_string('==========Training mean loss: %f ================' % (avl_loss))
        # log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 50 == 0:
            logger.info('Save model...')
            # pdb.set_trace()
            savepath = str(checkpoints_dir) + '/' + str(epoch) + 'model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            # total_correct = 0
            # total_seen = 0
            loss_sum = 0
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = data
                points = points.data.numpy()
                points = torch.Tensor(points)
                # target to tensor
                target = target.data.numpy()
                target = torch.Tensor(target)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                classifier = classifier.eval()
                reg_pred, trans_feat = classifier(points)
                # pred_val = reg_pred.contiguous().cpu().data.numpy()
                reg_pred = reg_pred.contiguous().view(-1)
                batch_label = target.cpu().data.numpy()
                # target = target.view(-1, 1)[:, 0]

                target = target[:, 0]

                loss = criterion(reg_pred, target.float(), trans_feat)
                loss_sum += loss
            avl_loss = loss_sum/num_batches
            log_string('===================Test mean loss: %f==============' % (avl_loss))

            writer.add_scalar('Test_Loss', avl_loss, epoch)

            # log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            if avl_loss <= best_loss:
                best_loss = avl_loss
                best_epoch = epoch

                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': loss_sum,
                    'model_state_dict': classifier.state_dict(),
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
