# TODO add more comments on what is happening
# TODO make logging look nicer or move it to tensorboard

import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import provider
from data_utils.point_dataset import PointDataset

# from data_utils.S3DISDataLoader import S3DISDataset
# from data_utils.DeDataLoader_600 import S3DISDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# classes = ['ground','low_vegetatation','medium_vegetation','high_vegetataion','building','water']
# classes = ['unclassified','ground','low_vegetatation','medium_vegetation','high_vegetataion','building','noise']

classes = ['ground', 'vegetatation', 'building', 'water']

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--data-path', type=str, default='./data/train_test_whole_class_1km',
                        help='path to data [default: "./data/train_test_whole_class_1km"]')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=128, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log-dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay-rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--points-per-sample', type=int, default=4096 * 2, help='points per sample [default: 4096 * 2]')
    parser.add_argument('--blocks-per-epoch', type=int, default=4096, help='blocks per epoch [default: 4096]')
    parser.add_argument('--block-size-x', type=float, default=0.05,
                        help='normalized block size for x coordinate [default: 0.05]')
    parser.add_argument('--block-size-y', type=float, default=0.05,
                        help='normalized block size for x coordinate [default: 0.05]')
    parser.add_argument('--step-size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr-decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test-area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--weighting', type=str, default="equal",
                        help='defines the weighting method, either "equal" or "class" [default: "equal"]')
    parser.add_argument('--n-data-worker', type=int, default=4,
                        help='data preprocessing threads [default: 4]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    # define log dir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
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

    batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # seen blocks per epoch (iterations)
    blocks_per_epoch = args.blocks_per_epoch
    # points per sample TODO think of variable way to do this
    points_per_sample = args.points_per_sample
    # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)
    block_size = (args.block_size_x, args.block_size_y)

    print("start loading training data ...")
    # get samples before training process
    # block size: your sample in a [block_x, block_y], ingore the Z axis, measuring with meter
    # Go to data_utils/GeoData_crop_1.py for details
    train_dataset = PointDataset(
        split='train', data_root=args.data_path, blocks_per_epoch=blocks_per_epoch,
        points_per_sample=points_per_sample, block_size=block_size, transform=None, training=True
    )
    print("start loading test data ...")
    test_dataset = PointDataset(
        split='test', data_root=args.data_path, blocks_per_epoch=blocks_per_epoch,
        points_per_sample=points_per_sample, block_size=block_size, transform=None, training=False
    )
    # test loader has to use batch size of 1 to allow for varying point clouds
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=args.n_data_worker,
        pin_memory=torch.cuda.is_available(), drop_last=True
    )
    # TODO this is not nice since there is still non-determistic sampling hapenning inside
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.n_data_worker,
        pin_memory=torch.cuda.is_available(), drop_last=False  # do not drop the last batches in test mode
    )
    # load already inverted weights TODO might be clearner to invert them here
    if args.weighting == "class":
        weights = torch.Tensor(train_dataset.label_weights).to(device)
    else:
        weights = None

    n_classes = train_dataset.n_classes

    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(test_dataset))

    '''MODEL LOADING'''
    # TODO this is not nice, why not just use import?
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    model = MODEL.get_model(n_classes).to(device)
    criterion = MODEL.get_loss().to(device)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # TODO this should be optional through parameters
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

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    # TODOS this should be parameters not constanstants
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    # TODO make this a function (eval and train)
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            optimizer.zero_grad()

            # apply model and make gradient step
            model = model.train()
            seg_pred, trans_feat = model(points)
            seg_pred = seg_pred.contiguous().view(-1, n_classes)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * points_per_sample)
            loss_sum += loss

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        # TODO disclaimer: not yet chopped and should be validation loader at this point
        with torch.no_grad():
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            class_distribution = np.zeros(n_classes)  # TODO later this should be constant
            total_seen_class = np.zeros(n_classes)
            total_correct_class = np.zeros(n_classes)
            total_iou_deno_class = np.zeros(n_classes)

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                points, target = data
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                # apply model
                model = model.eval()
                seg_pred, trans_feat = model(points)

                # calculate eval scores
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, n_classes)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                # # no need to use weighting in evaluation
                loss = criterion(seg_pred, target, trans_feat, None)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (batch_size * points_per_sample)
                counts, _ = np.histogram(batch_label, range(n_classes + 1))
                class_distribution += counts
                for l in range(n_classes):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            class_distribution = class_distribution.astype(np.float32) / np.sum(class_distribution.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(n_classes):
                iou_per_class_str += 'class %s ratio: %f, IoU: %f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), class_distribution[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen
                                                                    )))
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
