# TODO add more comments on what is happening
# TODO make logging look nicer or move it to tensorboard

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from
from utils import augmentations
from utils.point_dataset import PointDataset

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
    parser.add_argument('--log-dir', type=str, default="./log", help='Log path [default: "./log"]')
    parser.add_argument('--decay-rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--points-per-sample', type=int, default=4096 * 2, help='points per sample [default: 4096 * 2]')
    parser.add_argument('--blocks-per-epoch', type=int, default=4096, help='blocks per epoch [default: 4096]')
    parser.add_argument('--block-size-x', type=float, default=0.05,
                        help='normalized block size for x coordinate [default: 0.05]')
    parser.add_argument('--block-size-y', type=float, default=0.05,
                        help='normalized block size for x coordinate [default: 0.05]')
    parser.add_argument('--step-size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr-decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--weighting', type=str, default="equal",
                        help='defines the weighting method, either "equal" or "class" [default: "equal"]')
    parser.add_argument('--n-data-worker', type=int, default=4,
                        help='data preprocessing threads [default: 4]')

    return parser.parse_args()


def init_logging(log_dir):
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

    # init logger
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log(logger, 'Parameters ...')
    log(logger, args)
    return logger, experiment_dir


def log(logger, msg):
    logger.info(msg)
    print(msg)


def get_model(model_name: str, n_classes: int):
    model_name = model_name.lower().replace(" ", "").replace("_", "")
    if model_name == "pointnet":
        from models.pointnet import PointNet, Loss
        model = PointNet(n_classes)
        criterion = Loss(mat_diff_loss_scale=0.001)  # TODO replace magic constant
    elif model_name in ["pointnet++ssg", "pointnet2ssg"]:
        from models.pointnet2_msg import PointNet2MSG, Loss
        model = PointNet2MSG(n_classes)
        criterion = Loss()
    elif model_name in ["pointnet++msg", "pointnet2msg"]:
        from models.pointnet2_ssg import PointNet2SSG, Loss
        model = PointNet2SSG(n_classes)
        criterion = Loss()
    else:
        raise NotImplementedError(
            f"Chosen model ({model_name} is not available, must be in ('pointnet', 'pointnet++ssg', 'pointnet++msg')"
        )

    return model, criterion


def main(args):
    # set hyperparameter
    # set visible devices (how many GPUs are used for training)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    # seen blocks per epoch (iterations)
    blocks_per_epoch = args.blocks_per_epoch
    # points per sample TODO think of variable way to do this
    points_per_sample = args.points_per_sample
    # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)
    block_size = (args.block_size_x, args.block_size_y)

    logger, experiment_dir = init_logging(args.log_dir)

    # init data loader
    log(logger, "start loading training data ...")
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_data_worker,
        pin_memory=torch.cuda.is_available(), drop_last=True
    )
    # TODO this is not nice since there is still non-determistic sampling hapenning inside
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.n_data_worker,
        pin_memory=torch.cuda.is_available(), drop_last=False  # do not drop the last batches in test mode
    )
    # load already inverted weights TODO might be cleaner to invert them here
    if args.weighting == "class":
        weights = torch.Tensor(train_dataset.label_weights).to(device)
    else:
        weights = None

    n_classes = train_dataset.n_classes

    log(logger, "The number of training data is: %d" % len(train_dataset))
    log(logger, "The number of test data is: %d" % len(test_dataset))

    # init model
    model, criterion = get_model(args.model, n_classes)

    # push to correct device
    model = model.to(device)
    criterion = criterion.to(device)

    # TODO this should be optional through parameters
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log(logger, 'Use pretrain model')
    except:
        log(logger, 'No existing model, starting training from scratch...')
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
        log(logger, '**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log(logger, 'Learning rate:%f' % lr)
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
            points[:, :, :3] = augmentations.rotate_point_cloud_z(points[:, :, :3])
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

        log(logger, 'Training mean loss: %f' % (loss_sum / num_batches))
        log(logger, 'Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log(logger, 'Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log('Saving model....')

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

            log(logger, '---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
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
            log(logger, 'eval mean loss: %f' % (loss_sum / float(num_batches)))
            log(logger, 'eval point avg class IoU: %f' % (mIoU))
            log(logger, 'eval point accuracy: %f' % (total_correct / float(total_seen)))
            log(logger, 'eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(n_classes):
                iou_per_class_str += 'class %s ratio: %f, IoU: %f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), class_distribution[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log(logger, iou_per_class_str)
            log(logger, 'Eval mean loss: %f' % (loss_sum / num_batches))
            log(logger, 'Eval accuracy: %f' % (total_correct / float(total_seen
                                                                     )))
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log(logger, 'Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log(logger, 'Saving model....')
            log(logger, 'Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
