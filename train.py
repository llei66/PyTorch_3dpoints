# TODO add more comments on what is happening

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.util import weights_init, bn_momentum_adjust
from utils.point_dataset import PointDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ground', 'vegetation', 'building', 'water']

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
    parser.add_argument('--learning-rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log-dir', type=str, default="./log", help='Log path [default: "./log"]')
    parser.add_argument('--decay-rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--points-per-sample', type=int, default=4096, help='points per sample [default: 4096]')
    parser.add_argument('--eval-points-per-sample', type=int, default=10000,
                        help='points per sample during eval [default: 10000]')
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
    parser.add_argument('--no-rgb', action='store_true', default=False, help="ignores RBG if used")
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='internal of saving model [default: every 10 epochs]')

    return parser.parse_args()


class Trainer:
    def __init__(self, model_name, optimizer_name, n_classes, learning_rate, decay_rate,
                 step_size, class_weights, logger, checkpoint_dir, use_rgb, save_epoch):
        # set hyperparameter
        # set visible devices (how many GPUs are used for training)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_weights = class_weights
        if self.class_weights is not None:  # is a tensor
            self.class_weights = self.class_weights.to(self.device)
        self.n_classes = n_classes

        self.logger, self.checkpoints_dir = logger, checkpoint_dir

        # init model
        model, criterion = get_model(model_name, self.n_classes, use_rgb)

        # push to correct device
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        # TODO this should be optional through parameters (and will never happen since experiment folders are created with timestamp
        try:
            checkpoint = torch.load(str(self.checkpoints_dir) + '/best_model.pth')
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info('Use pretrain model')
        except:
            self.logger.info('No existing model, starting training from scratch...')
            self.start_epoch = 0
            self.model = model.apply(weights_init)

        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=decay_rate
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError(f"Only supports Adam and SGD for now, you selected: {self.optimizer}.")

        # TODO this should be parameters not constants
        self.learning_rate_clip = 1e-5
        self.momentum_original = 0.1
        self.momentum_decay = 0.5
        self.momentum_decay_step = step_size

    def adjust_lr(self, epoch):
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), self.learning_rate_clip)
        self.logger.info('Learning rate:%f' % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_bn_momentum(self, epoch):
        momentum = self.momentum_original * (self.momentum_decay ** (epoch // self.momentum_decay_step))
        if momentum < 0.01:
            momentum = 0.01
        self.logger.info('BN momentum updated to: %f' % momentum)
        self.model = self.model.apply(lambda x: bn_momentum_adjust(x, momentum))

    def save_model(self, epoch, mIoU, name="model"):
        savepath = f"{str(self.checkpoints_dir)}/{name}.pth"
        self.logger.info('Saving model to %s' % savepath)
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'miou': mIoU,
        }
        torch.save(state, savepath)

    def load_model(self, name):
        savepath = f"{str(self.checkpoints_dir)}/{name}.pth"
        self.logger.info('Load model from %s' % savepath)
        state = torch.load(savepath, map_location="cpu")
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def train(self, loader):
        '''Train on chopped scenes'''

        num_batches = len(loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        self.model.train()
        for i, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
            points, target = points.float().to(self.device), target.long().to(self.device)
            points = points.transpose(2, 1)
            bs = points.shape[0]
            self.optimizer.zero_grad()

            # apply model and make gradient step
            seg_pred, trans_feat = self.model(points)
            seg_pred = seg_pred.contiguous().view(-1, self.n_classes)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = self.criterion(seg_pred, target, trans_feat, self.class_weights)
            loss.backward()
            self.optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (bs * points.shape[-1])
            loss_sum += loss

        return loss_sum / num_batches, total_correct / float(total_seen)

    def eval(self, loader):
        '''Evaluate on all scenes'''
        num_batches = len(loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        class_distribution = np.zeros(self.n_classes)
        total_seen_class = np.zeros(self.n_classes)
        total_correct_class = np.zeros(self.n_classes)
        total_iou_deno_class = np.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():  # no gradient required
            for i, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
                # skip empty slices (no points)
                if points.shape[1] == 0:
                    num_batches -= 1  # since we skip a batch
                    continue
                points, target = points.float().to(self.device), target.long().to(self.device)
                points = points.transpose(2, 1)
                bs = points.shape[0]  # should be 1
                n_points = points.shape[-1]

                # apply model
                seg_pred, trans_feat = self.model(points)

                # calculate eval scores
                pred_val = seg_pred.cpu().data.numpy()
                seg_pred = seg_pred.view(-1, self.n_classes)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                # # no need to use weighting in evaluation (set to None)
                loss = self.criterion(seg_pred, target, trans_feat, None)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (bs * n_points)
                counts, _ = np.histogram(batch_label, range(self.n_classes + 1))
                class_distribution += counts
                for j in range(self.n_classes):
                    total_seen_class[j] += np.sum((batch_label == j))
                    total_correct_class[j] += np.sum((pred_val == j) & (batch_label == j))
                    total_iou_deno_class[j] += np.sum(((pred_val == j) | (batch_label == j)))

        eval_loss = loss_sum / float(num_batches)
        class_distribution = class_distribution.astype(np.float32) / np.sum(class_distribution.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        acc = total_correct / float(total_seen)
        class_acc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))
        return eval_loss, mIoU, acc, class_acc, total_correct_class, total_iou_deno_class, class_distribution


def get_model(model_name: str, n_classes: int, use_rgb: bool):
    model_name = model_name.lower().replace(" ", "").replace("_", "")
    info_channel = 0
    if use_rgb:
        info_channel = 3
    if model_name == "pointnet":
        from models.pointnet.model import PointNet
        from models.pointnet.loss import Loss
        model = PointNet(n_classes, info_channel)
        criterion = Loss(mat_diff_loss_scale=0.001)  # TODO replace magic constant
    elif model_name in ["pointnet++ssg", "pointnet2ssg"]:
        from models.pointnet2.model import PointNet2SSG
        from models.pointnet2.loss import Loss
        model = PointNet2SSG(n_classes, info_channel)
        criterion = Loss()
    elif model_name in ["pointnet++msg", "pointnet2msg"]:
        from models.pointnet2.model import PointNet2MSG
        from models.pointnet2.loss import Loss
        model = PointNet2MSG(n_classes, info_channel)
        criterion = Loss()
    elif model_name in ["pointnet#", "pointnetsharp"]:
        from models.pointnetsharp.model import PointNet
        from models.pointnetsharp.loss import Loss
        model = PointNet(n_classes, info_channel)
        criterion = Loss()
    else:
        raise NotImplementedError(
            f"Chosen model ({model_name} is not available, must be in ('pointnet', 'pointnet++ssg', 'pointnet++msg')"
        )

    return model, criterion


def get_data_loader(batch_size, blocks_per_epoch, points_per_sample,
                    block_size, data_path, split, use_rgb, training, global_z=None):
    dataset = PointDataset(
        split=split, data_root=data_path, blocks_per_epoch=blocks_per_epoch, use_rgb=use_rgb, global_z=global_z,
        points_per_sample=points_per_sample, block_size=block_size, transform=None, training=training
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=training, num_workers=args.n_data_worker,
        pin_memory=torch.cuda.is_available() and training, drop_last=training
    )
    return loader


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
             total_correct_class, total_iou_deno_class, class_distribution):
    logger.info('eval mean loss: %f' % eval_loss)
    logger.info('eval point avg class IoU: %f' % mIoU)
    logger.info('eval point accuracy: %f' % accuracy)
    logger.info('eval point avg class acc: %f' % class_acc)
    iou_per_class_str = '------- IoU --------\n'
    for j in range(len(class_distribution)):
        iou_per_class_str += 'class %s ratio: %f, IoU: %f \n' % (
            seg_label_to_cat[j] + ' ' * (14 - len(seg_label_to_cat[j])), class_distribution[j],
            total_correct_class[j] / float(total_iou_deno_class[j]))

    logger.info(iou_per_class_str)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = parse_args()

    # init logger
    logger, checkpoint_dir = init_logging(args.log_dir, args.model)
    save_epoch = args.save_epoch
    # init data loader
    # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)
    block_size = (args.block_size_x, args.block_size_y)
    logger.info("start loading train data ...")
    # TODO add back augmentations
    train_loader = get_data_loader(
        args.batch_size, args.blocks_per_epoch, args.points_per_sample, block_size, args.data_path, "train",
        use_rgb=not args.no_rgb, training=True
    )
    logger.info("start loading test data ...")
    # test loader has to use batch size of 1 to allow for varying point clouds
    test_loader = get_data_loader(
        1, args.blocks_per_epoch, args.eval_points_per_sample, block_size, args.data_path, "test",
        global_z=train_loader.dataset.get_global_z(), use_rgb=not args.no_rgb, training=False,
    )
    validate_loader = get_data_loader(
        1, args.blocks_per_epoch, args.eval_points_per_sample, block_size, args.data_path, "validate",
        global_z=train_loader.dataset.get_global_z(), use_rgb=not args.no_rgb, training=False,
    )

    # determine weighting method for loss function
    class_weights = None
    if args.weighting == "class":
        # load already inverted weights TODO might be cleaner to invert them here
        class_weights = torch.Tensor(train_loader.dataset.class_weights)

    ## import tensorboard to view

    writer = SummaryWriter(str(checkpoint_dir) + "/tensorboard")

    # init model and optimizer
    trainer = Trainer(
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        model_name=args.model,
        optimizer_name=args.optimizer,
        n_classes=train_loader.dataset.n_classes,
        learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        step_size=args.step_size,
        class_weights=class_weights,
        use_rgb=not args.no_rgb,
        save_epoch=args.save_epoch
    )
    logger.info('Parameters ...')
    logger.info(args)

    global_epoch = 0
    best_iou = 0
    for epoch in range(trainer.start_epoch, args.epoch):
        logger.info('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))

        # adjust learning rate and bn momentum
        trainer.adjust_lr(epoch)
        trainer.adjust_bn_momentum(epoch)

        loss, accuracy = trainer.train(train_loader)
        log_training(logger, loss, accuracy)

        # evaluate
        logger.info('---- Epoch %03d Evaluation ----' % (global_epoch + 1))
        (
            eval_loss, mIoU, accuracy, class_acc,
            total_correct_class, total_iou_deno_class, class_distribution
        ) = trainer.eval(validate_loader)

        log_eval(logger, eval_loss, mIoU, accuracy, class_acc,
                 total_correct_class, total_iou_deno_class, class_distribution)

        ## write to tensorboard

        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('eval_Loss', eval_loss, epoch)
        writer.add_scalar('mIoU', mIoU, epoch)

        # save as best model if mIoU is better
        if mIoU >= best_iou:
            best_iou = mIoU
            trainer.save_model(epoch, mIoU, "best_model")

        # save model every 5 epochs
        if epoch % save_epoch == 0:
            trainer.save_model(epoch, mIoU, str(epoch))

        logger.info('Best mIoU: %f' % best_iou)
        global_epoch += 1

    # load best model
    trainer.load_model("best_model")

    # evaluate on test data only once (everything else is cheating ;)
    logger.info('---- Epoch Test Evaluation ----')
    (
        eval_loss, mIoU, accuracy, class_acc,
        total_correct_class, total_iou_deno_class, class_distribution
    ) = trainer.eval(test_loader)

    log_eval(logger, eval_loss, mIoU, accuracy, class_acc,
             total_correct_class, total_iou_deno_class, class_distribution)
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
