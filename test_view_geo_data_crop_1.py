
import argparse
import os
# from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
# from data_utils.S3DISDataLoader_test_vis import S3DISDataset
from data_utils.GeoData_crop_test_view_1 import GeoData_crop_1


# from data_utils.indoor3d_util import g_label2color
from data_utils.Geodata3d_utils import g_label2color

import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
classes = ['ground','vegetatation','building','water']

# classes = ['unclassified','ground','low_vegetatation','medium_vegetation','high_vegetataion','building','noise', 'water']

# classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in testing [default: 32]')
    parser.add_argument('--root', type=str, default='data', help='input test dir')

    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_sem_seg', help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=5, help='Aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b,n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 4 #13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/PUNKTSKY_621_51_TIF_UTM32-ETRS89_txt_test_clear'
    root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_test_whole_class_200m/test_one_area'
    root = args.root
    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)

    # TEST_DATASET_WHOLE_SCENE = S3DISDataset(root, split='train', test_area=args.test_area, block_points=NUM_POINT)
    TEST_DATASET_WHOLE_SCENE = GeoData_crop_1(split='test_view', data_root=root, num_point=NUM_POINT, block_size=4.0, sample_rate=1.0, transform=None)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET_WHOLE_SCENE, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))


    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET_WHOLE_SCENE, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TEST_DATASET_WHOLE_SCENE.labelweights).cuda()

    output_name = root.split("/")[-1]
    log_string("The number of test data is: %d" %  len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    criterion = MODEL.get_loss().cuda()
    if args.visual:
        fout = open(os.path.join(visual_dir, output_name + '_pred.obj'), 'w')
        fout_gt = open(os.path.join(visual_dir, output_name + '_gt.obj'), 'w')

    whole_scene_data = np.array(TEST_DATASET_WHOLE_SCENE.room_points)
    # whole_scene_data = whole_scene_data.reshape(whole_scene_data[1], whole_scene_data[2])

    whole_label_predict = np.array([])
    whole_label_gt = np.array([])
    whole_label_predict_list = []
    whole_label_gt_list = []
    whole_points =  []
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss_sum += loss
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp
            ##
            # pdb.set_trace()
            # whole_label_predict_list = list(whole_label_predict)
            batch_label_list = list(batch_label.reshape([-1]))
            whole_label_gt_list.extend(batch_label_list)

            pred_val_list = list(pred_val.reshape([-1]))

            whole_label_predict_list.extend(pred_val_list)

            ##
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval point avg class IoU: %f' % (mIoU))
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        log_string('eval point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                total_correct_class[l] / float(total_iou_deno_class[l]))

        log_string(iou_per_class_str)
        log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
        print('----------------------------')

        filename = os.path.join(visual_dir, output_name + '.txt')
        # pdb.set_trace()

        with open(filename, 'w') as pl_save:
            for i in range(len(whole_label_predict_list)):
                pl_save.write(str(whole_label_predict_list[i]) + '\n')
            pl_save.close()

        for i in range(len(whole_label_predict_list)):
            # pdb.set_trace()
            # print(i)
            color = g_label2color[whole_label_predict_list[i]]
            color_gt = g_label2color[whole_label_gt_list[i]]
            if args.visual:
                fout.write('v %f %f %f %d %d %d %d\n' % (
                whole_scene_data[0, i, 0], whole_scene_data[0, i, 1], whole_scene_data[0, i, 2], color[0], color[1],
                color[2], whole_label_predict_list[i]))

                fout_gt.write(
                    'v %f %f %f %d %d %d\n' % (
                    whole_scene_data[0, i, 0], whole_scene_data[0, i, 1], whole_scene_data[0, i, 2], color_gt[0],
                    color_gt[1], color_gt[2]))
        # pdb.set_trace()
        if args.visual:
            fout.close()
            fout_gt.close()

        # IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        # iou_per_class_str = '------- IoU --------\n'
        # for l in range(NUM_CLASSES):
        #     iou_per_class_str += 'class %s, IoU: %.3f \n' % (
        #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
        #         total_correct_class[l] / float(total_iou_deno_class[l]))
        # log_string(iou_per_class_str)
        # log_string('eval point avg class IoU: %f' % np.mean(IoU))
        # log_string('eval whole scene point avg class acc: %f' % (
        #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        # log_string('eval whole scene point accuracy: %f' % (
        #             np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)
