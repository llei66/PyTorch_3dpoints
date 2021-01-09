import os
import numpy as np
from torch.utils.data import Dataset
import pdb
import random

class DublinData_1(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096,  block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        print(split)
        rooms = sorted(os.listdir(data_root))
        # rooms = [room for room in rooms if 'Area_' in room]
        # if split == 'train':
        #     rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        # else:
        #     rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        # pdb.set_trace()
        rooms_split = rooms

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        # labelweights = np.zeros(13)
        # labelweights = np.zeros(6)
        ## add water
        labelweights = np.zeros(3)

        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            # pdb.set_trace()

            # room_data = np.load(room_path)  # xyzrgbl, N*7
            print(room_path)
            room_data = np.loadtxt(room_path)  # xyzrgbl, N*7

            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            # tmp, _ = np.histogram(labels, range(14))
            tmp, _ = np.histogram(labels, range(4))

            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        # pdb.set_trace()

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)

        # self.labelweights = np.ones(3).astype(np.float32)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            # pdb.set_trace()

            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]
        count = 0
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]

            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]

            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        # pdb.set_trace()
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        # print("xxxx- get 2 ")
        # count += 1
        # print(idx)
        # print(count)
        # sample_name = "/home/SENSETIME/lilei/DEEPCROP/samples/sample_" +str(count) + ".txt"
        # current_labels_sample = current_labels[:, np.newaxis]
        # sample_np = np.concatenate((current_points,current_labels_sample), axis =1)
        # np.savetxt(sample_name,sample_np)


        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

if __name__ == '__main__':
    # data_root = '/data/REASEARCH/DEEPCROP/code/PYcode/point++_test/Pointnet_Pointnet2_pytorch/data/test_stanford_indoor3d/'
    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/building/clear_1'
    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/building/clear_1_diff_size'
    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/building/100m_test'

    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/test_whole_class_1km'
    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_test_whole_class_200m/test_one_area'
    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/trin_test_whole_class_200m/test_view'
    data_root = '/home/SENSETIME/lilei/DEEPCROP/DublinCity/sample/T_315500_234500_NW.bin/whole_label_copy'
    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_test_whole_class_200m/sub_areas'
    # num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01
    # num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01
    # num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01
    # num_point,  block_size, sample_rate = 4096, 5, 1.0, 0.01
    # num_point,  block_size, sample_rate = 2048, 5, 1.0, 0.01
    # num_point,  block_size, sample_rate = 2048, 5.0, 0.01
    # num_point,  block_size, sample_rate = 4096, 5.0, 1
    # num_point,  block_size, sample_rate = 4096, 5.0, 1
    num_point,  block_size, sample_rate = 4096, 2.0, 1
    # num_point,  block_size, sample_rate = 8192, 1.0, 1

    print(num_point,  block_size, sample_rate)
    # print("point_idxs.size > 1024")
    print("if point_idxs.size > 512:")


    point_data = DublinData_1(split='train', data_root=data_root, num_point=num_point, block_size=block_size, sample_rate=sample_rate, transform=None)


    print('point data size:', point_data.__len__())
    # pdb.set_trace()
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    for i in range( point_data.__len__()):
        i = np.random.randint(0,point_data.__len__())
        sample_name = "/home/SENSETIME/lilei/DEEPCROP/samples/sample_" + str(i) + ".txt"
        current_points = point_data.__getitem__(i)[0]
        current_labels_sample = point_data.__getitem__(i)[1][:, np.newaxis]
        sample_np = np.concatenate((current_points[:, 0:6], current_labels_sample), axis=1)
        np.savetxt(sample_name, sample_np)

    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()
