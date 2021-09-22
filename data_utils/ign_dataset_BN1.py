import os
import numpy as np
from torch.utils.data import Dataset
import pdb
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import warnings


class GeoData_crop_1(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096,  block_size=1.0,  transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms_split = sorted(os.listdir(data_root))

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        self.room_points_1, self.room_labels_1 = [], []
        self.intensity_all = []

        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            # room_data = np.load(room_path)  # xyzrgbl, N*7
            room_data = pd.read_csv(room_path, sep="\s+").values  # xyzrgbl, N*7
            # print(room_path)

            points, labels = room_data[:, 0:3], room_data[:, -1] 
            rgb = room_data[:,10:13]
            intensity = room_data[:,3].reshape(points.shape[0], 1)
            # self.intensity_all.append(intensity.tolist())
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

            # normalize
            # room_scale = (coord_max - coord_min)
            # points = (points - coord_min) / room_scale
            rgb = rgb / (256 * 255)

            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            # num_point_all.append(labels.size)
            points = np.concatenate((points, rgb, intensity), axis=1)
            # pdb.set_trace()
            self.room_points_1.append(points)
            for i in labels.tolist():
                self.room_labels_1.append(i)
            for j in intensity.tolist():
                self.intensity_all.append(j)


        # normalize points
        # pdb.set_trace()

        whole_coord_min, whole_coord_max = np.amin(self.room_coord_min, axis=0)[:3], np.amax(self.room_coord_max, axis=0)[:3]
        room_scale = (whole_coord_max - whole_coord_min)
        # normalize intensity
        intensity_mean = np.array(self.intensity_all).mean()
        intensity_std = np.array(self.intensity_all).std()

        # normalize label
        room_labels_mean = np.array(self.room_labels_1).mean()
        room_labels_std = np.array(self.room_labels_1).std()
        ### whole data 
        intensity_mean = 30.015351566593313
        intensity_std = 2.2168713538563276

        # def normal_dist(x, mean, sd):
        #     prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
        #     return prob_density
        #
        # pdb.set_trace()

        # pdf = normal_dist(np.array(self.intensity_all), intensity_mean, intensity_std)
        # plt.plot(np.array(self.intensity_all), pdf, color='red')
        # plt.xlabel('intensity_all')
        # plt.ylabel('Probability Density')
        # plt.show()

        # class norm1:
        #     def __init__(self, a1, b1, c1):
        #         self.a1 = a1
        #         self.b1 = b1
        #         self.c1 = c1
        #
        #     def dist_curve(self):
        #         plt.plot(self.c1, 1 / (self.b1 * np.sqrt(2 * np.pi)) *
        #                  np.exp(- (self.c1 - self.a1) ** 2 / (2 * self.b1 ** 2)), linewidth=2, color='y')
        #         plt.show()
        # x1 = np.array(self.intensity_all)
        # z1 = plt.hist(x1, normed=True, bins=100)  # hist
        # # w1, x1, z1 = plt.hist(x1, normed=True, bins=100)  # hist
        #
        # hist1 = norm1(intensity_mean, intensity_std, x1)
        # plot1 = hist1.dist_curve()
        # plot1.show()
        # normalize label
        room_labels_mean = 925.8869876854762
        room_labels_std =  499.07046037234073


        # ploat intensity mean and std
#pdb.set_trace()
#        plt.errorbar(1, intensity_mean, yerr = intensity_std, fmt="o")
#        plt.show
#        plt.errorbar(1, room_labels_mean, yerr = room_labels_std, fmt="o")
#        plt.show


        # retake

        # retake
        for room_name in rooms_split:
            room_path = os.path.join(data_root, room_name)
            # room_data = np.load(room_path)  # xyzrgbl, N*7
            room_data = pd.read_csv(room_path, sep="\s+").values  # xyzrgbl, N*7
            # print(room_path)

            points, labels = room_data[:, 0:3], room_data[:, -1]
            rgb = room_data[:, 10:13]
            intensity = room_data[:, 3].reshape(points.shape[0], 1)
            # pdb.set_trace()

            intensity = (intensity - intensity_mean) / intensity_std
            labels = (labels - room_labels_mean) / room_labels_std

            # normalize
            points = (points - whole_coord_min) / room_scale
            rgb = rgb / (256 * 255)

            # self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
            points = np.concatenate((points, rgb, intensity), axis=1)

            self.room_points.append(points), self.room_labels.append(labels)        # normalize rgb



        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index])
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))


    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]
        
        # center = points[np.random.choice(N_points)][:2]
        # block_min = center - [self.block_size / 2.0, self.block_size / 2.0]
        # block_max = center + [self.block_size / 2.0, self.block_size / 2.0]
        # point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0])
        #                       & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]


        # pdb.set_trace()

        # if point_idxs.size >= self.num_point:
        #     # selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        #     warnings.filterwarnings("ignore")
        #     nn = NearestNeighbors(self.num_point, algorithm="brute")
        #     nn.fit(points[point_idxs][:, :2])
        #     idx = nn.kneighbors(center[None, :], return_distance=False)[0]
        #     selected_point_idxs = point_idxs[idx]
        # if N_points >= self.num_point:
        #     selected_point_idxs = np.random.choice(N_points, self.num_point, replace=False)
        #
        # else:
        #     selected_point_idxs = np.random.choice(N_points, self.num_point, replace=True)
        #
        # selected_points = points[selected_point_idxs, :]  # num_point * 7
        # # selected_points = center_block(selected_points)
        # # pdb.set_trace()
        # current_points = selected_points
        # current_labels = labels[selected_point_idxs]
        # if self.transform is not None:
        #     pdb.set_trace()
        #     current_points, current_labels = self.transform(current_points, current_labels)

        return points, labels

    def __len__(self):
        return len(self.room_idxs)

def center_block(points):
    # center the samples around the center point
    selected_points = np.copy(points)  # make sure we work on a copy
    selected_points[:, :3] -= np.median(selected_points[:, :3], 0)

    # TODO see if this is better or worse
    # selected_points[:, :2] -= center # consider using the mean here
    # selected_points[:, 2] -= selected_points[:, 2].mean()

    return selected_points


if __name__ == '__main__':

    data_root = "/data/REASEARCH/DEEPCROP/PointCloudData/regression/20_july_2020/train"
    # data_root = "/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test/"
    # data_root = "/data/REASEARCH/DEEPCROP/PointCloudData/regression/split_lidar_clipping_train_test/test"

    num_point,  block_size = 400, 20

    point_data = GeoData_crop_1(split='train', data_root=data_root, num_point=num_point, block_size=block_size,  transform=None)


    print('point data size:', point_data.__len__())
#     pdb.set_trace()
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    for i in range( point_data.__len__()):
        i = np.random.randint(0,point_data.__len__())
        # pdb.set_trace()
        sample_name = "./reg/" + str(i) + ".txt"
        current_points = point_data.__getitem__(i)[0]
        current_labels_sample = point_data.__getitem__(i)[1][:, np.newaxis]
        sample_np = np.concatenate((current_points[:, 0:7], current_labels_sample), axis=1)
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
