import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


class GeoData_crop_1(Dataset):
    def __init__(self, split, data_root, blocks_per_epoch, points_per_sample, block_size=(1., 1., 1.), transform=None):
        super().__init__()
        if isinstance(block_size, float):
            block_size = [block_size] * 3
        self.block_size = np.array(block_size)
        self.blocks_per_epoch = blocks_per_epoch
        self.points_per_sample = points_per_sample

        self.transform = transform
        self.path = os.path.join(data_root, split)

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        ## add water
        total_class_counts = 0

        for room_name in os.listdir(self.path):
            room_path = os.path.join(self.path, room_name)

            print(room_path)

            # load data (pandas is way faster than numpy in this regard)
            room_data = pd.read_csv(room_path, sep=" ", header=None).values  # xyzrgbl, N*7

            # split into points and labels
            points, labels = room_data[:, :6], room_data[:, 6]  # xyzrgb, N*6; l, N

            # normalize
            # TODO FIX THIS, rooms will have different scaling in real world
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            # coordinates
            points[:, :3] = (points[:, :3] - coord_min) / (coord_max - coord_min)
            # rgb
            # TODO these seem always to be empty
            points[:, 3:] /= 255

            # get stats about occurances
            unique, counts = np.unique(labels, return_counts=True)
            total_class_counts += counts

            # save samples and stats
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        self.n_classes = len(unique)

        total_class_counts = total_class_counts.astype(np.float32)
        # invert the weights
        self.label_weights = 1 / total_class_counts
        # normalize them
        self.label_weights = self.label_weights / self.label_weights.sum()
        print(f"label weights: {self.label_weights}")
        print(f"class counts : {total_class_counts}")
        print(f"class distribution : {total_class_counts / total_class_counts.sum()}")

        sample_prob = num_point_all / np.sum(num_point_all)
        room_idxs = []
        for index in range(len(num_point_all)):
            room_idxs.extend([index] * int(round(sample_prob[index] * self.blocks_per_epoch)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        n_points = points.shape[0]

        # sample random point and area around it
        # TODO this is only ok for training. Idea: during testing we could divide the room into overlapping scenes
        center = points[np.random.choice(n_points)][:3]
        block_min = center - self.block_size / 2.0
        block_max = center + self.block_size / 2.0

        # query around points
        point_idxs = np.where(
            (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0])
            & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
            & (points[:, 2] >= block_min[2]) & (points[:, 2] <= block_max[2])
        )[0]

        # ensure same number of points per batch TODO either make this variable or large enough to not matter
        if point_idxs.size >= self.points_per_sample:
            # take closest to center points
            # TODO this might compromise the block size but at least doesn't change the point density
            nn = NearestNeighbors(self.points_per_sample, algorithm="brute")
            nn.fit(points[point_idxs][:, :3])
            idx = nn.kneighbors(center[None, :], return_distance=False)[0]
            point_idxs = point_idxs[idx]
        else:
            # oversample if too few points (this should only rarely happen, otherwise increase block size)
            point_idxs = np.random.choice(point_idxs, self.points_per_sample, replace=True)

        # select from query
        selected_points = points[point_idxs]
        selected_labels = labels[point_idxs]

        # apply transforms
        if self.transform is not None:
            selected_points, selected_labels = self.transform(selected_points, selected_labels)

        return selected_points, selected_labels

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    # TODO this could be used to create subset of the data
    # data_root = './data/train_test_whole_class_1km'
    data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_test_whole_class_1km/test_train_ll'
    save_dir = "./dataset"
    blocks_per_epoch = 4096  # seen blocks per epoch (iterations)
    points_per_sample = 4096 * 2  # points per sample TODO think of variable way to do this
    block_size = (0.05, 0.05, 1.)  # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)

    print(blocks_per_epoch, points_per_sample, block_size)

    point_data = GeoData_crop_1(split='train', data_root=data_root, blocks_per_epoch=blocks_per_epoch,
                                points_per_sample=points_per_sample, block_size=block_size, transform=None)

    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)

    # TODO this took too long
    # check avg distance between points
    # points = point_data.room_points[0]
    # from sklearn.neighbors import kneighbors_graph
    # distances = kneighbors_graph(points, 1, mode='distance', include_self=False, n_jobs=-1)
    # print(f"Median distance between closest points: {np.median(distances)}")

    # save a sample from the dataset
    for i, (points, labels) in enumerate(point_data):
        sample_name = f"{save_dir}/sample_{str(i)}.npy"
        sample_np = np.concatenate((points, labels[:, None]), axis=1)
        np.save(sample_name, sample_np)
        break

    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)
    # iterate 4 times through the data and test the running times
    for idx in range(4):
        start = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - start))
            start = time.time()
