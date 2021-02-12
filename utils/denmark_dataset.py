import os
from itertools import product

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch.cuda import is_available
from torch.utils.data import Dataset, DataLoader
import numpy as np

CLASSES = ['ground', 'vegetation', 'building', 'water']


class DenmarkDatasetBase(Dataset):
    def __init__(self, split, data_root, points_per_sample, use_rgb: bool, global_z=None):
        super().__init__()
        # init some constant to know label names
        self.classes = CLASSES

        # save args
        self.points_per_sample = points_per_sample

        self.path = os.path.join(data_root, split)

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        self.use_rgb = use_rgb
        self.room_names = []
        n_point_rooms = []
        total_class_counts = {}

        # load all room data
        for room_name in os.listdir(self.path):
            room_path = os.path.join(self.path, room_name)

            print(room_path)

            # load data (pandas is way faster than numpy in this regard)
            room_data = pd.read_csv(room_path, sep=" ", header=None).values  # xyzrgbl, N*7

            # split into points and labels
            points, labels = room_data[:, :6], room_data[:, 6]  # xyzrgb, N*6; l, N

            # stats for normalization
            # TODO FIX THIS, rooms will have different scaling in real world
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

            # remove rbg channel if not needed:
            if use_rgb:
                # rgb
                # TODO these seem always to be empty atm
                points[:, 3:] /= 255
            else:
                points = points[:, :3]

            # get stats about occurances
            unique, counts = np.unique(labels, return_counts=True)
            for i, unique_i in enumerate(unique):
                if unique_i not in total_class_counts:
                    total_class_counts[unique_i] = 0
                total_class_counts[unique_i] += counts[i]

            # save samples and stats
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            self.room_names.append(room_name)
            n_point_rooms.append(labels.size)
        self.n_point_rooms = n_point_rooms

        # give global_z from training set
        if global_z is None:
            global_z = np.min(np.array(self.room_coord_min)[:, 2], 0), np.max(np.array(self.room_coord_max)[:, 2], 0)
        min_z, max_z = self.global_z = global_z
        for points, coord_min, coord_max in zip(self.room_points, self.room_coord_min, self.room_coord_max):
            # override local z with global z
            coord_min[2] = min_z
            coord_max[2] = max_z
            # apply min-max normalization
            points[:, :3] = (points[:, :3] - coord_min) / (coord_max - coord_min)

        self.n_classes = len(total_class_counts)

        # sort keys and save as array
        total_class_counts = np.array([count[1] for count in sorted(total_class_counts.items())]).astype(np.float32)
        # class weighting 1/(C*|S_k|)
        self.class_weights = 1 / (total_class_counts * self.n_classes)
        print(f"label weights: {self.class_weights}")
        print(f"class counts : {total_class_counts}")
        print(f"class distribution : {total_class_counts / total_class_counts.sum()}")

    def get_global_z(self):
        return self.room_coord_min[0][2], self.room_coord_max[0][2]

    def get_denormalized_dataset(self):
        ''' denormalize data with saved stats '''
        rooms = []
        for points, coord_min, coord_max in zip(self.room_points, self.room_coord_min, self.room_coord_max):
            points = np.copy(points)
            points[:, :3] = points[:, :3] * (coord_max - coord_min) + coord_min
            if self.use_rgb:
                points[:, 3:] *= 255

            rooms.append(points)
        return rooms


class DenmarkDatasetTrain(DenmarkDatasetBase):
    def __init__(self, split, data_root, blocks_per_epoch, points_per_sample, use_rgb: bool,
                 block_size=(1., 1.), transform=None):
        super().__init__(split, data_root, points_per_sample, use_rgb, None)

        # save args
        if isinstance(block_size, float):
            block_size = [block_size] * 2
        self.block_size = np.array(block_size)
        self.blocks_per_epoch = blocks_per_epoch
        self.transform = transform

        room_idxs = []
        sample_prob = self.n_point_rooms / np.sum(self.n_point_rooms)
        for room_i in range(len(self.n_point_rooms)):
            room_idxs.extend([[room_i, i] for i in range(int(round(sample_prob[room_i] * self.blocks_per_epoch)))])
        self.room_idxs = room_idxs

        print("Total of {} samples in {} set.".format(len(self.room_idxs), split))

    # def get_partition_index:

    def __getitem__(self, idx):
        room_idx, sample_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        n_points = points.shape[0]

        # sample random point and area around it
        center = points[np.random.choice(n_points)][:2]
        block_min = center - self.block_size / 2.0
        block_max = center + self.block_size / 2.0

        # query around points
        point_idxs = np.where(
            (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0])
            & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1])
        )[0]

        # ensure same number of points per batch TODO either make this variable or large enough to not matter
        if point_idxs.size >= self.points_per_sample:
            # take closest to center points
            # TODO this might compromise the block size but at least doesn't change the point density
            nn = NearestNeighbors(self.points_per_sample, algorithm="brute")
            nn.fit(points[point_idxs][:, :2])
            idx = nn.kneighbors(center[None, :], return_distance=False)[0]
            point_idxs = point_idxs[idx]
        else:
            # oversample if too few points (this should only rarely happen, otherwise increase block size)
            point_idxs = np.random.choice(point_idxs, self.points_per_sample, replace=True)

        selected_points = points[point_idxs]
        selected_labels = labels[point_idxs]

        # center the samples around the center point
        selected_points[:, :2] = selected_points[:, :2] - center

        if self.transform is not None:
            # apply data augmentation TODO more than one transform should be possible
            selected_points, selected_labels = self.transform(selected_points, selected_labels)

        return selected_points, selected_labels

    def __len__(self):
        return len(self.room_idxs)


class DenmarkDatasetTest(DenmarkDatasetBase):
    def __init__(self, split, data_root, points_per_sample, use_rgb: bool, global_z, return_idx=False):
        super().__init__(split, data_root, points_per_sample, use_rgb, global_z)

        room_idxs = []
        # partition rooms
        self.split_values = []
        # each room is scaled between 0 and 1 so just try to have similar point counts
        for room_i, n_point_i in enumerate(self.n_point_rooms):
            n_split = n_point_i / points_per_sample
            n_split_2d = int(np.ceil(n_split ** .5))
            split_value = 1 / n_split_2d  # take the root since we want to partition in 2d
            room_idxs.extend([[room_i, (i, j)] for i, j in product(range(n_split_2d), range(n_split_2d))])
            # TODO test how many points are actually in the partitions and merge/expand them if necessary
            self.split_values.append(split_value)
        self.room_idxs = room_idxs
        self.return_idx = return_idx

        print("Total of {} samples in {} set.".format(len(self.room_idxs), split))

    # def get_partition_index:
    def __getitem__(self, idx):
        room_idx, sample_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N

        # load partition for testing (no augmentations here)
        i, j = sample_idx

        block_min = np.array([i * self.split_values[room_idx], j * self.split_values[room_idx]])
        block_max = np.array([(i + 1) * self.split_values[room_idx], (j + 1) * self.split_values[room_idx]])
        block_center = (block_min + block_max) / 2

        point_idxs = np.where(
            (points[:, 0] >= block_min[0]) & (points[:, 0] < block_max[0])
            & (points[:, 1] >= block_min[1]) & (points[:, 1] < block_max[1])
        )[0]

        selected_points = points[point_idxs]
        selected_labels = labels[point_idxs]

        # center around center of partitition
        selected_points[:, :2] = selected_points[:, :2] - block_center

        if self.return_idx:
            return selected_points, selected_labels, point_idxs, room_idx

        return selected_points, selected_labels

    def __len__(self):
        return len(self.room_idxs)


def get_data_loader(batch_size, points_per_sample, data_path, split, use_rgb, training, n_data_worker,
                    blocks_per_epoch=None, block_size=None, global_z=None, return_idx=False):
    if training:
        dataset = DenmarkDatasetTrain(
            split=split, data_root=data_path, blocks_per_epoch=blocks_per_epoch, use_rgb=use_rgb,
            points_per_sample=points_per_sample, block_size=block_size, transform=None
        )
    else:
        dataset = DenmarkDatasetTest(
            split=split, data_root=data_path, use_rgb=use_rgb, global_z=global_z,
            points_per_sample=points_per_sample, return_idx=return_idx
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=training, num_workers=n_data_worker,
        pin_memory=is_available() and training, drop_last=training
    )
    return loader


if __name__ == '__main__':
    # TODO this could be used to create subset of the data
    data_root = './data/train_test_whole_class_1km'
    # data_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_test_whole_class_1km/test_train_ll'
    save_dir = "./dataset"
    blocks_per_epoch = 4096  # seen blocks per epoch (iterations)
    points_per_sample = 4096 * 2  # points per sample
    block_size = (0.05, 0.05)  # tuple for normalized sampling area (e.g., if 1km = 1, 200m = 0.2)

    print(f"blocks per epoch: {blocks_per_epoch}, \n"
          f"points per sample: {points_per_sample}, \n"
          f"block size: {block_size}")

    point_data_train = DenmarkDatasetTrain(split='train', data_root=data_root, blocks_per_epoch=blocks_per_epoch,
                                           points_per_sample=points_per_sample, block_size=block_size,
                                           transform=None, use_rgb=False)

    point_data_test = DenmarkDatasetTest(split='test', data_root=data_root, points_per_sample=points_per_sample,
                                         use_rgb=False, global_z=point_data_train.get_global_z())

    # verify denorm works
    xx = pd.read_csv(point_data_test.room_names[0], sep=" ", header=None).values
    print(f"Denorm is close to original: {np.isclose(xx[:, :3], point_data_test.room_points[0]).all()}")

    # save a sample from the dataset
    for i, (points, labels) in enumerate(point_data_train):
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

    train_loader = DataLoader(
        point_data_train, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
    )
    start = time.time()
    for i, (input, target) in enumerate(train_loader):
        print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - start))
        start = time.time()

    test_loader = DataLoader(
        point_data_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=False
    )
    print("Checking average test sample size")
    samples = []
    for i, (input, target) in enumerate(test_loader):
        samples.append(input.shape[1])
    print(f"Number of test sample stats: {np.mean(samples)}+-{np.std(samples)} (median: {np.median(samples)})")
