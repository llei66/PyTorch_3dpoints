import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch.cuda import is_available
from torch.utils.data import Dataset, DataLoader
import warnings
CLASSES = ['ground', 'vegetation', 'building', 'water']


class DublinDatasetBase(Dataset):
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

        room_idxs = []
        sample_prob = self.n_point_rooms / np.sum(self.n_point_rooms)
        for room_i in range(len(self.n_point_rooms)):
            room_idxs.extend([[room_i, i] for i in range(int(round(sample_prob[room_i] * self.blocks_per_epoch)))])
        self.room_idxs = room_idxs

        print("Total of {} samples in {} set.".format(len(self.room_idxs), split))

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
            warnings.filterwarnings("ignore")
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
        selected_points[:, :2] -= center
        selected_points[:, 2] -= selected_points[:, 2].mean()

        if self.transform is not None:
            # apply data augmentation TODO more than one transform should be possible
            selected_points, selected_labels = self.transform(selected_points, selected_labels)

        return selected_points, selected_labels

    def __len__(self):
        return len(self.room_idxs)
