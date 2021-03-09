import os
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.cuda import is_available
from torch.utils.data import Dataset, DataLoader, Sampler
import warnings
CLASSES = ["Ground", "Low veg", "Medium Veg", "High Veg", "Building", "Water"]


class DenmarkDatasetBase(Dataset):
    def __init__(self, split, data_root, use_rgb: bool, block_size, global_z=None):
        super().__init__()
        # init some constant to know label names
        self.classes = CLASSES

        if isinstance(block_size, float):
            block_size = [block_size] * 2
        self.block_size = np.array(block_size)

        # save args
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
            points, labels = room_data[:, :-1], room_data[:, -1]  # xyzrgb, N*6; l, N

            # stats for normalization
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
            # choose the room with biggest spatial gap in height for scaling the z-axis
            max_room = np.argmax(np.array(self.room_coord_max)[:, 2] - np.array(self.room_coord_min)[:, 2], 0)
            global_z = np.array(self.room_coord_min)[max_room, 2], np.array(self.room_coord_max)[max_room, 2]
        min_z, max_z = self.global_z = global_z
        block_scale = np.concatenate([block_size, [1.]])
        self.room_coord_scale = []
        for points, coord_min, coord_max in zip(self.room_points, self.room_coord_min, self.room_coord_max):
            # override local z with global z
            coord_min[2] = min_z
            coord_max[2] = max_z
            # apply min-max normalization
            # center (makes it easier to do the block queries)
            points[:, :3] = points[:, :3] - coord_min
            # scale
            room_scale = (coord_max - coord_min)  # this shouldn't change for our 1k dataset
            points[:, :3] = points[:, :3] / (room_scale * block_scale)
            self.room_coord_scale.append(room_scale * block_scale)

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

    def get_descaled_dataset(self):
        ''' descale data with saved stats '''
        rooms = []
        for points, c_min, c_scale in zip(self.room_points, self.room_coord_min, self.room_coord_scale):
            points = np.copy(points)
            points[:, :3] = points[:, :3] * c_scale + c_min
            if self.use_rgb:
                points[:, 3:] *= 255

            rooms.append(points)
        return rooms


class DenmarkDatasetTrain(DenmarkDatasetBase):
    def __init__(self, split, data_root, points_per_sample, use_rgb: bool,
                 block_size=(1., 1.), transform=None, global_z=None):
        super().__init__(split, data_root, use_rgb, block_size, global_z)

        # save args
        self.points_per_sample = points_per_sample
        self.transform = transform

        self.room_idxs = []
        for room_i, n_point_i in enumerate(self.n_point_rooms):
            room_max = ((self.room_coord_max[room_i] - self.room_coord_min[room_i]) / self.room_coord_scale[room_i])[:2]
            point_list = []
            for sample_idx in range(n_point_i):
                point = self.room_points[room_i][sample_idx]
                # do not add edge points to the training set
                if (1 <= point[0] <= room_max[0] - 1) and (1 <= point[1] <= room_max[1] - 1):
                    point_list.append([room_i, sample_idx])
            self.room_idxs.extend(point_list)

        print("Total of {} samples in {} set.".format(len(self), split))

    # def get_partition_index:

    def __getitem__(self, idx):
        room_idx, sample_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N

        # sample random point and area around it
        block_center = points[sample_idx][:2]
        block_min = block_center - .5  # .5 is half the block size
        block_max = block_center + .5
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
            idx = nn.kneighbors(block_center[None, :], return_distance=False)[0]
            point_idxs = point_idxs[idx]
        else:
            # oversample if too few points (this should only rarely happen, otherwise increase block size)
            point_idxs = np.random.choice(point_idxs, self.points_per_sample, replace=True)
        # TODO could cache the points in a block to reduce querying (not an issue atm)

        selected_points = points[point_idxs]
        selected_labels = labels[point_idxs]

        # normalize per block
        selected_points = center_block(selected_points)

        if self.transform is not None:
            # apply data augmentation TODO more than one transform should be possible
            selected_points, selected_labels = self.transform(selected_points, selected_labels)

        selected_points = selected_points.astype(np.float32).transpose(1, 0)
        selected_labels = selected_labels.astype(np.longlong)

        return selected_points, selected_labels

    def __len__(self):
        return len(self.room_idxs)


class DenmarkDatasetTest(DenmarkDatasetBase):
    def __init__(self, split, data_root, use_rgb: bool, block_size, global_z, overlap, return_idx):
        super().__init__(split, data_root, use_rgb, block_size, global_z)
        # this works without taking dataset x, y scaling into account since we already scaled to (0, 1)
        self.overlap = overlap  # defines an overlap ratio
        self.overlap_value = np.array([overlap] * 2)
        self.overlap_difference = 1 - self.overlap_value

        room_idxs = []
        # partition rooms
        # each room is scaled between 0 and 1 so just try to have similar point counts
        for room_i, n_point_i in enumerate(self.n_point_rooms):
            # recover max room length from room and block scaling
            room_max = ((self.room_coord_max[room_i] - self.room_coord_min[room_i]) / self.room_coord_scale[room_i])[:2]
            n_split_2d = (np.ceil(room_max / self.overlap_difference)).astype(int)
            room_idxs.extend([[room_i, (i, j)] for i, j in product(range(n_split_2d[0]), range(n_split_2d[1]))])
            # TODO test how many points are actually in the partitions and merge/expand them if necessary
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

        block_min = np.array([i * self.overlap_difference[0], j * self.overlap_difference[1]])
        block_max = np.array([
            (i + 1) * self.overlap_difference[0] + self.overlap_value[0],
            (j + 1) * self.overlap_difference[1] + self.overlap_value[1]
        ])

        point_idxs = np.where(
            (points[:, 0] >= block_min[0]) & (points[:, 0] < block_max[0])
            & (points[:, 1] >= block_min[1]) & (points[:, 1] < block_max[1])
        )[0]

        selected_points = points[point_idxs]
        selected_labels = labels[point_idxs]

        # normalize per block
        selected_points = center_block(selected_points)

        selected_points = selected_points.astype(np.float32).transpose(1, 0)
        selected_labels = selected_labels.astype(np.longlong)

        if self.return_idx:
            return selected_points, selected_labels, point_idxs, room_idx

        return selected_points, selected_labels

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


def get_train_data_loader(batch_size, points_per_sample, data_path, split, use_rgb, n_data_worker,
                          steps_per_epoch=None, block_size=None, global_z=None):
    dataset = DenmarkDatasetTrain(
        split=split, data_root=data_path, use_rgb=use_rgb,
        points_per_sample=points_per_sample, block_size=block_size, transform=None, global_z=global_z
    )
    if steps_per_epoch == -1:
        num_samples = len(dataset)
    else:
        num_samples = steps_per_epoch * batch_size
    sampler = RandomSampler(dataset, num_samples=num_samples)
    loader = MultiEpochsDataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=n_data_worker,
        pin_memory=is_available(), drop_last=True
    )
    return loader


def get_test_data_loader(batch_size, data_path, split, use_rgb, n_data_worker, overlap,
                         block_size=None, global_z=None, return_idx=True):
    dataset = DenmarkDatasetTest(
        split=split, data_root=data_path, use_rgb=use_rgb, global_z=global_z,
        block_size=block_size, return_idx=return_idx, overlap=overlap
    )
    loader = MultiEpochsDataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_data_worker,
        pin_memory=False, drop_last=False
    )
    return loader


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    # we need this to avoid the destruction of the iterator (problematic with many datapoints)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = np.arange(len(data_source))
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        return iter(np.random.choice(self.idx, self.num_samples).tolist())

    def __len__(self):
        return self.num_samples


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

    point_data_train = DenmarkDatasetTrain(split='train', data_root=data_root, points_per_sample=points_per_sample,
                                           block_size=block_size, transform=None, use_rgb=False)
    point_data_test = DenmarkDatasetTest(split='test', data_root=data_root, block_size=block_size,
                                         use_rgb=False, global_z=None, overlap=.5, return_idx=True)

    # verify denorm works
    xx = pd.read_csv(os.path.join(data_root, "test", point_data_test.room_names[0]), sep=" ", header=None).values
    print(f"Scaled points close to original: {np.isclose(xx[:, :3], point_data_test.room_points[0]).all()}")
    print(f"Descale close to original: {np.isclose(xx[:, :3], point_data_test.get_descaled_dataset()[0]).all()}")

    # test if iterating through data throws errors
    for x in point_data_test: break

    # save a sample from the dataset
    for i, (points, labels) in enumerate(point_data_train):
        sample_name = f"{save_dir}/sample_{str(i)}.npy"
        sample_np = np.concatenate((points, labels[:, None]), axis=1)
        np.save(sample_name, sample_np)
        break

    import time, random

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
