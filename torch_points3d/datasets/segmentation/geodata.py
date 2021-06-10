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

### import torch3d related lib
from glob import glob
import multiprocessing
import logging
from torch_geometric.data import Dataset, Data

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.segmentation.geodata_util import DenmarkDatasetTrain, DenmarkDatasetTest, DenmarkDatasetBase

log = logging.getLogger(__name__)

class GeodataDataset(BaseDataset):
#class GeodataDataset(DenmarkDatasetBase):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        data_root = '/home/dmn774/Deepcrop/data/geodata/test_whole_one'
        blocks_per_epoch = 4096 
        points_per_sample = 4096 * 2
        block_size = (0.05, 0.05)
        self.train_dataset = DenmarkDatasetTrain(split='train', data_root=data_root, points_per_sample=points_per_sample, block_size=block_size, transform=None, use_rgb=False)

        self.val_dataset = DenmarkDatasetTest(split='validate', data_root=data_root, block_size=block_size, use_rgb=False, global_z=None, overlap=.5, return_idx=True)
        self.test_dataset = DenmarkDatasetTest(split='test', data_root=data_root, block_size=block_size,use_rgb=False, global_z=None, overlap=.5, return_idx=True)

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @property  # type: ignore
    def stuff_classes(self):
        """ Returns a list of classes that are not instances
        """
        return self.train_dataset.stuff_classes

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

