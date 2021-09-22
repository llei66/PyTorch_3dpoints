import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import laspy
from pathlib import Path


class BiomassDataset(Dataset):
    def __init__(self, root: Path, file_name, x_radius=15., y_radius=15., z_radius=20.):
        super().__init__()
        self.df = pd.read_csv(root / file_name)
        # difference between measurement and pointclouds taken
        self.df.eval("year_diff =  year - super", inplace=True)
        self.las_folder = root
        self.radius = np.array([[x_radius, y_radius, z_radius]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df = self.df.iloc[idx]

        las = laspy.read(self.las_folder / df.las_file)
        # only coordinates for now
        x = np.stack([las.x, las.y, las.z], 1)

        # interesting features for each point
        features = np.stack([
            las.intensity,
            las.classification,
            # las.scan_angle_rank, # sometimes missing
            las.edge_of_flight_line,
        ], 1)

        # if we need to differentiate between different measurement equipment
        machine = df.nfitype

        # TODO maybe use for inverse weighting (the further away a measurement, the smaller its weight)
        year_diff = df.year_diff

        # normalize
        x_center = (np.quantile(x, 0.99, axis=0, keepdims=True) + np.quantile(x, 0.01, axis=0, keepdims=True)) / 2
        x = (x - x_center) / self.radius

        # target
        y = df.BMbg_ha

        # TODO augmentations
        x.astype(np.float32)

        return x, y, features, machine, year_diff, df.las_file


def test():
    import sys
    try:
        root = Path(sys.argv[1])
    except IndexError:
        raise Exception("please give the path to the files as first argument")

    root = Path(sys.argv[1])
    dataset = BiomassDataset(root, "train_split.csv")
    for i, (x, y, features, machine, year_diff, las_file) in enumerate(dataset):
        print(f"sample: {i}\n"
              f"\tmax values \tx: {x.max(0)}\n"
              f"\tmin values \tx: {x.min(0)}\n"
              f"\tbio mass \t{y}\n"
              f"\tmachine \t {machine}\n"
              f"\tyear diff \t {year_diff}\n"
              f"\tfile \t {las_file}")

        if i == 10:
            break


if __name__ == "__main__":
    test()
