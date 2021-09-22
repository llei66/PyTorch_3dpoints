import torch.nn as nn
import torch.nn.functional as F
from .pointnet import PointNetEncoder


class get_model(nn.Module):
    def __init__(self, n_targets: int = 1, extra_channel: int = 0):
        super(get_model, self).__init__()
        channel = 3 + extra_channel
        # channel = 3
        print(channel)

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(256, n_targets)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x, trans_feat
