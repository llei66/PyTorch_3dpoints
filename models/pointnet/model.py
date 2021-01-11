import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from models.pointnet.util import PointNetEncoder


class PointNet(nn.Module):
    def __init__(self, n_classes, with_rgb=True):
        super(PointNet, self).__init__()
        if with_rgb:
            channel = 6
        else:
            channel = 3
        self.n_classes = n_classes
        self.act = nn.ReLU(inplace=True)

        # global feature extractor
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)

        # segmentation mlp part
        self.seg_mlp = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            self.act,
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            self.act,
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            self.act,
            torch.nn.Conv1d(128, self.n_classes, 1)
        )

    def forward(self, x):
        bs = x.shape[0]
        n_pts = x.shape[2]

        # extract global features
        x, trans, trans_feat = self.feat(x)

        # apply segmentation network to transformed features concatenated with global features (x)
        x = self.seg_mlp(x)

        # apply log softmax function to each point
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.n_classes), dim=-1)
        x = x.view(bs, n_pts, self.n_classes)

        return x, trans_feat


if __name__ == '__main__':
    model = PointNet(13, with_rgb=False)
    xyz = torch.rand(12, 3, 2048)
    model(xyz)
