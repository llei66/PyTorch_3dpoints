import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_BS1_point import PointNetEncoder, feature_transform_reguliarzer


import pdb

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 7
        else:
            channel = 3
        channel = 3
        print(channel)

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

#         self.fc1 = nn.Linear(1024, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, k)
        
        self.dropout = nn.Dropout(p=0.2)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        # pdb.set_trace()
        # x = F.log_softmax(x, dim=1)


        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # loss = F.nll_loss(pred, target)
        # target = (2000 -target) / 2000.
#         pdb.set_trace()
        loss = F.mse_loss(pred, target)
#         loss = F.smooth_l1_loss(pred, target)


        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

#         total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        # pdb.set_trace()
        total_loss = loss
        return total_loss

    # def forward(self, output, target, trans_feat):
    #     var_y = torch.var(target, unbiased=False)
    #     r2 = 1.0 - F.mse_loss(output, target, reduction="mean") / var_y
    #
    #     mat_diff_loss = feature_transform_reguliarzer(trans_feat)
    #     r2 = r2 + mat_diff_loss
    #     return r2