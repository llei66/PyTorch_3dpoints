import torch.nn as nn
import torch.nn.functional as F

from models.pointnet.util import feature_transform_regularizer


class Loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(Loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight=weight)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
