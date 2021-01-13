import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    iden = torch.eye(d).unsqueeze(0).to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - iden), dim=(1, 2)))
    return loss

class Loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(Loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight=weight)
        mat_diff_loss = feature_transform_regularizer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
