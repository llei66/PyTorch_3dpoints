import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight=weight)
        total_loss = loss
        return total_loss
