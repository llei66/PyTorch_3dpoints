import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss
