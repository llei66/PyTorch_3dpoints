import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


def point_block(in_dims, out_dims, activation=None):
    '''
    helper function to create a conv1d layer with kernel size 1, batch norm, and activation
    '''
    modules = [
        nn.Conv1d(in_dims, out_dims, 1),
        nn.BatchNorm1d(out_dims),
    ]

    if activation is not None:
        modules.append(activation)

    return nn.Sequential(*modules)


def mlp(in_dims, out_dims, activation):
    '''
    helper function to create a linear layer with batch norm and activation
    '''
    modules = [
        nn.Linear(in_dims, out_dims),
        nn.BatchNorm1d(out_dims)
    ]
    if activation is not None:
        modules.append(activation)
    return nn.Sequential(*modules)


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.point_blocks = nn.Sequential(
            point_block(channel, 64, self.act),
            point_block(64, 128, self.act),
            point_block(128, 1024, self.act),
        )
        self.mlps = nn.Sequential(
            mlp(1024, 512, self.act),
            mlp(512, 256, self.act),
        )
        self.out = nn.Linear(256, 9)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.point_blocks(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.mlps(x)
        x = self.out(x)

        # TODO this is inefficient
        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat(
            batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.point_blocks = nn.Sequential(
            point_block(k, 64, self.act),
            point_block(64, 128, self.act),
            point_block(128, 1024, self.act),
        )
        self.mlps = nn.Sequential(
            mlp(1024, 512, self.act),
            mlp(512, 256, self.act),
        )
        self.out = nn.Linear(256, k * k)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.point_blocks(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.mlps(x)
        x = self.out(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(
            batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = point_block(channel, 64, self.activation)
        self.conv2 = nn.Sequential(
            point_block(64, 128, self.activation),
            point_block(128, 1024, self.activation),
        )
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3, dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = self.conv1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.conv(2)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :].to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss
