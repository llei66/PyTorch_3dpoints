import torch
import torch.nn as nn
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


class STNkd(nn.Module):
    def __init__(self, info_channel=0, k=64, n_out=32, scaler=1):
        super(STNkd, self).__init__()
        self.act = nn.ReLU(inplace=True)
        n_neurons = [int(2 ** i * scaler) for i in range(6, 10)]
        self.point_blocks = nn.Sequential(
            point_block(k + info_channel, n_neurons[0], self.act),
            point_block(n_neurons[0], n_neurons[1], self.act),
            point_block(n_neurons[1], n_neurons[2], self.act),
        )
        self.mlps = nn.Sequential(
            point_block(n_neurons[2], n_neurons[1], self.act),
            point_block(n_neurons[1], n_neurons[0], self.act),
        )
        self.out = nn.Conv1d(n_neurons[0], n_out, 1)
        self.k = k

    def forward(self, x):
        x = self.point_blocks(x)
        # TODO this is too drastic
        x = torch.max(x, 2, keepdim=True)[0]

        x = self.mlps(x)
        x = self.out(x)

        return x


class STN3d(STNkd):
    def __init__(self, info_channel, n_out):
        super(STN3d, self).__init__(info_channel, k=3, n_out=n_out)
