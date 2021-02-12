import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv2d' in classname:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif 'Linear' in classname:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.momentum = momentum
