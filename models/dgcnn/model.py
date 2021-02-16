import torch
import torch.nn as nn

from .util import find_knn, get_graph_feature


# TODO this is the classification variant, we need the segmentation one
class DGCNN(nn.Module):
    def __init__(self, input_dim, output_dim,
                 k=20, do_multi_knn=False,
                 emb_dims=512, edge_dims=[64, 64, 128, 256], linear_dims=[512, 256],
                 dropout=.2, channels_last=False):
        super(DGCNN, self).__init__()
        self.k = k
        self.channels_last = channels_last
        self.do_multi_knn = do_multi_knn

        ### Create Edge-Conv Blocks
        self.edge_convs = []
        for i, o in zip([input_dim] + edge_dims[:-1], edge_dims):
            self.edge_convs.append(nn.Sequential(
                nn.Conv2d(2 * i, o, kernel_size=1, bias=False),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(negative_slope=0.2)
            ))
        self.edge_convs = nn.ModuleList(self.edge_convs)

        ### Create Embedded Conv1d
        self.emb_conv = nn.Sequential(nn.Conv1d(sum(edge_dims), emb_dims, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(emb_dims),
                                      nn.LeakyReLU(negative_slope=0.2)
                                      )

        ### Create Linear Blocks
        self.linears = []
        for i, o in zip([2 * emb_dims] + linear_dims[:-1], linear_dims):
            self.linears.append(nn.Sequential(
                nn.Linear(i, o, bias=False),
                nn.BatchNorm1d(o),
                nn.Dropout(p=dropout),
                nn.LeakyReLU(negative_slope=0.2)
            ))
        self.linears = nn.Sequential(*self.linears)
        self.final = nn.Linear(linear_dims[-1], output_dim)

    def forward(self, x):
        x = x.transpose(1, -1) if self.channels_last else x  # Transpose input if channels are last
        k = min(self.k, x.shape[-1])  # Small & dirty fix if x has less points than kNNs
        idx = None if self.do_multi_knn else find_knn(x, k)  # Calc. kNNs ones (!)

        ## Execute Edge-Convs
        xs = []  # Collect all x's
        for block in self.edge_convs:
            x = get_graph_feature(x, k, idx)  # Calc. Edge Features
            x = block(x).max(-1)[0]  # Execute Conv. + max-pool
            xs.append(x)

        ## Execute Embedding
        x = torch.cat(xs, dim=1)
        x = self.emb_conv(x)
        x = torch.cat((x.max(-1)[0], x.mean(-1)), dim=1)

        ## Execute Linears
        x = self.linears(x)

        ## Execute final linear Layer and return
        return self.final(x), None
