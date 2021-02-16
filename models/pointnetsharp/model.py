import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from .util import STN3d, STNkd, point_block


class PointNet(nn.Module):
    def __init__(self, n_classes, info_channel=0, n_layer=5):
        super(PointNet, self).__init__()
        self.n_classes = n_classes
        self.act = nn.ReLU(inplace=True)

        # global feature extractor
        n_out = 32
        n_in = 3
        self.feat = [STN3d(info_channel=info_channel, n_out=n_out)]
        for layer_i in range(n_layer - 1):
            n_in = n_out + n_in
            self.feat.append(STNkd(k=n_in, n_out=n_out, scaler=1 + layer_i * .25))
        self.feat = nn.ModuleList(self.feat)

        # segmentation mlp part
        self.seg_mlp = nn.Sequential(
            point_block(n_out + n_in, 512, self.act),
            point_block(512, 256, self.act),
            point_block(256, 128, self.act),
            nn.Conv1d(128, self.n_classes, 1)
        )

    def forward(self, x):
        bs = x.shape[0]
        n_pts = x.shape[2]

        # extract global features
        for feat in self.feat:
            f = feat(x)
            # concat with original points
            x = torch.cat([x, f.repeat_interleave(n_pts, 2)], 1)

        # apply segmentation network to transformed features concatenated with global features (x)
        x = self.seg_mlp(x)

        # apply log softmax function to each point
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.n_classes), dim=-1)
        x = x.view(bs, n_pts, self.n_classes)

        return x, None


if __name__ == '__main__':
    model = PointNet(13, 0)
    xyz = torch.rand(12, 3, 2048)
    model(xyz)
