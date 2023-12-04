from torch import nn

# resnet based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# and https://arxiv.org/pdf/1603.05027.pdf


def conv1xN(channels, N):
    return nn.Conv2d(channels, channels, kernel_size=(1, N), padding=(0, N // 2))


class Conv1DResidual(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        hdim = hparams['term_hidden_dim']
        self.bn1 = nn.BatchNorm2d(hdim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1xN(hdim, hparams['conv_filter'])
        self.bn2 = nn.BatchNorm2d(hdim)
        self.conv2 = conv1xN(hdim, hparams['conv_filter'])

    def forward(self, X):
        identity = X
        out = self.bn1(X)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out


class Conv1DResNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        blocks = [self._make_layer(hparams) for _ in range(hparams['matches_blocks'])]
        self.resnet = nn.Sequential(*blocks)

    def _make_layer(self, hparams):
        return Conv1DResidual(hparams)

    def forward(self, X):
        # X: num batches x num channels x TERM length x num alignments
        # out retains the shape of X
        # X = self.bn(X)
        if self.hparams['resnet_linear']:
            out = X
        else:
            out = self.resnet(X)

        # average along axis of alignments
        # out: num batches x hidden dim x TERM length
        out = out.mean(dim=-1)

        # put samples back in rows
        # out: num batches x TERM length x hidden dim
        out = out.transpose(1, 2)

        return out
