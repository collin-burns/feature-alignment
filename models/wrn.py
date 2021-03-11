import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, norm=nn.BatchNorm2d, ngroups=4, affine=True):
        super(BasicBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if norm is nn.BatchNorm2d or norm is nn.InstanceNorm2d:
            self.norm1 = norm(in_planes, affine=affine)
            self.norm2 = norm(out_planes, affine=affine)
        elif norm is nn.GroupNorm:
            self.norm1 = norm(ngroups, in_planes, affine=affine)
            self.norm2 = norm(ngroups, out_planes, affine=affine)
        elif norm is Identity:
            self.norm1 = norm()
            self.norm2 = norm()
        else:
            raise NotImplementedError

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.dropout = nn.Dropout(p=self.droprate)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.norm1(x))
        else:  
            out = self.relu1(self.norm1(x))
        if self.equalInOut:
            out = self.relu2(self.norm2(self.conv1(out)))
        else:
            out = self.relu2(self.norm2(self.conv1(x)))
        if self.droprate > 0:
            out = self.dropout(out)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, norm=nn.BatchNorm2d, ngroups=4, affine=True):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, norm, ngroups, affine)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, norm, ngroups, affine):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, norm, ngroups, affine))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, tinyimagenet=False, norm=nn.BatchNorm2d,
                 ngroups=4, affine=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, norm, ngroups, affine)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, norm, ngroups, affine)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, norm, ngroups, affine)
        # global average pooling and classifier

        if norm is nn.BatchNorm2d or norm is nn.InstanceNorm2d:
            self.norm1 = norm(nChannels[3], affine=affine)
        elif norm is nn.GroupNorm:
            self.norm1 = norm(ngroups, nChannels[3], affine=affine)
        elif norm is Identity:
            self.norm1 = norm()
        else:
            raise NotImplementedError

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.tinyimagenet = tinyimagenet

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm) and not isinstance(m, Identity):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.norm1(out))
        if self.tinyimagenet:
            out = F.avg_pool2d(out, 16)
        else:
            out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out
