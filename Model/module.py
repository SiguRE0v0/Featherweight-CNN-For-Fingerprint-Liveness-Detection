import torch
import torch.nn as nn
from typing import List


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act='relu', scales=List[int], se=False, invert=False):
        super(ResBlock, self).__init__()
        if scales is None:
            scales = [4]
        self.num_layers = len(scales)
        self.cur_channels = in_channels
        self.stride = 1
        self.shortcut = True
        if stride != 1 or in_channels != out_channels:
            self.shortcut = False
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                self.stride = stride
                self.cur_channels = out_channels
            if invert:
                layer = InvertedBottleneck(in_channels, self.cur_channels, stride=self.stride,
                                           act=act, scale=scales[i], se=se if scales[i] > 1 else False)
            else:
                layer = Bottleneck(in_channels, self.cur_channels, stride=self.stride, act=act, scale=scales[i])
            self.layers.append(layer)
        self.se_block = SEBlock(out_channels) if se and not invert else nn.Identity()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.se_block(out)
        if self.shortcut:
            out += x
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act='relu', scale=4, dropout=0):
        super(Bottleneck, self).__init__()
        self.middle_channels = out_channels // scale
        self.act = {'relu': nn.ReLU(inplace=True), 'swish': nn.SiLU(inplace=True)}[act]
        self.dropout = nn.Dropout(dropout)
        self.shortcut = True
        if stride != 1 or in_channels != out_channels:
            self.shortcut = False
        self.pwConv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.middle_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.middle_channels),
            self.act
        )
        self.pwConv2 = nn.Sequential(
            nn.Conv2d(self.middle_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.middle_channels),
            self.act
        )

    def forward(self, x):
        out = self.pwConv1(x)
        out = self.Conv(out)
        out = self.pwConv2(out)
        if self.shortcut:
            return out + x
        return out


class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act='relu', dropout=0, scale=4, se=False):
        super(InvertedBottleneck, self).__init__()
        self.act = {'relu': nn.ReLU(inplace=True), 'swish': nn.SiLU(inplace=True)}[act]
        self.dropout = nn.Dropout(dropout)
        self.shortcut = True
        self.middle_channels = out_channels * scale
        if stride != 1 or in_channels != out_channels:
            self.shortcut = False

        self.pwConv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.middle_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.middle_channels),
            self.act
        )
        self.pwConv2 = nn.Sequential(
            nn.Conv2d(self.middle_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dwConv = nn.Sequential(
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False, groups=self.middle_channels),
            nn.BatchNorm2d(self.middle_channels),
            self.act
        )
        self.se_block = SEBlock(self.middle_channels) if se else nn.Identity()

    def forward(self, x):
        out = self.pwConv1(x)
        out = self.dwConv(out)
        out = self.se_block(out)
        out = self.pwConv2(out)
        if self.shortcut:
            return out + x
        return out


class SpatialPyramidPoolingFast(nn.Module):
    def __init__(self, kernel=5):
        super(SpatialPyramidPoolingFast, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.pool(x)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        out = torch.cat([x, x1, x2, x3], dim=1)
        return self.avg_pool(out)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduce_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduce_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduce_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, h, w = x.size()
        x_avg = self.avg_pool(x).view(n, c)
        x_max = self.max_pool(x).view(n, c)
        out = self.fc(x_avg) + self.fc(x_max)
        c_weight = self.sigmoid(out)
        return x * c_weight.view(n, c, 1, 1)
