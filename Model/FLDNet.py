import torch
import torch.nn as nn
from Model.module import ResBlock, SpatialPyramidPoolingFast, SEBlock

class FLDNet(nn.Module):
    def __init__(self, in_channels, out_classes, enable_se=True, spp=True, invert=True):
        super(FLDNet, self).__init__()
        hidden_ch = 32
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_ch, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            ResBlock(in_channels=hidden_ch, out_channels=hidden_ch, stride=1, scales=[2, 2, 2], 
                     invert=invert, se=enable_se),
            ResBlock(in_channels=hidden_ch, out_channels=hidden_ch + 16, stride=2, scales=[1], 
                     invert=invert, se=enable_se)
        )
        self.stage2 = nn.Sequential(
            ResBlock(in_channels=hidden_ch + 16, out_channels=hidden_ch + 16, stride=1, scales=[2, 2], 
                     invert=invert, se=enable_se),
            ResBlock(in_channels=hidden_ch + 16, out_channels=hidden_ch + 24, stride=1, scales=[1],
                     invert=invert, se=enable_se),
        )
        self.stage3 = nn.Sequential(
            ResBlock(in_channels=hidden_ch + 24, out_channels=hidden_ch + 24, stride=1, scales=[2, 2, 2, 2],
                     invert=invert, se=enable_se),
            ResBlock(in_channels=hidden_ch + 24, out_channels=hidden_ch + 32, stride=2, scales=[1],
                     invert=invert, se=enable_se)
        )
        self.stage4 = nn.Sequential(
            ResBlock(in_channels=hidden_ch + 32, out_channels=hidden_ch + 32, stride=1, act='swish', scales=[2, 2],
                     invert=invert, se=enable_se),
            ResBlock(in_channels=hidden_ch + 32, out_channels=hidden_ch + 56, stride=2, act='swish', scales=[1],
                     invert=invert, se=enable_se)
        )
        self.stage5 = nn.Sequential(
            ResBlock(in_channels=hidden_ch + 56, out_channels=hidden_ch + 56, stride=1, act='swish', scales=[4, 4],
                     se=enable_se)
        )
        in_ch = (hidden_ch + 56) * 4 if spp else hidden_ch + 56
        self.pool = SpatialPyramidPoolingFast(kernel=3) if spp else nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=in_ch, out_features=out_classes)
    
    def forward(self, x):
        x = self.conv_input(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return logits
