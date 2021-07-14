"""
Resnet Blocks
1. Basic Block : Preserves the input shape
2. Bottle Neck block : Reduces the input shape
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    
    def __init__(self,
                in_channels : int,
                out_channels : int,
                stride : int = 1,
                downsample : nn.Module = None,
                groups : int = 1,
                base_width : int = 64,
                dilation : int = 1,
                norm_layer : nn.Module = None
                ) -> None:
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if in_channels != out_channels and downsample is None:
            raise ValueError("downsample cannot be None when in_channels != out_channels")
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.bn2 = norm_layer(out_channels)
    
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # residual connection
        out += identity

        out = self.relu(out)
        return out

class BottleNeckBlock(nn.Module):
    expansion: int = 4
    def __init__(self,
                in_channels : int,
                out_channels : int,
                stride : int = 1,
                downsample : nn.Module = None,
                groups : int = 1,
                base_width : int = 64,
                dilation : int = 1,
                norm_layer : nn.Module = None
                ) -> None:
        super(BottleNeckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if in_channels != out_channels and downsample is None:
            raise ValueError("downsample cannot be None when in_channels != out_channels")

        width = int(out_channels * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

if __name__ == "__main__":
    resnet = BottleNeckBlock(16, 32)
    print(resnet)