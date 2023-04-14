# -*- coding: utf-8 -*-
# @Time    : 2023/4/12 16:44
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : cnn_backbone.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torchsummary import summary

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBN, self).__init__()
        self.conv_bn = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        ])

    def forward(self, x):
        return self.conv_bn(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.main_path = nn.Sequential(
            ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            ConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.main_path(x)
        x = self.relu(x + residual)
        return x

class CNN_backbone(nn.Module):
    def __init__(self, in_channels):
        super(CNN_backbone, self).__init__()
        num_blocks = [2, 2, 2, 2]
        self.stage0 = nn.Sequential(
            ConvBN(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.in_channels = 64
        self.Block = None
        self.out_channels = None
        self.stage1 = self.get_stage(64, num_blocks[0], 1)
        self.stage2 = self.get_stage(128, num_blocks[0], 2)
        self.stage3 = self.get_stage(256, num_blocks[0], 1)
        self.stage4 = self.get_stage(512, num_blocks[0], 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = ConvBN(in_channels=512, out_channels=32, kernel_size=3, padding=1, stride=1)


    def get_stage(self, first_channels, block_num, stride=1):
        self.Block = BasicBlock
        self.out_channels = first_channels
        if stride != 2 or self.in_channels == first_channels:
            downsample = ConvBN(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            downsample = ConvBN(self.in_channels, self.out_channels, kernel_size=1, stride=stride, padding=0)

        stage_list = []
        stage_list.append(self.Block(self.in_channels, self.out_channels, stride=stride, downsample=downsample))
        self.in_channels = self.out_channels
        for _ in range(block_num - 1):
            stage_list.append(self.Block(self.in_channels, out_channels=self.out_channels, stride=1, downsample=None))
        return nn.Sequential(*stage_list)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.upsample(x)
        x = self.conv1(x)

        return x

    # def forward(self, x1, x2):
    #     x1 = self.forward_once(x1)
    #     x2 = self.forward_once(x2)
    #     return x1, x2


if __name__ == '__main__':
    model = CNN_backbone(in_channels=3)
    summary(model, (3, 224, 224), 1, 'cpu')
    # x1, x2 = torch.randn(size=(1, 3, 224, 224)), torch.randn(size=(1, 3, 224, 224))
    # x1, x2 = model(x1, x2)
    # print(x1.shape, x2.shape)
