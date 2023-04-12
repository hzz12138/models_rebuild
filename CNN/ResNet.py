# -*- coding: utf-8 -*-
# @Time    : 2023/4/12 13:50
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : ResNet.py
# @Software: PyCharm


import timm
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models


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


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.downsample = downsample
        self.main_path = nn.Sequential(
            ConvBN(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            ConvBN(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3, stride=stride,
                   padding=1),
            nn.ReLU(inplace=True),
            ConvBN(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.main_path(x)
        x = self.relu(x + residual)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, block_style, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.block_style = block_style
        self.stage0 = nn.Sequential(
            ConvBN(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_channels = 64
        self.stage1 = self.get_stage(64, block_num=num_blocks[0], stride=1)
        self.stage2 = self.get_stage(128, block_num=num_blocks[1], stride=2)
        self.stage3 = self.get_stage(256, block_num=num_blocks[2], stride=2)
        self.stage4 = self.get_stage(512, block_num=num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=self.out_channels, out_features=num_classes)

    def get_stage(self, first_channels, block_num, stride=1):
        if self.block_style == 'BottleBlock':
            self.Block = BottleNeck
            self.out_channels = first_channels * 4
        elif self.block_style == 'BasicBlock':
            self.Block = BasicBlock
            self.out_channels = first_channels
        else:
            self.Block = None
            self.out_channels = None

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

    # def get_basic_stage(self, first_channels, block_num, stride=1):
    #     if stride != 2 or self.in_channels == first_channels:
    #         downsample = ConvBN(self.in_channels, first_channels, kernel_size=1, stride=1, padding=0)
    #     else:
    #         downsample = ConvBN(self.in_channels, first_channels, kernel_size=1, stride=stride, padding=0)
    #     stage_list = []
    #     stage_list.append(self.Block(self.in_channels, first_channels, stride=stride, downsample=downsample))
    #     self.in_channels = first_channels
    #     for _ in range(block_num - 1):
    #         stage_list.append(self.Block(self.in_channels, out_channels=first_channels, stride=1, downsample=None))
    #     return nn.Sequential(*stage_list)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将第一维及之后的拉平
        x = self.fc(x)
        return x


def resnet18(num_classes=1000):
    return ResNet(in_channels=3, block_style='BasicBlock', num_blocks=[2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    return ResNet(in_channels=3, block_style='BasicBlock', num_blocks=[3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=1000):
    return ResNet(in_channels=3, block_style='BottleBlock', num_blocks=[3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=1000):
    return ResNet(in_channels=3, block_style='BottleBlock', num_blocks=[3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=1000):
    return ResNet(in_channels=3, block_style='BottleBlock', num_blocks=[3, 8, 36, 3], num_classes=num_classes)


if __name__ == '__main__':
    model1 = resnet50()
    summary(model1, (3, 224, 224), batch_size=1, device='cpu')
    model2 = models.resnet50()
    summary(model2, (3, 224, 224), batch_size=1, device='cpu')
