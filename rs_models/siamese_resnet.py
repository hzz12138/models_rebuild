# -*- coding: utf-8 -*-
# @Time    : 2023/4/6 11:03
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : siamese_resnet.py
# @Software: PyCharm


import torch
import torch.nn as nn
from torchsummary import summary

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Stage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 残差部分
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1_1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=1)
        self.conv2 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.stride == 2:
            residual = self.conv1_1(x)
            residual = self.pool1(residual)
        else:
            if self.in_channels == self.out_channels:
                residual = x
            else:
                residual = self.conv1_1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        residual = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + residual
        return x


class SiameseResnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_patches=4):
        super(SiameseResnet, self).__init__()
        # stage0
        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stage1-4
        self.stage1 = Stage(in_channels=64, out_channels=64, stride=1)
        self.stage2 = Stage(in_channels=64, out_channels=128, stride=2)
        self.stage3 = Stage(in_channels=128, out_channels=256, stride=1)
        self.stage4 = Stage(in_channels=256, out_channels=512, stride=1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = ConvBnRelu(in_channels=512, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # image_tokenizer
        self.conv3 = ConvBnRelu(in_channels=out_channels, out_channels=num_patches, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))  # stage0
        x = self.stage1(x)  # stage1
        x = self.stage2(x)  # stage2
        x = self.stage3(x)  # stage3
        x = self.stage4(x)  # stage4
        x = self.upsample(self.conv2(x))  # output
        x = self.get_semantic_token(x)  # image_tokenizer
        return x

    def get_semantic_token(self, x):
        b, c, h, w = x.shape
        x1 = x.reshape(b, c, -1)
        x1 = x1.transpose(-1, -2)
        x2 = self.conv3(x)
        b, c, h, w = x2.shape
        x2 = x2.reshape(b, c, -1)
        x2 = self.softmax(x2)
        x = torch.matmul(x2, x1)
        return x


if __name__ == '__main__':
    model = SiameseResnet(3, 32)
    summary(model, input_size=(3, 224, 224), batch_size=10, device='cpu')
    in_tensor_a = torch.randn(10, 3, 224, 224)
    in_tensor_b = torch.randn(10, 3, 224, 224)
    out_tensor_a = model(in_tensor_a)
    out_tensor_b = model(in_tensor_b)
    print(out_tensor_a.shape, out_tensor_b.shape)
