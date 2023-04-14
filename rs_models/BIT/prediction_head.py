# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 11:31
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : prediction_head.py
# @Software: PyCharm

import torch
import torch.nn as nn
from rs_models.BIT.cnn_backbone import CNN_backbone
from rs_models.BIT.bit_transformer import BIT_transformer
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


class PredictionHead(nn.Module):
    def __init__(self, image_size):
        super(PredictionHead, self).__init__()
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear')
        self.conv1 = ConvBN(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBN(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    in_tensor = torch.randn(size=(1, 32, 56, 56))
    model = PredictionHead()
    out_tensor = model(in_tensor)
    print(out_tensor.shape)