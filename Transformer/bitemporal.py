# -*- coding: utf-8 -*-
# @Time    : 2023/4/7 11:12
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : bitemporal.py
# @Software: PyCharm

import torch
import torch.nn as nn
from Transformer.siamese_resnet import SiameseResnet


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x1, x2):
        x = torch.concat([x1, x2], dim=1)
        return x


if __name__ == '__main__':
    model = Encoder()
    x1 = torch.rand(size=(10, 4, 32))
    x2 = torch.rand(size=(10, 4, 32))
    result = model(x1, x2)
    print(result.shape)
