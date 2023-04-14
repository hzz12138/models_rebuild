# -*- coding: utf-8 -*-
# @Time    : 2023/4/12 16:33
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : BIT.py
# @Software: PyCharm


import torch
import torch.nn as nn
from rs_models.BIT.cnn_backbone import CNN_backbone
from rs_models.BIT.bit_transformer import BIT_transformer
from rs_models.BIT.prediction_head import PredictionHead
from torchsummary import summary


class BIT(nn.Module):
    def __init__(self, in_channels=3, image_size=224, len_token=4, patch_dims=32, num_heads=8, decoder_depths=8, dropout_ratio=0.1):
        super(BIT, self).__init__()
        self.cnn_backbone = CNN_backbone(in_channels)
        self.bit_transformer = BIT_transformer(in_channels=32, len_token=len_token, patch_dims=patch_dims,
                                               num_heads=num_heads, decoder_depths=decoder_depths,
                                               dropout_ratio=dropout_ratio)
        self.prediction_head = PredictionHead(image_size=image_size)

    def forward(self, x1, x2):
        x1 = self.cnn_backbone(x1)
        x2 = self.cnn_backbone(x2)
        x1, x2 = self.bit_transformer(x1, x2)
        x = self.prediction_head(torch.abs(x1 - x2))
        return x


if __name__ == '__main__':
    model = BIT(in_channels=3, image_size=256, len_token=4, patch_dims=32, num_heads=8, decoder_depths=8, dropout_ratio=0.1)
    summary(model, [(3, 256, 256), (3, 256, 256)], 1, 'cpu')

    in_tensor_1 = torch.randn(size=(1, 3, 256, 256))
    in_tensor_2 = torch.randn(size=(1, 3, 256, 256))
    out_tensor = model(in_tensor_1, in_tensor_2)
    print(out_tensor.shape)
