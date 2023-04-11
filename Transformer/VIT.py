# -*- coding: utf-8 -*-
# @Time    : 2023/4/7 15:30
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : VIT.py
# @Software: PyCharm

import timm
import torch
import torch.nn as nn
from einops import rearrange
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, in_channels=3, patch_size=16, patch_dims=768):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.len_patches = image_size // patch_size
        self.patch_size = patch_size
        self.patch_dims = patch_dims
        # # 方法1，使用卷积
        # self.conv1 = nn.Conv2d(in_channels=in_channels,
        #                        out_channels=self.patch_dims,
        #                        kernel_size=patch_size,
        #                        padding=0,
        #                        stride=patch_size)
        self.fc1 = nn.Linear(patch_size ** 2 * in_channels, patch_dims)
        # 学习class token
        self.cls_token = nn.Parameter(torch.zeros(1, self.patch_dims))
        # 学习pos token
        self.pos_token = nn.Parameter(torch.zeros(1, self.len_patches ** 2 + 1, self.patch_dims))

    def forward(self, x):
        # 方法1
        # x = self.conv1(x)
        # x = torch.flatten(x, start_dim=2)
        # x = torch.transpose(x, 1, 2)

        # 方法2，使用rearrange
        x = rearrange(tensor=x,
                      pattern='B C (P1 L1) (P2 L2) -> B (L1 L2) (P1 P2 C)',
                      C=self.in_channels,
                      P1=self.patch_size,
                      P2=self.patch_size,
                      L1=self.len_patches,
                      L2=self.len_patches)
        x = self.fc1(x)
        # 类别编码
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        # 位置编码
        pos_token = self.pos_token
        x = x + pos_token

        return x


# 建立多头注意力机制
class MultiHeadAttn(nn.Module):
    def __init__(self, patch_dims, num_heads, dropout_ratio=0.1):
        super(MultiHeadAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dims = patch_dims // num_heads
        self.qkv = nn.Linear(in_features=patch_dims, out_features=patch_dims * 3)
        self.scale = self.head_dims ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(in_features=patch_dims, out_features=patch_dims)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        qkv = self.qkv(x)
        # B:batch_size P:num_patches H:num_heads C:channels d：head_dims
        qkv = rearrange(qkv, pattern="B P (C H d) -> C B H P d", C=3, H=self.num_heads, d=self.head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = rearrange(k, pattern="B H P d -> B H d P")
        attn = torch.matmul(q, k) * self.scale  # B H P P: batch_size, num_heads, num_patches, num_patches
        attn = self.softmax(attn)
        x = torch.matmul(attn, v)  # B H P d: batch_size, num_heads, num_patches, head_dims
        x = rearrange(x, "B P H d -> B H (P d)")
        x = self.proj(x)
        x = self.dropout(x)
        return x


# 全连接层
class MLP(nn.Module):
    def __init__(self, patch_dims, mlp_dims=None, dropout_ratio=0.1):
        super(MLP, self).__init__()
        # if not mlp_dims:
        #     mlp_dims = patch_dims * 4
        self.fc1 = nn.Linear(in_features=patch_dims, out_features=mlp_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc2 = nn.Linear(in_features=mlp_dims, out_features=patch_dims)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Transformer_Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, patch_dims=768, mlp_dims=None, num_heads=16, dropout_ratio=0.1):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=patch_dims)
        self.multi_head_attn = MultiHeadAttn(patch_dims=patch_dims, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.ln2 = nn.LayerNorm(normalized_shape=patch_dims)
        if not mlp_dims:
            mlp_dims = patch_dims * 4
        self.mlp = MLP(patch_dims=patch_dims, mlp_dims=mlp_dims, dropout_ratio=dropout_ratio)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.multi_head_attn(x)
        x = residual + x
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x


# MLP_head
class MLPHead(nn.Module):
    def __init__(self, patch_dims, num_classes):
        super(MLPHead, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=patch_dims)
        self.mlphead = nn.Linear(in_features=patch_dims, out_features=num_classes)

    def forward(self, x):
        x = self.ln1(x)
        cls = x[:, 0, :]
        x = self.mlphead(cls)
        return x


class VIT(nn.Module):
    """
    Args:
        image_size: 输入影像的大小
        in_channels: 输入影像的波段数
        patch_size: 切分patch块的大小
        patch_dims: 每一个patch的编码整合维度
        mlp_dims: Encoder中，mlp层的隐藏层维度
        num_heads: Encoder中，多头注意力的头数
        layers: Encoder中，Transformer_encoder块的数量
        dropout_ratio: dropout速率
        num_classes: 类别数量
    """

    def __init__(self, image_size=224, in_channels=3, patch_size=16, patch_dims=768, mlp_dims=None, num_heads=12,
                 num_layers=12, dropout_ratio=0.1, num_classes=1000):
        super(VIT, self).__init__()
        # patch_dims = patch_size ** 2 * in_channels
        self.patch_embedding = PatchEmbedding(image_size=image_size, in_channels=in_channels, patch_size=patch_size,
                                              patch_dims=patch_dims)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(patch_dims=patch_dims, mlp_dims=mlp_dims, num_heads=num_heads,
                                 dropout_ratio=dropout_ratio)] * num_layers)
        self.mlphead = MLPHead(patch_dims=patch_dims, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.mlphead(x)
        return x


def vit_base(num_classes=1000):
    return VIT(image_size=224, in_channels=3, patch_size=16, patch_dims=768, num_heads=12, num_layers=12,
               num_classes=num_classes, dropout_ratio=0.1)


def vit_large(num_classes=1000):
    return VIT(image_size=224, in_channels=3, patch_size=16, patch_dims=1024, num_heads=16, num_layers=24,
               num_classes=num_classes, dropout_ratio=0.1)


def vit_huge(num_classes=1000):
    return VIT(image_size=224, in_channels=3, patch_size=14, patch_dims=1280, num_heads=16, num_layers=32,
               num_classes=num_classes, dropout_ratio=0.1)


if __name__ == '__main__':
    model1 = vit_base(num_classes=1000)
    summary(model1, (3, 224, 224), batch_size=1, device='cpu')
    model2 = timm.models.vit_base_patch16_224()
    summary(model2, (3, 224, 224), batch_size=1, device='cpu')
