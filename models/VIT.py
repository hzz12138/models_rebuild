# -*- coding: utf-8 -*-
# @Time    : 2023/4/7 15:30
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : VIT.py
# @Software: PyCharm

import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, in_channels=3, patch_size=16):
        super(PatchEmbedding, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.len_patches = image_size//patch_size
        self.patch_size = patch_size
        self.emded_dims = self.patch_size ** 2 * self.in_channels
        # 方法1，使用卷积
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.emded_dims,
                               kernel_size=patch_size,
                               padding=0,
                               stride=patch_size)

        # 学习class token
        self.cls_token = nn.Parameter(torch.zeros(1, self.emded_dims))
        # 学习pos token
        self.pos_token = nn.Parameter(torch.zeros(1, self.len_patches ** 2 + 1, self.emded_dims))

    def forward(self,x):
        # 方法1
        x = self.conv1(x)
        x = torch.flatten(x,start_dim=2)
        x = torch.transpose(x, 1, 2)

        # 方法2，使用rearrange
        # x = rearrange(tensor=x,
        #               pattern='B C (P1 L1) (P2 L2) -> B (L1 L2) (P1 P2 C)',
        #               C=self.in_channels,
        #               P1=self.patch_size,
        #               P2=self.patch_size,
        #               L1=self.len_patches,
        #               L2=self.len_patches)

        # 类别编码
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token,x], dim=1)
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
        self.qkv = nn.Linear(in_features=patch_dims, out_features=patch_dims * 3, bias=False)
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
    def __init__(self, patch_dims, hidden_dims=None, dropout_ratio=0.1):
        super(MLP, self).__init__()
        if not hidden_dims:
            hidden_dims = patch_dims * 4
        self.fc1 = nn.Linear(in_features=patch_dims, out_features=hidden_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc2 = nn.Linear(in_features=hidden_dims, out_features=patch_dims)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Transformer_Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, patch_dims=768, num_heads=16, dropout_ratio=0.1):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=patch_dims)
        self.multi_head_attn = MultiHeadAttn(patch_dims=patch_dims, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.ln2 = nn.LayerNorm(normalized_shape=patch_dims)
        self.mlp = MLP(patch_dims=patch_dims, hidden_dims=patch_dims*4, dropout_ratio=dropout_ratio)

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
    def __init__(self, embed_dims, num_classes):
        super(MLPHead, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=embed_dims)
        self.mlphead = nn.Linear(in_features=embed_dims, out_features=num_classes)

    def forward(self, x):
        x = self.ln1(x)
        cls = x[:, 0, :]
        x = self.mlphead(cls)
        return x






if __name__ == '__main__':
    input_tensor = torch.randn(size=(10, 3, 224, 224))
    model1 = PatchEmbedding(
        image_size=224,
        in_channels=3,
        patch_size=16
    )
    output_tensor = model1(input_tensor)
    print(output_tensor.shape) # (batch_size, num_patches, patch_dims)

    model2 = TransformerEncoder(
        patch_dims=768,
        num_heads=16
    )
    output_tensor = model2(output_tensor)
    print(output_tensor.shape)

    model3 = MLPHead(
        embed_dims=768,
        num_classes=1024
    )
    output_tensor = model3(output_tensor)
    print(output_tensor.shape)
