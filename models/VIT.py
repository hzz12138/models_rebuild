# -*- coding: utf-8 -*-
# @Time    : 2023/4/7 15:30
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : VIT.py
# @Software: PyCharm

import torch
import torch.nn as nn
from einops import rearrange


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



class VIT_Encoder(nn.Module):
    def __init__(self, patch_dims):
        super(VIT_Encoder, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=patch_dims)
        self.multi_head_attn = MultiHeadAttn(patch_dims, num_heads=16)
        self.ln2 = nn.LayerNorm(normalized_shape=patch_dims)
        self.mlp = MLP(patch_dims=patch_dims)

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


if __name__ == '__main__':
    model = VIT_Encoder(patch_dims=1024)
    input_tensor = torch.randn(size=(1, 65, 1024))  # (batch_size, num_patches, patch_dims)
    output_tensor = model(input_tensor)

    print(output_tensor.shape)
