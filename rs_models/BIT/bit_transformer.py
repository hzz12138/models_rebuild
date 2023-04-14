# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 14:29
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : bit_transformer.py
# @Software: PyCharm


import torch
import torch.nn as nn
from einops import rearrange
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


class SemanticTokenizer(nn.Module):
    def __init__(self, in_channels, len_token, patch_dims):
        super(SemanticTokenizer, self).__init__()
        self.path_b = nn.Sequential(
            ConvBN(in_channels=in_channels, out_channels=len_token, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)
        # # 学习class token
        # self.cls_token = nn.Parameter(torch.zeros(1, self.patch_dims))
        # 学习pos token
        self.pos_token = nn.Parameter(torch.zeros(1, len_token, patch_dims))

    def forward(self, x):
        path_a = rearrange(x, "b c h w -> b (h w) c")
        path_b = self.path_b(x)
        path_b = rearrange(path_b, "b c h w -> b c (h w)")
        path_b = self.softmax(path_b)
        x = torch.matmul(path_b, path_a)
        pos_token = self.pos_token
        x = x + pos_token
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, patch_dims=32, num_heads=8, dropout_ratio=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(in_features=patch_dims, out_features=patch_dims*3)
        self.scale = (patch_dims//num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_ratio)


    def forward(self, x):
        # print(x.shape)
        qkv = self.qkv(x)
        # batch_size, num_patches, (channels, num_heads, head_dims) ->
        # channels, batch_size, num_heads, num_patches, head_dims
        # 注意：patch_dims = num_heads * head_dims ?
        qkv = rearrange(qkv, 'b p (c h d) -> c b h p d', c=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print('k:', k.shape)
        # print('v:', v.shape)

        k = rearrange(k, 'b h p d -> b h d p')
        attn = torch.matmul(q, k) * self.scale
        attn = self.softmax(attn)
        attn = torch.matmul(attn, v)  # batch_size, num_heads, num_patches, head_dims
        attn = rearrange(attn, 'b h p d -> b p (h d)')
        attn = self.dropout(attn)
        # print(attn.shape)
        return attn

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


class Encoder(nn.Module):
    def __init__(self, patch_dims, num_heads, dropout_ratio):
        super(Encoder, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=patch_dims)
        self.attention = MultiHeadAttention(patch_dims=patch_dims, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(normalized_shape=patch_dims)
        self.mlp = MLP(patch_dims=patch_dims, mlp_dims=patch_dims*4, dropout_ratio=dropout_ratio)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = x + residual
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, patch_dims=32, num_heads=8, dropout_ratio=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        # self.qkv = nn.Linear(in_features=patch_dims, out_features=patch_dims*3)
        self.to_q = nn.Linear(in_features=patch_dims, out_features=patch_dims)
        self.to_k = nn.Linear(in_features=patch_dims, out_features=patch_dims)
        self.to_v = nn.Linear(in_features=patch_dims, out_features=patch_dims)
        self.scale = (patch_dims//num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_ratio)


    def forward(self, x, token):

        q = self.to_q(x)
        k = self.to_k(token)
        v = self.to_v(token)
        q = rearrange(q, 'b p (h d) -> b h p d', h=self.num_heads)
        k = rearrange(k, 'b p (h d) -> b h d p', h=self.num_heads)
        v = rearrange(v, 'b p (h d) -> b h p d', h=self.num_heads)
        attn = torch.matmul(q, k) * self.scale
        attn = self.softmax(attn)
        x = torch.matmul(attn, v)  # batch_size, num_heads, num_patches, head_dims
        x = rearrange(x, 'b h p d -> b p (h d)')
        x = self.dropout(x)

        return x


class Decoder(nn.Module):
    def __init__(self, patch_dims=32, num_heads=2, depths=8, dropout_ratio=0.1):
        super(Decoder, self).__init__()
        self.depths = depths
        self.ln1 = nn.LayerNorm(normalized_shape=patch_dims)
        self.multiheadcrossattention = MultiHeadCrossAttention(patch_dims=patch_dims, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.ln2 = nn.LayerNorm(normalized_shape=patch_dims)
        self.mlp = MLP(patch_dims=patch_dims, mlp_dims=patch_dims*4, dropout_ratio=dropout_ratio)

    def forward(self, x, token):
        for _ in range(self.depths):
            b, c, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            residual = x
            x = self.ln1(x)
            x = self.multiheadcrossattention(x, token)
            x = residual + x
            residual = x
            x = self.ln2(x)
            x = self.mlp(x)
            x = residual + x
            x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x




class BIT_transformer(nn.Module):
    def __init__(self, in_channels=32, len_token=4, patch_dims=32, num_heads=8, decoder_depths=8, dropout_ratio=0.1):
        super(BIT_transformer, self).__init__()
        self.num_patches = len_token * 2
        self.patch_dims = patch_dims
        self.semantic_tokenizer = SemanticTokenizer(in_channels=in_channels, len_token=len_token, patch_dims=patch_dims)
        self.encoder = Encoder(patch_dims=patch_dims, num_heads=num_heads, dropout_ratio=dropout_ratio)
        self.decoder = Decoder(patch_dims=patch_dims, num_heads=num_heads, depths=decoder_depths, dropout_ratio=dropout_ratio)

    def forward(self, x1, x2):
        token_x1 = self.semantic_tokenizer(x1)
        token_x2 = self.semantic_tokenizer(x2)
        token = torch.concat([token_x1, token_x2], dim=1)
        token = self.encoder(token)
        token_x1, token_x2 = torch.chunk(token, chunks=2, dim=1)
        x1 = self.decoder(x1, token_x1)
        x2 = self.decoder(x2, token_x2)
        return x1, x2


if __name__ == '__main__':
    in_tensor_1 = torch.randn(size=(1, 32, 56, 56))
    in_tensor_2 = torch.randn(size=(1, 32, 56, 56))
    model = BIT_transformer()
    out_tensor_1, out_tensor_2 = model(in_tensor_1, in_tensor_2)
    print(out_tensor_1.shape, out_tensor_2.shape)
    # summary(model, (3, 224, 224), 1, 'cpu')
