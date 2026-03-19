# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/15 下午8:51
import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F
import math
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_
# from Prelude_env.ExploreModel.model import MHSA
# from CountMambaModel.model_mamba2 import Mamba2
# from lxj_utils_sys import timer

# from Prelude_env.ExploreRun.test_module import *





class Attention_base(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        # self.attn_drop = nn.Sequential(
        #     nn.Softmax(dim=-1),
        # )
        # self.proj_drop = nn.Sequential(
        #     nn.Linear(dim, dim),
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class Attention_Causal(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)

        # 如果需要输出投影层，可以取消注释
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(0.1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape

        # 1. 计算 Q, K, V
        # Shape: (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 计算注意力分数 (Attention Scores)
        # Shape: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ==================== 核心修改：添加因果掩码 ====================
        # 生成一个下三角矩阵 (Lower Triangular Matrix)
        # torch.tril 会将上三角（不包含对角线）置为 0，下三角和对角线保持为 1
        # mask shape: (N, N)
        mask = torch.ones(N, N, device=x.device).tril()

        # 使用 masked_fill 将 mask 为 0 的位置（即未来位置）填充为 -inf
        # 这样 Softmax 之后，这些位置的概率会趋近于 0
        attn = attn.masked_fill(mask == 0, float('-inf'))
        # ==============================================================

        # 3. 归一化 (Softmax)
        attn = F.softmax(attn, dim=-1)

        # 4. 聚合 Value
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 如果有输出投影层
        # x = self.proj(x)
        # x = self.proj_drop(x)

        return x

class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, atten_config):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = eval(f'Attention_{atten_config["name"]}')(dim=embed_dim, num_heads=nhead, **atten_config)

        #self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        #self.attn = LinearAttentionOptimized(embed_dim=embed_dim, value_dim=embed_dim, approx_dim=embed_dim)
        #self.attn = MultiHeadLinearAttention(num_heads=nhead,embed_dim=embed_dim, value_dim=embed_dim, approx_dim=embed_dim)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, atten_config):
        super().__init__()
        self.nets = nn.ModuleList(
            [MHSA_Block(embed_dim, num_heads, dim_feedforward, atten_config) for _ in range(num_mhsa_layers)])
    def forward(self, x, pos_embed=0):
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output




def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=np.float32)
    # print(grid.shape)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(1, 1))

    def forward(self, x):
        x = self.proj(x)
        return x


class CausalCNN(nn.Module):
    def __init__(self, in_channels, mid_channel, kernel_size=5):
        super(CausalCNN, self).__init__()

        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn1 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn2 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn3 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn4 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x.squeeze(2)

        x = self.conv1(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.transpose(1, 2)

        return x, None


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


# class LocalProfiling(nn.Module):
#     """ Local Profiling module in ARES """
#
#     def __init__(self, in_channels, mid_channel):
#         super().__init__()
#
#         self.dividing = nn.Sequential(
#             Rearrange('b c (n p) -> (b n) c p', n=4),
#         )
#         self.combination = nn.Sequential(
#             Rearrange('(b n) c p -> b c (n p)', n=4),
#         )
#
#         self.net = nn.Sequential(
#             ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=128, out_channels=mid_channel, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#         )
#
#     def forward(self, x):
#         x = x.squeeze(2)
#
#         if self.training and False:
#             sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
#             x = torch.roll(x, shifts=sliding_size, dims=-1)
#         else:
#             sliding_size = 0
#
#         x = self.dividing(x)
#         x = self.net(x)
#         x = self.combination(x)
#
#         x = x.permute(0, 2, 1)
#         return x, sliding_size
#
#
#
# class LocalProfiling_SAP(nn.Module):
#     """ Local Profiling module in ARES """
#
#     def __init__(self, in_channels, mid_channel):
#         super().__init__()
#         num_segments = 4
#         self.slicer = SaliencySlicer(in_channels, num_segments=num_segments)
#         self.aggregator = nn.Sequential(
#             SaliencyAggregator(num_segments)
#         )
#         self.net = nn.Sequential(
#             ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=128, out_channels=mid_channel, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#         )
#
#     def forward(self, x):
#         x = x.squeeze(2)
#
#         x = self.slicer(x)
#         x = self.net(x)
#         x = self.aggregator(x)
#
#         x = x.permute(0, 2, 1)
#         return x, None
#
#
# class LocalProfiling_LR(nn.Module):
#     """ Local Profiling module in ARES """
#
#     def __init__(self, in_channels, mid_channel):
#         super().__init__()
#
#         self.dividing = nn.Sequential(
#             Rearrange('b c (n p) -> (b n) c p', n=4),
#         )
#         self.combination = nn.Sequential(
#             Rearrange('(b n) c p -> b c (n p)', n=4),
#         )
#
#         self.net = nn.Sequential(
#             ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#             ConvBlock1d(in_channels=128, out_channels=mid_channel, kernel_size=7),
#             nn.MaxPool1d(kernel_size=8, stride=4),
#             nn.Dropout(p=0.1),
#         )
#
#         self.learn_roll = LearnableRolling(in_channels=in_channels, seq_len=7200, max_shift=1 + 7200 // 4, num_candidates=16)
#     def forward(self, x):
#         x = x.squeeze(2)
#
#         # if self.training:
#         #     sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
#         #     x = torch.roll(x, shifts=sliding_size, dims=-1)
#         # else:
#         #     sliding_size = 0
#
#         x = self.learn_roll(x)
#         sliding_size = 0
#         x = self.dividing(x)
#         x = self.net(x)
#         x = self.combination(x)
#
#         x = x.permute(0, 2, 1)
#         return x, sliding_size

class SlidingWindowSplit(nn.Module):
    def __init__(self, segment_len, overlap_ratio=0.0, padding_value=0):
        super().__init__()
        self.segment_len = segment_len
        self.padding_value = padding_value
        # 步长计算
        self.stride = int(segment_len * (1 - overlap_ratio))
        if self.stride < 1:
            self.stride = 1

    def calculate_n(self, l):
        """
        根据输入长度 l，计算分段后的窗口数量 n
        """
        if l < self.segment_len:
            # 长度不足一个片段，padding 到 segment_len，窗口数为 1
            return 1

        # 计算 padding 后的长度 (逻辑同 forward)
        remainder = (l - self.segment_len) % self.stride
        pad_len = 0 if remainder == 0 else self.stride - remainder
        l_pad = l + pad_len

        # 计算窗口数 n: (L_pad - segment_len) / stride + 1
        n = (l_pad - self.segment_len) // self.stride + 1
        return n

    def forward(self, x):
        b, c, l = x.shape

        # 1. 自动计算 padding
        if l < self.segment_len:
            pad_len = self.segment_len - l
        else:
            remainder = (l - self.segment_len) % self.stride
            pad_len = 0 if remainder == 0 else self.stride - remainder

        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=self.padding_value)

        # 2. 滑动窗口拆分
        # 此时的 x 维度: (b, c, l_pad)
        x = x.unfold(dimension=-1, size=self.segment_len, step=self.stride)

        # 3. 获取窗口数 n (x 维度变为 b, c, n, p)
        n = x.shape[2]

        # 验证逻辑：确保 calculate_n 的结果和实际 unfold 的维度一致
        # assert n == self.calculate_n(l)

        x = rearrange(x, 'b c n p -> (b n) c p')
        return x

class LocalProfiling_overlap(nn.Module):
    """ Local Profiling module in ARES """

    def __init__(self, in_channels, mid_channel, overlap_ratio=0):
        super().__init__()
        print(f"重复率：{overlap_ratio}")
        segment_len = 1800
        self.dividing = SlidingWindowSplit(segment_len=segment_len,overlap_ratio=overlap_ratio)

        self.net = nn.Sequential(
            ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=mid_channel, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.squeeze(2)

        is_roll = False# 是否随机滑动

        if self.training and is_roll:
            sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
            x = torch.roll(x, shifts=sliding_size, dims=-1)
        else:
            sliding_size = 0

        x = self.dividing(x)
        x = self.net(x)
        x = Rearrange('(b n) c p -> b c (n p)', b=B)(x)

        x = x.permute(0, 2, 1)
        return x, sliding_size


def compute_out_size(length):
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    return length

# class ExploreModel_Mamba(nn.Module):
#     def __init__(self,
#                  num_classes,
#                  patch_size,
#                  num_tabs,
#                  max_matrix_len=1800,
#                  drop_path_rate=0.1,
#                  embed_dim=256,
#                  depth=4,
#                  early_stage=False,
#                  fine_predict=False):
#         super().__init__()
#
#         self.early_stage = early_stage
#         self.num_tabss = num_tabs
#         self.fine_predict = fine_predict
#         if fine_predict:
#             assert self.num_tabss == 2
#
#         assert num_tabs == 1 or not early_stage
#
#         skip_localfit_flag = True
#
#         if num_tabs == 1 or skip_localfit_flag:
#             assert max_matrix_len % 6 == 0
#             num_patches = max_matrix_len // 6
#         else:
#             assert max_matrix_len % 4 == 0
#             num_patches = compute_out_size(max_matrix_len // 4) * 4
#         self.num_patches = num_patches
#
#         self.patch_embed = PatchEmbed(patch_size=(patch_size, 1), in_chans=1, embed_dim=embed_dim)
#
#         if num_tabs == 1 or skip_localfit_flag:
#             self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
#         else:
#             self.local_model = LocalProfiling(in_channels=embed_dim, mid_channel=embed_dim)
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
#                                       requires_grad=False)
#
#         #self.blocks = nn.ModuleList([MHSA(embed_dim, embed_dim // 4) for i in range(depth)])
#         self.blocks = nn.ModuleList([
#             Mamba2(layer_idx=i, d_model=embed_dim, headdim=embed_dim // 4)
#             for i in range(depth)])
#
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.droppaths = nn.ModuleList([
#             DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
#             for i in range(depth)])
#         self.fc_norm = nn.LayerNorm(embed_dim)
#
#         if fine_predict:
#             self.fc = nn.Linear(embed_dim, num_classes)
#             self.fc_fine = nn.Linear(embed_dim, num_classes + 1)
#         else:
#             self.fc = nn.Linear(embed_dim, num_classes)
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'dist_token'}
#
#     def forward(self, x, idx=None):
#         # embed patches
#         x = self.patch_embed(x)
#
#         # local feature
#         x, sliding_size = self.local_model(x)
#
#         # add pos embed w/o cls token
#         x = x + self.pos_embed[:, 1:, :]
#
#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # apply Transformer blocks
#         for blk, drop in zip(self.blocks, self.droppaths):
#             x = drop(blk(x)) + x
#         x = self.fc_norm(x)
#         x = x[:, 1:, :]
#
#         if self.early_stage and self.training:
#             # early-stage training
#             return self.fc(x)
#         if self.fine_predict:
#             # fine-predict training
#             x0 = self.fc_fine(x)
#
#             x = x.mean(dim=1)
#             x = self.fc(x)
#
#             return x, x0, sliding_size
#
#         if (self.early_stage and not self.training) or (not self.early_stage and not self.fine_predict):
#             if idx is None:
#                 # early-stage inference or normal inference
#                 x = x.mean(dim=1)
#                 x = self.fc(x)
#                 return x
#             else:
#                 aggregate_idx = torch.floor(idx / 6).long()
#                 batch_size = x.size(0)
#
#                 selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
#                 x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
#                 x = self.fc(x)
#                 return x
class ExploreModel_EM1(nn.Module):
    def __init__(self,
                 num_classes,
                 patch_size,
                 num_tabs,
                 max_matrix_len=1800,
                 drop_path_rate=0.1,
                 embed_dim=256,
                 depth=4,
                 early_stage=False,
                 fine_predict=False,
                 overlap_ratio=0,
                 **kwargs):
        super().__init__()

        self.early_stage = early_stage
        self.num_tabss = num_tabs
        self.fine_predict = fine_predict
        if fine_predict:
            assert self.num_tabss == 2

        assert num_tabs == 1 or not early_stage

        skip_localfit_flag = False

        if num_tabs == 1 or skip_localfit_flag:
            assert max_matrix_len % 6 == 0
            num_patches = max_matrix_len // 6
        else:
            assert max_matrix_len % 4 == 0
            num_patches = compute_out_size(max_matrix_len // 4) * 4
        self.num_patches = num_patches

        self.patch_embed = PatchEmbed(patch_size=(patch_size, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or skip_localfit_flag:
            self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
        else:
            self.local_model = LocalProfiling_overlap(in_channels=embed_dim, mid_channel=embed_dim, overlap_ratio=overlap_ratio)
            num_patches = self.local_model.dividing.calculate_n(7200) * compute_out_size(1800)
            print(f"从组后:{num_patches+1}")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        #self.blocks = nn.ModuleList([MHSA(embed_dim, embed_dim // 4) for i in range(depth)])
        # self.blocks = nn.ModuleList([
        #     #Mamba2(layer_idx=i, d_model=embed_dim, headdim=embed_dim // 4)
        #     #nn.Identity()
        #     MHSA(embed_dim, embed_dim // 4, 1, embed_dim*4,
        #          atten_config={'name':'base'}
        #          )
        #     for i in range(depth)])
        self.atten = MHSA(embed_dim, embed_dim // 4, depth, embed_dim*4,
                 atten_config={'name':'Causal'}
                 )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim)

        if fine_predict:
            self.fc = nn.Linear(embed_dim, num_classes)
            self.fc_fine = nn.Linear(embed_dim, num_classes + 1)
        else:
            self.fc = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
    def forward(self, x, idx=None):
        # embed patches
        x = self.patch_embed(x)

        # local feature
        x, sliding_size = self.local_model(x)

        # add pos embed w/o cls token
        x = x #+ self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token# + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # for blk, drop in zip(self.blocks, self.droppaths):
        #     x = drop(blk(x)) + x
        x = self.atten(x, self.pos_embed)
        x = self.fc_norm(x)
        x = x[:, 1:, :]

        if self.early_stage and self.training:
            # early-stage training
            return self.fc(x)
        if self.fine_predict:
            # fine-predict training
            x0 = self.fc_fine(x)

            x = x.mean(dim=1)
            x = self.fc(x)

            return x, x0, sliding_size

        if (self.early_stage and not self.training) or (not self.early_stage and not self.fine_predict):
            if idx is None:
                # early-stage inference or normal inference
                x = x.mean(dim=1)
                x = self.fc(x)
                return x
            else:
                aggregate_idx = torch.floor(idx / 6).long()
                batch_size = x.size(0)
                selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
                x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
                x = self.fc(x)
                return x




class GateNet(nn.Module):
    def __init__(self):
        super(GateNet, self).__init__()
        self.feature_dim = 512
        # RF_model 现在会返回确定的特征维度
        self.model = RF_model(out_feature_dim=self.feature_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.gating_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.model(x)
        x = self.gap(feat)
        x = x.view(x.size(0), -1)
        h = self.gating_fc(x)
        return h


class RF_model(nn.Module):
    def __init__(self, out_feature_dim=512):
        super(RF_model, self).__init__()
        self.first_layer = make_first_layers()

        # 核心修正：make_first_layers 最终输出的是 64 个通道
        # 如果经过 view 转换，我们需要确保 1D 卷积的输入通道匹配
        self.features = make_layers([128, 128, 'M', 256, 256, 'M', 512, out_feature_dim], in_channels=64)

        self._initialize_weights()

    def forward(self, x):
        # 1. 经过 2D 卷积层: input [B, 1, 5, 1800] -> output [B, 64, H', W']
        x = self.first_layer(x)

        # 2. 核心修正：处理维度以适配 1D 卷积
        # 这里的 H' 因为卷积和池化会变小。报错显示你的 H' 此时被 view 压进了通道。
        # 我们使用 mean 或者是 view 重新排列，确保通道数依然是 64
        # 假设我们只关心时间维度上的特征映射：
        x = torch.mean(x, dim=2)  # 将高度维度 H' 压缩，保持通道数为 64, 形状变为 [B, 64, W']

        # 3. 经过 1D 卷积层
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=64):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(3, stride=2, padding=1), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d((1, 3)),
        nn.Dropout(0.1),
        nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),  # 经过这一步，Height 维度通常会变小
        nn.Dropout(0.1)
    )

def get_model(num_classes, num_tabs, patch_size, model_name = "EM1", **kwargs):
    model = eval('ExploreModel_{}'.format(model_name))(num_classes=num_classes, num_tabs=num_tabs, patch_size=patch_size, **kwargs)
    return model

if __name__ == '__main__':
    from torchinfo import summary
    model_name = 'EM1'
    model = get_model(num_classes=95, num_tabs=5, patch_size=5, model_name=model_name, max_matrix_len=7200)
    data = torch.rand((20,1,5,7200))
    model(data)
    summary(model, input_size=(1, 1, 5, 7200), depth=float('5'), col_names=["input_size", "output_size", "num_params", "kernel_size"])


    B, C, T = 20, 5, 1800
    model = GateNet()
    # 打印模型结构验证
    summary(model, input_size=(B, 1, C, T))
