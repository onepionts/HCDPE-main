# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from torch.nn.utils import weight_norm
from torch import Tensor, Size
from typing import Union, List
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import logging

from mmpose.models.builder import BACKBONES
from mmcv.utils import get_logger

from utils.ckpt_load import load_checkpoint_swin


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use `get_logger` method in mmcv to get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmpose".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


_shape_t = Union[int, List[int], Size]


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=None):#norm_layer默认使用 nn.Identity() 作为规范化层。
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNormFP32(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNormFP32, self).__init__(normalized_shape, eps, elementwise_affine)
#前向传播方法，对输入张量应用层归一化。首先，将输入张量转换为浮点型，然后使用 F.layer_norm 函数应用层归一化，并将结果转换回与输入张量相同的数据类型。
    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)



#该类的主要目的是将输入张量、权重和偏置（如果存在）转换为浮点型，并应用线性变换，以确保输出与输入张量具有相同的数据类型。
class LinearFP32(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearFP32, self).__init__(in_features, out_features, bias)
    #：前向传播方法，对输入张量应用线性变换。
    # 首先，将输入张量、权重和偏置（如果存在）转换为浮点型，然后使用 F.linear 函数进行线性变换。
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float(),
                        self.bias.float() if self.bias is not None else None)




#用于实现多层感知机（MLP）模型。实现了一个简单的多层感知机结构，可以用于在深度学习模型中构建基本的前馈神经网络。
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=None, mlpfp32=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlpfp32 = mlpfp32

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features)
        else:
            self.norm = None
#前向传播方法。首先将输入 x 通过第一个全连接层，并通过激活函数和 Dropout 层。如果提供了归一化层，则在此之后应用归一化。
# 然后将结果通过第二个全连接层，再次应用激活函数和 Dropout 层。如果 mlpfp32 为 True，则将权重和偏置转换为浮点型进行计算。
    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        if self.mlpfp32:
            x = self.fc2.float()(x.type(torch.float32))
            x = self.drop.float()(x)
            # print(f"======>[MLP FP32]")
        else:
            x = self.fc2(x)
            x = self.drop(x)
        return x

#用于实现结合了卷积和多层感知机（MLP）的模型。
class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=None, mlpfp32=False, proj_ln=False):
        super().__init__()
        self.mlp = Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features,
                       act_layer=act_layer, drop=drop, norm_layer=norm_layer, mlpfp32=mlpfp32)
        self.conv_proj = nn.Conv2d(in_features,
                                   in_features,
                                   kernel_size=3,
                                   padding=1,
                                   stride=1,
                                   bias=False,
                                   groups=in_features)
        self.proj_ln = LayerNorm2D(in_features, LayerNormFP32) if proj_ln else None
    #前向传播方法。首先，将输入 x 重塑为形状为 (B, H, W, C) 的张量，并将通道维度移到第二个位置。
    # 然后，通过卷积层 conv_proj 对输入进行卷积。如果设置了 proj_ln，则在卷积输出上应用 LayerNorm2D 层。
    # 接下来，将结果重塑回形状 (B, L, C)。最后，通过 MLP 模块 mlp 处理重塑后的输入，返回处理后的结果。
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
        x = self.conv_proj(x)
        if self.proj_ln:
            x = self.proj_ln(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = x.reshape(B, L, C)
        x = self.mlp(x, H, W)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    #首先，根据输入张量的形状，将其重塑为 (B, H // window_size, window_size, W // window_size, window_size, C) 的形状，即将高度和宽度分别划分为 H // window_size 和 W // window_size 个子窗口，每个子窗口的大小为 window_size x window_size。
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    #对重塑后的张量进行维度置换操作，将维度顺序调整为 (B, num_windows_H, num_windows_W, window_size, window_size, C)，其中 num_windows_H 和 num_windows_W 分别表示在高度和宽度方向上的子窗口数量。
   #将维度展平，得到形状为 (num_windows*B, window_size, window_size, C) 的输出张量，即划分后的窗口。
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# 实现过程：
#
# 首先，根据输入的窗口张量的形状，计算出批大小 B。
# 然后，将窗口张量重新重塑为 (B, H // window_size, W // window_size, window_size, window_size, C) 的形状，即将窗口张量的维度重新组织，恢复出每个窗口的形状。
# 接着，对重塑后的张量进行维度置换操作，将维度顺序调整为 (B, H // window_size, window_size, W // window_size, window_size, C)。
# 将维度展平，得到形状为 (B, H, W, C) 的输出张量，即恢复后的图像张量。

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#这是一个窗口注意力机制模块，用于处理长序列输入
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlpfp32=False, pretrain_window_size=-1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.mlpfp32 = mlpfp32
        self.attn_type = attn_type
        self.rpe_output_type = rpe_output_type
        self.relative_coords_table_type = relative_coords_table_type

        if self.attn_type == 'cosine_mh':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        elif self.attn_type == 'normal':
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5
        else:
            raise NotImplementedError()
        if self.relative_coords_table_type != "none":
            # mlp to generate table of relative position bias
            self.rpe_mlp = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                         nn.ReLU(inplace=True),
                                         LinearFP32(rpe_hidden_dim, num_heads, bias=False))

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = torch.stack(
                torch.meshgrid([relative_coords_h,
                                relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
            if relative_coords_table_type == 'linear':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            elif relative_coords_table_type == 'linear_bylayer':
                print(f"norm8_log_bylayer: [{self.window_size}] ==> [{pretrain_window_size}]")
                relative_coords_table[:, :, :, 0] /= (pretrain_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrain_window_size - 1)
            elif relative_coords_table_type == 'norm8_log':
                relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
                relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_192to640':
                if self.window_size[0] == 40:
                    relative_coords_table[:, :, :, 0] /= (11)
                    relative_coords_table[:, :, :, 1] /= (11)
                elif self.window_size[0] == 20:
                    relative_coords_table[:, :, :, 0] /= (5)
                    relative_coords_table[:, :, :, 1] /= (5)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            # check
            elif relative_coords_table_type == 'norm8_log_256to640':
                if self.window_size[0] == 40:
                    relative_coords_table[:, :, :, 0] /= (15)
                    relative_coords_table[:, :, :, 1] /= (15)
                elif self.window_size[0] == 20:
                    relative_coords_table[:, :, :, 0] /= (7)
                    relative_coords_table[:, :, :, 1] /= (7)
                else:
                    raise NotImplementedError
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            elif relative_coords_table_type == 'norm8_log_bylayer':
                print(f"norm8_log_bylayer: [{self.window_size}] ==> [{pretrain_window_size}]")
                relative_coords_table[:, :, :, 0] /= (pretrain_window_size - 1)
                relative_coords_table[:, :, :, 1] /= (pretrain_window_size - 1)
                relative_coords_table *= 8  # normalize to -8, 8
                relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
            else:
                raise NotImplementedError
            self.register_buffer("relative_coords_table", relative_coords_table)
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if self.attn_type == 'cosine_mh':
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            #print(torch.tensor(1. / 0.01).device)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).cuda())).exp()
            attn = (q @ k.transpose(-2, -1)) * logit_scale.float()
        elif self.attn_type == 'normal':
            q = q * self.scale
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            raise NotImplementedError()

        if self.relative_coords_table_type != "none":
            # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
            relative_position_bias_table = self.rpe_mlp(self.relative_coords_table).view(-1, self.num_heads)
        else:
            relative_position_bias_table = self.relative_position_bias_table
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        if self.rpe_output_type == 'normal':
            pass
        elif self.rpe_output_type == 'sigmoid':
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        else:
            raise NotImplementedError

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = attn.type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        if self.mlpfp32:
            x = self.proj.float()(x.type(torch.float32))
            x = self.proj_drop.float()(x)
            # print(f"======>[ATTN FP32]")
        else:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
# 这是`WindowAttention`类的前向传播方法和额外表示方法。
#
# - **前向传播方法 (`forward`)：**
#   - 接收输入张量 `x`，以及可选的掩码张量 `mask`。
#   - 首先，根据查询、键、值的线性映射计算注意力矩阵。
#   - 如果采用余弦多头注意力 (`cosine_mh`)，则对查询和键进行归一化，并根据缩放因子缩放注意力矩阵。
#   - 如果采用普通多头注意力 (`normal`)，则对查询进行缩放，并计算注意力矩阵。
#   - 根据相对位置编码表计算相对位置偏置，并将其添加到注意力矩阵中。
#   - 如果存在掩码，则将其应用于注意力矩阵。
#   - 对注意力矩阵进行 softmax 操作，然后应用 dropout。
#   - 将注意力矩阵乘以值张量，并进行线性投影。
#   - 如果 `mlpfp32` 为 True，则将输出转换为 32 位浮点型并应用 dropout。
#   - 返回处理后的张量。
#
# - **额外表示方法 (`extra_repr`)：**
#   - 返回类的额外表示，包括维度 (`dim`)、窗口大小 (`window_size`) 和注意力头数 (`num_heads`)。
#
# - **FLOPs 方法 (`flops`)：**
#   - 计算模型的浮点运算数 (FLOPs)，用于评估模型的计算成本。

class SwinTransformerBlockPost(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 use_mlp_norm=False, endnorm=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlp_type='normal', mlpfp32=False,
                 pretrain_window_size=-1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type, rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim, attn_type=attn_type, mlpfp32=mlpfp32,
            pretrain_window_size=pretrain_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if mlp_type == 'normal':
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv_ln':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32, proj_ln=True)

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x

        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.norm1.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.mlp(x, H, W)
        if self.mlpfp32:
            x = self.norm2.float()(x)
            x = x.type(orig_type)
        else:
            x = self.norm2(x)
        x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x
#它表示Swin Transformer模块，用于处理输入特征。。
# 这是一个名为`SwinTransformerBlockPost`的类，它表示Swin Transformer模块。
#
# - **参数：**
#   - `dim`：输入通道数。
#   - `num_heads`：注意力头数。
#   - `window_size`：窗口大小。
#   - `shift_size`：用于SW-MSA的移位大小。
#   - `mlp_ratio`：MLP隐藏层维度与嵌入维度的比率。
#   - `qkv_bias`：是否对查询、键、值添加可学习的偏置。
#   - `qk_scale`：如果设置，则覆盖默认的qk缩放。
#   - `drop`：Dropout率。
#   - `attn_drop`：注意力权重的dropout率。
#   - `drop_path`：随机深度率。
#   - `act_layer`：激活函数。
#   - `norm_layer`：规范化层。
#   - `relative_coords_table_type`：相对位置编码表类型。
#   - `rpe_hidden_dim`：RPE的隐藏层维度。
#   - `rpe_output_type`：RPE输出类型。
#   - `attn_type`：注意力类型。
#   - `mlp_type`：MLP类型。
#   - `mlpfp32`：MLP是否使用32位浮点精度。
#   - `pretrain_window_size`：预训练窗口大小。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化模块的各个参数，并创建所需的子模块，包括注意力模块、DropPath模块和MLP模块。
#
# - **前向传播方法 (`forward`)：**
#   - 对输入特征进行归一化，并通过注意力模块处理特征。
#   - 根据窗口大小和移位大小对特征进行分块和重排。
#   - 将分块的特征传递给注意力模块，得到处理后的特征。
#   - 将处理后的特征通过MLP模块，得到最终的输出特征。
#   - 如果设置了`endnorm`参数，则对输出特征进行归一化。
#
# 该类实现了一个Swin Transformer块，用于处理输入特征。

class SwinTransformerBlockPre(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 use_mlp_norm=False, endnorm=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 init_values=None, relative_coords_table_type='norm8_log', rpe_hidden_dim=512,
                 rpe_output_type='normal', attn_type='normal', mlp_type='normal', mlpfp32=False,
                 pretrain_window_size=-1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        self.endnorm = endnorm
        self.mlpfp32 = mlpfp32
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            relative_coords_table_type=relative_coords_table_type, rpe_output_type=rpe_output_type,
            rpe_hidden_dim=rpe_hidden_dim, attn_type=attn_type, mlpfp32=mlpfp32,
            pretrain_window_size=pretrain_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if mlp_type == 'normal':
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32)
        elif mlp_type == 'conv_ln':
            self.mlp = ConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                               norm_layer=norm_layer if self.use_mlp_norm else None, mlpfp32=mlpfp32, proj_ln=True)

        if init_values is not None and init_values >= 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = 1.0, 1.0

        if self.endnorm:
            self.enorm = norm_layer(dim)
        else:
            self.enorm = None

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        orig_type = x.dtype  # attn may force to fp32
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        if self.mlpfp32:
            x = self.gamma_1 * x
            x = x.type(orig_type)
        else:
            x = self.gamma_1 * x
        x = shortcut + self.drop_path(x)
        shortcut = x

        orig_type = x.dtype
        x = self.norm2(x)
        if self.mlpfp32:
            x = self.gamma_2 * self.mlp(x, H, W)
            x = x.type(orig_type)
        else:
            x = self.gamma_2 * self.mlp(x, H, W)
        x = shortcut + self.drop_path(x)

        if self.endnorm:
            x = self.enorm(x)

        return x
#这是一个名为`SwinTransformerBlockPre`的类，它表示Swin Transformer模块。

# - **参数：**
#   - `dim`：输入通道数。
#   - `num_heads`：注意力头数。
#   - `window_size`：窗口大小。
#   - `shift_size`：用于SW-MSA的移位大小。
#   - `mlp_ratio`：MLP隐藏层维度与嵌入维度的比率。
#   - `qkv_bias`：是否对查询、键、值添加可学习的偏置。
#   - `qk_scale`：如果设置，则覆盖默认的qk缩放。
#   - `drop`：Dropout率。
#   - `attn_drop`：注意力权重的dropout率。
#   - `drop_path`：随机深度率。
#   - `act_layer`：激活函数。
#   - `norm_layer`：规范化层。
#   - `init_values`：初始化值。
#   - `relative_coords_table_type`：相对位置编码表类型。
#   - `rpe_hidden_dim`：RPE的隐藏层维度。
#   - `rpe_output_type`：RPE输出类型。
#   - `attn_type`：注意力类型。
#   - `mlp_type`：MLP类型。
#   - `mlpfp32`：MLP是否使用32位浮点精度。
#   - `pretrain_window_size`：预训练窗口大小。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化模块的各个参数，并创建所需的子模块，包括注意力模块、DropPath模块和MLP模块。
#   - 可以通过`init_values`参数指定初始化值。
#
# - **前向传播方法 (`forward`)：**
#   - 对输入特征进行归一化，并通过注意力模块处理特征。
#   - 根据窗口大小和移位大小对特征进行分块和重排。
#   - 将分块的特征传递给注意力模块，得到处理后的特征。
#   - 将处理后的特征通过MLP模块，得到最终的输出特征。
#   - 如果设置了`endnorm`参数，则对输出特征进行归一化。
#
# 该类实现了一个Swin Transformer块，用于处理输入特征。

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x
# 用于实现图像块合并操作。
# 这是一个名为`PatchMerging`的类，用于实现图像块合并操作。
#
# - **参数：**
#   - `dim`：输入通道数。
#   - `norm_layer`：规范化层。
#   - `postnorm`：是否在合并之前进行规范化。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化模块的参数，并创建线性层和规范化层。
#
# - **前向传播方法 (`forward`)：**
#   - 接收输入特征张量 `x` 和空间分辨率 `H` 和 `W`。
#   - 将输入特征张量 `x` 重塑为形状 `(B, H, W, C)`，其中 `B` 是批量大小， `H` 和 `W` 是空间维度， `C` 是通道数。
#   - 如果输入特征的高度或宽度为奇数，则在需要的维度上进行填充，以确保能够进行合并操作。
#   - 将输入特征沿高度和宽度方向按步长为2进行切片，得到4个子图像 `x0`、`x1`、`x2` 和 `x3`。
#   - 将这4个子图像沿通道维度连接起来，得到形状 `(B, H/2, W/2, 4*C)` 的张量 `x`。
#   - 将 `x` 重塑为形状 `(B, H/2*W/2, 4*C)`。
# #   - 如果设置了 `postnorm` 参数，则先将 `x` 经过线性层和规范化层处理；否则，先进行规范化，然后再经过线性层处理。
# #   - 返回处理后的特征张量 `x`。
#
# 该类实现了图像块合并操作，用于将输入特征的每4个相邻块合并为一个块。

class PatchReduction1C(nn.Module):
    r""" Patch Reduction Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)

        return x
# 这是一个名为 `PatchReduction1C` 的类，用于实现输入通道数减少的操作。
#
# - **参数：**
#   - `dim`：输入通道数。
#   - `norm_layer`：规范化层。
#   - `postnorm`：是否在操作之后进行规范化。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化模块的参数，并创建线性层和规范化层。
#
# - **前向传播方法 (`forward`)：**
#   - 接收输入特征张量 `x`、高度 `H` 和宽度 `W`。
#   - 如果设置了 `postnorm` 参数，则先将 `x` 经过线性层和规范化层处理；否则，先进行规范化，然后再经过线性层处理。
#   - 返回处理后的特征张量 `x`。
#
# 该类实现了输入通道数减少的操作，通常用于减少特征图的通道数。

class ConvPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(2 * dim) if postnorm else norm_layer(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        if self.postnorm:
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x).flatten(2).transpose(1, 2)  # B H//2*W//2 2*C

        return x
# 这是一个名为 `ConvPatchMerging` 的类，用于通过卷积操作合并特征图中的相邻像素。
#
# - **参数：**
#   - `dim`：输入通道数。
#   - `norm_layer`：规范化层。
#   - `postnorm`：是否在操作之后进行规范化。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化模块的参数，并创建卷积层和规范化层。
#
# - **前向传播方法 (`forward`)：**
#   - 接收输入特征张量 `x`、高度 `H` 和宽度 `W`。
#   - 将输入特征张量 `x` 格式化为 `(B, H, W, C)` 的形状。
#   - 根据是否需要后置规范化，选择不同的处理顺序：
#     - 如果需要后置规范化，则先将输入特征张量转换为 `(B, C, H, W)` 的形状，然后经过卷积操作、展平和转置处理，最后进行规范化。
#     - 如果不需要后置规范化，则先进行规范化，然后将输入特征张量转换为 `(B, C, H, W)` 的形状，经过卷积操作、展平和转置处理。
#   - 返回处理后的特征张量 `x`。
#
# 该类通过卷积操作将输入特征图的高度和宽度减半，同时将通道数增加一倍。

class BasicLayer(nn.Module):#将Block和pathc Merging封装到basiclayer中。

    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        use_shift (bool): Whether to use shifted window. Default: True.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 checkpoint_blocks=255,
                 init_values=None,
                 endnorm_interval=-1,
                 use_mlp_norm=False,
                 use_shift=True,
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 rpe_output_type='normal',
                 attn_type='normal',
                 mlp_type='normal',
                 mlpfp32_blocks=[-1],
                 postnorm=True,
                 pretrain_window_size=-1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.checkpoint_blocks = checkpoint_blocks
        self.init_values = init_values if init_values is not None else 0.0
        self.endnorm_interval = endnorm_interval
        self.mlpfp32_blocks = mlpfp32_blocks
        self.postnorm = postnorm

        # build blocks
        if self.postnorm:
            self.blocks = nn.ModuleList([
                SwinTransformerBlockPost(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) or (not use_shift) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_mlp_norm=use_mlp_norm,
                    endnorm=True if ((i + 1) % endnorm_interval == 0) and (
                            endnorm_interval > 0) else False,
                    relative_coords_table_type=relative_coords_table_type,
                    rpe_hidden_dim=rpe_hidden_dim,
                    rpe_output_type=rpe_output_type,
                    attn_type=attn_type,
                    mlp_type=mlp_type,
                    mlpfp32=True if i in mlpfp32_blocks else False,
                    pretrain_window_size=pretrain_window_size)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                SwinTransformerBlockPre(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) or (not use_shift) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_mlp_norm=use_mlp_norm,
                    endnorm=True if ((i + 1) % endnorm_interval == 0) and (
                            endnorm_interval > 0) else False,
                    relative_coords_table_type=relative_coords_table_type,
                    rpe_hidden_dim=rpe_hidden_dim,
                    rpe_output_type=rpe_output_type,
                    attn_type=attn_type,
                    mlp_type=mlp_type,
                    mlpfp32=True if i in mlpfp32_blocks else False,
                    pretrain_window_size=pretrain_window_size)
                for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, postnorm=postnorm)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for idx, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if isinstance(self.downsample, PatchReduction1C):
                return x, H, W, x_down, H, W
            else:
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
                return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

    def _init_block_norm_weights(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, self.init_values)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, self.init_values)
# 这是一个名为 `BasicLayer` 的类，它实现了基本的 Swin Transformer 层。
#
# - **参数：**
#   - `dim`：特征通道数。
#   - `depth`：层的深度。
#   - `num_heads`：注意力头的数量。
#   - `window_size`：局部窗口大小，默认为 7。
#   - `mlp_ratio`：MLP 隐藏维度与嵌入维度的比率，默认为 4。
#   - `qkv_bias`：是否为 query、key 和 value 添加可学习的偏置。
#   - `qk_scale`：query 和 key 的缩放因子。
#   - `drop`：Dropout 率。
#   - `attn_drop`：注意力 Dropout 率。
#   - `drop_path`：随机深度率。
#   - `norm_layer`：规范化层。
#   - `downsample`：下采样层，在层末端对特征进行下采样。
#   - `use_checkpoint`：是否使用检查点以节省内存，默认为 False。
#   - `checkpoint_blocks`：检查点的块数。
#   - `init_values`：初始化值，默认为 0.0。
#   - `endnorm_interval`：是否在特定层数后使用 LayerNorm。
#   - `use_mlp_norm`：是否在 MLP 中使用规范化。
#   - `use_shift`：是否使用移位窗口。
#   - `relative_coords_table_type`：相对坐标表的类型。
#   - `rpe_hidden_dim`：相对位置编码隐藏维度。
#   - `rpe_output_type`：相对位置编码的输出类型。
#   - `attn_type`：注意力类型。
#   - `mlp_type`：MLP 类型。
#   - `mlpfp32_blocks`：MLP 是否使用 FP32 的块索引。
#   - `postnorm`：是否在每个块之后进行规范化。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化各个参数，并创建多个 Swin Transformer 块。
#
# - **前向传播方法 (`forward`)：**
#   - 计算用于 SW-MSA 的注意力掩码。
#   - 对每个块进行前向传播，根据需要使用检查点。
#   - 如果有下采样层，对下采样层进行前向传播并返回结果。
#
# - **辅助方法 (`_init_block_norm_weights`)：**
#   - 初始化块中的规范化层权重。
#
# 该类用于构建 Swin Transformer 模型的基本层，每个基本层由多个 Swin Transformer 块组成。

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x
#
# 这是一个名为 `PatchEmbed` 的类，它用于将图像转换为补丁嵌入。
#
# - **参数：**
#   - `patch_size`：补丁的大小。
#   - `in_chans`：输入图像的通道数。
#   - `embed_dim`：线性投影输出通道数。
#   - `norm_layer`：规范化层。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化各个参数，并创建一个卷积层用于投影。
#   - 如果提供了规范化层，则初始化规范化层。
#
# - **前向传播方法 (`forward`)：**
#   - 对输入图像进行填充，以使其能够被完全分割成大小为 `patch_size` 的补丁。
#   - 将填充后的图像通过投影卷积层进行投影。
#   - 如果提供了规范化层，则对投影结果进行规范化并返回。
#
# 该类用于将输入图像转换为补丁表示，以供后续的 Swin Transformer 模型处理。
class ResNetDLNPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(4)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = nn.Sequential(nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False),
                                   LayerNorm2D(64, norm_layer),
                                   nn.GELU(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                                   LayerNorm2D(64, norm_layer),
                                   nn.GELU(),
                                   nn.Conv2d(64, embed_dim, 3, stride=1, padding=1, bias=False))
        self.norm = LayerNorm2D(embed_dim, norm_layer if norm_layer is not None else LayerNormFP32)  # use ln always
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.maxpool(x)
        # x = x.flatten(2).transpose(1, 2)
        return x
#
# 这是一个名为 `ResNetDLNPatchEmbed` 的类，它是一个基于 ResNet 架构的补丁嵌入层，结合了深度可分离卷积 (Depthwise Separable Convolution) 和 Layer Normalization。
#
# - **参数：**
#   - `in_chans`：输入图像的通道数。
#   - `embed_dim`：嵌入维度。
#   - `norm_layer`：规范化层。
#
# - **初始化方法 (`__init__`)：**
#   - 创建一系列卷积层和规范化层，其中第一个卷积层使用 3x3 的卷积核，stride 为 2，用于对输入图像进行下采样。
#   - 使用 Layer Normalization 对卷积结果进行规范化。
#   - 使用 GELU 作为激活函数。
#   - 第二个卷积层保持输入大小不变，继续对图像特征进行卷积。
#   - 最后一个卷积层将特征映射到嵌入维度。
#   - 创建一个额外的 Layer Normalization 层，对整个嵌入维度进行规范化。
#   - 使用 GELU 作为激活函数。
#   - 添加一个最大池化层，用于进一步降低特征图的大小。
#
# - **前向传播方法 (`forward`)：**
#   - 对输入图像进行填充，以使其能够被完全分割成大小为 `patch_size` 的补丁。
#   - 将填充后的图像通过一系列卷积层和规范化层进行特征提取和处理。
#   - 最后通过最大池化层对特征图进行池化以减小尺寸。
@BACKBONES.register_module()
class SwinV2TransformerRPE2FC(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        use_shift (bool): Whether to use shifted window. Default: True.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,#线性投影输出通道数
                 depths=[2, 2, 6, 2],#每个 Swin Transformer 阶段的深度。
                 num_heads=[3, 6, 12, 24],
                 window_size=7,#窗口大小。默认为7。
                 mlp_ratio=4.,#mlp 隐藏维度与嵌入维度的比率。默认为4。
                 qkv_bias=True,#如果为 True，将查询、键、值添加可学习的偏置。默认为 True。
                 qk_scale=None,
                 drop_rate=0.,#Dropout 比率。
                 attn_drop_rate=0.,#注意力 Dropout 比率
                 drop_path_rate=0.1,#随机深度比率。默认为0.2。
                 norm_layer=partial(LayerNormFP32, eps=1e-6),#norm_layer=nn.LayerNorm
                 ape=False,# 如果为 True，将绝对位置嵌入添加到补丁嵌入中
                 patch_norm=True,#如果为 True，在补丁嵌入后添加归一化。
                 use_checkpoint=False,
                 init_values=1e-5,
                 endnorm_interval=-1,
                 use_mlp_norm_layers=[],
                 relative_coords_table_type='norm8_log',
                 rpe_hidden_dim=512,
                 attn_type='cosine_mh',
                 rpe_output_type='sigmoid',
                 rpe_wd=False,
                 postnorm=True,
                 mlp_type='normal',
                 patch_embed_type='normal',
                 patch_merge_type='normal',
                 strid16=False,
                 checkpoint_blocks=[255, 255, 255, 255],
                 mlpfp32_layer_blocks=[[-1], [-1], [-1], [-1]],
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_shift=True,
                 rpe_interpolation='geo',
                 pretrain_window_size=[-1, -1, -1, -1],
                 **kwargs):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.rpe_interpolation = rpe_interpolation
        self.mlp_ratio = mlp_ratio
        self.endnorm_interval = endnorm_interval
        self.use_mlp_norm_layers = use_mlp_norm_layers
        self.relative_coords_table_type = relative_coords_table_type
        self.rpe_hidden_dim = rpe_hidden_dim
        self.rpe_output_type = rpe_output_type
        self.rpe_wd = rpe_wd
        self.attn_type = attn_type
        self.postnorm = postnorm
        self.mlp_type = mlp_type
        self.strid16 = strid16

        if isinstance(window_size, list):
            pass
        elif isinstance(window_size, int):
            window_size = [window_size] * self.num_layers
        else:
            raise TypeError("We only support list or int for window size")

        if isinstance(use_shift, list):
            pass
        elif isinstance(use_shift, bool):
            use_shift = [use_shift] * self.num_layers
        else:
            raise TypeError("We only support list or bool for use_shift")

        if isinstance(use_checkpoint, list):
            pass
        elif isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint] * self.num_layers
        else:
            raise TypeError("We only support list or bool for use_checkpoint")

        # split image into non-overlapping patches
        if patch_embed_type == 'normal':
            self.patch_embed = PatchEmbed(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        elif patch_embed_type == 'resnetdln':
            assert patch_size == 4, "check"
            self.patch_embed = ResNetDLNPatchEmbed(
                in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        elif patch_embed_type == 'resnetdnf':
            assert patch_size == 4, "check"
            self.patch_embed = ResNetDLNPatchEmbed(
                in_chans=in_chans, embed_dim=embed_dim, norm_layer=None)
        else:
            raise NotImplementedError()
        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        if patch_merge_type == 'normal':
            downsample_layer = PatchMerging
        elif patch_merge_type == 'conv':
            downsample_layer = ConvPatchMerging
        else:
            raise NotImplementedError()
        # build layers
        self.layers = nn.ModuleList()
        num_features = []
        for i_layer in range(self.num_layers):
            cur_dim = int(embed_dim * 2 ** (i_layer - 1)) \
                if (i_layer == self.num_layers - 1 and strid16) else \
                int(embed_dim * 2 ** i_layer)
            num_features.append(cur_dim)
            if i_layer < self.num_layers - 2:
                cur_downsample_layer = downsample_layer
            elif i_layer == self.num_layers - 2:
                if strid16:
                    cur_downsample_layer = PatchReduction1C
                else:
                    cur_downsample_layer = downsample_layer
            else:
                cur_downsample_layer = None
            layer = BasicLayer(
                dim=cur_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=cur_downsample_layer,
                use_checkpoint=use_checkpoint[i_layer],
                checkpoint_blocks=checkpoint_blocks[i_layer],
                init_values=init_values,
                endnorm_interval=endnorm_interval,
                use_mlp_norm=True if i_layer in use_mlp_norm_layers else False,
                use_shift=use_shift[i_layer],
                relative_coords_table_type=self.relative_coords_table_type,
                rpe_hidden_dim=self.rpe_hidden_dim,
                rpe_output_type=self.rpe_output_type,
                attn_type=self.attn_type,
                mlp_type=self.mlp_type,
                mlpfp32_blocks=mlpfp32_layer_blocks[i_layer],
                postnorm=self.postnorm,
                pretrain_window_size=pretrain_window_size[i_layer]
            )
            self.layers.append(layer)

        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.norm3.eval()
        for param in self.norm3.parameters():
            param.requires_grad = False

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        for bly in self.layers:
            bly._init_block_norm_weights()

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint_swin(self, pretrained, strict=False, map_location='cpu',
                                 logger=logger, rpe_interpolation=self.rpe_interpolation)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
            
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer.float()(x_out.float())

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinV2TransformerRPE2FC, self).train(mode)
        self._freeze_stages()
# `SwinV2TransformerRPE2FC` 是一个用于图像分类任务的 Swin Transformer 模型，该模型通过堆叠多个基本层 (BasicLayer) 来构建一个完整的 Transformer 结构。
# 
# - **参数：**
#   - `pretrain_img_size`：预训练模型输入图像的大小，默认为 224。
#   - `patch_size`：补丁大小。
#   - `in_chans`：输入图像的通道数。
#   - `embed_dim`：线性投影输出通道数。
#   - `depths`：每个 Swin Transformer 阶段的深度。
#   - `num_heads`：每个阶段的注意力头数。
#   - `window_size`：窗口大小。
#   - `mlp_ratio`：MLP 隐藏维度与嵌入维度之比。
#   - `qkv_bias`：是否为查询、键和值添加可学习的偏置。
#   - `qk_scale`：如果设置，则覆盖默认的 qk 缩放比例。
#   - `drop_rate`：Dropout 比率。
#   - `attn_drop_rate`：注意力 dropout 比率。
#   - `drop_path_rate`：随机深度率。
#   - `norm_layer`：规范化层。
#   - `ape`：是否将绝对位置嵌入添加到补丁嵌入中。
#   - `patch_norm`：是否在补丁嵌入后添加规范化。
#   - `out_indices`：输出的阶段索引。
#   - `frozen_stages`：要冻结的阶段。
#   - `use_checkpoint`：是否使用检查点以节省内存。
#   - `use_shift`：是否使用移位窗口。
#   - 其他参数包括用于各种组件的参数，如相对位置编码 (RPE)、注意力类型、MLP 类型等。
#
# - **初始化方法 (`__init__`)：**
#   - 初始化各个组件，包括补丁嵌入层、绝对位置嵌入、位置丢弃层等。
#   - 根据配置构建多个基本层，并将它们存储在 `layers` 中。
#   - 为每个输出阶段添加规范化层。
#
# - **前向传播方法 (`forward`)：**
#   - 通过补丁嵌入层将输入图像转换为补丁表示。
#   - 如果启用了绝对位置嵌入，则将其添加到补丁表示中。
#   - 对补丁表示进行位置丢弃，并将其传递给每个基本层。
#   - 将每个输出阶段的结果添加到输出列表中并返回。
#
# - **冻结方法 (`_freeze_stages`)：**
#   - 根据 `frozen_stages` 参数冻结模型的部分参数。
#
# - **训练方法 (`train`)：**
#   - 覆盖训练方法，确保在训练模式下也会冻结指定的层参数。