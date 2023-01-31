
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


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


class SwinformerEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        self.window_size = 7

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_windows(self, hidden_states):
        #hidden_states
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()

        B, A, L, C = hidden_states.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        hidden_states = hidden_states.view(B, A, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = hidden_states.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, self.window_size, self.window_size, C)

        return windows

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_windows(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # Apply sequence reduction
            hidden_states = self.sr(hidden_states)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_windows(self.key(hidden_states))
        value_layer = self.transpose_for_windows(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class WMSA_reduction(nn.Module):
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

    def __init__(self, dim, num_heads, sequence_reduction_ratio, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, height, width, output_attentions=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = x.shape
            # Reshape to (batch_size, num_channels, height, width)
            x = x.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # Apply sequence reduction
            x = self.sr(x)
            # Reshape back to (batch_size, seq_len, num_channels)
            x = x.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            x = self.layer_norm(x)

            batch_size, seq_len, num_channels = x.shape
            qkv_ = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, num_channels // self.num_heads).permute(2, 0, 3, 1, 4)
            _, k, v = qkv_[0], qkv_[1], qkv_[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        outputs = (x, attn) if output_attentions else (x,)

        return outputs

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

class WindowAttention_reduction(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    # def __init__(self, config, hidden_size, input_resolution, num_attention_heads, sequence_reduction_ratio):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):

    # def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
    #              mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
    #              act_layer=nn.GELU, norm_layer=nn.LayerNorm,
    #              fused_window_process=False):
        super().__init__()
        self.dim = hidden_size
        # self.input_resolution = input_resolution
        self.num_heads = num_attention_heads
        self.window_size = 7

        # self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            # self.window_size = min(self.input_resolution)

        self.attn = WMSA_reduction(
            hidden_size, num_heads=num_attention_heads, sequence_reduction_ratio=sequence_reduction_ratio,
            qkv_bias=True, qk_scale=None, attn_drop=config.attention_probs_dropout_prob, proj_drop=config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width, output_attentions=False):
        # H, W = self.input_resolution
        H, W = (height, width)

        B, L, C = hidden_states.shape
        assert L == H * W, "input feature has wrong size"

        x = hidden_states

        x = x.view(B, H, W, C)
        
        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA
        attn_windows = self.attn(x_windows, self.window_size, self.window_size, output_attentions)  # nW*B, window_size*window_size, C

        # merge windows
        windows_attn = attn_windows[0].view(-1, self.window_size, self.window_size, C)

        # reverse partition
        attention_output = window_reverse(windows_attn, self.window_size, H, W)  # B H' W' C

        attention_output = attention_output.view(B, H * W, C)
        
        if output_attentions:
            output_attn = attn_windows[1:].view(-1, self.window_size, self.window_size, C)
            output_attn = window_reverse(output_attn, self.window_size, H, W)  # B H' W' C
            # TODO: output_attn reshape 해줘야 할 것 같음.
            outputs = (attention_output,) + output_attn  # add attentions if we output them
        else:
            outputs = (attention_output,)

        return outputs