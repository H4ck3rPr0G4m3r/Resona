"""VITS relative-position transformer encoder (layout compatible with RVC checkpoints)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm over the channel dim of a [B, C, T] tensor."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels, self.eps = channels, eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = F.layer_norm(x.transpose(1, -1), (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int, n_heads: int, window_size: int = 10):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads, self.window_size = n_heads, window_size
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        rel = self.k_channels**-0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel)
        self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel)

    def forward(self, x, attn_mask):
        q, k, v = self.conv_q(x), self.conv_k(x), self.conv_v(x)
        b, d, t = k.shape
        h, kc = self.n_heads, self.k_channels
        q = q.view(b, h, kc, t).transpose(2, 3) / math.sqrt(kc)
        k = k.view(b, h, kc, t).transpose(2, 3)
        v = v.view(b, h, kc, t).transpose(2, 3)
        scores = torch.matmul(q, k.transpose(-2, -1))
        rel_k = self._relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(q, rel_k.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._rel_to_abs(rel_logits)
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        p = F.softmax(scores, dim=-1)
        out = torch.matmul(p, v)
        rel_v = self._relative_embeddings(self.emb_rel_v, t)
        out = out + torch.matmul(self._abs_to_rel(p), rel_v.unsqueeze(0))
        out = out.transpose(2, 3).contiguous().view(b, d, t)
        return self.conv_o(out)

    def _relative_embeddings(self, emb, length: int):
        w = self.window_size
        pad = max(length - (w + 1), 0)
        start = max((w + 1) - length, 0)
        if pad > 0:
            emb = F.pad(emb, [0, 0, pad, pad, 0, 0])
        return emb[:, start : start + 2 * length - 1]

    @staticmethod
    def _rel_to_abs(x):
        b, h, l, _ = x.shape
        x = F.pad(x, [0, 1])
        x = x.view(b, h, l * 2 * l)
        x = F.pad(x, [0, l - 1])
        return x.view(b, h, l + 1, 2 * l - 1)[:, :, :l, l - 1 :]

    @staticmethod
    def _abs_to_rel(x):
        b, h, l, _ = x.shape
        x = F.pad(x, [0, l - 1])
        x = x.view(b, h, l * l + l * (l - 1))
        x = F.pad(x, [l, 0])
        return x.view(b, h, l, 2 * l)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, filter_ch: int, kernel_size: int):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_ch, filter_ch, kernel_size)
        self.conv_2 = nn.Conv1d(filter_ch, out_ch, kernel_size)
        self.pad = ((kernel_size - 1) // 2, kernel_size // 2)

    def forward(self, x, x_mask):
        x = torch.relu(self.conv_1(F.pad(x * x_mask, self.pad)))
        x = self.conv_2(F.pad(x * x_mask, self.pad))
        return x * x_mask


class Encoder(nn.Module):
    def __init__(self, hidden: int, filter_ch: int, n_heads: int, n_layers: int, kernel_size: int, window_size=10):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden, hidden, n_heads, window_size))
            self.norm_layers_1.append(LayerNorm(hidden))
            self.ffn_layers.append(FFN(hidden, hidden, filter_ch, kernel_size))
            self.norm_layers_2.append(LayerNorm(hidden))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, n1, ffn, n2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            x = n1(x + attn(x, attn_mask))
            x = n2(x + ffn(x, x_mask))
        return x * x_mask
