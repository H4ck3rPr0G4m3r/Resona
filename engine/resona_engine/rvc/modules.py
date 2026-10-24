"""Flow (WaveNet coupling) and HiFi-GAN building blocks, weight norm folded at load time."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class WN(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int):
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden * n_layers, 1)
        for i in range(n_layers):
            dil = dilation_rate**i
            self.in_layers.append(
                nn.Conv1d(hidden, 2 * hidden, kernel_size, dilation=dil, padding=get_padding(kernel_size, dil))
            )
            res_skip = 2 * hidden if i < n_layers - 1 else hidden
            self.res_skip_layers.append(nn.Conv1d(hidden, res_skip, 1))

    def forward(self, x, x_mask, g):
        out = torch.zeros_like(x)
        g = self.cond_layer(g)
        h = self.hidden
        for i in range(self.n_layers):
            a = self.in_layers[i](x) + g[:, i * 2 * h : (i + 1) * 2 * h]
            acts = torch.tanh(a[:, :h]) * torch.sigmoid(a[:, h:])
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + rs[:, :h]) * x_mask
                out = out + rs[:, h:]
            else:
                out = out + rs
        return out * x_mask


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden, kernel_size, dilation_rate, n_layers, gin_channels):
        super().__init__()
        self.half = channels // 2
        self.pre = nn.Conv1d(self.half, hidden, 1)
        self.enc = WN(hidden, kernel_size, dilation_rate, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden, self.half, 1)  # mean-only

    def reverse(self, x, x_mask, g):
        x0, x1 = x[:, : self.half], x[:, self.half :]
        h = self.enc(self.pre(x0) * x_mask, x_mask, g)
        m = self.post(h) * x_mask
        return torch.cat([x0, (x1 - m) * x_mask], 1)


class Flip(nn.Module):
    def reverse(self, x, x_mask, g):
        return torch.flip(x, [1])


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden, kernel_size, dilation_rate, n_layers, gin_channels))
            self.flows.append(Flip())

    def reverse(self, x, x_mask, g):
        for flow in reversed(self.flows):
            x = flow.reverse(x, x_mask, g)
        return x


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
            for _ in dilation
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
            for d in dilation
        )

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU_SLOPE)) + x
        return x


def fold_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace weight_g/weight_v (or parametrizations.weight.original0/1) pairs with a plain weight."""
    out: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            g, v = val, state[base + ".weight_v"]
        elif key.endswith(".parametrizations.weight.original0"):
            base = key[: -len(".parametrizations.weight.original0")]
            g, v = val, state[base + ".parametrizations.weight.original1"]
        elif key.endswith((".weight_v", ".parametrizations.weight.original1")):
            continue
        else:
            out[key] = val
            continue
        dims = tuple(range(1, v.dim()))
        out[base + ".weight"] = g * (v / v.norm(dim=dims, keepdim=True))
    return out
