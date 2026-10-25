"""Rational polyphase resampler on the GPU (Kaiser-windowed sinc), fixed-shape and graph-capturable.

Every voice is resampled to one output rate inside the synth, so the host layout never depends
on the voice and voices of different sample rates can be crossfaded in the engine.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_SR = 48000


class Resampler(nn.Module):
    def __init__(self, sr_in: int, sr_out: int = OUT_SR, half_taps: int = 48, beta: float = 8.6, rolloff: float = 0.97):
        super().__init__()
        g = math.gcd(sr_in, sr_out)
        self.L, self.M = sr_out // g, sr_in // g
        self.identity = self.L == self.M
        n = 2 * half_taps * self.L + 1
        self.delay = half_taps * self.L  # linear-phase centre, in upsampled samples
        t = np.arange(n) - self.delay
        fc = rolloff * 0.5 / max(self.L, self.M)  # cutoff in cycles per upsampled sample
        h = 2 * fc * np.sinc(2 * fc * t) * np.kaiser(n, beta)
        h *= self.L / h.sum()
        self.register_buffer("h", torch.from_numpy(h.astype(np.float32)).view(1, 1, -1), persistent=False)

    def out_len(self, n_in: int) -> int:
        return n_in * self.L // self.M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n] float32 → [n * L / M] float32, time-aligned (output k ↔ input time k·M/L)."""
        if self.identity:
            return x
        n_out = self.out_len(x.shape[-1])
        y = F.conv_transpose1d(x.view(1, 1, -1), self.h, stride=self.L)  # zero-stuff + lowpass
        return y[0, 0, self.delay : self.delay + n_out * self.M : self.M]
