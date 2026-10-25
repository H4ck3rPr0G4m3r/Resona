"""Stateful, phase-continuous NSF harmonic source with time-locked noise.

RVC's SineGen restarts its phase at 0 on every call, so consecutive streaming blocks
disagree in the overlap region and the crossfade smears two out-of-phase sines ("warble").
Here phase is carried across calls keyed by absolute sample index, and every random draw
comes from a fixed buffer indexed by absolute sample, so overlapping regions are identical.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn

NOISE_LEN = 1 << 20  # ~22 s at 48 kHz before the noise pattern repeats


@functools.lru_cache(maxsize=8)
def fixed_noise(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Deterministic noise, generated once per process (every voice shares the same pattern)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


class PhaseContinuousSource(nn.Module):
    def __init__(self, sr: int, upp: int, sine_amp: float = 0.1, noise_std: float = 0.003):
        super().__init__()
        self.sr, self.upp = sr, upp
        self.sine_amp, self.noise_std = sine_amp, noise_std
        self.l_linear = nn.Linear(1, 1)  # checkpoint: dec.m_source.l_linear
        self.register_buffer("noise", fixed_noise((NOISE_LEN,), seed=0x5E50), persistent=False)
        self.reset()

    def reset(self) -> None:
        self._prev_start: int | None = None  # absolute sample index of previous region start
        self._prev_cum: torch.Tensor | None = None  # [n+1] phase (cycles) before each sample

    def _phase0(self, start: int) -> float:
        if self._prev_start is None or self._prev_cum is None:
            return 0.0
        off = start - self._prev_start
        if 0 <= off < self._prev_cum.shape[0]:
            return float(self._prev_cum[off])
        return 0.0

    def noise_at(self, start: int, n: int) -> torch.Tensor:
        idx = (torch.arange(n, device=self.noise.device) + start) % NOISE_LEN
        return self.noise[idx]

    @torch.no_grad()
    def forward(self, f0: torch.Tensor, abs_frame0: int, commit: bool = True) -> torch.Tensor:
        """f0: [T] Hz at frame rate (0 = unvoiced). Returns excitation [1, 1, T*upp] float32.

        `commit=False` computes without updating phase state (used by offline/reference paths
        that render the same region twice).
        """
        f0 = f0.float()
        start = abs_frame0 * self.upp
        f_up = torch.repeat_interleave(f0, self.upp)
        inc = (f_up / self.sr).double()
        cum = torch.empty(inc.shape[0] + 1, dtype=torch.float64, device=f0.device)
        cum[0] = self._phase0(start)
        cum[1:] = cum[0] + torch.cumsum(inc, 0)
        cum = torch.remainder(cum, 1.0)
        sine = torch.sin(2 * math.pi * cum[1:]).float() * self.sine_amp
        uv = (f_up > 0).float()
        noise_amp = uv * self.noise_std + (1 - uv) * (self.sine_amp / 3)
        x = sine * uv + noise_amp * self.noise_at(start, sine.shape[0])
        if commit:
            self._prev_start, self._prev_cum = start, cum
        out = torch.tanh(self.l_linear(x.unsqueeze(-1).to(self.l_linear.weight.dtype)))
        return out.float().view(1, 1, -1)
