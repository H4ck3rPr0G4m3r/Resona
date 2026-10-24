"""Stateful F0 post-processing for streaming.

Frames are addressed by absolute index. Frames before the commit point are frozen: later
calls reuse the committed value so overlapping blocks see identical pitch (and therefore
identical excitation phase). Uncommitted frames (crossfade tail, lookahead) are recomputed
each call with the most context available.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class F0PostConfig:
    median: int = 5  # median filter width in frames (odd, 1 = off)
    octave_fix: bool = True
    octave_window: int = 15  # frames of raw voiced history for the reference median (~150 ms)
    min_segment: int = 3  # voiced blips / unvoiced gaps shorter than this are removed / filled
    glide_ms: float = 0.0  # one-pole pitch slew in log domain (0 = off)


def _median_filter(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or x.shape[0] == 0:
        return x.copy()
    r = width // 2
    padded = np.pad(x, (r, r), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def _segments(voiced: np.ndarray):
    """Yield (start, end, is_voiced) runs."""
    n = voiced.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n and voiced[j] == voiced[i]:
            j += 1
        yield i, j, bool(voiced[i])
        i = j


class F0Post:
    HISTORY = 64

    def __init__(self, cfg: F0PostConfig | None = None):
        self.cfg = cfg or F0PostConfig()
        self.reset()

    def reset(self) -> None:
        self._committed: deque[tuple[int, float]] = deque(maxlen=self.HISTORY)  # (abs frame, f0)
        self._raw_voiced: deque[float] = deque(maxlen=self.cfg.octave_window)
        self._commit_upto: int | None = None
        self._pending: dict[int, tuple[float, float]] = {}  # abs -> (processed, raw)

    def _committed_tail(self, n: int) -> np.ndarray:
        return np.array([v for _, v in list(self._committed)[-n:]], dtype=np.float64)

    def process(self, abs_frame0: int, raw: np.ndarray) -> np.ndarray:
        """raw: [T] Hz (0 = unvoiced) for frames abs_frame0.. → processed [T]."""
        raw = np.asarray(raw, dtype=np.float64)
        out = np.zeros_like(raw)
        k0 = 0
        if self._commit_upto is not None and self._committed:
            k0 = int(np.clip(self._commit_upto - abs_frame0, 0, raw.shape[0]))
            lookup = dict(self._committed)
            for k in range(k0):
                out[k] = lookup.get(abs_frame0 + k, raw[k])
        new = raw[k0:]
        if new.shape[0] == 0:
            return out
        ctx = self._committed_tail(self.cfg.median // 2 + self.cfg.min_segment)
        seq = np.concatenate([ctx, new])
        seq = _median_filter(seq, self.cfg.median)
        if self.cfg.octave_fix:
            seq = self._fix_octaves(seq, np.concatenate([ctx, new]))
        seq = self._clean_segments(seq, n_ctx=ctx.shape[0])
        if self.cfg.glide_ms > 0:
            seq = self._glide(seq)
        processed = seq[ctx.shape[0] :]
        out[k0:] = processed
        self._pending = {abs_frame0 + k0 + i: (processed[i], new[i]) for i in range(new.shape[0])}
        return out

    def commit(self, upto_abs: int) -> None:
        """Freeze all frames with absolute index < upto_abs."""
        for a in sorted(self._pending):
            if a >= upto_abs:
                break
            val, raw = self._pending[a]
            if self._committed and a <= self._committed[-1][0]:
                continue
            self._committed.append((a, float(val)))
            if raw > 0:
                self._raw_voiced.append(float(raw))
        self._pending = {a: v for a, v in self._pending.items() if a >= upto_abs}
        self._commit_upto = upto_abs

    def _fix_octaves(self, seq: np.ndarray, raw_seq: np.ndarray) -> np.ndarray:
        hist = deque(self._raw_voiced, maxlen=self.cfg.octave_window)
        out = seq.copy()
        for i, f in enumerate(seq):
            if f <= 0:
                continue
            if len(hist) >= 3:
                ref = float(np.median(hist))
                cents = 1200 * np.log2(f / ref)
                if abs(cents - 1200) < 300:
                    out[i] = f / 2
                elif abs(cents + 1200) < 300:
                    out[i] = f * 2
            if raw_seq[i] > 0:
                hist.append(float(raw_seq[i]))  # reference follows raw, so sustained jumps are accepted
        return out

    def _clean_segments(self, seq: np.ndarray, n_ctx: int) -> np.ndarray:
        m = self.cfg.min_segment
        if m <= 1:
            return seq
        out = seq.copy()
        voiced = seq > 0
        segs = list(_segments(voiced))
        for idx, (s, e, v) in enumerate(segs):
            if e - s >= m or e <= n_ctx:
                continue  # long enough, or entirely inside frozen context
            is_last = idx == len(segs) - 1
            if v and not is_last:
                out[s:e] = 0.0  # short voiced blip
            elif not v and 0 < idx and not is_last:
                left, right = seq[s - 1], seq[e]
                out[s:e] = np.geomspace(left, right, e - s + 2)[1:-1]  # fill short gap
        return out

    def _glide(self, seq: np.ndarray) -> np.ndarray:
        a = 1.0 - np.exp(-10.0 / self.cfg.glide_ms)
        out = seq.copy()
        prev = None
        for i, f in enumerate(seq):
            if f <= 0:
                prev = None
                continue
            lf = np.log(f)
            prev = lf if prev is None else prev + a * (lf - prev)
            out[i] = np.exp(prev)
        return out


def to_coarse(f0: np.ndarray | "torch.Tensor"):
    """RVC mel-scale coarse pitch (1..255) from Hz."""
    import torch

    f0 = torch.as_tensor(f0)
    mel_min = 1127 * np.log(1 + 50 / 700)
    mel_max = 1127 * np.log(1 + 1100 / 700)
    mel = 1127 * torch.log(1 + f0 / 700)
    mel = torch.where(mel > 0, (mel - mel_min) * 254 / (mel_max - mel_min) + 1, mel)
    return torch.clamp(torch.round(mel), 1, 255).long()
