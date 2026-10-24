"""SOLA crossfade (reference implementation; the host has the real-time Rust twin)."""

from __future__ import annotations

import numpy as np


class Sola:
    def __init__(self, crossfade: int, search: int):
        self.X, self.S = crossfade, search
        t = np.linspace(0.0, 1.0, crossfade, endpoint=False) + 0.5 / max(crossfade, 1)
        # Raised-cosine (equal-gain): the aligned signals are correlated, so equal-power would
        # bump the level by up to +3 dB at each seam.
        self.fade_in = (np.sin(0.5 * np.pi * t) ** 2).astype(np.float32)
        self.fade_out = (1.0 - self.fade_in).astype(np.float32)
        self.tail: np.ndarray | None = None
        self.last_offset = search // 2
        self.seam_stats: list[tuple[float, float, float]] = []  # (ncc, mismatch_db, tail_rms)

    def reset(self) -> None:
        self.tail = None
        self.seam_stats.clear()

    def best_offset(self, head: np.ndarray) -> int:
        """Offset k in [0, S] maximizing normalized cross-correlation of head[k:k+X] with the tail."""
        if self.tail is None or self.X == 0:
            return self.S // 2
        X, S = self.X, self.S
        win = np.lib.stride_tricks.sliding_window_view(head[: X + S], X)  # [S+1, X]
        num = win @ self.tail
        den = np.sqrt((win * win).sum(1) + 1e-8)
        return int(np.argmax(num / den))

    def process(self, y: np.ndarray, block: int) -> np.ndarray:
        """y: block + X + S samples. Returns `block` samples of continuous output."""
        k = self.best_offset(y)
        self.last_offset = k
        out = y[k : k + block].copy()
        if self.tail is not None and self.X:
            head = y[k : k + self.X]
            te = float((self.tail**2).sum())
            mis = float(((head - self.tail) ** 2).sum())
            ncc = float(head @ self.tail / np.sqrt((head**2).sum() * te + 1e-12))
            self.seam_stats.append((ncc, 10 * np.log10((mis + 1e-12) / (te + 1e-12)), np.sqrt(te / self.X)))
            out[: self.X] = out[: self.X] * self.fade_in + self.tail * self.fade_out
        self.tail = y[k + block : k + block + self.X].copy()
        return out
