"""Audio file I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        x = soxr.resample(x, file_sr, sr, quality="VHQ")
    return x.astype(np.float32)


def save(path: str | Path, x: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="FLOAT")
