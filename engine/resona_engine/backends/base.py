"""Backend-neutral streaming contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol

import torch

FRAME_MS = 20  # all stream sizes are multiples of one ContentVec frame


@dataclass
class StreamConfig:
    block_ms: int = 60
    context_ms: int = 1000
    lookahead_ms: int = 40
    crossfade_ms: int = 10
    sola_search_ms: int = 10  # host searches ±search/2 around the nominal offset
    dec_pad_ms: int = 80  # decoder left padding (receptive field warm-up)
    edge_audio_ms: int = 40  # mirror-pad the window's right edge (better-conditioned block tail)
    f0_method: str = "rmvpe"  # "rmvpe" | "fcpe"

    def validate(self) -> "StreamConfig":
        for name in ("block_ms", "context_ms", "lookahead_ms"):
            v = getattr(self, name)
            if v <= 0 or v % FRAME_MS:
                raise ValueError(f"{name}={v} must be a positive multiple of {FRAME_MS} ms")
        if self.edge_audio_ms < 0 or self.edge_audio_ms % FRAME_MS:
            raise ValueError(f"edge_audio_ms={self.edge_audio_ms} must be a non-negative multiple of {FRAME_MS} ms")
        for name in ("crossfade_ms", "sola_search_ms", "dec_pad_ms"):
            v = getattr(self, name)
            if v < 0 or v % 10:
                raise ValueError(f"{name}={v} must be a non-negative multiple of 10 ms")
        if self.crossfade_ms + self.sola_search_ms // 2 > self.lookahead_ms:
            raise ValueError("crossfade + search/2 must fit inside the lookahead")
        if self.f0_method not in ("rmvpe", "fcpe"):
            raise ValueError(f"unknown f0_method {self.f0_method!r}")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "StreamConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names}).validate()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LiveParams:
    pitch: float = 0.0  # semitones
    formant: float = 0.0  # semitones of spectral-envelope shift
    index_rate: float = 0.5
    index_unvoiced: float = 0.5  # index-rate multiplier on unvoiced frames
    protect: float = 0.33  # 0.5 = off
    f0_median: int = 5
    f0_octave_fix: bool = True
    f0_glide_ms: float = 0.0
    f0_threshold: float = 0.03

    @classmethod
    def from_dict(cls, d: dict, base: "LiveParams | None" = None) -> "LiveParams":
        out = LiveParams(**asdict(base)) if base else LiveParams()
        names = {f.name for f in fields(cls)}
        for k, v in d.items():
            if k in names:
                setattr(out, k, type(getattr(out, k))(v))
        return out

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StreamLayout:
    """Derived sizes the host needs to drive SOLA. Sample counts are at the model rate."""

    model_sr: int
    block_in: int  # 16 kHz samples per input block
    block_out: int  # model-rate samples per output block (after SOLA)
    out_len: int  # model-rate samples returned per process() call
    crossfade: int
    search: int
    nominal_offset: int  # SOLA offset that would be exact with no correction

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StageTimes:
    features_ms: float = 0.0
    f0post_ms: float = 0.0
    synth_ms: float = 0.0
    total_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class VoiceBackend(Protocol):
    def load(self, path: Path, opts: dict) -> dict: ...
    def configure(self, cfg: StreamConfig) -> StreamLayout: ...
    def process(self, x16k: torch.Tensor, p: LiveParams) -> torch.Tensor: ...
    def reset(self) -> None: ...
