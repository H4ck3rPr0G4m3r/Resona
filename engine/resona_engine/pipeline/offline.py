"""Whole-file conversion (the non-streaming reference) and a streaming simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..backends.base import LiveParams, StreamConfig
from ..backends.rvc_v2 import HOP_IN, FeatureStack, RVCv2Backend, Voice, build_phone
from .f0post import F0Post, F0PostConfig, to_coarse
from .sola import Sola

PAD = 1600  # 100 ms reflect padding at each end


@dataclass
class StreamResult:
    audio: np.ndarray
    times: list
    seams: list[tuple[float, float, float]]  # (ncc, mismatch_db, tail_rms) per seam


@torch.no_grad()
def convert_offline(features: FeatureStack, voice: Voice, wav16k: np.ndarray, p: LiveParams,
                    f0_method: str = "rmvpe") -> np.ndarray:
    """Convert a whole 16 kHz clip in one pass (infinite context). Returns model-rate audio."""
    n = wav16k.shape[0]
    x = np.pad(wav16k.astype(np.float32), (PAD, PAD), mode="reflect")
    x = np.pad(x, (0, (-x.shape[0]) % 320))
    xt = torch.from_numpy(x).to(features.device)
    T = xt.shape[0] // HOP_IN
    feats50 = features.content(xt, p.formant)
    f0_raw = features.f0(xt, f0_method, p.f0_threshold)
    post = F0Post(F0PostConfig(median=p.f0_median, octave_fix=p.f0_octave_fix, glide_ms=p.f0_glide_ms))
    f0p = torch.from_numpy(post.process(0, f0_raw.cpu().numpy())).float().to(features.device)
    f0s = f0p * (2.0 ** (p.pitch / 12.0))
    phone = build_phone(feats50, voice, f0p > 0, p, T)
    pitch = to_coarse(f0s).to(features.device).unsqueeze(0)
    voice.synth.reset()
    wav = voice.synth.infer(phone, pitch, f0s, -PAD // HOP_IN, 0, T, commit_phase=False)
    upp = voice.synth.upp
    start = PAD // HOP_IN * upp
    out_len = int(round(n * voice.info.sr / 16000))
    return wav[start : start + out_len].cpu().numpy()


def convert_streaming(features: FeatureStack, voice: Voice, wav16k: np.ndarray, p: LiveParams,
                      cfg: StreamConfig, profile: bool = False):
    """Simulate the real-time path block by block (engine + host SOLA).

    Returns audio aligned to the input timeline (model rate), per-block StageTimes, and per-seam
    SOLA consistency stats.
    """
    backend = RVCv2Backend(features, voice)
    backend.profile = profile
    layout = backend.configure(cfg)
    sola = Sola(layout.crossfade, layout.search)
    B = layout.block_in
    L = cfg.lookahead_ms * 16
    x = wav16k.astype(np.float32)
    # Feed lookahead worth of trailing silence so the final block is emitted.
    x = np.concatenate([x, np.zeros(L + (-x.shape[0]) % B + B, dtype=np.float32)])
    outs, times = [], []
    for i in range(0, x.shape[0] - B + 1, B):
        y = backend.process(torch.from_numpy(x[i : i + B]), p).cpu().numpy()
        outs.append(sola.process(y, layout.block_out))
        times.append(backend.last_times)
    y = np.concatenate(outs)
    # Call k emits input time [k*B - L, (k+1)*B - L): output lags the input by the lookahead.
    delay = int(round(cfg.lookahead_ms * layout.model_sr / 1000))
    out_len = int(round(wav16k.shape[0] * layout.model_sr / 16000))
    return StreamResult(y[delay : delay + out_len], times, list(sola.seam_stats))
