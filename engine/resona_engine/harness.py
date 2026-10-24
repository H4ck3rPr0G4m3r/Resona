"""Streaming ≈ offline quality harness.

Renders each clip offline (one pass, infinite context) and through the streaming path
(block by block + SOLA), then compares them:

- lsd: mean log-spectral distance (dB) between the two renders
- boundary_excess: LSD in ±10 ms windows around block seams ÷ LSD elsewhere (1.0 = seams invisible)
- f0_rmse_cents / octave_error_rate: pitch of the streamed output vs the offline output
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soxr
import torch

from .audio import load_mono, save
from .backends.base import LiveParams, StreamConfig
from .backends.rvc_v2 import FeatureStack, Voice
from .pipeline.offline import convert_offline, convert_streaming

N_FFT = 1024


def _logspec(x: np.ndarray, hop: int) -> np.ndarray:
    t = torch.from_numpy(x.astype(np.float32))
    spec = torch.stft(t, N_FFT, hop, N_FFT, window=torch.hann_window(N_FFT), center=True, return_complex=True)
    return (20 * torch.log10(spec.abs().clamp_min(1e-5))).numpy()  # [F, frames]


def compare(ref: np.ndarray, test: np.ndarray, sr: int, seam_times_s: np.ndarray) -> dict:
    n = min(len(ref), len(test))
    hop = sr // 200  # 5 ms
    a, b = _logspec(ref[:n], hop), _logspec(test[:n], hop)
    energy = a.max(axis=0)
    active = energy > energy.max() - 50  # ignore near-silent frames
    lsd = np.sqrt(((a - b) ** 2).mean(axis=0))  # per frame
    t = np.arange(lsd.shape[0]) * hop / sr
    near = np.zeros_like(active)
    for s in seam_times_s:
        near |= np.abs(t - s) <= 0.010
    seam = active & near
    rest = active & ~near
    lsd_seam = float(lsd[seam].mean()) if seam.any() else float("nan")
    lsd_rest = float(lsd[rest].mean()) if rest.any() else float("nan")
    return {
        "lsd": float(lsd[active].mean()),
        "lsd_seam": lsd_seam,
        "lsd_rest": lsd_rest,
        "boundary_excess": lsd_seam / lsd_rest if lsd_rest > 0 else float("nan"),
    }


def pitch_error(features: FeatureStack, ref: np.ndarray, test: np.ndarray, sr: int) -> dict:
    n = min(len(ref), len(test))
    f = []
    for x in (ref[:n], test[:n]):
        x16 = soxr.resample(x, sr, 16000).astype(np.float32)
        f.append(features.rmvpe.infer(torch.from_numpy(x16).to(features.device)).cpu().numpy())
    m = min(len(f[0]), len(f[1]))
    a, b = f[0][:m], f[1][:m]
    both = (a > 0) & (b > 0)
    if not both.any():
        return {"f0_rmse_cents": float("nan"), "octave_error_rate": float("nan"), "voicing_agreement": 0.0}
    cents = 1200 * np.log2(b[both] / a[both])
    return {
        "f0_rmse_cents": float(np.sqrt(np.mean(np.clip(cents, -600, 600) ** 2))),
        "octave_error_rate": float(np.mean(np.abs(cents) > 600)),
        "voicing_agreement": float(np.mean((a > 0) == (b > 0))),
    }


def seam_summary(seams: list[tuple[float, float, float]]) -> dict:
    """Consistency of consecutive blocks in the crossfade region (only seams with signal)."""
    s = np.array(seams, dtype=np.float64).reshape(-1, 3)
    if s.shape[0] == 0:
        return {"seam_ncc": float("nan"), "seam_mismatch_db": float("nan")}
    loud = s[:, 2] > max(1e-3, 0.05 * s[:, 2].max())
    s = s[loud] if loud.any() else s
    return {"seam_ncc": float(np.mean(s[:, 0])), "seam_mismatch_db": float(np.median(s[:, 1]))}


def run_harness(features: FeatureStack, voice: Voice, clips: list[Path], p: LiveParams,
                cfg: StreamConfig, out_dir: Path | None = None) -> dict:
    sr = voice.info.sr
    results = {}
    for clip in clips:
        x = load_mono(clip, 16000)
        ref = convert_offline(features, voice, x, p, cfg.f0_method)
        res = convert_streaming(features, voice, x, p, cfg, profile=True)
        test, times = res.audio, res.times
        dur = len(x) / 16000
        B, L = cfg.block_ms / 1000, cfg.lookahead_ms / 1000
        seams = np.arange(B - L, dur, B)
        r = compare(ref, test, sr, seams)
        r.update(seam_summary(res.seams))
        r.update(pitch_error(features, ref, test, sr))
        tot = sorted(t.total_ms for t in times[3:]) or [0.0]
        r["block_ms_p50"] = tot[len(tot) // 2]
        r["block_ms_p99"] = tot[min(int(len(tot) * 0.99), len(tot) - 1)]
        results[clip.name] = r
        if out_dir:
            save(out_dir / f"{clip.stem}.offline.wav", ref, sr)
            save(out_dir / f"{clip.stem}.stream.wav", test, sr)
    keys = ("lsd", "boundary_excess", "seam_ncc", "seam_mismatch_db", "f0_rmse_cents", "octave_error_rate")
    vals = list(results.values())
    results["_mean"] = {k: float(np.nanmean([v[k] for v in vals])) for k in keys}
    results["_model"] = voice.info.name
    results["_config"] = cfg.to_dict()
    return results
