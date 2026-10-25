"""Per-stage latency benchmark of the streaming backend.

uv run python -m resona_engine.bench --model M.pth [--index I] [--blocks 300] [--block 60]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import assets
from .backends.base import LiveParams, StreamConfig
from .backends.rvc_v2 import FeatureStack, RVCv2Backend, Voice


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(len(xs) * q), len(xs) - 1)]


def run(model: str, index: str | None, blocks: int, block_ms: int, cfg_extra: dict | None = None) -> dict:
    dev = torch.device("cuda")
    fs = FeatureStack(assets.resolve(), dev)
    voice = Voice.load(model, index, dev)
    be = RVCv2Backend(fs, voice)
    be.profile = True
    lay = be.configure(StreamConfig(block_ms=block_ms, **(cfg_extra or {})))
    p = LiveParams(index_rate=0.5 if index else 0.0)
    rng = np.random.default_rng(0)
    t = np.arange(lay.block_in * blocks) / 16000
    x = (0.2 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)
    times = []
    for i in range(blocks):
        be.process(torch.from_numpy(x[i * lay.block_in : (i + 1) * lay.block_in]), p)
        if i >= 10:
            times.append(be.last_times)
    out = {}
    for k in ("features_ms", "f0post_ms", "synth_ms", "total_ms"):
        v = [getattr(s, k) for s in times]
        out[k] = {"p50": round(pct(v, 0.5), 2), "p99": round(pct(v, 0.99), 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index")
    ap.add_argument("--blocks", type=int, default=300)
    ap.add_argument("--block", type=int, default=60)
    a = ap.parse_args()
    print(json.dumps(run(a.model, a.index, a.blocks, a.block), indent=1))


if __name__ == "__main__":
    main()
