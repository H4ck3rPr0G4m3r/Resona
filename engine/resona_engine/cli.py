"""`resona-engine` command line: convert, harness, assets, inspect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(args):
    from . import assets
    from .backends.rvc_v2 import FeatureStack, Voice

    dev = _device()
    half = dev.type == "cuda"
    features = FeatureStack(assets.resolve(), dev, half)
    voice = Voice.load(args.model, args.index, dev, half)
    return features, voice


def _params(args):
    from .backends.base import LiveParams

    return LiveParams(pitch=args.pitch, formant=args.formant, index_rate=args.index_rate, protect=args.protect)


def _stream_cfg(args):
    from .backends.base import StreamConfig

    return StreamConfig(block_ms=args.block, context_ms=args.context, lookahead_ms=args.lookahead,
                        f0_method=args.f0).validate()


def cmd_convert(args) -> int:
    from .audio import load_mono, save
    from .pipeline.offline import convert_offline, convert_streaming

    features, voice = _load(args)
    x = load_mono(args.input, 16000)
    p = _params(args)
    if args.streaming:
        res = convert_streaming(features, voice, x, p, _stream_cfg(args), profile=True)
        y, times = res.audio, res.times
        tot = sorted(t.total_ms for t in times[3:]) or [0.0]
        print(f"blocks={len(times)} p50={tot[len(tot) // 2]:.1f}ms p99={tot[int(len(tot) * 0.99)]:.1f}ms")
    else:
        y = convert_offline(features, voice, x, p, args.f0)
    save(args.output, y, voice.synth.out_sr)
    print(f"wrote {args.output} ({voice.synth.out_sr} Hz, {len(y) / voice.synth.out_sr:.2f}s)")
    return 0


def cmd_harness(args) -> int:
    from .harness import run_harness

    features, voice = _load(args)
    report = run_harness(features, voice, [Path(c) for c in args.clips], _params(args), _stream_cfg(args),
                         out_dir=Path(args.out) if args.out else None)
    print(json.dumps(report, indent=2))
    return 0


def cmd_inspect(args) -> int:
    from .rvc.checkpoint import inspect

    print(json.dumps(inspect(args.model).to_dict(), indent=2))
    return 0


def cmd_assets(args) -> int:
    from . import assets

    if args.action == "fetch":
        assets.import_from_vonovox()
        assets.fetch(progress=lambda rel, d, t: print(f"\r{rel}: {d >> 20}/{t >> 20} MiB", end="", file=sys.stderr))
        print(file=sys.stderr)
    gone = assets.missing()
    print(json.dumps({"root": str(assets.assets_root()), "missing": [a.rel for a in gone]}))
    return 0 if not gone else 1


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True)
    p.add_argument("--index")
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--formant", type=float, default=0.0)
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--f0", default="rmvpe", choices=["rmvpe", "fcpe"])
    p.add_argument("--block", type=int, default=60)
    p.add_argument("--context", type=int, default=1000)
    p.add_argument("--lookahead", type=int, default=40)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resona-engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert a WAV file")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--streaming", action="store_true", help="simulate the real-time path")
    _common(c)
    c.set_defaults(fn=cmd_convert)
    h = sub.add_parser("harness", help="streaming vs offline quality metrics")
    h.add_argument("clips", nargs="+")
    h.add_argument("--out", help="write offline/streaming WAVs here")
    _common(h)
    h.set_defaults(fn=cmd_harness)
    i = sub.add_parser("inspect", help="print model metadata")
    i.add_argument("model")
    i.set_defaults(fn=cmd_inspect)
    a = sub.add_parser("assets", help="check or fetch feature-extractor assets")
    a.add_argument("action", choices=["status", "fetch"])
    a.set_defaults(fn=cmd_assets)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
