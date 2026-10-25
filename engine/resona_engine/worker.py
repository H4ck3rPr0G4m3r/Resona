"""Engine worker process: JSON-RPC control on stdio, audio over shared-memory rings.

Launched by the host:  python -m resona_engine.worker --shm <prefix> [--stub]

Rings/events (created by the host before launch):
  <prefix>-in / <prefix>-in-ev    16 kHz input blocks   (host → engine)
  <prefix>-out / <prefix>-out-ev  model-rate output     (engine → host)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .ipc.rpc import RpcError, RpcServer
from .ipc.shm import NamedEvent, ShmRing

PROTOCOL = 1
MAX_BACKLOG = 2  # blocks; older input is skipped to catch up
STATS_INTERVAL = 0.1
VOICE_CACHE = 3  # voices kept resident in VRAM
XFADE_MS = 50.0  # voice-switch crossfade


def log(*a) -> None:
    print("[engine]", *a, file=sys.stderr, flush=True)


class StubBackend:
    """Torch-free stand-in used to test the host: echoes the input with the real timing contract."""

    def __init__(self):
        from .backends.base import StreamConfig

        self.cfg = StreamConfig()
        self.last_times = None
        self.last_f0 = None

    def configure(self, cfg):
        from .backends.base import StageTimes, StreamLayout

        self.cfg = cfg.validate()
        ms = lambda v: v * 16  # noqa: E731
        self.B, self.L, self.X, self.S = ms(cfg.block_ms), ms(cfg.lookahead_ms), ms(cfg.crossfade_ms), ms(cfg.sola_search_ms)
        self.hist = np.zeros(self.B + self.L + self.X + self.S + self.B, np.float32)
        self.last_times = StageTimes()
        self.layout = StreamLayout(16000, self.B, self.B, self.B + self.X + self.S, self.X, self.S, self.S // 2)
        return self.layout

    def reset(self):
        self.hist[:] = 0

    def skip(self, x):
        self.process(x, None)

    def process(self, x, p):
        x = np.asarray(x, np.float32)
        self.hist = np.concatenate([self.hist[len(x):], x])
        end = len(self.hist) - self.L + self.X + self.S // 2
        start = end - (self.B + self.X + self.S)
        return self.hist[start:end].copy()


class Engine:
    def __init__(self, rpc: RpcServer, shm_prefix: str | None, stub: bool):
        self.rpc = rpc
        self.stub = stub
        self.prefix = shm_prefix
        self.lock = threading.RLock()  # guards backend/voice/config swaps vs. the stream loop
        self.running = threading.Event()
        self.quit = threading.Event()
        self.thread: threading.Thread | None = None
        self.features = None
        self.backend = None
        self.voices: dict[str, object] = {}
        self.voice_order: deque[str] = deque()
        self.active: str | None = None
        self.params = None
        self.cfg = None
        self.layout = None
        self.times: deque = deque(maxlen=100)
        self.dropped = 0
        self.processed = 0
        self.ring_in = self.ring_out = self.ev_in = self.ev_out = None
        self._register()

    # -- helpers -----------------------------------------------------------------------------
    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_features(self):
        if self.features is None:
            from . import assets
            from .backends.rvc_v2 import FeatureStack

            dev = self._device()
            t = time.perf_counter()
            self.features = FeatureStack(assets.resolve(), dev, half=dev.type == "cuda")
            log(f"feature extractors loaded in {time.perf_counter() - t:.1f}s on {dev}")
        return self.features

    def _ensure_rings(self):
        if self.ring_in is None:
            if not self.prefix:
                raise RpcError("worker started without --shm")
            self.ring_in = ShmRing.open(f"{self.prefix}-in")
            self.ring_out = ShmRing.open(f"{self.prefix}-out")
            self.ev_in = NamedEvent.open(f"{self.prefix}-in-ev")
            self.ev_out = NamedEvent.open(f"{self.prefix}-out-ev")

    # -- RPC methods ------------------------------------------------------------------------
    def _register(self):
        m = self.rpc.method

        @m("hello")
        def hello(_):
            info = {"protocol": PROTOCOL, "pid": os.getpid(), "stub": self.stub}
            if not self.stub:
                torch = self._torch()
                info["torch"] = torch.__version__
                info["cuda"] = torch.cuda.is_available()
                info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                info["vram_total_mb"] = (
                    torch.cuda.get_device_properties(0).total_memory >> 20 if torch.cuda.is_available() else 0
                )
            return info

        @m("assets.status")
        def assets_status(_):
            from . import assets

            assets.import_from_vonovox()
            return {"root": str(assets.assets_root()), "missing": [a.rel for a in assets.missing()]}

        @m("assets.fetch")
        def assets_fetch(_):
            from . import assets

            assets.import_from_vonovox()
            assets.fetch(progress=lambda rel, d, t: self.rpc.notify("assets.progress", {"file": rel, "done": d, "total": t}))
            return {"missing": [a.rel for a in assets.missing()]}

        @m("model.inspect")
        def model_inspect(p):
            from .rvc.checkpoint import CheckpointError, inspect

            try:
                return inspect(p["path"]).to_dict()
            except CheckpointError as e:
                raise RpcError(str(e), -32010)

        @m("model.load")
        def model_load(p):
            return self.load_voice(p["id"], p["path"], p.get("index_path"))

        @m("model.activate")
        def model_activate(p):
            return self.activate(p["id"], p.get("params"), float(p.get("xfade_ms", XFADE_MS)))

        @m("model.unload")
        def model_unload(p):
            with self.lock:
                if p["id"] == self.active:
                    raise RpcError("cannot unload the active voice")
                self._forget(self.voices.pop(p["id"], None))
                if p["id"] in self.voice_order:
                    self.voice_order.remove(p["id"])
            return True

        @m("stream.configure")
        def stream_configure(p):
            return self.configure(p.get("config") or {})

        @m("engine.calibrate")
        def engine_calibrate(p):
            return self.calibrate(p.get("blocks") or [40, 60, 80, 100], int(p.get("n", 80)), float(p.get("budget", 0.6)))

        @m("stream.start")
        def stream_start(_):
            self.start()
            return True

        @m("stream.stop")
        def stream_stop(_):
            self.stop()
            return True

        @m("params.set")
        def params_set(p):
            from .backends.base import LiveParams

            self.params = LiveParams.from_dict(p, self.params)
            return self.params.to_dict()

        @m("engine.shutdown")
        def shutdown(_):
            self.stop()
            self.quit.set()
            threading.Timer(0.05, lambda: os._exit(0)).start()
            return True

    # -- model / stream management ------------------------------------------------------------
    def load_voice(self, vid: str, path: str, index_path: str | None) -> dict:
        if self.stub:
            return {"name": Path(path).stem, "sr": 16000, "stub": True}
        from .backends.rvc_v2 import Voice
        from .rvc.checkpoint import CheckpointError

        self._ensure_features()
        cached = self.voices.get(vid)
        if cached is not None and str(cached.info.path) == str(Path(path)):
            info = cached.info.to_dict()  # already resident (switching back to a recent voice)
            info["has_index"] = cached.index is not None
            info["index_size"] = cached.index.size if cached.index is not None else 0
            return info
        t = time.perf_counter()
        try:
            voice = Voice.load(path, index_path, self.features.device, self.features.half)
        except (CheckpointError, ValueError) as e:
            raise RpcError(str(e), -32010)
        with self.lock:
            self.voices[vid] = voice
            if vid in self.voice_order:
                self.voice_order.remove(vid)
            self.voice_order.append(vid)
            while len(self.voice_order) > VOICE_CACHE:
                old = next((v for v in self.voice_order if v != self.active and v != vid), None)
                if old is None:
                    break
                self.voice_order.remove(old)
                self._forget(self.voices.pop(old, None))
        if self.backend is not None and self.cfg is not None:
            # Capture now so activation is seamless; outside the lock so streaming continues.
            self.backend.prepare_voice(voice, lock=self.lock)
        if not self.running.is_set():
            self._torch().cuda.empty_cache()
        log(f"voice {vid} loaded in {time.perf_counter() - t:.2f}s ({voice.info.name}, {voice.info.sr} Hz)")
        info = voice.info.to_dict()
        info["has_index"] = voice.index is not None
        info["index_size"] = voice.index.size if voice.index is not None else 0
        return info

    def activate(self, vid: str, params: dict | None = None, xfade_ms: float = 0.0) -> dict:
        """Make `vid` the active voice. While streaming this is seamless: same layout (output is
        always 48 kHz), input history kept, and old -> new voice crossfaded over `xfade_ms`."""
        from .backends.base import LiveParams

        with self.lock:
            if self.stub:
                self.active = vid
                return {"active": vid}
            voice = self.voices.get(vid)
            if voice is None:
                raise RpcError(f"voice {vid!r} is not loaded", -32011)
            from .backends.rvc_v2 import RVCv2Backend

            old_params = self.params
            if params is not None:
                self.params = LiveParams.from_dict(params, self.params)
            if self.backend is None:
                self.backend = RVCv2Backend(self._ensure_features(), voice)
            else:
                streaming = self.running.is_set()
                self.backend.set_voice(voice, xfade_ms if streaming else 0.0, old_params)
            self.active = vid
            return {"active": vid, "layout": self.layout.to_dict() if self.layout else None}

    def configure(self, cfg_dict: dict) -> dict:
        from .backends.base import LiveParams, StreamConfig

        try:
            cfg = StreamConfig.from_dict(cfg_dict)
        except (TypeError, ValueError) as e:
            raise RpcError(str(e), -32602)
        with self.lock:
            self.cfg = cfg
            if self.params is None:
                self.params = LiveParams()
            if self.stub:
                self.backend = self.backend or StubBackend()
            elif self.backend is None:
                raise RpcError("load and activate a voice first", -32011)
            self.layout = self.backend.configure(cfg)
            self._prepare_cached()
            self._warmup()
            return self.layout.to_dict()

    def _forget(self, voice) -> None:
        if voice is not None and self.backend is not None and hasattr(self.backend, "forget"):
            self.backend.forget(voice)

    def _prepare_cached(self) -> None:
        if hasattr(self.backend, "prepare_voice"):
            for v in self.voices.values():
                self.backend.prepare_voice(v)

    def calibrate(self, blocks: list[int], n: int, budget: float) -> dict:
        """Time each block size on this GPU; recommend the smallest with p99 <= budget * block."""
        if self.stub:
            return {"results": [], "recommended": self.cfg.block_ms if self.cfg else 60}
        if self.running.is_set():
            raise RpcError("stop the stream before calibrating", -32014)
        from dataclasses import replace

        import torch

        from .backends.base import LiveParams, StreamConfig

        with self.lock:
            if self.backend is None:
                raise RpcError("load and activate a voice first", -32011)
            base = self.cfg or StreamConfig()
            params = self.params or LiveParams()
            rng = np.random.default_rng(0)
            results = []
            try:
                for b in blocks:
                    try:
                        cfg = replace(base, block_ms=int(b)).validate()
                    except ValueError:
                        continue
                    lay = self.backend.configure(cfg)
                    t = np.arange(lay.block_in * (n + 5)) / 16000
                    x = (0.2 * np.sin(2 * np.pi * 160 * t) + 0.02 * rng.standard_normal(t.shape)).astype(np.float32)
                    times = []
                    for i in range(n + 5):
                        t0 = time.perf_counter()
                        self.backend.process(torch.from_numpy(x[i * lay.block_in : (i + 1) * lay.block_in]), params).cpu()
                        if i >= 5:
                            times.append((time.perf_counter() - t0) * 1e3)
                    times.sort()
                    p50, p99 = times[len(times) // 2], times[min(int(len(times) * 0.99), len(times) - 1)]
                    results.append({"block_ms": int(b), "p50_ms": round(p50, 2), "p99_ms": round(p99, 2),
                                    "ok": p99 <= budget * b})
                    log(f"calibrate block {b} ms: p50 {p50:.1f} p99 {p99:.1f}")
            finally:
                self.layout = self.backend.configure(base) if self.cfg else self.layout
                if self.cfg:
                    self._prepare_cached()
            ok = [r["block_ms"] for r in results if r["ok"]]
            rec = min(ok) if ok else max((r["block_ms"] for r in results), default=base.block_ms)
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            return {"results": results, "recommended": rec, "budget": budget, "gpu": gpu}

    def _warmup(self, n: int = 3) -> None:
        if self.stub:
            return
        import torch

        x = torch.zeros(self.layout.block_in)
        for _ in range(n):
            self.backend.process(x, self.params)
        self.backend.reset()

    def start(self) -> None:
        self._ensure_rings()
        with self.lock:
            if self.layout is None:
                raise RpcError("stream not configured", -32012)
            if self.thread and self.thread.is_alive():
                return
            if self.layout.out_len > self.ring_out.slot_samples or self.layout.block_in > self.ring_in.slot_samples:
                raise RpcError("ring slots too small for this configuration", -32013)
            self.backend.reset()
            self.ring_in_drain()
            self.running.set()
            self.thread = threading.Thread(target=self._loop, name="stream", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        gc.enable()
        gc.collect()

    def ring_in_drain(self) -> None:
        while self.ring_in.pop() is not None:
            pass

    # -- streaming loop ---------------------------------------------------------------------
    def _loop(self) -> None:
        gc.collect()
        gc.freeze()
        gc.disable()
        last_stats = time.perf_counter()
        pitch_in, pitch_out = [], []
        try:
            while self.running.is_set():
                if not self.ev_in.wait(50) and len(self.ring_in) == 0:
                    continue
                while self.running.is_set() and len(self.ring_in):
                    backlog = len(self.ring_in)
                    item = self.ring_in.pop()
                    if item is None:
                        break
                    seq, _flags, data = item
                    with self.lock:
                        if backlog > MAX_BACKLOG:
                            self.backend.skip(self._to_tensor(data))
                            self.dropped += 1
                            continue
                        t0 = time.perf_counter()
                        out = self.backend.process(self._to_tensor(data), self.params)
                        out = out.cpu().numpy() if hasattr(out, "cpu") else out
                        dt = (time.perf_counter() - t0) * 1e3
                        if self.backend.last_f0 is not None:
                            pitch_in.extend(self.backend.last_f0[0].tolist())
                            pitch_out.extend(self.backend.last_f0[1].tolist())
                    if not self.ring_out.push(seq, out):
                        self.dropped += 1
                    self.ev_out.set()
                    self.times.append(dt)
                    self.processed += 1
                now = time.perf_counter()
                if now - last_stats >= STATS_INTERVAL:
                    last_stats = now
                    self._emit_stats(pitch_in, pitch_out)
                    pitch_in, pitch_out = [], []
        except Exception as e:  # surface to host, keep process alive for diagnostics
            import traceback

            traceback.print_exc(file=sys.stderr)
            self.rpc.notify("stream.error", {"message": f"{type(e).__name__}: {e}"})
            self.running.clear()
        finally:
            gc.enable()

    def _to_tensor(self, data: np.ndarray):
        if self.stub:
            return data
        import torch

        return torch.from_numpy(data)

    def _emit_stats(self, pitch_in: list, pitch_out: list) -> None:
        t = sorted(self.times)
        stats = {
            "processed": self.processed,
            "dropped": self.dropped,
            "backlog": len(self.ring_in),
            "p50_ms": t[len(t) // 2] if t else 0.0,
            "p99_ms": t[min(int(len(t) * 0.99), len(t) - 1)] if t else 0.0,
        }
        if self.backend is not None and getattr(self.backend, "last_times", None) is not None:
            lt = self.backend.last_times
            stats.update(features_ms=lt.features_ms, f0post_ms=lt.f0post_ms, synth_ms=lt.synth_ms)
        if not self.stub:
            torch = self._torch()
            if torch.cuda.is_available():
                stats["vram_mb"] = torch.cuda.memory_allocated() >> 20
        self.rpc.notify("stats", stats)
        if pitch_in:
            self.rpc.notify("pitch", {"input": [round(v, 1) for v in pitch_in], "output": [round(v, 1) for v in pitch_out]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resona_engine.worker")
    ap.add_argument("--shm", help="shared-memory name prefix created by the host")
    ap.add_argument("--stub", action="store_true", help="echo backend (no torch), for host testing")
    args = ap.parse_args(argv)
    # Keep stdout exclusively for RPC: anything else that prints goes to stderr.
    rpc_out = sys.stdout
    sys.stdout = sys.stderr
    rpc = RpcServer(stdin=sys.stdin, stdout=rpc_out)
    engine = Engine(rpc, args.shm, args.stub)
    log(f"ready (protocol {PROTOCOL}, stub={args.stub})")
    rpc.notify("ready", {"protocol": PROTOCOL})
    try:
        rpc.serve()
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
