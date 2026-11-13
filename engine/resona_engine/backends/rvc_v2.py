"""RVC v2 streaming backend.

Each `process()` call receives one block of 16 kHz input and returns
`block + crossfade + search` model-rate samples covering the block that ends `lookahead`
before the newest input sample. See docs/specs §4 for the window model.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..assets import AssetPaths
from ..pipeline import graphs
from ..pipeline.f0post import F0Post, F0PostConfig, to_coarse
from ..rvc.checkpoint import ModelInfo
from ..rvc.contentvec import ContentVec
from ..rvc.index import GpuIndex
from ..rvc.rmvpe import RMVPE
from ..rvc.resample import OUT_SR
from ..rvc.synth import Synth
from .base import LiveParams, StageTimes, StreamConfig, StreamLayout

SR_IN = 16000
HOP_IN = 160  # 16 kHz samples per 100 fps frame


class FeatureStack:
    """Shared, voice-independent extractors (loaded once per engine)."""

    def __init__(self, assets: AssetPaths, device: torch.device, half: bool = True):
        self.device, self.half = device, half
        self.contentvec = ContentVec(assets.contentvec_dir, device, half)
        self.rmvpe = RMVPE(assets.rmvpe, device, half)
        self._fcpe = None

    @property
    def fcpe(self):
        if self._fcpe is None:
            from torchfcpe import spawn_bundled_infer_model

            with contextlib.redirect_stdout(io.StringIO()):
                self._fcpe = spawn_bundled_infer_model(str(self.device))
        return self._fcpe

    @torch.no_grad()
    def content(self, x16k: torch.Tensor, formant: float = 0.0) -> torch.Tensor:
        """[W] → [1, W/320, 768]. A formant shift resamples the embedder input only."""
        n = x16k.shape[0] // 320
        if abs(formant) < 1e-3:
            return self.contentvec.extract(x16k)
        r = 2.0 ** (formant / 12.0)
        m = max(int(round(x16k.shape[0] / r / 320)) * 320, 640)
        xs = F.interpolate(x16k.view(1, 1, -1), size=m, mode="linear", align_corners=False).view(-1)
        feats = self.contentvec.extract(xs)  # [1, m/320, 768]
        feats = F.interpolate(feats.transpose(1, 2).float(), size=n, mode="linear", align_corners=False)
        return feats.transpose(1, 2).to(self.contentvec.model.dtype)

    @torch.no_grad()
    def f0(self, x16k: torch.Tensor, method: str = "rmvpe", threshold: float = 0.03) -> torch.Tensor:
        n = x16k.shape[0] // HOP_IN
        if method == "fcpe":
            f = self.fcpe.infer(
                x16k.float().view(1, -1, 1), sr=SR_IN, decoder_mode="local_argmax", threshold=0.006,
                f0_min=50, f0_max=1100, interp_uv=False, output_interp_target_length=n,
            )
            return f.view(-1)[:n].float()
        return self.rmvpe.infer(x16k, threshold)


@dataclass
class Voice:
    info: ModelInfo
    synth: Synth
    index: GpuIndex | None

    @classmethod
    def load(cls, path: str | Path, index_path: str | Path | None, device, half=True) -> "Voice":
        synth = Synth.from_checkpoint(path, device, half)
        index = GpuIndex.load(index_path, device) if index_path else None
        return cls(synth.info, synth, index)


def build_phone(
    feats50: torch.Tensor, voice: Voice, voiced: torch.Tensor, p: LiveParams, n_frames: int
) -> torch.Tensor:
    """Index blend (with unvoiced gating) + consonant protection at 100 fps. → [1, T, 768]"""
    index = voice.index if p.index_rate > 0 else None
    protect = p.protect if p.protect < 0.5 else None
    return mix_phone(feats50, index, voiced, p.index_rate, p.index_unvoiced, protect, n_frames)


def mix_phone(feats50, index: GpuIndex | None, voiced, rate, unvoiced, protect, n_frames: int) -> torch.Tensor:
    """`build_phone` with the rates as floats or 0-d tensors (so a CUDA graph can take them as
    inputs). `index=None` / `protect=None` turn the blend / the protection off; so does a rate
    of 0 / a protect of 1, with an identical result."""
    raw = feats50.repeat_interleave(2, dim=1)[:, :n_frames]
    mixed = raw
    if index is not None:
        retr = index.retrieve(feats50[0]).unsqueeze(0).repeat_interleave(2, dim=1)[:, :n_frames]
        gate = torch.where(voiced, 1.0, unvoiced).to(raw.dtype).view(1, -1, 1)
        r = rate * gate
        mixed = retr * r + raw * (1 - r)
    if protect is not None:
        w = torch.where(voiced, 1.0, protect).to(raw.dtype).view(1, -1, 1)
        mixed = mixed * w + raw * (1 - w)
    return mixed


class RVCv2Backend:
    def __init__(self, features: FeatureStack, voice: Voice | None = None):
        self.features = features
        self.device = features.device
        self.voice = voice
        self.cfg: StreamConfig | None = None
        self.f0post = F0Post()
        self.profile = False
        self.last_times = StageTimes()
        self.last_f0: tuple[np.ndarray, np.ndarray] | None = None  # (input f0, output f0) for the block
        self.use_graphs = graphs.available(self.device) and os.environ.get("RESONA_GRAPHS", "1") != "0"
        self._g_content: graphs.Graph | None = None
        self._g_rmvpe: graphs.Graph | None = None
        self._g_synth: dict[int, graphs.Graph] = {}  # id(voice.synth) -> graph
        self._side = torch.cuda.Stream(self.device) if self.use_graphs else None
        # CUDA spin-waits by default: a whole CPU core, twice per block, taken from a game that
        # is running beside us. A blocking event lets the thread sleep until the GPU is done.
        self._done = torch.cuda.Event(blocking=True) if self.device.type == "cuda" else None

    # -- configuration ---------------------------------------------------------------------
    def set_voice(self, voice: Voice, xfade_ms: float = 0.0, old_params: LiveParams | None = None) -> None:
        """Switch voices without touching the input timeline.

        The layout is voice-independent (output is always OUT_SR), so streaming continues. With
        `xfade_ms`, the next blocks render both voices and crossfade them (equal power, the voices
        are uncorrelated) keyed by absolute output sample, so overlapping blocks agree.
        """
        old = self.voice
        self.voice = voice
        if self.cfg is None:
            return
        self.prepare_voice(voice)
        voice.synth.reset()
        self._xfade = None
        if old is not None and old is not voice and xfade_ms > 0 and self.n_in >= self.W:
            next_frame0 = (self.n_in + self.B - self.W) // HOP_IN
            start = (next_frame0 + self.C // HOP_IN) * self.hop_out + self.X + self.S
            self._xfade = (old, old_params, start, max(int(xfade_ms * OUT_SR / 1000), 1))

    def forget(self, voice: Voice) -> None:
        """Release the captured graph of an evicted voice."""
        self._g_synth.pop(id(voice.synth), None)

    def configure(self, cfg: StreamConfig) -> StreamLayout:
        if self.voice is None:
            raise RuntimeError("no voice loaded")
        self.cfg = cfg.validate()
        ms = lambda v: v * SR_IN // 1000  # noqa: E731
        self.C, self.B, self.L = ms(cfg.context_ms), ms(cfg.block_ms), ms(cfg.lookahead_ms)
        self.W = self.C + self.B + self.L
        self.T = self.W // HOP_IN
        mso = lambda v: v * OUT_SR // 1000  # noqa: E731
        self.hop_out = OUT_SR // 100
        self.X, self.S = mso(cfg.crossfade_ms), mso(cfg.sola_search_ms)
        self.Bo = mso(cfg.block_ms)
        self.out_len = self.Bo + self.X + self.S
        self.pad_f = cfg.dec_pad_ms // 10
        self.ext = cfg.edge_audio_ms // 10
        self.T_all = self.T + self.ext
        # Output region (OUT_SR samples, window-relative) starts S/2 before the block start.
        self.region_start = (self.C // HOP_IN) * self.hop_out - self.S // 2
        self.dec_first = max(self.region_start // self.hop_out - self.pad_f, 0)
        self.layout = StreamLayout(
            model_sr=OUT_SR, block_in=self.B, block_out=self.Bo, out_len=self.out_len,
            crossfade=self.X, search=self.S, nominal_offset=self.S // 2,
        )
        self.reset()
        self._build_graphs()
        return self.layout

    # -- CUDA graphs -----------------------------------------------------------------------
    # A block is three graph launches: ContentVec and RMVPE (+ decode) side by side, then one
    # per-voice graph for everything after the F0 post-processing (index blend, coarse pitch,
    # noise windows, NSF source, enc_p -> flow -> decoder). A game holding the GPU makes every
    # separate launch wait for a time slice, so the eager glue between graphs (~130 small
    # kernels) used to cost as much as a whole graph. Live parameters enter as tensors.
    def _build_graphs(self) -> None:
        if not self.use_graphs:
            return
        synth_key = (self.T_all, self.dec_first, self.B, self.region_start, self.out_len, self.ext)
        if synth_key != getattr(self, "_synth_key", None):
            self._synth_key = synth_key
            self._g_synth.clear()
        feat_key = (self.W, self.ext, self.cfg.f0_method)
        if feat_key != getattr(self, "_feat_key", None):
            self._feat_key = feat_key
            self._g_content = self._g_rmvpe = None
            buf = torch.zeros(self.W, device=self.device)
            fs, T = self.features, self.T
            self._g_content = graphs.Graph(lambda b: fs.contentvec.extract(self._mirror(b)), [buf])
            if self.cfg.f0_method == "rmvpe":
                thr = torch.tensor(0.03, device=self.device)
                rmvpe = lambda b, t: fs.rmvpe.decode(fs.rmvpe.salience(self._mirror(b)), t)[:T]  # noqa: E731
                self._g_rmvpe = graphs.Graph(rmvpe, [buf, thr])
        self.prepare_voice(self.voice)

    def _mirror(self, buf: torch.Tensor) -> torch.Tensor:
        """Mirrored continuation so the window's right edge isn't conditioned on zeros."""
        if not self.ext:
            return buf
        return torch.cat([buf, torch.flip(buf[-self.ext * HOP_IN :], [0])])

    def _render_fn(self, voice: Voice):
        """Synthesis for one voice at the current layout as a pure function of tensors, so the
        same code runs eagerly or captured: (content [1, T_all/2, 768], post-processed f0 [T],
        abs_frame0 (0-d long), params [pitch factor, index rate, index unvoiced, protect])
        -> audio [out_len] at OUT_SR. Advances the voice's NSF phase by one block."""
        synth, d0, T_all, ext = voice.synth, self.dec_first, self.T_all, self.ext
        dec_len = T_all - d0
        off = self.region_start - d0 * self.hop_out
        out_len, advance = self.out_len, self.B // HOP_IN
        g = synth.speaker(0)

        def render(feats50, f0p, abs_frame0, prm):
            if ext:
                f0p = torch.cat([f0p, torch.flip(f0p[-ext:], [0])])
            voiced = f0p > 0
            f0s = f0p * prm[0]
            phone = mix_phone(feats50, voice.index, voiced, prm[1], prm[2], prm[3], T_all)
            pitch = to_coarse(f0s).unsqueeze(0)
            noise, exc = synth.prepare_step(T_all, f0s, abs_frame0, d0, dec_len, advance)
            wav = synth.core(phone, pitch, noise, exc, g, d0, dec_len)
            return wav[off : off + out_len]

        return render

    def prepare_voice(self, voice: Voice, lock=None) -> None:
        """Capture the synth graph for `voice` at the current layout (no streaming state touched).

        With `lock`, the capture runs outside it (only the cache insert is locked), so a stream
        thread holding that lock between blocks is never stalled by a voice preload.
        """
        if not self.use_graphs or self.cfg is None or id(voice.synth) in self._g_synth:
            return
        dev, key = self.device, self._synth_key
        ex = [
            torch.zeros(1, self.T_all // 2, 768, device=dev, dtype=self.features.contentvec.model.dtype),
            torch.zeros(self.T, device=dev),
            torch.zeros((), device=dev, dtype=torch.long),
            torch.tensor([1.0, 0.5, 0.5, 0.33], device=dev),
        ]
        src = voice.synth.dec.m_source
        saved = src.phase.clone(), src.phase_at.clone()
        graph = graphs.Graph(self._render_fn(voice), ex)
        src.phase.copy_(saved[0])  # the capture's warm-up runs advanced the NSF phase
        src.phase_at.copy_(saved[1])
        with lock if lock is not None else contextlib.nullcontext():
            if key == self._synth_key:  # layout unchanged meanwhile
                self._g_synth[id(voice.synth)] = graph

    def reset(self) -> None:
        self.buf = torch.zeros(self.W if self.cfg else 0, device=self.device)
        self.n_in = 0
        self._xfade = None
        self.f0post.reset()
        if self.voice:
            self.voice.synth.reset()

    # -- streaming -------------------------------------------------------------------------
    @torch.no_grad()
    def skip(self, x16k: torch.Tensor) -> None:
        """Advance the input timeline without inference (catch-up after a stall).

        Continuity state is dropped, so the next processed block starts a fresh phase/pitch history.
        """
        self.buf = torch.cat([self.buf[self.B :], x16k.to(self.device, torch.float32)])
        self.n_in += self.B
        self._xfade = None
        self.f0post.reset()
        self.voice.synth.reset()

    def _features(self, p: LiveParams) -> tuple[torch.Tensor, torch.Tensor]:
        """→ (raw f0 [T] Hz, content [1, T_all/2, 768]) of the current window. Graphs run on
        parallel streams."""
        fs, method = self.features, self.cfg.f0_method
        g_content = self._g_content if abs(p.formant) < 1e-3 else None
        content = lambda: g_content(self.buf) if g_content is not None else fs.content(self._mirror(self.buf), p.formant)  # noqa: E731
        if self._g_rmvpe is not None and method == "rmvpe":
            cur = torch.cuda.current_stream(self.device)
            self._side.wait_stream(cur)
            with torch.cuda.stream(self._side):
                f0 = self._g_rmvpe(self.buf, torch.tensor(p.f0_threshold))
            feats50 = content()
            cur.wait_stream(self._side)
            return f0, feats50
        return fs.f0(self._mirror(self.buf), method, p.f0_threshold)[: self.T], content()

    def _render(self, voice: Voice, p: LiveParams, feats50, f0p: torch.Tensor, abs_frame0: int):
        """Synthesize the output region for one voice. f0p: post-processed f0 [T] on the CPU.
        → audio [out_len] at OUT_SR (a fresh tensor)."""
        protect = p.protect if p.protect < 0.5 else 1.0
        prm = torch.tensor([2.0 ** (p.pitch / 12.0), p.index_rate, p.index_unvoiced, protect])
        frame0 = torch.tensor(abs_frame0, dtype=torch.long)
        graph = self._g_synth.get(id(voice.synth))
        if graph is not None:
            return graph(feats50, f0p, frame0, prm).clone()  # graph outputs are reused buffers
        dev = self.device
        return self._render_fn(voice)(feats50, f0p.to(dev), frame0.to(dev), prm.to(dev)).clone()

    def _sync(self):
        if self.profile and self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def _wait(self) -> None:
        """Wait for the work queued on the current stream, sleeping instead of spinning."""
        if self._done is not None:
            self._done.record()
            self._done.synchronize()

    @torch.no_grad()
    def process(self, x16k: torch.Tensor, p: LiveParams) -> torch.Tensor:
        if self.cfg is None or self.voice is None:
            raise RuntimeError("backend not configured")
        if x16k.shape[0] != self.B:
            raise ValueError(f"expected {self.B} samples, got {x16k.shape[0]}")
        t0 = self._sync()
        x16k = x16k.to(self.device, torch.float32)
        self.buf = torch.cat([self.buf[self.B :], x16k])
        self.n_in += self.B
        abs_frame0 = (self.n_in - self.W) // HOP_IN

        self.f0post.cfg = F0PostConfig(median=p.f0_median, octave_fix=p.f0_octave_fix, glide_ms=p.f0_glide_ms)
        f0_raw, feats50 = self._features(p)
        t1 = self._sync()

        self._wait()  # features done: the copy below no longer has to wait on the GPU
        f0_raw_np = f0_raw.cpu().numpy()
        f0_np = self.f0post.process(abs_frame0, f0_raw_np)
        f0p = torch.from_numpy(f0_np).float()
        t2 = self._sync()

        out = self._render(self.voice, p, feats50, f0p, abs_frame0)
        if self._xfade is not None:
            old, old_p, start, n = self._xfade
            region0 = abs_frame0 * self.hop_out + self.region_start
            if region0 >= start + n:
                self._xfade = None
            else:
                prev = self._render(old, old_p or p, feats50, f0p, abs_frame0)
                idx = torch.arange(self.out_len, device=self.device) + region0
                w = ((idx - start).float() / n).clamp(0, 1) * (math.pi / 2)
                out = prev * torch.cos(w) + out * torch.sin(w)
        self._wait()  # so the caller's copy of `out` doesn't spin-wait either
        commit_upto = abs_frame0 + (self.C + self.B) // HOP_IN
        self.f0post.commit(commit_upto)
        t3 = self._sync()

        # The pitch trace comes from the CPU copies, not two more GPU->CPU syncs per block.
        bs, be = self.C // HOP_IN, (self.C + self.B) // HOP_IN
        self.last_f0 = (f0_raw_np[bs:be], (f0_np[bs:be] * 2.0 ** (p.pitch / 12.0)).astype(np.float32))
        self.last_times = StageTimes(
            features_ms=(t1 - t0) * 1e3, f0post_ms=(t2 - t1) * 1e3, synth_ms=(t3 - t2) * 1e3,
            total_ms=(t3 - t0) * 1e3,
        )
        return out
