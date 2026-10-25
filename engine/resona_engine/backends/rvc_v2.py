"""RVC v2 streaming backend.

Each `process()` call receives one block of 16 kHz input and returns
`block + crossfade + search` model-rate samples covering the block that ends `lookahead`
before the newest input sample. See docs/specs §4 for the window model.
"""

from __future__ import annotations

import contextlib
import io
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
from ..rvc.synth import FLOW_CONTEXT, Synth
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
    raw = feats50.repeat_interleave(2, dim=1)[:, :n_frames]
    mixed = raw
    if voice.index is not None and p.index_rate > 0:
        retr = voice.index.retrieve(feats50[0]).unsqueeze(0).repeat_interleave(2, dim=1)[:, :n_frames]
        gate = torch.where(voiced, 1.0, p.index_unvoiced).to(raw.dtype).view(1, -1, 1)
        rate = p.index_rate * gate
        mixed = retr * rate + raw * (1 - rate)
    if p.protect < 0.5:
        w = torch.where(voiced, 1.0, p.protect).to(raw.dtype).view(1, -1, 1)
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

    # -- configuration ---------------------------------------------------------------------
    def set_voice(self, voice: Voice) -> None:
        self.voice = voice
        if self.cfg:
            self.configure(self.cfg)

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
        sr = self.voice.info.sr
        msr = lambda v: v * sr // 1000  # noqa: E731
        self.upp = self.voice.synth.upp
        self.X, self.S = msr(cfg.crossfade_ms), msr(cfg.sola_search_ms)
        self.Bo = msr(cfg.block_ms)
        self.out_len = self.Bo + self.X + self.S
        self.pad_f = cfg.dec_pad_ms // 10
        self.ext = cfg.edge_audio_ms // 10
        self.T_all = self.T + self.ext
        self.region_start = self._region_start(sr, self.upp)
        self.dec_first = self._dec_first(sr, self.upp)
        self.layout = StreamLayout(
            model_sr=sr, block_in=self.B, block_out=self.Bo, out_len=self.out_len,
            crossfade=self.X, search=self.S, nominal_offset=self.S // 2,
        )
        self.reset()
        self._build_graphs()
        return self.layout

    def _region_start(self, sr: int, upp: int) -> int:
        """Model-rate sample (window-relative) where the returned region starts: S/2 before the block."""
        return (self.C // HOP_IN) * upp - (self.cfg.sola_search_ms * sr // 1000) // 2

    def _dec_first(self, sr: int, upp: int) -> int:
        return max(self._region_start(sr, upp) // upp - self.pad_f, 0)

    # -- CUDA graphs -----------------------------------------------------------------------
    def _build_graphs(self) -> None:
        if not self.use_graphs:
            return
        key = (self.T_all, self.cfg.f0_method)
        if key == getattr(self, "_graph_key", None):
            self.prepare_voice(self.voice)  # voice switch: feature graphs stay valid
            return
        self._graph_key = key
        self._g_content = self._g_rmvpe = None
        self._g_synth.clear()
        n = self.T_all * HOP_IN
        x = torch.zeros(n, device=self.device)
        fs = self.features
        self._g_content = graphs.Graph(fs.contentvec.extract, [x])
        if self.cfg.f0_method == "rmvpe":
            self._g_rmvpe = graphs.Graph(fs.rmvpe.salience, [x])
        self.prepare_voice(self.voice)

    def prepare_voice(self, voice: Voice) -> None:
        """Capture the synth graph for `voice` at the current layout (no streaming state touched)."""
        if not self.use_graphs or self.cfg is None or id(voice.synth) in self._g_synth:
            return
        synth, dev = voice.synth, self.device
        dtype = self.features.contentvec.model.dtype
        d0 = self._dec_first(voice.info.sr, synth.upp)
        flow_head = max(d0 - FLOW_CONTEXT, 0)
        dec_len = self.T_all - d0
        inter = synth.z_noise.shape[0]
        ex = [
            torch.zeros(1, self.T_all, 768, device=dev, dtype=dtype),
            torch.ones(1, self.T_all, device=dev, dtype=torch.long),
            torch.zeros(1, inter, self.T_all - flow_head, device=dev, dtype=synth.emb_g.weight.dtype),
            torch.zeros(1, 1, dec_len * synth.upp, device=dev),
            synth.speaker(0),
        ]
        self._g_synth[id(synth)] = graphs.Graph(lambda *a: synth.core(*a, d0, dec_len), ex)

    def reset(self) -> None:
        self.buf = torch.zeros(self.W if self.cfg else 0, device=self.device)
        self.n_in = 0
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
        self.f0post.reset()
        self.voice.synth.reset()

    def _features(self, xin: torch.Tensor, p: LiveParams) -> tuple[torch.Tensor, torch.Tensor]:
        """→ (raw f0 [T_all] Hz, content [1, T_all/2, 768]). Graphs run on parallel streams."""
        fs, method = self.features, self.cfg.f0_method
        g_content = self._g_content if abs(p.formant) < 1e-3 else None
        if self._g_rmvpe is not None and method == "rmvpe":
            cur = torch.cuda.current_stream(self.device)
            self._side.wait_stream(cur)
            with torch.cuda.stream(self._side):
                sal = self._g_rmvpe(xin)
            feats50 = g_content(xin) if g_content is not None else fs.content(xin, p.formant)
            cur.wait_stream(self._side)
            f0 = fs.rmvpe.decode(sal, p.f0_threshold)[: xin.shape[0] // HOP_IN]
        else:
            feats50 = g_content(xin) if g_content is not None else fs.content(xin, p.formant)
            f0 = fs.f0(xin, method, p.f0_threshold)
        return f0, feats50

    def _sync(self):
        if self.profile and self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

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
        T, ext = self.T, self.ext
        xin = self.buf
        if ext:  # mirrored continuation so the right edge isn't conditioned on zeros
            xin = torch.cat([self.buf, torch.flip(self.buf[-ext * HOP_IN :], [0])])
        f0_raw_all, feats50 = self._features(xin, p)
        f0_raw = f0_raw_all[:T]
        t1 = self._sync()

        f0_np = self.f0post.process(abs_frame0, f0_raw.cpu().numpy())
        f0p = torch.from_numpy(f0_np).float().to(self.device)
        if ext:
            f0p = torch.cat([f0p, torch.flip(f0p[-ext:], [0])])
        f0s = f0p * (2.0 ** (p.pitch / 12.0))
        t2 = self._sync()

        voiced = f0p > 0
        phone = build_phone(feats50, self.voice, voiced, p, T + ext)
        pitch = to_coarse(f0s).to(self.device).unsqueeze(0)
        T_all = phone.shape[1]

        # Region (model-rate samples, window-relative): starts S/2 before the block start.
        synth, d0 = self.voice.synth, self.dec_first
        dec_len = T_all - d0
        noise, exc, g = synth.prepare(T_all, f0s, abs_frame0, d0, dec_len, commit_phase=True)
        graph = self._g_synth.get(id(synth))
        if graph is not None:
            wav = graph(phone, pitch, noise, exc, g)
        else:
            wav = synth.core(phone, pitch, noise, exc, g, d0, dec_len)
        off = self.region_start - d0 * self.upp
        out = wav[off : off + self.out_len].clone()
        commit_upto = abs_frame0 + (self.C + self.B) // HOP_IN
        self.f0post.commit(commit_upto)
        t3 = self._sync()

        bs, be = self.C // HOP_IN, (self.C + self.B) // HOP_IN
        self.last_f0 = (f0_raw[bs:be].cpu().numpy(), f0s[bs:be].cpu().numpy())
        self.last_times = StageTimes(
            features_ms=(t1 - t0) * 1e3, f0post_ms=(t2 - t1) * 1e3, synth_ms=(t3 - t2) * 1e3,
            total_ms=(t3 - t0) * 1e3,
        )
        return out
