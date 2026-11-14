"""RVC v2 synthesizer (SynthesizerTrnMs768NSFsid inference path), split for streaming.

Module names mirror RVC checkpoints so weights load strictly. The NSF decoder is split into
our stateful source (`dec.m_source`, fp32) and the convolutional body, which takes the
excitation as an input.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import attentions
from .checkpoint import ModelInfo, load_state
from .resample import OUT_SR, Resampler
from .modules import LRELU_SLOPE, ResBlock1, ResBlock2, ResidualCouplingBlock, fold_weight_norm, to_2d
from .source import fixed_noise, PhaseContinuousSource

@contextlib.contextmanager
def skip_init():
    """Make nn.init's in-place initializers no-ops while modules are constructed."""
    names = ("kaiming_uniform_", "uniform_", "normal_", "xavier_uniform_", "zeros_", "ones_")
    saved = {n: getattr(nn.init, n) for n in names}
    for n in names:
        setattr(nn.init, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(nn.init, n, f)


Z_NOISE_FRAMES = 1 << 14  # time-locked flow noise, ~164 s at 100 fps
FLOW_CONTEXT = 24  # frames of left context the flow needs (4 couplings × ±6 receptive field)


class TextEncoder(nn.Module):
    def __init__(self, out_ch, hidden, filter_ch, n_heads, n_layers, kernel_size):
        super().__init__()
        self.out_ch, self.hidden = out_ch, hidden
        self.emb_phone = nn.Linear(768, hidden)
        self.emb_pitch = nn.Embedding(256, hidden)
        self.encoder = attentions.Encoder(hidden, filter_ch, n_heads, n_layers, kernel_size)
        self.proj = nn.Conv1d(hidden, out_ch * 2, 1)

    def forward(self, phone, pitch, skip_head: int = 0):
        x = (self.emb_phone(phone) + self.emb_pitch(pitch)) * math.sqrt(self.hidden)
        x = F.leaky_relu(x, 0.1).transpose(1, -1)  # [B, H, T]
        x = self.encoder(x)
        if skip_head:
            x = x[:, :, skip_head:]
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_ch, dim=1)
        return m, logs


class GeneratorNSFBody(nn.Module):
    def __init__(self, initial_ch, resblock, rk, rd, upsample_rates, uic, uk, gin_channels, sr):
        super().__init__()
        self.num_kernels = len(rk)
        self.upp = math.prod(upsample_rates)
        self.m_source = PhaseContinuousSource(sr, self.upp)
        self.conv_pre = nn.Conv1d(initial_ch, uic, 7, 1, padding=3)
        block = ResBlock1 if str(resblock) == "1" else ResBlock2
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, uk)):
            c_cur = uic // (2 ** (i + 1))
            self.ups.append(nn.ConvTranspose1d(uic // (2**i), c_cur, k, u, padding=(k - u) // 2))
            if i + 1 < len(upsample_rates):
                s = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=s * 2, stride=s, padding=s // 2))
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = uic
        for i in range(len(self.ups)):
            ch = uic // (2 ** (i + 1))
            for k, d in zip(rk, rd):
                self.resblocks.append(block(ch, k, d))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(gin_channels, uic, 1)

    def forward(self, x, har_source, g):
        """x [1, C, 1, T], har_source [1, 1, 1, T·upp], g [1, gin, 1, 1] → [1, 1, 1, T·upp]."""
        x = self.conv_pre(x) + self.cond(g)
        har_source = har_source.to(x.dtype)
        for i, (up, nc) in enumerate(zip(self.ups, self.noise_convs)):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            x = x + nc(har_source)
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


class Synth(nn.Module):
    def __init__(self, info: ModelInfo):
        super().__init__()
        (_spec, _seg, inter, hidden, filter_ch, n_heads, n_layers, kernel_size, _p, resblock,
         rk, rd, ur, uic, uk, spk_embed_dim, gin, sr) = info.config
        self.info = info
        self.sr = info.sr
        self.enc_p = TextEncoder(inter, hidden, filter_ch, n_heads, n_layers, kernel_size)
        self.dec = GeneratorNSFBody(inter, resblock, rk, rd, ur, uic, uk, gin, info.sr)
        self.flow = ResidualCouplingBlock(inter, hidden, 5, 1, 3, gin_channels=gin)
        self.emb_g = nn.Embedding(spk_embed_dim, gin)
        self.register_buffer("z_noise", fixed_noise((inter, Z_NOISE_FRAMES), seed=0x2E50), persistent=False)
        self.upp = self.dec.upp
        if self.upp * 100 != self.sr:
            raise ValueError(f"upsample product {self.upp} does not match {self.sr} Hz at 100 fps")
        self.out_sr = OUT_SR
        self.out_upp = OUT_SR // 100  # output samples per frame
        self.resampler = Resampler(self.sr, OUT_SR)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device="cuda", half: bool = True) -> "Synth":
        info, state = load_state(path)
        state = fold_weight_norm({k: v.float() for k, v in state.items()})
        with skip_init():  # every weight comes from the checkpoint; random init is wasted time
            model = cls(info)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            raise ValueError(f"{Path(path).name}: unsupported model layout ({info.vocoder_architecture}): {e}") from e
        # Flow and decoder run as 2-D convolutions over [1, C, 1, T], channels-last (see `to_2d`).
        to_2d(model.flow)
        to_2d(model.dec)
        model.eval().to(device)
        model.flow.to(memory_format=torch.channels_last)
        model.dec.to(memory_format=torch.channels_last)
        model.half_precision = half
        if half:
            model.half()
            model.dec.m_source.float()  # excitation stays fp32
            model.dec.m_source.phase = model.dec.m_source.phase.double()
            model.z_noise = model.z_noise.float()
            model.resampler.float()
        return model

    @property
    def device(self) -> torch.device:
        return self.emb_g.weight.device

    def speaker(self, sid: int = 0) -> torch.Tensor:
        return self.emb_g(torch.tensor([sid], device=self.device)).unsqueeze(-1)

    def z_noise_at(self, abs_frame0: int | torch.Tensor, n: int) -> torch.Tensor:
        idx = (torch.arange(n, device=self.device) + abs_frame0) % Z_NOISE_FRAMES
        return self.z_noise[:, idx].unsqueeze(0)

    @torch.no_grad()
    def infer(
        self,
        phone: torch.Tensor,  # [1, T, 768]
        pitch: torch.Tensor,  # [1, T] long, coarse 1..255
        f0: torch.Tensor,  # [T] Hz (shifted), 0 = unvoiced
        abs_frame0: int,  # absolute frame index of window frame 0
        dec_start: int,  # first window frame to decode (incl. left padding)
        dec_len: int,  # number of frames to decode
        commit_phase: bool = True,
        sid: int = 0,
    ) -> torch.Tensor:
        """Returns OUT_SR audio [dec_len * out_upp] float32 for window frames [dec_start, dec_start+dec_len)."""
        noise, exc, g = self.prepare(phone.shape[1], f0, abs_frame0, dec_start, dec_len, commit_phase, sid)
        return self.core(phone, pitch, noise, exc, g, dec_start, dec_len)

    def prepare(self, n_frames, f0, abs_frame0, dec_start, dec_len, commit_phase=True, sid=0):
        """Stateful, position-dependent inputs of `core` (time-locked noise, NSF excitation)."""
        flow_head = max(dec_start - FLOW_CONTEXT, 0)
        noise = self.z_noise_at(abs_frame0 + flow_head, n_frames - flow_head).to(self.emb_g.weight.dtype)
        exc = self.dec.m_source(f0[dec_start : dec_start + dec_len], abs_frame0 + dec_start, commit_phase)
        return noise, exc, self.speaker(sid)

    def prepare_step(self, n_frames, f0, abs_frame0: torch.Tensor, dec_start, dec_len, advance_frames: int):
        """`prepare` for streaming, with the position as a long tensor: no CPU round trip, so it
        can be captured in a CUDA graph. The speaker embedding is left to the caller."""
        flow_head = max(dec_start - FLOW_CONTEXT, 0)
        noise = self.z_noise_at(abs_frame0 + flow_head, n_frames - flow_head).to(self.emb_g.weight.dtype)
        exc = self.dec.m_source.step(f0[dec_start : dec_start + dec_len], abs_frame0 + dec_start, advance_frames)
        return noise, exc

    def core(self, phone, pitch, noise, exc, g, dec_start: int, dec_len: int) -> torch.Tensor:
        """Pure, fixed-shape part (enc_p → flow → decoder); safe to capture as a CUDA graph."""
        dtype = self.emb_g.weight.dtype
        flow_head = max(dec_start - FLOW_CONTEXT, 0)
        m_p, logs_p = self.enc_p(phone.to(dtype), pitch, flow_head)
        z_p = m_p + torch.exp(logs_p) * noise * 0.66666
        g = g.unsqueeze(2)
        z = self.flow.reverse(z_p.unsqueeze(2).contiguous(memory_format=torch.channels_last), g)
        off = dec_start - flow_head
        z = z[..., off : off + dec_len]
        return self.resampler(self.dec(z, exc.unsqueeze(2), g).reshape(-1).float())

    def reset(self) -> None:
        self.dec.m_source.reset()
