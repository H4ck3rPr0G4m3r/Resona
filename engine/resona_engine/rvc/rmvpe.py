"""RMVPE pitch estimator (Wei et al., 2023), module layout compatible with `rmvpe.pt`.

Architecture follows the MIT-licensed RVC-WebUI implementation.
Input: 16 kHz mono audio. Output: F0 in Hz at 100 fps (hop 160), 0 where unvoiced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
HOP = 160
N_FFT = 1024
N_MELS = 128
FMIN, FMAX = 30.0, 8000.0
N_CLASSES = 360
CENTS_OFFSET = 1997.3794084376191


class ConvBlockRes(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
        )
        self.has_shortcut = in_ch != out_ch
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x) + (self.shortcut(x) if self.has_shortcut else x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList(
            [ConvBlockRes(in_ch, out_ch, momentum)]
            + [ConvBlockRes(out_ch, out_ch, momentum) for _ in range(n_blocks - 1)]
        )
        self.pool = nn.AvgPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x):
        for c in self.conv:
            x = c(x)
        if self.pool is None:
            return x
        return x, self.pool(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, in_size, n_encoders, kernel_size, n_blocks, out_ch=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_ch, out_ch, kernel_size, n_blocks, momentum))
            in_ch, out_ch, in_size = out_ch, out_ch * 2, in_size // 2
        self.out_channel = out_ch

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_ch, out_ch, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResEncoderBlock(in_ch, out_ch, None, n_blocks, momentum)]
            + [ResEncoderBlock(out_ch, out_ch, None, n_blocks, momentum) for _ in range(n_inters - 1)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride, 1, out_padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [ConvBlockRes(out_ch * 2, out_ch, momentum)]
            + [ConvBlockRes(out_ch, out_ch, momentum) for _ in range(n_blocks - 1)]
        )

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for c in self.conv2:
            x = c(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(in_ch, in_ch // 2, stride, n_blocks, momentum))
            in_ch //= 2

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_ch=1, en_out_ch=16):
        super().__init__()
        self.encoder = Encoder(in_ch, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_ch)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class BiGRU(nn.Module):
    def __init__(self, in_features, hidden, num_layers):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class E2E(nn.Module):
    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2)):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks)
        self.cnn = nn.Conv2d(16, 3, 3, padding=1)
        self.fc = nn.Sequential(
            BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASSES), nn.Dropout(0.25), nn.Sigmoid()
        )

    def forward(self, mel):  # mel: [B, n_mels, T]
        x = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(x)).transpose(1, 2).flatten(-2)
        return self.fc(x)  # [B, T, 360]


def mel_filterbank() -> torch.Tensor:
    """HTK-scale, Slaney-normalized mel filterbank (matches librosa htk=True). [n_mels, n_fft//2+1]"""

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = N_FFT // 2 + 1
    fftfreqs = np.linspace(0, SR / 2, n_freqs)
    mel_pts = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2)
    hz_pts = mel_to_hz(mel_pts)
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_pts[2 : N_MELS + 2] - hz_pts[:N_MELS])
    weights *= enorm[:, None]
    return torch.from_numpy(weights.astype(np.float32))


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Fold each eval-mode `conv → BatchNorm2d` pair of an `nn.Sequential` into the convolution.

    Same function, a third fewer kernels to launch per block. In place; returns `model`."""
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        items = list(seq.named_children())
        for (_, conv), (bn_name, bn) in zip(items, items[1:]):
            if not (isinstance(bn, nn.BatchNorm2d) and isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d))):
                continue
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
            # Conv2d weights are [out, in, kh, kw], ConvTranspose2d weights are [in, out, kh, kw].
            shape = (1, -1, 1, 1) if isinstance(conv, nn.ConvTranspose2d) else (-1, 1, 1, 1)
            bias = shift if conv.bias is None else conv.bias * scale + shift
            conv.weight = nn.Parameter(conv.weight * scale.view(shape), requires_grad=False)
            conv.bias = nn.Parameter(bias, requires_grad=False)
            setattr(seq, bn_name, nn.Identity())
    return model


class RMVPE(nn.Module):
    """Pitch estimator. `infer(x16k)` returns Hz per 10 ms frame (len(x)//160 frames)."""

    def __init__(self, path: str | Path, device: torch.device | str = "cuda", half: bool = True):
        super().__init__()
        self.model = E2E()
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        with torch.no_grad():
            fold_batchnorm(self.model.unet)  # in fp32, before any cast
        # The U-Net is ~110 small 2-D convolutions. In fp16 cuDNN runs them about twice as fast
        # channels-last (5.8 → 3.2 ms per block on an RTX 3060 Ti), and closer to the fp32 result.
        self.model.to(memory_format=torch.channels_last)
        self.register_buffer("mel_basis", mel_filterbank(), persistent=False)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        cents = 20 * torch.arange(N_CLASSES, dtype=torch.float32) + CENTS_OFFSET
        self.register_buffer("cents_map", F.pad(cents, (4, 4)), persistent=False)
        self.half_precision = half
        self.to(device)
        if half:
            self.model.half()

    @property
    def device(self) -> torch.device:
        return self.mel_basis.device

    def mel(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, N_FFT, HOP, N_FFT, window=self.window, center=True, return_complex=True
        )
        mag = spec.abs()
        mel = torch.matmul(self.mel_basis, mag)
        return torch.log(torch.clamp(mel, min=1e-5))  # [n_mels, T]

    def salience(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T] float32 audio → [frames, 360] salience (frames = len//160 + 1)."""
        mel = self.mel(x.float()).unsqueeze(0)
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        if self.half_precision:
            mel = mel.half()
        hidden = self.model(mel)
        return hidden[0, :n_frames].float()

    def decode(self, sal: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
        """Local weighted average of cents around the argmax (±4 bins)."""
        center = torch.argmax(sal, dim=1)
        sal_p = F.pad(sal, (4, 4))
        idx = (center + 4).unsqueeze(1) + torch.arange(-4, 5, device=sal.device).unsqueeze(0)
        w = torch.gather(sal_p, 1, idx)
        c = self.cents_map[idx]
        cents = (w * c).sum(1) / w.sum(1).clamp_min(1e-8)
        maxx = sal.max(dim=1).values
        f0 = 10.0 * torch.pow(2.0, cents / 1200.0)
        return torch.where(maxx > threshold, f0, torch.zeros_like(f0))

    @torch.no_grad()
    def infer(self, x16k: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
        n = x16k.shape[-1] // HOP
        return self.decode(self.salience(x16k), threshold)[:n]
