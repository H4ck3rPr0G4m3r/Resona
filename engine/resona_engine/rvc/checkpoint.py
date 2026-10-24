"""RVC checkpoint inspection and normalized loading.

Handles both Applio-style metadata (int `sr`, `vocoder`, `embedder_model`) and the original
RVC-WebUI format (`sr: "40k"`, `info`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

_SR_ALIASES = {"32k": 32000, "40k": 40000, "48k": 48000}
SUPPORTED_SR = (32000, 40000, 48000)


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    name: str
    sr: int
    version: str
    f0: bool
    vocoder: str
    vocoder_architecture: str | None
    embedder: str
    config: tuple
    hash: str  # first 16 hex chars of the file's sha256
    epoch: int | None = None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "name": self.name,
            "sr": self.sr,
            "version": self.version,
            "f0": self.f0,
            "vocoder": self.vocoder,
            "vocoder_architecture": self.vocoder_architecture,
            "embedder": self.embedder,
            "hash": self.hash,
            "epoch": self.epoch,
        }


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _parse_sr(raw) -> int:
    if isinstance(raw, str):
        sr = _SR_ALIASES.get(raw.lower())
        if sr is None and raw.isdigit():
            sr = int(raw)
    else:
        sr = int(raw)
    if sr not in SUPPORTED_SR:
        raise CheckpointError(f"unsupported sample rate {raw!r}")
    return sr


def _read(path: Path) -> dict:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # corrupt file, pickled code, etc.
        raise CheckpointError(f"cannot read checkpoint {path.name}: {e}") from e
    if not isinstance(ck, dict) or "weight" not in ck or "config" not in ck:
        raise CheckpointError(f"{path.name} is not an RVC inference checkpoint")
    return ck


def _info(path: Path, ck: dict) -> ModelInfo:
    version = str(ck.get("version", "v1"))
    if version != "v2":
        raise CheckpointError(f"{path.name}: only RVC v2 models are supported (got {version})")
    f0 = bool(int(ck.get("f0", 1)))
    if not f0:
        raise CheckpointError(f"{path.name}: models without pitch guidance are not supported")
    config = tuple(ck["config"])
    if len(config) != 18:
        raise CheckpointError(f"{path.name}: unexpected config length {len(config)}")
    sr = _parse_sr(ck.get("sr", config[-1]))
    epoch = ck.get("epoch")
    return ModelInfo(
        path=path,
        name=str(ck.get("model_name") or path.stem),
        sr=sr,
        version=version,
        f0=f0,
        vocoder=str(ck.get("vocoder", "HiFi-GAN")),
        vocoder_architecture=ck.get("vocoder_architecture"),
        embedder=str(ck.get("embedder_model", "contentvec")),
        config=config,
        hash=file_hash(path),
        epoch=int(epoch) if isinstance(epoch, (int, float)) else None,
    )


def inspect(path: str | Path) -> ModelInfo:
    path = Path(path)
    return _info(path, _read(path))


def load_state(path: str | Path) -> tuple[ModelInfo, dict[str, torch.Tensor]]:
    path = Path(path)
    ck = _read(path)
    info = _info(path, ck)
    if info.vocoder != "HiFi-GAN":
        raise CheckpointError(f"{path.name}: unsupported vocoder variant {info.vocoder!r}")
    if info.embedder != "contentvec":
        raise CheckpointError(f"{path.name}: unsupported embedder {info.embedder!r}")
    state = {k: v for k, v in ck["weight"].items() if not k.startswith("enc_q.")}
    return info, state
