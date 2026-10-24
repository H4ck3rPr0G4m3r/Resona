import os
from pathlib import Path

import pytest
import torch

REFERENCE_MODELS = Path(
    os.environ.get("RESONA_REFERENCE_MODELS", r"C:\Pedro\AiVoiceChanger\Vonovox-1.6.9\models")
)


def reference_models() -> list[Path]:
    return sorted(REFERENCE_MODELS.rglob("*.pth")) if REFERENCE_MODELS.is_dir() else []


def index_for(pth: Path) -> Path | None:
    idx = sorted(pth.parent.glob("*.index"))
    return idx[0] if idx else None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
requires_models = pytest.mark.skipif(not reference_models(), reason="reference models not present")


@pytest.fixture(scope="session")
def assets():
    from resona_engine import assets as a

    try:
        return a.resolve()
    except a.AssetError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def rmvpe(assets, device):
    from resona_engine.rvc.rmvpe import RMVPE

    return RMVPE(assets.rmvpe, device, half=device.type == "cuda")


@pytest.fixture(scope="session")
def contentvec(assets, device):
    from resona_engine.rvc.contentvec import ContentVec

    return ContentVec(assets.contentvec_dir, device, half=device.type == "cuda")


def harmonic_tone(f0: float | torch.Tensor, seconds: float, sr: int = 16000, n_harm: int = 12) -> torch.Tensor:
    """A buzzy, voice-like harmonic tone (1/k amplitude rolloff)."""
    n = int(seconds * sr)
    if isinstance(f0, (int, float)):
        f = torch.full((n,), float(f0))
    else:
        f = f0
    phase = torch.cumsum(2 * torch.pi * f / sr, 0)
    x = sum(torch.sin(k * phase) / k for k in range(1, n_harm + 1))
    return (0.3 * x / x.abs().max()).float()
