import math

import pytest
import torch

from resona_engine.rvc.source import PhaseContinuousSource
from resona_engine.rvc.synth import Synth
from tests.conftest import reference_models, requires_models


@requires_models
@pytest.mark.parametrize("pth", reference_models(), ids=lambda p: p.stem[:24])
def test_load_all_reference_models(pth, device):
    synth = Synth.from_checkpoint(pth, device, half=device.type == "cuda")
    t = 60
    phone = torch.randn(1, t, 768, device=device)
    pitch = torch.full((1, t), 60, dtype=torch.long, device=device)
    f0 = torch.full((t,), 200.0, device=device)
    wav = synth.infer(phone, pitch, f0, abs_frame0=0, dec_start=40, dec_len=12)
    assert wav.shape == (12 * synth.upp,)
    assert torch.isfinite(wav).all()
    assert wav.abs().max() <= 1.0


def _source(sr=40000, upp=400):
    src = PhaseContinuousSource(sr, upp)
    with torch.no_grad():
        src.l_linear.weight.fill_(1.0)
        src.l_linear.bias.zero_()
    return src


def test_source_phase_continuous_across_calls():
    """Rendering [0, 20) then [10, 30) must agree with a single render of [0, 30) on the overlap."""
    src = _source()
    f0 = torch.full((30,), 220.0)
    whole = src(f0, abs_frame0=0, commit=False)[0, 0]
    src.reset()
    src(f0[:20], abs_frame0=0)
    second = src(f0[10:], abs_frame0=10)[0, 0]
    overlap_whole = whole[10 * 400 :]
    assert torch.allclose(second, overlap_whole, atol=1e-5)


def test_source_seam_has_no_discontinuity():
    src = _source()
    f0 = torch.full((10,), 180.0)
    a = src(f0, abs_frame0=0)[0, 0]
    b = src(f0, abs_frame0=10)[0, 0]
    seam = (b[0] - a[-1]).abs()
    typical = (a[1:] - a[:-1]).abs().max()
    assert seam <= typical * 1.05


def test_source_noise_is_time_locked():
    src = _source()
    f0 = torch.zeros(10)  # unvoiced: pure noise
    a = src(f0, abs_frame0=5, commit=False)[0, 0]
    b = src(f0, abs_frame0=5, commit=False)[0, 0]
    assert torch.equal(a, b)
    c = src(f0, abs_frame0=6, commit=False)[0, 0]
    assert torch.allclose(a[400:], c[:-400])
