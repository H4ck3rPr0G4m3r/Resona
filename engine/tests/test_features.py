import numpy as np
import pytest
import torch

from resona_engine.rvc import rmvpe as rmvpe_mod
from tests.conftest import harmonic_tone


def test_mel_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    ours = rmvpe_mod.mel_filterbank().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-6)


@pytest.mark.parametrize("f0", [110.0, 220.0, 330.0])
def test_rmvpe_tracks_steady_tone(rmvpe, device, f0):
    x = harmonic_tone(f0, 1.0).to(device)
    f = rmvpe.infer(x)
    assert f.shape[0] == x.shape[0] // 160
    mid = f[20:-20]
    assert (mid > 0).all(), "steady tone should be voiced"
    cents = 1200 * torch.log2(mid / f0)
    assert cents.abs().median() < 25


def test_rmvpe_silence_is_unvoiced(rmvpe, device):
    f = rmvpe.infer(torch.zeros(16000, device=device))
    assert (f == 0).float().mean() > 0.95


def test_contentvec_shape(contentvec, device):
    x = harmonic_tone(150.0, 1.0).to(device)
    feats = contentvec.extract(x)
    assert feats.shape == (1, 50, 768)
    assert torch.isfinite(feats).all()
