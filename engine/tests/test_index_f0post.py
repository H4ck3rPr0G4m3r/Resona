import numpy as np
import pytest
import torch

from resona_engine.pipeline.f0post import F0Post, F0PostConfig, to_coarse
from resona_engine.rvc.index import GpuIndex, load_vectors
from tests.conftest import index_for, reference_models, requires_models


def test_gpu_retrieve_matches_bruteforce(device):
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((2000, 768)).astype(np.float16)
    idx = GpuIndex(vecs, device)
    q = torch.from_numpy(rng.standard_normal((7, 768)).astype(np.float32)).to(device)
    got = idx.retrieve(q).float().cpu().numpy()
    v = vecs.astype(np.float32)
    qq = q.cpu().numpy()
    d2 = ((qq[:, None, :] - v[None]) ** 2).sum(-1)
    nn_idx = np.argsort(d2, axis=1)[:, :8]
    w = 1 / np.take_along_axis(d2, nn_idx, 1) ** 2
    w /= w.sum(1, keepdims=True)
    ref = (v[nn_idx] * w[..., None]).sum(1)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)


@requires_models
def test_load_reference_index(tmp_path):
    pth = next(p for p in reference_models() if index_for(p))
    vecs = load_vectors(index_for(pth), cache_dir=tmp_path)
    assert vecs.ndim == 2 and vecs.shape[1] == 768 and vecs.shape[0] > 100
    again = load_vectors(index_for(pth), cache_dir=tmp_path)  # from cache
    assert np.array_equal(vecs, again)


def test_octave_glitch_is_folded():
    post = F0Post(F0PostConfig(median=1, min_segment=1))
    f = np.full(40, 200.0)
    f[25:28] = 400.0  # 30 ms octave-up glitch
    out = post.process(0, f)
    assert np.allclose(out[25:28], 200.0)


def test_sustained_octave_change_is_accepted():
    post = F0Post(F0PostConfig(median=1, min_segment=1, octave_window=15))
    f = np.concatenate([np.full(30, 200.0), np.full(60, 400.0)])
    out = post.process(0, f)
    assert np.allclose(out[-20:], 400.0)


def test_short_voiced_blip_removed_and_gap_filled():
    post = F0Post(F0PostConfig(median=1, octave_fix=False, min_segment=3))
    f = np.zeros(40)
    f[5:7] = 150.0  # 2-frame blip in silence
    f[10:30] = 200.0
    f[18] = 0.0  # 1-frame dropout
    out = post.process(0, f)
    assert (out[5:7] == 0).all()
    assert out[18] == pytest.approx(200.0)


def test_committed_frames_are_frozen():
    post = F0Post(F0PostConfig(median=1, octave_fix=False, min_segment=1))
    first = post.process(0, np.full(20, 200.0))
    post.commit(10)
    second = post.process(5, np.full(20, 300.0))  # frames 5..9 are committed
    assert np.allclose(second[:5], first[5:10])
    assert np.allclose(second[5:], 300.0)


def test_coarse_range():
    c = to_coarse(torch.tensor([0.0, 50.0, 1100.0, 5000.0]))
    assert c.tolist()[0] == 1 and c.tolist()[-1] == 255
    assert 1 <= c[1] <= 2 and c[2] == 255
