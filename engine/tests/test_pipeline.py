from pathlib import Path

import numpy as np
import pytest
import torch

from resona_engine.audio import load_mono
from resona_engine.backends.base import LiveParams, StreamConfig
from resona_engine.backends.rvc_v2 import FeatureStack, RVCv2Backend, Voice
from resona_engine.harness import seam_summary
from resona_engine.pipeline.offline import convert_offline, convert_streaming
from tests.conftest import harmonic_tone, index_for, reference_models, requires_cuda, requires_models

CLIPS = Path(__file__).parent / "clips"

pytestmark = [requires_cuda, requires_models]


@pytest.fixture(scope="module")
def features(assets):
    return FeatureStack(assets, torch.device("cuda"))


@pytest.fixture(scope="module")
def voice():
    pth = next(p for p in reference_models() if index_for(p) and "Trump" in str(p))
    return Voice.load(pth, index_for(pth), torch.device("cuda"))


def test_streaming_block_shapes(features, voice):
    be = RVCv2Backend(features, voice)
    lay = be.configure(StreamConfig())
    assert lay.model_sr == 48000 and lay.block_in == 960 and lay.block_out == 60 * 48
    out = be.process(torch.zeros(lay.block_in), LiveParams())
    assert out.shape == (lay.out_len,)


def test_pipeline_silence_is_finite_and_quiet(features, voice):
    be = RVCv2Backend(features, voice)
    lay = be.configure(StreamConfig())
    for _ in range(20):
        out = be.process(torch.zeros(lay.block_in), LiveParams(pitch=12))
    assert torch.isfinite(out).all()
    assert out.abs().max() < 0.05
    assert abs(float(out.mean())) < 1e-2  # no DC


def test_no_index_path(features, voice):
    x = harmonic_tone(150.0, 1.0).numpy()
    no_index = Voice(voice.info, voice.synth, None)
    a = convert_offline(features, no_index, x, LiveParams(index_rate=0.7))
    b = convert_offline(features, voice, x, LiveParams(index_rate=0.0))
    np.testing.assert_allclose(a, b, atol=2e-3)


def test_wrong_block_size_rejected(features, voice):
    be = RVCv2Backend(features, voice)
    be.configure(StreamConfig())
    with pytest.raises(ValueError):
        be.process(torch.zeros(100), LiveParams())


def test_invalid_stream_config():
    with pytest.raises(ValueError):
        StreamConfig(block_ms=50).validate()
    with pytest.raises(ValueError):
        StreamConfig(lookahead_ms=20, crossfade_ms=20).validate()


@pytest.mark.parametrize("method", ["rmvpe", "fcpe"])
def test_streaming_output_length_and_seams(features, voice, method):
    x = load_mono(CLIPS / "speech.wav", 16000)[: 16000 * 3]
    res = convert_streaming(features, voice, x, LiveParams(), StreamConfig(f0_method=method))
    assert abs(len(res.audio) - len(x) * 48000 / 16000) <= 1
    assert np.isfinite(res.audio).all()
    s = seam_summary(res.seams)
    assert s["seam_ncc"] > 0.8, s  # regression guard (naive RVC streaming scores ~0.78)


def test_formant_shift_runs(features, voice):
    x = load_mono(CLIPS / "speech.wav", 16000)[:16000]
    y = convert_offline(features, voice, x, LiveParams(formant=3.0))
    assert np.isfinite(y).all() and np.abs(y).max() > 1e-3


def _stream(be, x, lay, p):
    return torch.cat([be.process(torch.from_numpy(x[i : i + lay.block_in]), p) for i in range(0, len(x) - lay.block_in + 1, lay.block_in)])


@pytest.mark.parametrize("formant", [0.0, 2.0])
def test_cuda_graphs_match_eager(features, voice, formant):
    x = load_mono(CLIPS / "speech.wav", 16000)
    x = x[: 16000 * 3]
    p = LiveParams(index_rate=0.5, formant=formant)
    eager = RVCv2Backend(features, voice)
    eager.use_graphs = False
    lay = eager.configure(StreamConfig())
    graph = RVCv2Backend(features, voice)
    graph.configure(StreamConfig())
    assert graph._g_synth and graph._g_content is not None
    a, b = _stream(eager, x, lay, p), _stream(graph, x, lay, p)
    assert a.shape == b.shape
    err = (a - b).abs()
    assert float(err.max()) < 2e-2 and float(err.mean()) < 1e-3, (float(err.max()), float(err.mean()))


def test_voice_switch_reuses_feature_graphs(features, voice):
    be = RVCv2Backend(features, voice)
    be.configure(StreamConfig())
    g_content = be._g_content
    other = Voice(voice.info, Voice.load(next(p for p in reference_models() if "Trump" not in str(p)), None, torch.device("cuda")).synth, None)
    be.prepare_voice(other)
    assert len(be._g_synth) == 2
    be.set_voice(other)
    assert be._g_content is g_content
    out = be.process(torch.zeros(960), LiveParams())
    assert torch.isfinite(out).all()
    be.forget(other)
    assert len(be._g_synth) == 1


def test_seamless_switch_across_sample_rates(features, voice):
    """Mid-stream switch 40k -> other-rate voice: same layout, crossfade, no clicks at the seam."""
    from resona_engine.pipeline.sola import Sola

    from resona_engine.rvc.checkpoint import inspect

    other_pth = next(p for p in reference_models() if index_for(p) and inspect(p).sr != voice.info.sr)
    other = Voice.load(other_pth, index_for(other_pth), torch.device("cuda"))
    x = load_mono(CLIPS / "speech.wav", 16000)[: 16000 * 4]
    be = RVCv2Backend(features, voice)
    lay = be.configure(StreamConfig())
    be.prepare_voice(other)
    sola = Sola(lay.crossfade, lay.search)
    p = LiveParams()
    outs, switch_at = [], 30
    for k, i in enumerate(range(0, len(x) - lay.block_in + 1, lay.block_in)):
        if k == switch_at:
            be.set_voice(other, xfade_ms=50, old_params=p)
            assert be.layout == lay and be._xfade is not None
        outs.append(sola.process(be.process(torch.from_numpy(x[i : i + lay.block_in]), p).cpu().numpy(), lay.block_out))
        if k == switch_at + 3:
            assert be._xfade is None  # fade finished within a few blocks
    y = np.concatenate(outs)
    assert np.isfinite(y).all()
    jump = np.abs(np.diff(y))
    seam = slice(switch_at * lay.block_out - 960, (switch_at + 2) * lay.block_out + 960)
    outside = np.concatenate([jump[: seam.start], jump[seam.stop :]])
    assert jump[seam].max() < 1.25 * outside.max(), (jump[seam].max(), outside.max())  # a click would be far larger
