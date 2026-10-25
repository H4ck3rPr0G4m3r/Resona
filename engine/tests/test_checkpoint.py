from dataclasses import replace
from pathlib import Path

import pytest

from resona_engine.rvc import checkpoint
from tests.conftest import reference_models, requires_models


@requires_models
@pytest.mark.parametrize("pth", reference_models(), ids=lambda p: p.stem[:24])
def test_inspect_reference_models(pth):
    info = checkpoint.inspect(pth)
    assert info.sr in (32000, 40000, 48000)
    assert info.version == "v2" and info.f0
    assert len(info.hash) == 16


def test_parse_old_style_sample_rate():
    assert checkpoint._parse_sr("40k") == 40000
    assert checkpoint._parse_sr(48000) == 48000
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint._parse_sr("22k")


def test_unsupported_variants_are_refused():
    ok = checkpoint.ModelInfo(
        path=Path("v.pth"), name="v", sr=40000, version="v2", f0=True, vocoder="HiFi-GAN",
        vocoder_architecture=None, embedder="contentvec", config=(), hash="0" * 16,
    )
    assert checkpoint.check_supported(ok) is ok
    for field, value in [("vocoder", "RefineGAN"), ("embedder", "spin")]:
        with pytest.raises(checkpoint.CheckpointError, match=value):
            checkpoint.check_supported(replace(ok, **{field: value}))


def test_rejects_non_checkpoint(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.inspect(bad)
