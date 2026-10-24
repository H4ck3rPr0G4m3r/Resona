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


def test_rejects_non_checkpoint(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.inspect(bad)
