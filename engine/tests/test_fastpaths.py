"""The load-time rewrites that make a block cheaper must not change what it computes."""

import torch
import torch.nn as nn

from resona_engine.rvc.modules import ResBlock1, ResidualCouplingBlock, to_2d
from resona_engine.rvc.rmvpe import ConvBlockRes, ResDecoderBlock, fold_batchnorm


def _randomize_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_(0, 0.5)


def test_fold_batchnorm_keeps_the_function():
    torch.manual_seed(0)
    for block, x, extra in (
        (ConvBlockRes(4, 8), torch.randn(1, 4, 16, 12), ()),
        (ResDecoderBlock(8, 4, (2, 2)), torch.randn(1, 8, 8, 6), (torch.randn(1, 4, 16, 12),)),
    ):
        block.eval()
        _randomize_bn(block)
        with torch.no_grad():
            ref = block(x, *extra)
            fold_batchnorm(block)
            out = block(x, *extra)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in block.modules())
        assert torch.allclose(out, ref, atol=1e-5), float((out - ref).abs().max())


def test_to_2d_matches_the_1d_layers():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 50)
    res = ResBlock1(16, 3, (1, 3, 5)).eval()
    up = nn.ConvTranspose1d(16, 8, 16, 8, padding=4).eval()
    flow = ResidualCouplingBlock(16, 24, 5, 1, 3, gin_channels=6).eval()
    g = torch.randn(1, 6, 1)
    with torch.no_grad():
        ref = res(x), up(x), flow.reverse(x, g)
        holder = nn.ModuleDict({"res": res, "up": up, "flow": flow})
        to_2d(holder)
        x4, g4 = x.unsqueeze(2), g.unsqueeze(2)
        out = holder["res"](x4), holder["up"](x4), holder["flow"].reverse(x4, g4)
    assert not any(isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)) for m in holder.modules())
    for a, b in zip(ref, out):
        assert torch.allclose(a, b.squeeze(2), atol=1e-5), float((a - b.squeeze(2)).abs().max())
