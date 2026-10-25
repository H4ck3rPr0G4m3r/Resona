"""ContentVec feature extractor (HuBERT-base architecture, layer-12 output, 768-d at 50 fps)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

HOP = 320  # 20 ms at 16 kHz


class ContentVec(nn.Module):
    def __init__(self, model_dir: str | Path, device: torch.device | str = "cuda", half: bool = True):
        super().__init__()
        from transformers import HubertConfig, HubertModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        model_dir = Path(model_dir)
        config = HubertConfig.from_pretrained(model_dir, attn_implementation=os.environ.get("RESONA_ATTN", "sdpa"))
        model = HubertModel(config)
        state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        # content-vec-best ships a `final_proj` head that RVC v2 does not use.
        state = {k.removeprefix("hubert."): v for k, v in state.items() if not k.startswith("final_proj")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if "masked_spec_embed" not in k]
        if missing or unexpected:
            raise RuntimeError(f"ContentVec weights mismatch: missing={missing} unexpected={unexpected}")
        model.eval()
        self.model = model.to(device)
        self.half_precision = half
        if half:
            self.model.half()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def extract(self, x16k: torch.Tensor) -> torch.Tensor:
        """x16k: [T] float32 → [1, T//320, 768]. The last frame is repeated so frame i ↔ [20i, 20i+20) ms."""
        x = x16k.unsqueeze(0)
        x = x.half() if self.half_precision else x.float()
        feats = self.model(x).last_hidden_state  # [1, T//320 - 1, 768]
        n = x16k.shape[-1] // HOP
        if feats.shape[1] < n:
            feats = torch.cat([feats, feats[:, -1:].expand(1, n - feats.shape[1], -1)], dim=1)
        return feats[:, :n]
