"""Exact feature retrieval from an RVC `.index` file, on the GPU.

The faiss index is read once (via deserialize, so non-ASCII paths work), its vectors are
reconstructed and cached as fp16 `.npy`, and search is an exact top-k over a matmul.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..assets import data_root
from .checkpoint import file_hash

TOP_K = 8


def _reconstruct(path: Path) -> np.ndarray:
    import faiss

    index = faiss.deserialize_index(np.fromfile(path, dtype=np.uint8))
    n = index.ntotal
    try:
        return index.reconstruct_n(0, n)
    except RuntimeError:
        ivf = faiss.extract_index_ivf(index)
        ivf.make_direct_map()
        return index.reconstruct_n(0, n)


def load_vectors(path: str | Path, cache_dir: Path | None = None) -> np.ndarray:
    path = Path(path)
    cache_dir = cache_dir or data_root() / "cache" / "index"
    cache = cache_dir / f"{file_hash(path)}.f16.npy"
    if cache.is_file():
        return np.load(cache)
    vecs = _reconstruct(path).astype(np.float16)
    if vecs.ndim != 2 or vecs.shape[1] != 768:
        raise ValueError(f"{path.name}: expected 768-d vectors, got shape {vecs.shape}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, vecs)
    tmp.replace(cache)
    return vecs


class GpuIndex:
    def __init__(self, vectors: np.ndarray, device: torch.device | str):
        self.vectors = torch.from_numpy(vectors).to(device)  # [N, 768] fp16
        self.norms = self.vectors.float().pow(2).sum(1)  # [N]

    @classmethod
    def load(cls, path: str | Path, device="cuda") -> "GpuIndex":
        return cls(load_vectors(path), device)

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    @torch.no_grad()
    def retrieve(self, feats: torch.Tensor, k: int = TOP_K) -> torch.Tensor:
        """feats: [T, 768] → inverse-square-distance weighted mean of the k nearest vectors."""
        q = feats.to(self.vectors.dtype)
        d2 = (q.float().pow(2).sum(1, keepdim=True) - 2 * (q @ self.vectors.T).float() + self.norms)
        d2, idx = torch.topk(d2.clamp_min(0), min(k, self.size), dim=1, largest=False)
        w = 1.0 / d2.clamp_min(1e-6).pow(2)
        w = w / w.sum(1, keepdim=True)
        picked = self.vectors[idx].float()  # [T, k, 768]
        return (picked * w.unsqueeze(-1)).sum(1).to(feats.dtype)
