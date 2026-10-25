"""CUDA-graph capture of the fixed-shape inference stages.

A streaming session has a fixed window, so every stage sees the same shapes on every block.
Capturing them once removes per-kernel launch overhead (hundreds of small kernels per block),
which is most of the eager cost and most of its jitter.

Outputs of a graph are static buffers that the next replay overwrites; consume or clone them
before replaying again.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch


class Graph:
    def __init__(self, fn: Callable[..., torch.Tensor | tuple], inputs: Sequence[torch.Tensor], warmup: int = 2):
        self.fn = fn
        self.inputs = [x.detach().clone() for x in inputs]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(warmup):  # lazy init (cuBLAS/cuFFT plans, allocator) outside capture
                fn(*self.inputs)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        # A dedicated capture stream gives this graph its own cuBLAS workspace (keyed per stream);
        # graphs captured on a shared stream race on it when replayed concurrently. Thread-local
        # capture lets the stream thread keep running while a new voice is captured.
        with torch.no_grad(), torch.cuda.graph(self.graph, stream=side, capture_error_mode="thread_local"):
            self.outputs = fn(*self.inputs)

    def __call__(self, *inputs: torch.Tensor):
        for dst, src in zip(self.inputs, inputs):
            dst.copy_(src, non_blocking=True)
        self.graph.replay()
        return self.outputs


def available(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()
