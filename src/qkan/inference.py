# Copyright (c) 2026, Jiun-Cheng Jiang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inference helpers: CUDA graph capture for QKAN.

At small batch sizes QKAN inference is CPU-bound — roughly 8 kernel launches per
forward with ~2 us of Python/launch overhead each. A CUDA graph replays the
captured stream of kernels as a single submission, eliminating almost all of
that overhead. For GPT-style HQKAN blocks this typically yields a 2-3x speedup.

Friendly API (recommended)
--------------------------
``compile_inference`` wraps any module with lazy, per-shape CUDA-graph
capture. First call at a new shape captures; subsequent calls replay.
Training / ``requires_grad`` inputs transparently fall back to eager::

    import qkan
    model = MyModel(...).cuda().eval()
    model = qkan.compile_inference(model)          # drop-in wrapper
    with torch.no_grad():
        y = model(x)                               # captures on first call
        y2 = model(x)                              # replays (2-3x faster)

For multi-block transformers, install per-block graphs in one shot::

    qkan.graph_submodules(transformer, sample_input, predicate=lambda m: isinstance(m, MyMLP))

Low-level API
-------------
``make_graphed_inference(module, sample)`` captures a single-shape graph and
returns a bare callable — useful when you know the shape ahead of time and
don't want the shape-dispatch overhead.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

__all__ = [
    "make_graphed_inference",
    "compile_inference",
    "CompiledInference",
    "graph_submodules",
]


# ---------------------------------------------------------------------------
# Low-level: single-shape graph capture
# ---------------------------------------------------------------------------


def make_graphed_inference(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup: int = 3,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Capture a CUDA graph for ``model(sample_input)`` and return a replay fn.

    The input to the returned callable must match ``sample_input`` in shape,
    dtype, and device. The output tensor is reused across calls — clone it
    before issuing the next replay if you need to keep the value around.

    Args:
        model: a module in eval() mode. Parameters must not change between
            capture and replay.
        sample_input: representative input — its shape/dtype/device fixes the
            capture.
        warmup: warmup forward passes on a side stream before capture.
            PyTorch recommends >=3 to stabilise cuBLAS/cuDNN selection.

    Returns:
        A callable ``fn(x) -> y`` that replays the captured graph.
    """
    if not sample_input.is_cuda:
        raise ValueError("make_graphed_inference requires a CUDA input")

    model.eval()
    was_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        static_input = torch.empty_like(sample_input)
        static_input.copy_(sample_input)

        # Warmup on a side stream so the main stream stays clean for capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(1, warmup)):
                model(static_input)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model(static_input)
    finally:
        torch.set_grad_enabled(was_grad)

    sample_shape = sample_input.shape
    sample_dtype = sample_input.dtype

    def replay(x: torch.Tensor) -> torch.Tensor:
        if x.shape != sample_shape or x.dtype != sample_dtype:
            raise ValueError(
                f"Graphed callable requires shape {tuple(sample_shape)} "
                f"dtype {sample_dtype}; got shape {tuple(x.shape)} "
                f"dtype {x.dtype}"
            )
        static_input.copy_(x)
        graph.replay()
        return static_output

    return replay


# ---------------------------------------------------------------------------
# Friendly API: lazy, multi-shape drop-in wrapper
# ---------------------------------------------------------------------------


def _input_key(x: torch.Tensor) -> Optional[tuple]:
    """Hashable key for graph-cache lookup; None when graphs don't apply."""
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return None
    return (tuple(x.shape), x.dtype, x.device.index)


class CompiledInference(nn.Module):
    """Transparent wrapper that captures CUDA graphs lazily per input shape.

    Behaves exactly like the wrapped module in training (``self.training`` is
    True) or when grad is enabled or the input is not a single CUDA tensor —
    so you can wrap a model once and keep using it normally::

        model = CompiledInference(model)
        model.train(); model(x).sum().backward()   # eager, grad flows
        model.eval()
        with torch.no_grad():
            model(x)                               # captures + replays

    On each eval/no-grad forward, the (shape, dtype, device) of the input is
    used as the cache key. A miss triggers a capture (up to ``max_shapes``);
    a hit replays the captured graph. When the cache is full the fallback is
    eager execution — graphs are not evicted.

    Args:
        module: the module to wrap. ``forward`` must take one tensor arg.
        max_shapes: maximum number of distinct input shapes to cache.
        warmup: warmup forward passes before each capture.
    """

    def __init__(
        self,
        module: nn.Module,
        max_shapes: int = 8,
        warmup: int = 3,
    ) -> None:
        super().__init__()
        self.module = module
        self.max_shapes = int(max_shapes)
        self.warmup = int(warmup)
        # key -> (graph, static_input, static_output)
        self._cache: dict[
            tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]
        ] = {}

    # Delegate train/eval/to/state_dict transparently.
    def train(self, mode: bool = True):
        self.module.train(mode)
        # Param changes invalidate captured graphs.
        self._cache.clear()
        return super().train(mode)

    def clear_cache(self) -> None:
        """Drop all captured graphs. Call after editing parameters in-place."""
        self._cache.clear()

    @torch.no_grad()
    def _capture(
        self, sample: torch.Tensor
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
        static_input = torch.empty_like(sample)
        static_input.copy_(sample)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(1, self.warmup)):
                self.module(static_input)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.module(static_input)
        return graph, static_input, static_output

    def forward(self, x: torch.Tensor, *extra: Any, **kwargs: Any) -> torch.Tensor:
        # Any path that can't be safely graphed falls through to eager.
        if (
            self.training
            or torch.is_grad_enabled()
            or extra
            or kwargs
            or not isinstance(x, torch.Tensor)
            or not x.is_cuda
            or x.requires_grad
        ):
            return self.module(x, *extra, **kwargs)

        key = _input_key(x)
        if key is None:
            return self.module(x)

        entry = self._cache.get(key)
        if entry is None:
            if len(self._cache) >= self.max_shapes:
                # Cache is saturated — fall back to eager rather than evict
                # (evicting mid-inference would trash ongoing replays).
                return self.module(x)
            entry = self._capture(x)
            self._cache[key] = entry

        graph, static_input, static_output = entry
        static_input.copy_(x)
        graph.replay()
        return static_output


def compile_inference(
    module: nn.Module,
    max_shapes: int = 8,
    warmup: int = 3,
) -> CompiledInference:
    """Shortcut: ``compile_inference(m) == CompiledInference(m)``.

    Drop-in replacement that uses CUDA graphs for inference and falls back to
    eager execution for training or gradient-tracking forwards. See
    ``CompiledInference`` for details.
    """
    return CompiledInference(module, max_shapes=max_shapes, warmup=warmup)


# ---------------------------------------------------------------------------
# Helper: apply CUDA graphs to selected submodules (e.g. every MLP block)
# ---------------------------------------------------------------------------


def graph_submodules(
    model: nn.Module,
    sample_input: torch.Tensor,
    predicate: Callable[[nn.Module], bool],
    max_shapes: int = 8,
    warmup: int = 3,
) -> nn.Module:
    """Wrap every submodule matching ``predicate`` with ``CompiledInference``.

    Useful for transformer-style models where full-model graph capture fails
    (e.g. SDPA backends increment an RNG counter even at dropout_p=0). Wrap
    each MLP block instead — QKAN's launch-bound cost is concentrated there.

    Example::

        qkan.graph_submodules(
            gpt2, sample_idx,
            predicate=lambda m: isinstance(m, HQKANMLP),
        )

    The model is modified in place; replaced submodules wrap the originals so
    they share parameters (``named_parameters`` is unchanged).
    """
    # Collect matches first to avoid mutating during traversal.
    targets: list[tuple[nn.Module, str, nn.Module]] = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if predicate(child):
                targets.append((parent, name, child))
    for parent, name, child in targets:
        setattr(
            parent, name, CompiledInference(child, max_shapes=max_shapes, warmup=warmup)
        )
    # Trigger an initial capture pass with the provided sample so the first
    # live forward doesn't pay the compile cost.
    model.eval()
    with torch.no_grad():
        model(sample_input)
    return model
