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
    "make_graphed_train_step",
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
# Low-level: train-mode graph capture (forward + loss + backward)
# ---------------------------------------------------------------------------


def _reallocate_params_on_stream(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Clone every trainable Parameter onto the current CUDA stream.

    CUDA-graph capture binds an autograd accumulator to whichever stream the
    Parameter's storage was allocated on. When a model is constructed on the
    default stream and capture is later attempted on a side stream, the
    accumulator launches a kernel on the legacy stream and capture fails with
    ``cudaErrorStreamCaptureImplicit``. Re-allocating the Parameters on the
    capture stream re-creates the accumulators with the right affinity.

    The optimizer's ``param_groups`` are rewired to the new Parameter objects;
    any existing ``optimizer.state`` is re-keyed best-effort (state tensors
    survive but are NOT migrated to a new stream — re-init momentum state if
    you saw any training before).
    """
    old_to_new: dict[int, torch.nn.Parameter] = {}
    for name, p in list(model.named_parameters()):
        if not p.requires_grad:
            continue
        new_p = torch.nn.Parameter(p.data.clone(), requires_grad=True)
        # Walk dotted name to find the parent module and the attribute.
        parent: nn.Module = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_p)
        old_to_new[id(p)] = new_p
    # Rebuild any view/alias caches the model maintains over its Parameters.
    for sub in model.modules():
        init_fn = getattr(sub, "_init_view_caches", None)
        if callable(init_fn):
            init_fn()
    # Update optimizer references.
    new_state: dict = {}
    for group in optimizer.param_groups:
        new_params: list = []
        for p in group["params"]:
            new_p_opt = old_to_new.get(id(p))
            if new_p_opt is None:
                new_params.append(p)
            else:
                new_params.append(new_p_opt)
                if p in optimizer.state:
                    new_state[new_p_opt] = optimizer.state.pop(p)
        group["params"] = new_params
    if new_state:
        optimizer.state.update(new_state)
    # QKAN-aware optimizers (QKANMuon, QKANAdamMini, QKANBeliefMini,
    # QKANSpectralMini) route updates by parameter NAME via an
    # id(param)-keyed side dict; migrate it so routing survives the swap.
    # Popping old ids matters: once the old Parameters are collected,
    # CPython can reuse their ids for unrelated tensors.
    param_names = getattr(optimizer, "_param_names", None)
    if isinstance(param_names, dict):
        for old_id, new_p in old_to_new.items():
            name = param_names.pop(old_id, None)
            if name is not None:
                param_names[id(new_p)] = name
    # QKANMuon caches per-group (muon, adamw) partitions of the old
    # Parameter objects; drop the cache so it rebuilds on the next step.
    partition_cache = getattr(optimizer, "_partition_cache", None)
    if isinstance(partition_cache, dict):
        partition_cache.clear()


def make_graphed_train_step(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    warmup: int = 3,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Capture forward + loss + backward into a CUDA graph for one train step.

    The captured region is everything up to (but excluding) ``optimizer.step``:
    ``optimizer.zero_grad(set_to_none=False)``, ``forward``, ``loss_fn``,
    ``loss.backward()``. The optimizer step is left to the caller because
    optimisers like Adam contain host-side scalar work that breaks capture.

    The returned callable accepts ``(input, target)``, copies them into the
    static buffers, replays the graph, and returns the loss tensor (a view
    into a static buffer — clone before mutating). After replay, every
    parameter's ``.grad`` holds the captured-step gradient and ``optimizer
    .step()`` can be called as usual.

    Side effects on ``model`` and ``optimizer``:
        - Every trainable Parameter is re-allocated on the capture stream so
          the autograd accumulator gets the right stream affinity. References
          held by the caller (e.g. ``my_layer.theta``) ARE updated; references
          taken BEFORE calling this function go stale. Re-fetch
          ``model.parameters()`` if you cached anything.
        - ``optimizer.param_groups`` is rewired to point at the new
          Parameters; ``optimizer.state`` is re-keyed best-effort but state
          tensors are NOT migrated across streams. Adam users running this on
          a freshly-built optimizer (no momentum yet) are unaffected.

    Constraints:
        - ``sample_input`` / ``sample_target`` fix the shapes and dtypes
          of subsequent calls; new shapes require a fresh capture.
        - All parameters of ``model`` must be on CUDA. ``set_to_none=False``
          inside the graph zeroes grads in-place — never swap ``.grad`` for a
          new tensor outside the graph (e.g. don't call
          ``optimizer.zero_grad(set_to_none=True)`` between replays).
        - ``loss_fn`` must produce a scalar tensor.

    Args:
        model: a module in train() mode with CUDA parameters.
        sample_input: representative input tensor.
        sample_target: representative target tensor.
        loss_fn: ``loss_fn(output, target) -> scalar`` (e.g. ``nn.MSELoss()``).
        optimizer: the optimiser whose grads will be zeroed inside the graph.
            ``optimizer.step()`` is NOT captured — call it after each replay.
        warmup: warmup forward/backward passes on a side stream before capture.

    Returns:
        A callable ``train_step(x, y) -> loss``. Loss is a view into a
        static buffer; clone it if you need to retain past the next replay.
    """
    if not sample_input.is_cuda or not sample_target.is_cuda:
        raise ValueError("make_graphed_train_step requires CUDA input/target")

    model.train()

    # Static buffers for I/O — copy_'d on each replay.
    static_input = torch.empty_like(sample_input)
    static_input.copy_(sample_input)
    static_target = torch.empty_like(sample_target)
    static_target.copy_(sample_target)

    # Side stream for warmup + capture. Everything stream-bound below (param
    # reallocation, .grad allocation via first backward, capture itself) must
    # happen here.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # Re-allocate trainable Parameters on the side stream so their grad
        # accumulators bind to the capture stream rather than the legacy
        # default. Required for any model that aliases Parameters via views
        # at __init__ (e.g. QKANLayer p_dim=2).
        _reallocate_params_on_stream(model, optimizer)
        # Warmup: run a full train step (incl. backward) so all .grad
        # buffers are allocated on the side stream, and any lazy state in
        # the model is materialised.
        for _ in range(max(1, warmup)):
            optimizer.zero_grad(set_to_none=True)
            out = model(static_input)
            loss = loss_fn(out, static_target)
            loss.backward()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # Capture: zero_grad(set_to_none=False) keeps grad tensors allocated and
    # zeros them in-place, so the graph sees the same .grad pointers on
    # every replay. Capture stream MUST be the same one we warmed up on.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        optimizer.zero_grad(set_to_none=False)
        static_output = model(static_input)
        static_loss = loss_fn(static_output, static_target)
        static_loss.backward()

    sample_in_shape = sample_input.shape
    sample_in_dtype = sample_input.dtype
    sample_tgt_shape = sample_target.shape
    sample_tgt_dtype = sample_target.dtype

    def train_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != sample_in_shape or x.dtype != sample_in_dtype:
            raise ValueError(
                f"Graphed train_step requires input shape {tuple(sample_in_shape)} "
                f"dtype {sample_in_dtype}; got shape {tuple(x.shape)} "
                f"dtype {x.dtype}"
            )
        if y.shape != sample_tgt_shape or y.dtype != sample_tgt_dtype:
            raise ValueError(
                f"Graphed train_step requires target shape {tuple(sample_tgt_shape)} "
                f"dtype {sample_tgt_dtype}; got shape {tuple(y.shape)} "
                f"dtype {y.dtype}"
            )
        static_input.copy_(x)
        static_target.copy_(y)
        graph.replay()
        return static_loss

    return train_step


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
