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

Usage::

    model = HQKANBlock(...).eval()
    sample = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
    graphed = make_graphed_inference(model, sample)
    y = graphed(input_with_matching_shape)

The returned output tensor is the same static buffer on every call, so clone
it before issuing another replay if you need to keep the value around.
"""

from __future__ import annotations

from typing import Callable

import torch


def make_graphed_inference(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    warmup: int = 3,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Capture a CUDA graph for `model(sample_input)` and return a replay fn.

    The input to the returned callable must have the same shape, dtype, and
    device as `sample_input`. The output tensor is reused across calls.

    Args:
        model: an already-configured module in eval() mode. Parameters must
            remain unchanged between capture and replay.
        sample_input: a representative input used to trace the graph. Its
            shape/dtype/device fixes the capture.
        warmup: number of warmup forward passes on a side stream before
            capture (PyTorch recommends at least 3 to force cuBLAS/cuDNN
            kernel selection).

    Returns:
        A callable that takes an input tensor, copies it into the captured
        input buffer, replays the graph, and returns the captured output.
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
