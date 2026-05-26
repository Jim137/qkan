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

"""Triton-backed elementwise activation as an autograd-aware nn.Module.

Wraps the Triton kernels from ``qkan.triton_activations`` so the activation
runs on the same backend as ``solver="flash"`` (and benefits from the same
fused-kernel launch profile).
"""

from typing import Any, Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .triton_activations import (
        SUPPORTED_KINDS,
        triton_activation_backward,
        triton_activation_forward,
    )

    _TRITON_ACT_AVAILABLE = True
except ImportError:
    _TRITON_ACT_AVAILABLE = False
    SUPPORTED_KINDS = (
        "silu",
        "gelu_exact",
        "gelu_tanh",
        "relu",
        "tanh",
        "sigmoid",
    )

# Element-count threshold below which the inference-time path falls back to
# the torch built-in. Picked from `_bench_activation_backends.py` on bf16,
# RTX 5090: Triton's ~6 us launch overhead loses to torch up to ~1M elements
# (4096 x 256). Above this the Triton kernel starts catching up.
_TRITON_FAST_PATH_NUMEL = 1 << 20  # 1,048,576

# Direct functional refs for the inference fast path. Caching the function
# object avoids ~2 us per call vs constructing a fresh nn.Module each time.
_TORCH_FUNCTIONAL: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": F.silu,
    "gelu_exact": lambda x: F.gelu(x, approximate="none"),
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class _TritonActivation(torch.autograd.Function):
    """Custom autograd function for Triton elementwise activations."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, kind: str) -> torch.Tensor:  # type: ignore[override]
        ctx.kind = kind
        ctx.save_for_backward(x)
        return triton_activation_forward(x, kind)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        grad_x = triton_activation_backward(grad_output, x, ctx.kind)
        return grad_x, None


class TritonActivation(nn.Module):
    """Drop-in elementwise activation backed by Triton kernels.

    Args:
        kind: one of ``"silu"``, ``"gelu_exact"``, ``"gelu_tanh"``, ``"relu"``,
            ``"tanh"``, ``"sigmoid"``.

    The module falls back to the pure-PyTorch reference implementation when the
    input is on CPU or when Triton is not installed.
    """

    def __init__(self, kind: str = "silu") -> None:
        super().__init__()
        if kind not in SUPPORTED_KINDS:
            raise ValueError(
                f"Unsupported activation kind: {kind!r}. "
                f"Supported: {sorted(SUPPORTED_KINDS)}"
            )
        self.kind = kind

    def extra_repr(self) -> str:
        return f"kind={self.kind!r}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _TRITON_ACT_AVAILABLE or not x.is_cuda:
            return _TORCH_FUNCTIONAL[self.kind](x)
        # Small-shape forward-only fast path: the Triton kernel carries a
        # ~6 us fixed launch overhead that loses to torch's pointwise op for
        # all bf16/f32 sizes typical in QKAN per-edge activations (up to
        # ~1M elements on a 5090). Only kicks in during inference, since
        # training needs the autograd-aware path.
        if (
            not torch.is_grad_enabled() or not x.requires_grad
        ) and x.numel() < _TRITON_FAST_PATH_NUMEL:
            return _TORCH_FUNCTIONAL[self.kind](x)
        return cast(torch.Tensor, _TritonActivation.apply(x, self.kind))


__all__ = ["TritonActivation"]
