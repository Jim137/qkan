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
CuTe-backed pointwise activations for the QKAN base path.

QKANLayer combines a quantum solver output with ``base_activation(x) * base_weight``.
When the solver is on cuTe kernels, this module provides the matching base-path
activations as a single drop-in ``nn.Module`` that doesn't bounce through generic
torch ops. f32 and bf16 I/O are both supported; compute is always f32.

Usage::

    from qkan.solver.cute_activation import CuTeActivation
    base = CuTeActivation("silu")
    y = base(x)  # autograd-aware
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from ..cute_ops import (
        _ACTIVATION_KIND_MAP,
        _CUTE_KERNELS_AVAILABLE,
        cute_activation_backward,
        cute_activation_forward,
    )

    _CUTE_ACTIVATION_AVAILABLE = _CUTE_KERNELS_AVAILABLE
except ImportError:
    _CUTE_ACTIVATION_AVAILABLE = False
    _ACTIVATION_KIND_MAP = {
        "silu": 0,
        "gelu_exact": 1,
        "gelu_tanh": 2,
        "relu": 3,
        "tanh": 4,
        "sigmoid": 5,
    }


__all__ = ["CuTeActivation"]


class _CuTeActivation(torch.autograd.Function):
    """Autograd glue around the cuTe activation kernels."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, kind: str) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.kind = kind
        return cute_activation_forward(x, kind, c_dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        # grad_y must match x's dtype for the kernel; torch normally provides
        # this automatically, but cast defensively in case autocast intervenes.
        if grad_y.dtype != x.dtype:
            grad_y = grad_y.to(x.dtype)
        grad_x = cute_activation_backward(
            grad_y.contiguous(), x, ctx.kind, c_dtype=x.dtype
        )
        return grad_x, None


class CuTeActivation(nn.Module):
    """Drop-in nn.Module for QKAN base activations using cuTe kernels.

    Forward and backward run as a single CUDA kernel each (one per
    activation kind × dtype). f32 and bf16 I/O are supported; compute is
    always done in f32.

    Args:
        kind: One of ``"silu"``, ``"gelu_exact"``, ``"gelu_tanh"``, ``"relu"``,
            ``"tanh"``, ``"sigmoid"``.
    """

    def __init__(self, kind: str = "silu") -> None:
        super().__init__()
        if kind not in _ACTIVATION_KIND_MAP:
            raise ValueError(
                f"Unknown CuTeActivation kind '{kind}'. "
                f"Supported: {sorted(_ACTIVATION_KIND_MAP)}"
            )
        if not _CUTE_ACTIVATION_AVAILABLE:
            raise ImportError(
                "CuTe activation kernels not available. Install with: "
                "CUTLASS_PATH=/path/to/cutlass pip install -e .[cute]"
            )
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bypass autograd machinery when no grad is needed (matches the
        # cute solver pattern in cute.py — saves ~6-10 us of Python overhead).
        if not torch.is_grad_enabled() or not x.requires_grad:
            return cute_activation_forward(x, self.kind, c_dtype=x.dtype)
        return _CuTeActivation.apply(x, self.kind)

    def extra_repr(self) -> str:
        return f"kind={self.kind!r}"
