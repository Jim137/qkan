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

"""autograd.Function + nn.Module wrapper for cuTile activations."""

from typing import Any

import torch
import torch.nn as nn

try:
    from ..cutile_activations import (
        _VALID_KINDS,
        cutile_activation_backward,
        cutile_activation_forward,
    )

    _CUTILE_ACTIVATION_AVAILABLE = True
except ImportError:
    _CUTILE_ACTIVATION_AVAILABLE = False
    _VALID_KINDS = ("silu", "gelu_exact", "gelu_tanh", "relu", "tanh", "sigmoid")


class _CuTileActivation(torch.autograd.Function):
    """Custom autograd function dispatching to cuTile activation kernels."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, kind: str) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.kind = kind
        ctx.c_dtype = x.dtype
        return cutile_activation_forward(x, kind, x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        (x,) = ctx.saved_tensors
        grad_x = cutile_activation_backward(grad_y, x, ctx.kind, ctx.c_dtype)
        return grad_x, None


class CuTileActivation(nn.Module):
    """Drop-in replacement for ``torch.nn.SiLU`` / ``GELU`` / etc., backed by cuTile.

    Args:
        kind: activation name. One of
            ``{"silu", "gelu_exact", "gelu_tanh", "relu", "tanh", "sigmoid"}``.
    """

    def __init__(self, kind: str = "silu") -> None:
        super().__init__()
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"Unsupported activation kind '{kind}'. Valid: {_VALID_KINDS}"
            )
        if not _CUTILE_ACTIVATION_AVAILABLE:
            raise ImportError(
                "cuda.tile is required for CuTileActivation. "
                "Install with: pip install cuda-tile"
            )
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bypass autograd machinery if no grad needed (matches the cuTile
        # solver pattern in cutile.py — saves ~6-10 us of Python overhead).
        if not torch.is_grad_enabled() or not x.requires_grad:
            return cutile_activation_forward(x, self.kind, x.dtype)
        return _CuTileActivation.apply(x, self.kind)

    def extra_repr(self) -> str:
        return f"kind={self.kind!r}"
