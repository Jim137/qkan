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
FlyDSL-backed pointwise activations for the QKAN base path.

QKANLayer combines a quantum solver output with ``base_activation(x) * base_weight``.
When the solver is ``flydsl``, this module provides the matching base-path
activations as a single drop-in ``nn.Module``. f32 and bf16 I/O are both
supported; compute is always f32.

Usage::

    from qkan.solver.flydsl.flydsl_activation import FlyDSLActivation
    base = FlyDSLActivation("silu")
    y = base(x)  # autograd-aware
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .flydsl_ops import _FLYDSL_KERNELS_AVAILABLE, _launch

__all__ = ["FlyDSLActivation"]

_ACTIVATION_KINDS = ("silu", "gelu_exact", "gelu_tanh", "relu", "tanh", "sigmoid")

_ACT_BLOCK = 256
_GELU_TANH_K0 = 0.7978845608028654  # sqrt(2/pi)
_GELU_TANH_K1 = 0.044715
_INV_SQRT2 = 0.7071067811865476
_INV_SQRT_2PI = 0.3989422804014327  # N(0,1) pdf coefficient


if _FLYDSL_KERNELS_AVAILABLE:
    import flydsl.compiler as flyc  # type: ignore[import-not-found, import-untyped]
    import flydsl.expr as fx  # type: ignore[import-not-found, import-untyped]
    from flydsl.expr import const_expr  # type: ignore[import-not-found, import-untyped]

    def _sigmoid(x):
        return 1.0 / (1.0 + fx.math.exp(0.0 - x))

    def _tanh(x):
        # math.tanh has no ROCDL lowering ("no libcall available for ftanh").
        # exp overflow saturates this cleanly to +/-1.
        return 1.0 - 2.0 / (fx.math.exp(2.0 * x) + 1.0)

    def _act_forward(x, KIND):
        if const_expr(KIND == "silu"):
            return x * _sigmoid(x)
        if const_expr(KIND == "gelu_exact"):
            return 0.5 * x * (1.0 + fx.math.erf(x * _INV_SQRT2))
        if const_expr(KIND == "gelu_tanh"):
            inner = _GELU_TANH_K0 * (x + _GELU_TANH_K1 * x * x * x)
            return 0.5 * x * (1.0 + _tanh(inner))
        if const_expr(KIND == "relu"):
            return fx.max(x, 0.0)
        if const_expr(KIND == "tanh"):
            return _tanh(x)
        return _sigmoid(x)

    def _act_derivative(x, KIND):
        if const_expr(KIND == "silu"):
            s = _sigmoid(x)
            return s * (1.0 + x * (1.0 - s))
        if const_expr(KIND == "gelu_exact"):
            cdf = 0.5 * (1.0 + fx.math.erf(x * _INV_SQRT2))
            pdf = _INV_SQRT_2PI * fx.math.exp(-0.5 * x * x)
            return cdf + x * pdf
        if const_expr(KIND == "gelu_tanh"):
            x2 = x * x
            t = _tanh(_GELU_TANH_K0 * (x + _GELU_TANH_K1 * x2 * x))
            du = _GELU_TANH_K0 * (1.0 + 3.0 * _GELU_TANH_K1 * x2)
            return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
        if const_expr(KIND == "relu"):
            return (x > 0.0).select(fx.Float32(1.0), fx.Float32(0.0))
        if const_expr(KIND == "tanh"):
            t = _tanh(x)
            return 1.0 - t * t
        s = _sigmoid(x)
        return s * (1.0 - s)

    def _load(T, idx, BF16):
        v = T[idx]
        if const_expr(BF16):
            v = v.to(fx.Float32)
        return v

    def _store(T, idx, v, BF16):
        if const_expr(BF16):
            v = v.to(fx.BFloat16)
        T[idx] = v

    @flyc.kernel
    def _act_forward_kernel(
        X: fx.Tensor,
        Y: fx.Tensor,
        n: fx.Int64,
        KIND: fx.Constexpr[str],
        BF16: fx.Constexpr[bool],
    ):
        idx = fx.Int64(fx.block_idx.x) * _ACT_BLOCK + fx.thread_idx.x
        if idx < n:
            _store(Y, idx, _act_forward(_load(X, idx, BF16), KIND), BF16)

    @flyc.kernel
    def _act_backward_kernel(
        GY: fx.Tensor,
        X: fx.Tensor,
        GX: fx.Tensor,
        n: fx.Int64,
        KIND: fx.Constexpr[str],
        BF16: fx.Constexpr[bool],
    ):
        idx = fx.Int64(fx.block_idx.x) * _ACT_BLOCK + fx.thread_idx.x
        if idx < n:
            x = _load(X, idx, BF16)
            gy = _load(GY, idx, BF16)
            _store(GX, idx, gy * _act_derivative(x, KIND), BF16)

    @flyc.jit
    def _launch_act_forward(
        X: fx.Tensor,
        Y: fx.Tensor,
        n: fx.Int64,
        n_blocks: fx.Int32,
        KIND: fx.Constexpr[str],
        BF16: fx.Constexpr[bool],
        stream: fx.Stream = fx.Stream(None),
    ):
        _act_forward_kernel(X, Y, n, KIND, BF16).launch(
            grid=(n_blocks, 1, 1), block=(_ACT_BLOCK, 1, 1), stream=stream
        )

    @flyc.jit
    def _launch_act_backward(
        GY: fx.Tensor,
        X: fx.Tensor,
        GX: fx.Tensor,
        n: fx.Int64,
        n_blocks: fx.Int32,
        KIND: fx.Constexpr[str],
        BF16: fx.Constexpr[bool],
        stream: fx.Stream = fx.Stream(None),
    ):
        _act_backward_kernel(GY, X, GX, n, KIND, BF16).launch(
            grid=(n_blocks, 1, 1), block=(_ACT_BLOCK, 1, 1), stream=stream
        )


def _io_view(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Flat contiguous f32/bf16 view of ``t`` and whether it is bf16."""
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(-1), True
    return t.to(torch.float32).contiguous().view(-1), False


def flydsl_activation_forward(x: torch.Tensor, kind: str) -> torch.Tensor:
    """FlyDSL pointwise activation; returns a tensor shaped like ``x``.

    bf16 inputs keep bf16 I/O; any other dtype is computed and returned in f32
    and cast back to ``x.dtype``.
    """
    xf, bf16 = _io_view(x)
    y = torch.empty_like(xf)
    n = xf.numel()
    if n:
        _launch(
            _launch_act_forward,
            (kind, bf16),
            xf,
            y,
            n,
            -(-n // _ACT_BLOCK),
            kind,
            bf16,
            torch.cuda.current_stream(x.device),
        )
    return y.view(x.shape).to(x.dtype)


def flydsl_activation_backward(
    grad_y: torch.Tensor, x: torch.Tensor, kind: str
) -> torch.Tensor:
    """Gradient of :func:`flydsl_activation_forward` with respect to ``x``."""
    xf, bf16 = _io_view(x)
    gyf = grad_y.to(xf.dtype).contiguous().view(-1)
    gx = torch.empty_like(xf)
    n = xf.numel()
    if n:
        _launch(
            _launch_act_backward,
            (kind, bf16),
            gyf,
            xf,
            gx,
            n,
            -(-n // _ACT_BLOCK),
            kind,
            bf16,
            torch.cuda.current_stream(x.device),
        )
    return gx.view(x.shape).to(x.dtype)


class _FlyDSLActivation(torch.autograd.Function):
    """Autograd glue around the FlyDSL activation kernels."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, kind: str) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.kind = kind
        return flydsl_activation_forward(x, kind)

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        return flydsl_activation_backward(grad_y, x, ctx.kind), None


class FlyDSLActivation(nn.Module):
    """Drop-in nn.Module for QKAN base activations using FlyDSL kernels.

    Forward and backward run as a single kernel each (one per activation
    kind × dtype). f32 and bf16 I/O are supported; compute is always f32.

    Args:
        kind: One of ``"silu"``, ``"gelu_exact"``, ``"gelu_tanh"``, ``"relu"``,
            ``"tanh"``, ``"sigmoid"``.
    """

    def __init__(self, kind: str = "silu") -> None:
        super().__init__()
        if kind not in _ACTIVATION_KINDS:
            raise ValueError(
                f"Unknown FlyDSLActivation kind '{kind}'. "
                f"Supported: {sorted(_ACTIVATION_KINDS)}"
            )
        if not _FLYDSL_KERNELS_AVAILABLE:
            raise ImportError(
                "FlyDSL activation kernels need the flydsl package and a ROCm "
                "build of PyTorch: pip install qkan[flydsl]"
            )
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() or not x.requires_grad:
            return flydsl_activation_forward(x, self.kind)
        return _FlyDSLActivation.apply(x, self.kind)

    def extra_repr(self) -> str:
        return f"kind={self.kind!r}"
