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
Triton feature map of the Fourier-GEMM solver (``solver="te"``).

``[cos kx, sin kx, act(x)]`` written straight into the GEMM operand, in its
dtype and already padded, by one forward kernel; one backward kernel
recomputes the trig from ``x``.  All math runs in float32 inside the kernels
(cast on load, cast on store).  Second derivatives (``create_graph=True``) go
through torch ops instead of the backward kernel.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from triton.language.extra import libdevice  # type: ignore

from .flash.triton_activation import _TORCH_FUNCTIONAL
from .flash.triton_activations import (
    _GELU_TANH_K0,
    _GELU_TANH_K1,
    _INV_SQRT2,
    _INV_SQRT_2PI,
    _check_kind,
)

# What the kernels read and write.
_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

# Elements of x per program, tiled (BLOCK_B rows, BLOCK_I columns).  Both
# kernels run at memory bandwidth, so the exact size and num_warps hardly matter.
_BLOCK = 1024


@triton.jit
def _act(x, KIND: tl.constexpr):
    """Activation ``KIND`` of float32 ``x``; formulas of flash/triton_activations.py."""
    if KIND == "silu":
        y = x * tl.sigmoid(x)
    elif KIND == "gelu_exact":
        y = 0.5 * x * (1.0 + tl.erf(x * _INV_SQRT2))
    elif KIND == "gelu_tanh":
        inner = _GELU_TANH_K0 * (x + _GELU_TANH_K1 * x * x * x)
        y = 0.5 * x * (1.0 + libdevice.tanh(inner))
    elif KIND == "relu":
        y = tl.where(x > 0.0, x, 0.0)
    elif KIND == "tanh":
        y = libdevice.tanh(x)
    else:  # sigmoid
        y = tl.sigmoid(x)
    return y


@triton.jit
def _act_grad(x, KIND: tl.constexpr):
    """Derivative of ``_act`` at float32 ``x``."""
    if KIND == "silu":
        s = tl.sigmoid(x)
        dydx = s * (1.0 + x * (1.0 - s))
    elif KIND == "gelu_exact":
        cdf = 0.5 * (1.0 + tl.erf(x * _INV_SQRT2))
        pdf = _INV_SQRT_2PI * tl.exp(-0.5 * x * x)
        dydx = cdf + x * pdf
    elif KIND == "gelu_tanh":
        x2 = x * x
        t = libdevice.tanh(_GELU_TANH_K0 * (x + _GELU_TANH_K1 * x * x2))
        du_dx = _GELU_TANH_K0 * (1.0 + 3.0 * _GELU_TANH_K1 * x2)
        dydx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
    elif KIND == "relu":
        dydx = tl.where(x > 0.0, 1.0, 0.0)
    elif KIND == "tanh":
        t = libdevice.tanh(x)
        dydx = 1.0 - t * t
    else:  # sigmoid
        s = tl.sigmoid(x)
        dydx = s * (1.0 - s)
    return dydx


# out_stride is not specialized: knowing it is a multiple of 16 while in_dim is
# not (any align=32 output of an odd width), Triton gives each output stream
# its own thread layout, and converting between them makes the forward a third
# slower.  Nor is batch: the last, partial batch of an epoch would compile
# both kernels again.
@triton.jit(do_not_specialize=["out_stride", "batch"])
def _features_fwd_kernel(
    x_ptr,  # [Batch, In_Dim]
    out_ptr,  # [>=Batch, Out_Stride], Out_Stride >= (2*REPS+1) * In_Dim
    batch,
    in_dim,
    out_stride,
    REPS: tl.constexpr,
    KIND: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Frequency-major features of a (BLOCK_B, BLOCK_I) tile of x.

    Grid: (cdiv(batch, BLOCK_B) * cdiv(in_dim, BLOCK_I),).
    out[b, f*In_Dim + i]: cos((f+1) x) for f < REPS, sin((f-REPS+1) x) for
    f < 2*REPS, act(x) for f = 2*REPS.
    """
    pid = tl.program_id(0)
    tiles_i = tl.cdiv(in_dim, BLOCK_I)
    # int64 rows: the output passes 2^31 elements at realistic sizes (batch 1M
    # x 3072 features), and tl.program_id and the int arguments are int32
    # (#31).  Same in the backward kernel.
    b = (pid // tiles_i).to(tl.int64) * BLOCK_B + tl.arange(0, BLOCK_B)
    i = (pid % tiles_i) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (b[:, None] < batch) & (i[None, :] < in_dim)

    x = tl.load(x_ptr + b[:, None] * in_dim + i[None, :], mask=mask).to(tl.float32)
    out = out_ptr + b[:, None] * out_stride + i[None, :]

    # One sin/cos per element; harmonic k+1 by angle addition from harmonic k.
    c1 = tl.cos(x)
    s1 = tl.sin(x)
    c = c1
    s = s1
    for k in tl.static_range(REPS):
        tl.store(out + k * in_dim, c, mask=mask)
        tl.store(out + (REPS + k) * in_dim, s, mask=mask)
        c, s = c * c1 - s * s1, s * c1 + c * s1
    tl.store(out + 2 * REPS * in_dim, _act(x, KIND), mask=mask)


@triton.jit(do_not_specialize=["batch"])
def _features_bwd_kernel(
    x_ptr,  # [Batch, In_Dim]
    g_ptr,  # [>=Batch, G_Stride], gradient of the forward output
    gx_ptr,  # [Batch, In_Dim]
    batch,
    in_dim,
    g_stride,
    REPS: tl.constexpr,
    KIND: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Input gradient of ``_features_fwd_kernel``, same tiling and grid.

    gx = sum_k k * (cos(kx) * g_sin_k - sin(kx) * g_cos_k) + act'(x) * g_act
    """
    pid = tl.program_id(0)
    tiles_i = tl.cdiv(in_dim, BLOCK_I)
    b = (pid // tiles_i).to(tl.int64) * BLOCK_B + tl.arange(0, BLOCK_B)
    i = (pid % tiles_i) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (b[:, None] < batch) & (i[None, :] < in_dim)

    x_offs = b[:, None] * in_dim + i[None, :]
    x = tl.load(x_ptr + x_offs, mask=mask).to(tl.float32)
    g = g_ptr + b[:, None] * g_stride + i[None, :]

    g_act = tl.load(g + 2 * REPS * in_dim, mask=mask).to(tl.float32)
    gx = _act_grad(x, KIND) * g_act
    c1 = tl.cos(x)
    s1 = tl.sin(x)
    c = c1
    s = s1
    for k in tl.static_range(REPS):
        g_cos = tl.load(g + k * in_dim, mask=mask).to(tl.float32)
        g_sin = tl.load(g + (REPS + k) * in_dim, mask=mask).to(tl.float32)
        gx += (k + 1) * (c * g_sin - s * g_cos)
        c, s = c * c1 - s * s1, s * c1 + c * s1
    tl.store(gx_ptr + x_offs, gx, mask=mask)


def _tiling(batch: int, in_dim: int) -> tuple[tuple[int], int, int]:
    """Grid and (BLOCK_B, BLOCK_I): whole rows up to ``_BLOCK`` columns."""
    block_i = min(triton.next_power_of_2(in_dim), _BLOCK)
    block_b = _BLOCK // block_i
    grid = (triton.cdiv(batch, block_b) * triton.cdiv(in_dim, block_i),)
    return grid, block_b, block_i


def _features_forward(
    x: torch.Tensor, reps: int, kind: str, dtype: torch.dtype, align: int
) -> torch.Tensor:
    """Contiguous x (B, I) -> features (Bp, Kp) of ``dtype``, see ``fourier_features``."""
    batch, in_dim = x.shape
    k = (2 * reps + 1) * in_dim
    rows, cols = -batch % align, -k % align
    out = torch.empty((batch + rows, k + cols), device=x.device, dtype=dtype)
    # Only the padding strips; the kernel writes everything else.
    if rows:
        out[batch:].zero_()
    if cols:
        out[:batch, k:].zero_()
    if x.numel() == 0:
        return out
    grid, block_b, block_i = _tiling(batch, in_dim)
    _features_fwd_kernel[grid](
        x,
        out,
        batch,
        in_dim,
        out.shape[1],
        REPS=reps,
        KIND=kind,
        BLOCK_B=block_b,
        BLOCK_I=block_i,
    )
    return out


def _features_backward(
    x: torch.Tensor, grad_out: torch.Tensor, reps: int, kind: str
) -> torch.Tensor:
    """Contiguous x (B, I), grad_out (Bp, Kp) -> grad_x (B, I) in x's dtype."""
    grad_out = grad_out.contiguous()
    batch, in_dim = x.shape
    grad_x = torch.empty_like(x)
    if x.numel() == 0:
        return grad_x
    grid, block_b, block_i = _tiling(batch, in_dim)
    _features_bwd_kernel[grid](
        x,
        grad_out,
        grad_x,
        batch,
        in_dim,
        grad_out.shape[1],
        REPS=reps,
        KIND=kind,
        BLOCK_B=block_b,
        BLOCK_I=block_i,
    )
    return grad_x


def _features_backward_torch(
    x: torch.Tensor, grad_out: torch.Tensor, reps: int, kind: str
) -> torch.Tensor:
    """``_features_backward`` in torch ops, which autograd can differentiate again."""
    k = torch.arange(1, reps + 1, device=x.device, dtype=x.dtype)
    kx = k[:, None] * x[:, None, :]
    act = _TORCH_FUNCTIONAL[kind](x)
    features = torch.cat([kx.cos(), kx.sin(), act[:, None]], dim=1).flatten(1)
    grad_out = grad_out[: x.shape[0], : features.shape[1]]
    (grad_x,) = torch.autograd.grad(features, x, grad_out, create_graph=True)
    return grad_x


class _FourierFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, reps, kind, dtype, align):
        ctx.save_for_backward(x)
        ctx.reps, ctx.kind = reps, kind
        return _features_forward(x, reps, kind, dtype, align)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # Grad mode is on in a backward only under create_graph=True (a PINN
        # loss on du/dx).  The kernel's result would be a constant there and
        # the second-order term would vanish without an error.
        if torch.is_grad_enabled():
            grad_x = _features_backward_torch(x, grad_out, ctx.reps, ctx.kind)
        else:
            grad_x = _features_backward(x, grad_out, ctx.reps, ctx.kind)
        return grad_x, None, None, None, None


def fourier_features(
    x: torch.Tensor, reps: int, kind: str, dtype: torch.dtype, align: int = 1
) -> torch.Tensor:
    """``[cos kx, sin kx, act(x)]`` as a GEMM operand, differentiable in ``x``.

    Args:
        x: (B, I) CUDA tensor, float32 / bfloat16 / float16, any contiguity.
        reps: number of harmonics R >= 1.
        kind: activation, one of "silu", "gelu_exact", "gelu_tanh", "relu",
            "tanh", "sigmoid".
        dtype: output dtype, float32 / bfloat16 / float16.
        align: both output dims are padded with zeros up to a multiple of it.

    Returns:
        (Bp, Kp) tensor of ``dtype``, K = (2R+1)*I, frequency-major:
        ``out[b, f*I + i]`` is ``cos((f+1) x[b, i])`` for f < R,
        ``sin((f-R+1) x[b, i])`` for R <= f < 2R and ``act(x[b, i])`` for
        f = 2R; rows >= B and columns >= K are zero.
    """
    _check_kind(kind)
    # int(): a numpy integer is not a valid Triton constexpr.  The copy of a
    # strided x is made here, once, so the backward reuses it.
    return _FourierFeatures.apply(x.contiguous(), int(reps), kind, dtype, align)
