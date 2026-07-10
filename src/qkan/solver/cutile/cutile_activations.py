# mypy: ignore-errors
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
cuTile elementwise activation kernels for QKAN base path.

Provides forward and backward cuTile kernels for the standard activation
set used by ``QKANLayer.base_activation``: silu, gelu (exact + tanh),
relu, tanh, sigmoid. f32 and bf16 IO supported; compute is always f32 for
numerical accuracy and bf16 outputs are cast at scatter time (mirroring
the existing fused QKAN cuTile kernels).

Kind dispatch uses integer codes because cuTile's launch wrapper rejects
``str`` arguments even when the kernel parameter is annotated
``Constant[str]``. Integer codes annotated as ``Constant[int]`` constant-
fold the branch at JIT-compile time, yielding kernel specializations
equivalent to having one kernel per kind.
"""

import math

import cuda.tile as ct  # type: ignore
import torch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# ── erf approximation (Abramowitz & Stegun 7.1.26) ──────────────────────────
#
# Used by gelu_exact. cuTile does not expose ``erf`` natively, so we inline
# a high-accuracy polynomial. Maximum absolute error ~1.5e-7 for f32 — safe
# for the < 1e-5 tolerance the spec demands on f32 IO.
_ERF_A1 = 0.254829592
_ERF_A2 = -0.284496736
_ERF_A3 = 1.421413741
_ERF_A4 = -1.453152027
_ERF_A5 = 1.061405429
_ERF_P = 0.3275911


# ── kind codes (must match the integer literals in the kernel) ──────────────
_KIND_SILU = 0
_KIND_GELU_EXACT = 1
_KIND_GELU_TANH = 2
_KIND_RELU = 3
_KIND_TANH = 4
_KIND_SIGMOID = 5

_KIND_TO_CODE = {
    "silu": _KIND_SILU,
    "gelu_exact": _KIND_GELU_EXACT,
    "gelu_tanh": _KIND_GELU_TANH,
    "relu": _KIND_RELU,
    "tanh": _KIND_TANH,
    "sigmoid": _KIND_SIGMOID,
}
_VALID_KINDS = tuple(_KIND_TO_CODE.keys())


# Tile-block size for elementwise ops. 1024 picks a sweet spot: large
# enough to amortize launch / per-block indexing overhead at f32 speeds
# (4 KiB per block) yet small enough to keep many CTAs resident.
_BLOCK_SIZE = 1024


# ── forward kernel ──────────────────────────────────────────────────────────


@ct.kernel
def _activation_forward_kernel(
    x,  # [N]
    out,  # [N]
    n_elems: ConstInt,
    KIND: ConstInt,
    COMPUTE_BF16: ConstBool,
    BLOCK_SIZE: ConstInt,
):
    """Elementwise activation forward.

    Grid: (cdiv(N, BLOCK_SIZE),). One program processes BLOCK_SIZE elements.
    Compute is always f32; output is cast back to bf16 if ``COMPUTE_BF16``.
    """
    pid = ct.bid(0)
    offs = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = offs < n_elems

    x_vals = ct.astype(ct.gather(x, (offs,), mask=mask, padding_value=0.0), ct.float32)

    if KIND == 0:  # silu
        # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        sig = 1.0 / (1.0 + ct.exp(-x_vals))
        y = x_vals * sig
    elif KIND == 1:  # gelu_exact
        # gelu_exact(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        INV_SQRT2 = 0.7071067811865476
        z = x_vals * INV_SQRT2
        sign = ct.where(z >= 0.0, 1.0, -1.0)
        az = ct.where(z >= 0.0, z, -z)
        t = 1.0 / (1.0 + _ERF_P * az)
        # Horner evaluation of A&S 7.1.26 polynomial.
        poly = _ERF_A5
        poly = poly * t + _ERF_A4
        poly = poly * t + _ERF_A3
        poly = poly * t + _ERF_A2
        poly = poly * t + _ERF_A1
        erf_val = sign * (1.0 - poly * t * ct.exp(-az * az))
        y = 0.5 * x_vals * (1.0 + erf_val)
    elif KIND == 2:  # gelu_tanh
        # gelu_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        SQRT_2_OVER_PI = 0.7978845608028654
        inner = SQRT_2_OVER_PI * (x_vals + 0.044715 * x_vals * x_vals * x_vals)
        y = 0.5 * x_vals * (1.0 + ct.tanh(inner))
    elif KIND == 3:  # relu
        y = ct.where(x_vals > 0.0, x_vals, 0.0)
    elif KIND == 4:  # tanh
        y = ct.tanh(x_vals)
    else:  # sigmoid (KIND == 5)
        y = 1.0 / (1.0 + ct.exp(-x_vals))

    if COMPUTE_BF16:
        ct.scatter(out, (offs,), ct.astype(y, ct.bfloat16), mask=mask)
    else:
        ct.scatter(out, (offs,), y, mask=mask)


# ── backward kernel ─────────────────────────────────────────────────────────


@ct.kernel
def _activation_backward_kernel(
    x,  # [N]
    grad_y,  # [N]
    grad_x,  # [N]
    n_elems: ConstInt,
    KIND: ConstInt,
    COMPUTE_BF16: ConstBool,
    BLOCK_SIZE: ConstInt,
):
    """Elementwise activation backward.

    grad_x[i] = grad_y[i] * d activation(x[i]) / d x.
    Compute is always f32; output is cast back to bf16 if ``COMPUTE_BF16``.
    """
    pid = ct.bid(0)
    offs = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = offs < n_elems

    x_vals = ct.astype(ct.gather(x, (offs,), mask=mask, padding_value=0.0), ct.float32)
    gy = ct.astype(ct.gather(grad_y, (offs,), mask=mask, padding_value=0.0), ct.float32)

    if KIND == 0:  # silu
        # d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = 1.0 / (1.0 + ct.exp(-x_vals))
        d = sig * (1.0 + x_vals * (1.0 - sig))
        gx = gy * d
    elif KIND == 1:  # gelu_exact
        # d/dx [0.5 x (1 + erf(x/sqrt(2)))] = 0.5 (1 + erf(x/sqrt(2))) + x * phi(x)
        # where phi(x) = (1/sqrt(2*pi)) exp(-x^2/2)
        INV_SQRT2 = 0.7071067811865476
        INV_SQRT_2PI = 0.3989422804014327
        z = x_vals * INV_SQRT2
        sign = ct.where(z >= 0.0, 1.0, -1.0)
        az = ct.where(z >= 0.0, z, -z)
        t = 1.0 / (1.0 + _ERF_P * az)
        poly = _ERF_A5
        poly = poly * t + _ERF_A4
        poly = poly * t + _ERF_A3
        poly = poly * t + _ERF_A2
        poly = poly * t + _ERF_A1
        erf_val = sign * (1.0 - poly * t * ct.exp(-az * az))
        phi = INV_SQRT_2PI * ct.exp(-0.5 * x_vals * x_vals)
        d = 0.5 * (1.0 + erf_val) + x_vals * phi
        gx = gy * d
    elif KIND == 2:  # gelu_tanh
        # Let u = sqrt(2/pi)*(x + 0.044715*x^3), s = tanh(u).
        # y = 0.5*x*(1 + s); du/dx = sqrt(2/pi)*(1 + 3*0.044715*x^2)
        # dy/dx = 0.5*(1 + s) + 0.5*x*(1 - s^2)*du/dx
        SQRT_2_OVER_PI = 0.7978845608028654
        x2 = x_vals * x_vals
        inner = SQRT_2_OVER_PI * (x_vals + 0.044715 * x_vals * x2)
        s = ct.tanh(inner)
        du_dx = SQRT_2_OVER_PI * (1.0 + 0.134145 * x2)
        d = 0.5 * (1.0 + s) + 0.5 * x_vals * (1.0 - s * s) * du_dx
        gx = gy * d
    elif KIND == 3:  # relu
        gx = ct.where(x_vals > 0.0, gy, 0.0)
    elif KIND == 4:  # tanh
        # d/dx tanh(x) = 1 - tanh(x)^2
        th = ct.tanh(x_vals)
        gx = gy * (1.0 - th * th)
    else:  # sigmoid (KIND == 5)
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        sig = 1.0 / (1.0 + ct.exp(-x_vals))
        gx = gy * sig * (1.0 - sig)

    if COMPUTE_BF16:
        ct.scatter(grad_x, (offs,), ct.astype(gx, ct.bfloat16), mask=mask)
    else:
        ct.scatter(grad_x, (offs,), gx, mask=mask)


# ── host-side launch wrappers ───────────────────────────────────────────────


def _kind_code(kind: str) -> int:
    if kind not in _KIND_TO_CODE:
        raise ValueError(f"Unsupported activation kind '{kind}'. Valid: {_VALID_KINDS}")
    return _KIND_TO_CODE[kind]


def _resolve_io_dtype(c_dtype: torch.dtype) -> tuple[torch.dtype, bool]:
    """Return (io_dtype, compute_bf16). f32 stays f32; everything else is bf16 IO."""
    if c_dtype == torch.float32:
        return torch.float32, False
    return torch.bfloat16, True


def cutile_activation_forward(
    x: torch.Tensor, kind: str, c_dtype: torch.dtype
) -> torch.Tensor:
    """Launch the cuTile elementwise activation forward kernel.

    Args:
        x: input tensor on CUDA, any shape; will be flattened internally.
        kind: one of ``_VALID_KINDS``.
        c_dtype: compute / storage dtype (``torch.float32`` or ``torch.bfloat16``).

    Returns:
        Tensor of the same shape as ``x`` in the chosen IO dtype.
    """
    code = _kind_code(kind)
    if not x.is_cuda:
        raise ValueError("cutile_activation_forward requires a CUDA tensor")

    io_dtype, compute_bf16 = _resolve_io_dtype(c_dtype)
    x_io = x if x.dtype == io_dtype else x.to(io_dtype)
    x_flat = x_io.contiguous().view(-1)
    n_elems = x_flat.numel()
    out_flat = torch.empty(n_elems, device=x.device, dtype=io_dtype)

    if n_elems == 0:
        return out_flat.view_as(x)

    grid = (math.ceil(n_elems / _BLOCK_SIZE), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _activation_forward_kernel,
        (x_flat, out_flat, n_elems, code, compute_bf16, _BLOCK_SIZE),
    )
    return out_flat.view_as(x)


def cutile_activation_backward(
    grad_y: torch.Tensor, x: torch.Tensor, kind: str, c_dtype: torch.dtype
) -> torch.Tensor:
    """Launch the cuTile elementwise activation backward kernel.

    Args:
        grad_y: upstream gradient with the same shape as ``x``.
        x: input tensor saved from the forward pass.
        kind: one of ``_VALID_KINDS``.
        c_dtype: compute / storage dtype (``torch.float32`` or ``torch.bfloat16``).

    Returns:
        ``grad_x`` with the same shape as ``x`` in the chosen IO dtype.
    """
    code = _kind_code(kind)
    if not x.is_cuda or not grad_y.is_cuda:
        raise ValueError("cutile_activation_backward requires CUDA tensors")
    if grad_y.shape != x.shape:
        raise ValueError(
            f"grad_y shape {tuple(grad_y.shape)} != x shape {tuple(x.shape)}"
        )

    io_dtype, compute_bf16 = _resolve_io_dtype(c_dtype)
    x_io = x if x.dtype == io_dtype else x.to(io_dtype)
    grad_y_io = grad_y if grad_y.dtype == io_dtype else grad_y.to(io_dtype)
    x_flat = x_io.contiguous().view(-1)
    grad_y_flat = grad_y_io.contiguous().view(-1)
    n_elems = x_flat.numel()
    grad_x_flat = torch.empty(n_elems, device=x.device, dtype=io_dtype)

    if n_elems == 0:
        return grad_x_flat.view_as(x)

    grid = (math.ceil(n_elems / _BLOCK_SIZE), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _activation_backward_kernel,
        (
            x_flat,
            grad_y_flat,
            grad_x_flat,
            n_elems,
            code,
            compute_bf16,
            _BLOCK_SIZE,
        ),
    )
    return grad_x_flat.view_as(x)
