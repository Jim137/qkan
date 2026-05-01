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
Triton-jitted activation kernels (forward + backward) for QKAN base path.

Activations supported (kind):
    "silu", "gelu_exact", "gelu_tanh", "relu", "tanh", "sigmoid"

I/O dtypes: float32 and bfloat16. All math runs in float32 inside the kernel
(cast on load, cast on store) to keep accuracy uniform across dtypes.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from triton.language.extra import libdevice  # type: ignore

# Constants — must be wrapped with tl.constexpr so @triton.jit can read them.
_GELU_TANH_K0 = tl.constexpr(0.7978845608028654)  # sqrt(2/pi)
_GELU_TANH_K1 = tl.constexpr(0.044715)
_INV_SQRT2 = tl.constexpr(0.7071067811865476)  # 1/sqrt(2)
_INV_SQRT_2PI = tl.constexpr(0.3989422804014327)  # 1/sqrt(2*pi)  (N(0,1) pdf coef)


# ── Forward kernels ────────────────────────────────────────────────────────


@triton.jit
def _silu_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(x)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _gelu_exact_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.erf(x * _INV_SQRT2))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _gelu_tanh_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    inner = _GELU_TANH_K0 * (x + _GELU_TANH_K1 * x * x * x)
    y = 0.5 * x * (1.0 + libdevice.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _relu_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.where(x > 0.0, x, 0.0)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _tanh_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = libdevice.tanh(x)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _sigmoid_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.sigmoid(x)
    tl.store(y_ptr + offs, y, mask=mask)


# ── Backward kernels ───────────────────────────────────────────────────────


@triton.jit
def _silu_bwd_kernel(x_ptr, gy_ptr, gx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=mask).to(tl.float32)
    s = tl.sigmoid(x)
    # d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    dydx = s * (1.0 + x * (1.0 - s))
    tl.store(gx_ptr + offs, gy * dydx, mask=mask)


@triton.jit
def _gelu_exact_bwd_kernel(x_ptr, gy_ptr, gx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=mask).to(tl.float32)
    # dydx = 0.5*(1+erf(x/sqrt2)) + x * pdf(x), pdf = (1/sqrt(2*pi))*exp(-x^2/2)
    cdf = 0.5 * (1.0 + tl.erf(x * _INV_SQRT2))
    pdf = _INV_SQRT_2PI * tl.exp(-0.5 * x * x)
    dydx = cdf + x * pdf
    tl.store(gx_ptr + offs, gy * dydx, mask=mask)


@triton.jit
def _gelu_tanh_bwd_kernel(x_ptr, gy_ptr, gx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=mask).to(tl.float32)
    # u = sqrt(2/pi)*(x + 0.044715*x^3); t = tanh(u); du/dx = sqrt(2/pi)*(1 + 3*0.044715*x^2)
    # dy/dx = 0.5*(1+t) + 0.5*x*(1 - t^2)*du/dx
    x2 = x * x
    u = _GELU_TANH_K0 * (x + _GELU_TANH_K1 * x * x2)
    t = libdevice.tanh(u)
    du_dx = _GELU_TANH_K0 * (1.0 + 3.0 * _GELU_TANH_K1 * x2)
    dydx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
    tl.store(gx_ptr + offs, gy * dydx, mask=mask)


@triton.jit
def _relu_bwd_kernel(x_ptr, gy_ptr, gx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=mask).to(tl.float32)
    dydx = tl.where(x > 0.0, 1.0, 0.0)
    tl.store(gx_ptr + offs, gy * dydx, mask=mask)


@triton.jit
def _tanh_bwd_kernel(x_ptr, gy_ptr, gx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=mask).to(tl.float32)
    t = libdevice.tanh(x)
    dydx = 1.0 - t * t
    tl.store(gx_ptr + offs, gy * dydx, mask=mask)


@triton.jit
def _sigmoid_bwd_kernel(x_ptr, gy_ptr, gx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    gy = tl.load(gy_ptr + offs, mask=mask).to(tl.float32)
    s = tl.sigmoid(x)
    dydx = s * (1.0 - s)
    tl.store(gx_ptr + offs, gy * dydx, mask=mask)


# ── Dispatch tables ────────────────────────────────────────────────────────


_FWD_KERNELS = {
    "silu": _silu_fwd_kernel,
    "gelu_exact": _gelu_exact_fwd_kernel,
    "gelu_tanh": _gelu_tanh_fwd_kernel,
    "relu": _relu_fwd_kernel,
    "tanh": _tanh_fwd_kernel,
    "sigmoid": _sigmoid_fwd_kernel,
}

_BWD_KERNELS = {
    "silu": _silu_bwd_kernel,
    "gelu_exact": _gelu_exact_bwd_kernel,
    "gelu_tanh": _gelu_tanh_bwd_kernel,
    "relu": _relu_bwd_kernel,
    "tanh": _tanh_bwd_kernel,
    "sigmoid": _sigmoid_bwd_kernel,
}

SUPPORTED_KINDS = tuple(_FWD_KERNELS.keys())


def _check_kind(kind: str) -> None:
    if kind not in _FWD_KERNELS:
        raise ValueError(
            f"Unsupported activation kind: {kind!r}. "
            f"Supported: {sorted(_FWD_KERNELS.keys())}"
        )


def triton_activation_forward(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Element-wise activation forward via Triton.

    Args:
        x: input tensor, any shape, dtype float32 or bfloat16, on CUDA.
        kind: one of "silu", "gelu_exact", "gelu_tanh", "relu", "tanh", "sigmoid".

    Returns:
        Tensor with same shape and dtype as ``x``.
    """
    _check_kind(kind)
    if not x.is_cuda:
        raise RuntimeError("triton_activation_forward requires CUDA tensors")
    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    if n == 0:
        return y
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _FWD_KERNELS[kind][grid](x_c, y, n, BLOCK=BLOCK)
    return y


def triton_activation_backward(
    grad_y: torch.Tensor, x: torch.Tensor, kind: str
) -> torch.Tensor:
    """Element-wise activation backward via Triton: returns dL/dx given dL/dy.

    Args:
        grad_y: upstream gradient, same shape/dtype as forward output.
        x: input tensor saved from forward.
        kind: one of the supported kinds.

    Returns:
        Tensor with same shape/dtype as ``x`` (== forward output dtype).
    """
    _check_kind(kind)
    if not x.is_cuda:
        raise RuntimeError("triton_activation_backward requires CUDA tensors")
    x_c = x.contiguous()
    gy_c = grad_y.contiguous()
    if gy_c.dtype != x_c.dtype:
        gy_c = gy_c.to(x_c.dtype)
    gx = torch.empty_like(x_c)
    n = x_c.numel()
    if n == 0:
        return gx
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _BWD_KERNELS[kind][grid](x_c, gy_c, gx, n, BLOCK=BLOCK)
    return gx
