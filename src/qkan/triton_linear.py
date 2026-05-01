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
Triton-backed nn.Linear drop-in.

Computes y = x @ W^T + b with a block-tiled Triton matmul kernel. f32 accumulation
internally; bf16 / f32 inputs and outputs are supported. Backward uses torch.matmul
for grad_x / grad_W (correctness is the goal, not beating cuBLAS).

Triton is imported lazily so that ``from qkan.triton_linear import TritonLinear``
works on systems without Triton; the ImportError is raised only when a kernel
launch is actually attempted. This mirrors the ``_FLASH_AVAILABLE`` gating used
elsewhere in QKAN.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Triton kernel (only compiled when triton is importable)
# --------------------------------------------------------------------------- #

if _TRITON_AVAILABLE:

    @triton.jit
    def _linear_forward_kernel(
        x_ptr,  # [M, K]
        w_ptr,  # [N, K]   (nn.Linear weight stored as out × in)
        b_ptr,  # [N] or null
        y_ptr,  # [M, N]
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        HAS_BIAS: tl.constexpr,
        IS_F32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Block-tiled GEMM: y = x @ W^T (+ b).

        Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N)).

        Mirrors the standard Triton matmul tutorial; only twist is W is laid out
        as [N, K] (matching torch's nn.Linear weight) so the K axis is contiguous
        for both x and W loads. For f32 inputs we ask for IEEE precision so we
        match the default torch.nn.Linear path (TF32 off in cuBLAS by default);
        for bf16 we let tl.dot pick its native bf16 tensor-core path.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_block_ptr = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_block_ptr = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_mask = offs_k[None, :] < (K - k)

            x_tile = tl.load(
                x_block_ptr,
                mask=(offs_m[:, None] < M) & k_mask,
                other=0.0,
            )
            w_tile = tl.load(
                w_block_ptr,
                mask=(offs_n[:, None] < N) & k_mask,
                other=0.0,
            )

            # x_tile: [BM, BK]; w_tile: [BN, BK]; want acc += x @ w^T -> [BM, BN]
            if IS_F32:
                acc += tl.dot(
                    x_tile.to(tl.float32),
                    tl.trans(w_tile).to(tl.float32),
                    input_precision="ieee",
                )
            else:
                acc += tl.dot(x_tile, tl.trans(w_tile))

            x_block_ptr += BLOCK_K * stride_xk
            w_block_ptr += BLOCK_K * stride_wk

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        y_block_ptr = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_block_ptr, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


# --------------------------------------------------------------------------- #
# Python launchers
# --------------------------------------------------------------------------- #


def _check_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise ImportError(
            "Triton is required for TritonLinear. "
            "Install triton (e.g. `pip install triton`) to use this module."
        )


def _select_blocks(M: int, N: int, K: int) -> tuple[int, int, int]:
    """Pick BLOCK_M / BLOCK_N / BLOCK_K honoring tl.dot's >=16 constraint."""
    bm = 16 if M < 32 else (32 if M < 64 else 64)
    bn = 16 if N < 32 else (32 if N < 64 else 64)
    bk = 16 if K < 32 else (32 if K < 64 else 64)
    return bm, bn, bk


def triton_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute y = x @ weight^T (+ bias) with a Triton GEMM kernel.

    x:      (M, K) — caller is responsible for flattening leading dims.
    weight: (N, K) — nn.Linear convention.
    bias:   (N,) or None.

    Returns y: (M, N) in x.dtype.
    """
    _check_triton()
    assert x.is_cuda and weight.is_cuda, "TritonLinear requires CUDA tensors"
    assert x.dim() == 2, f"expected 2D x, got shape {tuple(x.shape)}"
    assert weight.dim() == 2, f"expected 2D weight, got shape {tuple(weight.shape)}"
    assert x.shape[1] == weight.shape[1], (
        f"in_features mismatch: x.shape[1]={x.shape[1]} vs weight.shape[1]={weight.shape[1]}"
    )

    M, K = x.shape
    N = weight.shape[0]

    x_c = x.contiguous()
    w_c = weight.contiguous()
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    if bias is not None:
        b_c = bias.contiguous()
        b_ptr: Any = b_c
        has_bias = True
    else:
        # Triton needs a tensor to take ptr from even when unused.
        b_c = torch.empty(1, device=x.device, dtype=x.dtype)
        b_ptr = b_c
        has_bias = False

    BLOCK_M, BLOCK_N, BLOCK_K = _select_blocks(M, N, K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _linear_forward_kernel[grid](
        x_c,
        w_c,
        b_ptr,
        y,
        M,
        N,
        K,
        x_c.stride(0),
        x_c.stride(1),
        w_c.stride(0),
        w_c.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=has_bias,
        IS_F32=(x.dtype == torch.float32),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y


def triton_linear_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    needs_grad_x: bool = True,
    needs_grad_w: bool = True,
    needs_grad_b: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Backward for y = x @ W^T + b.

    grad_x = grad_y @ W
    grad_W = grad_y^T @ x
    grad_b = grad_y.sum(dim=0)

    Uses torch.matmul (cuBLAS) for the matmuls; the autograd.Function still owns
    the stream/graph behaviour, which is what callers need for CUDA-graph capture.
    """
    grad_x = torch.matmul(grad_y, weight) if needs_grad_x else None
    grad_w = torch.matmul(grad_y.transpose(0, 1), x) if needs_grad_w else None
    grad_b = grad_y.sum(dim=0) if needs_grad_b else None
    return grad_x, grad_w, grad_b


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #


class _TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return triton_linear_forward(x, weight, bias)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        x, weight = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        grad_x, grad_w, grad_b = triton_linear_backward(
            grad_y,
            x,
            weight,
            needs_grad_x=ctx.needs_input_grad[0],
            needs_grad_w=ctx.needs_input_grad[1],
            needs_grad_b=ctx.has_bias and ctx.needs_input_grad[2],
        )
        return grad_x, grad_w, grad_b


# --------------------------------------------------------------------------- #
# nn.Module
# --------------------------------------------------------------------------- #


class TritonLinear(nn.Module):
    """nn.Linear drop-in computing y = x @ W^T + b via Triton kernels.

    Behaviour matches ``torch.nn.Linear`` for >=2D inputs (leading dims are
    flattened, matmul is performed, then reshaped back). Internal compute is
    fp32; outputs are cast back to the input dtype.

    Initialization matches nn.Linear exactly: kaiming_uniform_(a=sqrt(5)) for
    weight and uniform_(-1/sqrt(fan_in), 1/sqrt(fan_in)) for bias.

    Under ``torch.no_grad()`` the autograd.Function wrapper is bypassed for a
    small inference-time speedup.
    """

    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)  # type: ignore[arg-type]
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)  # type: ignore[arg-type]
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.Linear exactly. See pytorch/pytorch#57109 for the sqrt(5)
        # rationale: it reduces to uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(
                f"TritonLinear expects >=2D input, got shape {tuple(x.shape)}"
            )

        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        if torch.is_grad_enabled() and (
            x_flat.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        ):
            y_flat = _TritonLinearFunction.apply(x_flat, self.weight, self.bias)
        else:
            y_flat = triton_linear_forward(x_flat, self.weight, self.bias)

        return y_flat.reshape(*leading_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
