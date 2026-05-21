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
cuTile matmul kernel for the QKAN-style ``nn.Linear`` drop-in.

Implements the forward of ``y = x @ W^T + b`` as a single tiled GEMM
written in cuTile. f32 and bf16 IO are supported; the accumulator is
always f32 and cast back at store time.

The backward path uses ``torch.matmul`` (cuBLAS) for ``grad_x`` and
``grad_W`` because beating cuBLAS is explicitly out of scope — the goal
is consistent stream / CUDA-graph behaviour for users who are already
running on the cuTile solver. ``grad_b`` is a trivial reduction handled
by torch.
"""

import math
from typing import Any

import cuda.tile as ct  # type: ignore
import torch
import torch.nn as nn
from torch.nn import init

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# ── tile shape ──────────────────────────────────────────────────────────────
#
# 64 × 64 × 32 is a comfortable default for bf16 mma: it fills a couple of
# warps' worth of accumulator registers and gives the compiler enough K
# slack to keep fragments resident across iterations. The Linear shapes
# we care about (HQKAN: d=128–512, els=64–256) are large enough that
# fixed tiles are fine; we don't autotune.
_BM = 64
_BN = 64
_BK = 32


# ── forward kernel ──────────────────────────────────────────────────────────


@ct.kernel
def _linear_forward_kernel(
    x,  # [M, K] — input (M = leading dims flattened, K = in_features)
    w,  # [N, K] — weight
    bias,  # [N] — bias (zeroed when HAS_BIAS is False; never read)
    out,  # [M, N] — output
    M: ConstInt,
    N: ConstInt,
    K: ConstInt,
    HAS_BIAS: ConstBool,
    OUT_BF16: ConstBool,
    BM: ConstInt,
    BN: ConstInt,
    BK: ConstInt,
):
    """Tiled GEMM: ``out = x @ w^T + bias``.

    Grid: ``(cdiv(M, BM), cdiv(N, BN))``. Each block accumulates a
    ``(BM, BN)`` output tile in f32 by streaming the K dimension in
    ``BK``-wide chunks. The weight tile is loaded with ``order=(1, 0)``
    so the transpose comes for free at the load.
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.zeros((BM, BN), dtype=ct.float32)
    n_k_tiles = ct.cdiv(K, BK)
    for ki in range(n_k_tiles):
        # x_tile[m, k] = x[pid_m*BM + m, ki*BK + k]
        x_tile = ct.load(
            x, (pid_m, ki), shape=(BM, BK), padding_mode=ct.PaddingMode.ZERO
        )
        # w is (N, K) but we want the (K, N) view tile for the GEMM. Load
        # with order=(1, 0): index (ki, pid_n), shape (BK, BN) — yields
        # w_tile[k, n] = w[pid_n*BN + n, ki*BK + k] (i.e. W^T tile).
        w_tile = ct.load(
            w,
            (ki, pid_n),
            shape=(BK, BN),
            order=(1, 0),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(x_tile, w_tile, acc)

    if HAS_BIAS:
        # bias[n] broadcast across the BM rows of the tile.
        bias_tile = ct.load(
            bias, (pid_n,), shape=(BN,), padding_mode=ct.PaddingMode.ZERO
        )
        bias_f32 = ct.broadcast_to(ct.astype(bias_tile, ct.float32), (BM, BN))
        acc = acc + bias_f32

    if OUT_BF16:
        ct.store(out, (pid_m, pid_n), ct.astype(acc, ct.bfloat16))
    else:
        ct.store(out, (pid_m, pid_n), acc)


# ── host-side launch ────────────────────────────────────────────────────────


def _resolve_io_dtype(dtype: torch.dtype) -> tuple[torch.dtype, bool]:
    """Return ``(io_dtype, out_bf16)``. f32 stays f32; bf16 stays bf16.

    We deliberately do not promote f16 → bf16 silently — Linear inputs in
    QKAN are bf16 or f32 only. Other dtypes raise.
    """
    if dtype == torch.float32:
        return torch.float32, False
    if dtype == torch.bfloat16:
        return torch.bfloat16, True
    raise ValueError(
        f"CuTileLinear only supports float32 and bfloat16, got {dtype}. "
        "Cast inputs/parameters before the call."
    )


def cutile_linear_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Compute ``y = x @ weight.T + bias`` with a cuTile kernel.

    Args:
        x: ``(*, in_features)`` on CUDA. Leading dimensions are flattened
           internally and restored on output.
        weight: ``(out_features, in_features)`` on CUDA.
        bias: ``(out_features,)`` on CUDA, or ``None``.

    Returns:
        Tensor of shape ``(*, out_features)`` matching ``x``'s dtype.
    """
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("cutile_linear_forward requires CUDA tensors")
    if bias is not None and not bias.is_cuda:
        raise ValueError("cutile_linear_forward requires bias on CUDA")

    io_dtype, out_bf16 = _resolve_io_dtype(x.dtype)

    leading_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = weight.shape[0]
    if weight.shape[1] != in_features:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} incompatible with "
            f"x shape {tuple(x.shape)}"
        )

    x_flat = x.contiguous().view(-1, in_features)
    M = x_flat.shape[0]

    weight_c = weight.to(io_dtype).contiguous()
    if bias is not None:
        bias_c = bias.to(io_dtype).contiguous()
    else:
        # Pass a 1-element placeholder so the kernel signature stays uniform.
        # HAS_BIAS=False prevents any read of this tensor.
        bias_c = torch.empty(1, device=x.device, dtype=io_dtype)

    out_flat = torch.empty(M, out_features, device=x.device, dtype=io_dtype)

    if M == 0:
        return out_flat.view(*leading_shape, out_features)

    grid = (math.ceil(M / _BM), math.ceil(out_features / _BN), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _linear_forward_kernel,
        (
            x_flat,
            weight_c,
            bias_c,
            out_flat,
            M,
            out_features,
            in_features,
            bias is not None,
            out_bf16,
            _BM,
            _BN,
            _BK,
        ),
    )
    return out_flat.view(*leading_shape, out_features)


__all__ = ["CuTileLinear"]


# ── autograd glue ───────────────────────────────────────────────────────────


class _CuTileLinearFunction(torch.autograd.Function):
    """Forward via cuTile kernel; backward via cuBLAS (``torch.matmul``).

    Beating cuBLAS on the backward GEMMs is explicitly out of scope — the
    point of the cuTile path is consistent stream/CUDA-graph behaviour
    next to the cuTile QKAN solver, not raw speed.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return cutile_linear_forward(x, weight, bias)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        x, weight = ctx.saved_tensors
        grad_y_c = grad_y.contiguous()

        # grad_x = grad_y @ weight, shape (*, in_features). Flatten leading
        # dims for the matmul, restore on return.
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        leading = grad_y_c.shape[:-1]
        gy_flat = grad_y_c.view(-1, out_features)
        x_flat = x.contiguous().view(-1, in_features)

        grad_x = (
            (gy_flat @ weight).view(*leading, in_features)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_w = gy_flat.t() @ x_flat if ctx.needs_input_grad[1] else None
        grad_b = (
            gy_flat.sum(dim=0) if (ctx.has_bias and ctx.needs_input_grad[2]) else None
        )
        return grad_x, grad_w, grad_b


# ── nn.Module wrapper ───────────────────────────────────────────────────────


class CuTileLinear(nn.Module):
    """Drop-in replacement for ``torch.nn.Linear`` backed by cuTile.

    Computes ``y = x @ W.T + b`` using a cuTile-decorated forward kernel
    matched in style to the rest of the cuTile backend in QKAN. The
    backward path uses ``torch.matmul`` (cuBLAS) for ``grad_x`` /
    ``grad_W`` plus a torch reduction for ``grad_b`` — the goal is
    correctness and consistent stream behaviour, not beating cuBLAS.

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: if ``True``, learn an additive bias.
        device: device for parameters. Defaults to current device.
        dtype: parameter dtype. Defaults to ``torch.get_default_dtype()``.

    Shape:
        - Input: ``(*, in_features)``.
        - Output: ``(*, out_features)``.

    Notes:
        Initialisation matches ``nn.Linear`` exactly:
        ``init.kaiming_uniform_(W, a=sqrt(5))`` and ``uniform_`` for
        ``b`` within ``±1/sqrt(fan_in)``.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

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
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match ``nn.Linear.reset_parameters`` exactly.

        ``kaiming_uniform_(a=sqrt(5))`` on weight is equivalent to
        ``uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))``. See
        https://github.com/pytorch/pytorch/issues/57109 for the rationale.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference fast path — same trick as CuTileActivation. autograd.Function
        # saves tensors and adds Python overhead even when grad isn't needed,
        # so we skip it under no_grad / inference_mode and call the kernel
        # launcher directly.
        if not torch.is_grad_enabled() or not (
            x.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        ):
            return cutile_linear_forward(x, self.weight, self.bias)
        return _CuTileLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
