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


import torch

from ._base import QKANSolver, register
from ._utils import _cast_grads_to_dtype
from .exact import torch_exact_solver

try:
    from ..cutile_ops import (
        cutile_pz_backward,
        cutile_pz_forward,
        cutile_real_backward,
        cutile_real_forward,
        cutile_rpz_backward,
        cutile_rpz_forward,
    )

    _CUTILE_AVAILABLE = True
except ImportError:
    _CUTILE_AVAILABLE = False


_SUPPORTED_CUTILE_ANSATZES = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}


class _CuTileFlashFunction(torch.autograd.Function):
    """
    Custom autograd function: cuTile forward and backward.

    Forward dispatches to the appropriate cuTile kernel based on ansatz.
    Backward uses direct cuTile kernels with forward recomputation.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        theta,
        preacts_w,
        preacts_b,
        reps,
        fast_measure,
        preacts_trainable,
        out_dim,
        c_dtype,
        ansatz,
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable
        ctx.out_dim = out_dim
        ctx.c_dtype = c_dtype
        ctx.ansatz = ansatz

        if ansatz in ("pz_encoding", "pz"):
            return cutile_pz_forward(
                x,
                theta,
                preacts_w,
                preacts_b,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            return cutile_rpz_forward(
                x,
                theta,
                preacts_w,
                preacts_b,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz == "real":
            return cutile_real_forward(
                x,
                theta,
                preacts_w,
                preacts_b,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        else:
            raise ValueError(f"Unsupported ansatz for cutile: {ansatz}")

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        ansatz = ctx.ansatz

        if ansatz in ("pz_encoding", "pz"):
            grad_x, grad_theta, grad_pw, grad_pb = cutile_pz_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.preacts_trainable,
                ctx.fast_measure,
                c_dtype=ctx.c_dtype,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            grad_x, grad_theta, grad_pw, grad_pb = cutile_rpz_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.fast_measure,
                c_dtype=ctx.c_dtype,
            )
        elif ansatz == "real":
            grad_x, grad_theta, grad_pw, grad_pb = cutile_real_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.preacts_trainable,
                ctx.fast_measure,
                c_dtype=ctx.c_dtype,
            )
        else:
            raise ValueError(f"Unsupported ansatz for cutile backward: {ansatz}")

        if ctx.c_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            grad_x, grad_theta, grad_pw, grad_pb = _cast_grads_to_dtype(
                grad_x, grad_theta, grad_pw, grad_pb, x.dtype
            )

        return (
            grad_x,
            grad_theta,
            grad_pw,
            grad_pb,
            None,  # reps
            None,  # fast_measure
            None,  # preacts_trainable
            None,  # out_dim
            None,  # c_dtype
            None,  # ansatz
        )


def cutile_flash_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    cuTile-accelerated exact solver. Drop-in replacement for flash_exact_solver.

    Uses fused cuTile kernels for pz_encoding, rpz_encoding, and real ansatzes.
    Falls back to torch_exact_solver for unsupported ansatzes.

    Args:
        Same as torch_exact_solver.

    Returns:
        torch.Tensor, shape: (batch_size, out_dim, in_dim)
    """
    if not _CUTILE_AVAILABLE:
        raise ImportError(
            "cuTile fused kernels not available. Install cuda-tile to use cutile solver."
        )

    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])
    c_dtype = kwargs.get("dtype", torch.complex64)
    batch, in_dim = x.shape

    # Fallback for unsupported ansatzes
    if ansatz not in _SUPPORTED_CUTILE_ANSATZES:
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )

    # Broadcasting logic (mirrors torch_exact_solver)
    if len(theta.shape) != 4:
        theta = theta.unsqueeze(0)
    if theta.shape[1] != in_dim:
        repeat_out = out_dim
        repeat_in = in_dim // theta.shape[1] + 1
        theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]

    # rpz always needs encoded_x; others only when preacts_trainable
    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    if _needs_encoded_x:
        if len(preacts_weight.shape) != 3:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]

    # Check if gradients are needed (training)
    # Bypass autograd.Function wrapper when torch.no_grad() / inference_mode
    # is active. Saves ~6-10 μs of Python overhead per inference forward.
    if not torch.is_grad_enabled():
        needs_grad = False
    else:
        needs_grad = theta.requires_grad or x.requires_grad
        if _needs_encoded_x or preacts_trainable:
            needs_grad = (
                needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad
            )

    if needs_grad:
        return _CuTileFlashFunction.apply(
            x,
            theta,
            preacts_weight,
            preacts_bias,
            reps,
            fast_measure,
            preacts_trainable,
            out_dim,
            c_dtype,
            ansatz,
        )
    else:
        if ansatz in ("pz_encoding", "pz"):
            return cutile_pz_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            return cutile_rpz_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz == "real":
            return cutile_real_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        else:
            raise NotImplementedError


class CuTileSolver(QKANSolver):
    """cuTile (Tile Language) solver (registered as ``"cutile"``)."""

    name = "cutile"

    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor:
        if not _CUTILE_AVAILABLE:
            raise ImportError(
                "cuda.tile is required for solver='cutile'. "
                "Install with: pip install cuda-tile"
            )
        return cutile_flash_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )


register(CuTileSolver())
