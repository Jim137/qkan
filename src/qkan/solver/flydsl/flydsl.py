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

from .._base import QKANSolver, register
from ..exact import torch_exact_solver

try:
    from .flydsl_ops import (
        _FLYDSL_KERNELS_AVAILABLE,
        flydsl_backward,
        flydsl_forward,
    )

    _FLYDSL_AVAILABLE = _FLYDSL_KERNELS_AVAILABLE
except ImportError:
    _FLYDSL_AVAILABLE = False


_SUPPORTED_FLYDSL_ANSATZES = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}


def _grad_like(grad, ref):
    """Cast an f32 kernel gradient back to its input's dtype."""
    if grad is None or grad.dtype == ref.dtype:
        return grad
    return grad.to(ref.dtype)


class _FlyDSLFunction(torch.autograd.Function):
    """
    Custom autograd function: FlyDSL forward and backward.

    The backward kernel recomputes the forward state per sample and walks the
    circuit in reverse, so nothing but the inputs is saved for backward.
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
        c_dtype,
        ansatz,
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable
        ctx.ansatz = ansatz
        return flydsl_forward(
            x,
            theta,
            preacts_w,
            preacts_b,
            reps,
            ansatz,
            preacts_trainable,
            fast_measure,
            c_dtype=c_dtype,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        grad_x, grad_theta, grad_pw, grad_pb = flydsl_backward(
            x,
            theta,
            preacts_w,
            preacts_b,
            grad_output,
            ctx.reps,
            ctx.ansatz,
            ctx.preacts_trainable,
            ctx.fast_measure,
        )
        return (
            _grad_like(grad_x, x),
            _grad_like(grad_theta, theta),
            _grad_like(grad_pw, preacts_w),
            _grad_like(grad_pb, preacts_b),
            None,  # reps
            None,  # fast_measure
            None,  # preacts_trainable
            None,  # c_dtype
            None,  # ansatz
        )


def flydsl_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    FlyDSL-accelerated exact solver for AMD GPUs (ROCm).

    Drop-in replacement for flash_exact_solver / cute_exact_solver. Uses fused
    FlyDSL kernels for pz_encoding, rpz_encoding, and real ansatzes and falls
    back to torch_exact_solver for the rest.

    Key properties:
      - ``reps`` is unrolled at compile time; theta trig is computed once per
        thread and reused across its batch chunk
      - Backward is state-free: gates are un-applied in reverse, so no
        per-layer state checkpoints are written to memory

    Args:
        Same as torch_exact_solver.

    Returns:
        torch.Tensor, shape: (batch_size, out_dim, in_dim)
    """
    if not _FLYDSL_AVAILABLE:
        raise ImportError(
            "FlyDSL fused kernels not available. Install flydsl and a ROCm "
            "build of PyTorch: pip install qkan[flydsl]"
        )

    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])
    c_dtype = kwargs.get("dtype", torch.complex64)
    batch, in_dim = x.shape

    # Fallback for unsupported ansatzes
    if ansatz not in _SUPPORTED_FLYDSL_ANSATZES:
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

    # Check if gradients are needed. Under torch.no_grad() / inference_mode,
    # skip the autograd.Function wrapper to avoid its overhead even when
    # parameter tensors still have requires_grad=True.
    if not torch.is_grad_enabled():
        needs_grad = False
    else:
        needs_grad = theta.requires_grad or x.requires_grad
        if _needs_encoded_x:
            needs_grad = (
                needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad
            )

    if needs_grad:
        return _FlyDSLFunction.apply(
            x,
            theta,
            preacts_weight,
            preacts_bias,
            reps,
            fast_measure,
            preacts_trainable,
            c_dtype,
            ansatz,
        )
    return flydsl_forward(
        x,
        theta,
        preacts_weight,
        preacts_bias,
        reps,
        ansatz,
        preacts_trainable,
        fast_measure,
        c_dtype=c_dtype,
    )


class FlyDSLSolver(QKANSolver):
    """FlyDSL ROCm solver (registered as ``"flydsl"``)."""

    name = "flydsl"

    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor:
        return flydsl_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )


register(FlyDSLSolver())
