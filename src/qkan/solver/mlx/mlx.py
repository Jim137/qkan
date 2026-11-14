# Copyright (c) 2026, Jiun-Cheng Jiang and the QKAN contributors. All rights reserved.
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


from typing import Optional

import torch

from .._base import QKANSolver, register
from .._utils import check_ansatz, check_c_dtype
from ..flydsl.flydsl_ops import _first_order_only
from .mlx_ops import (
    _ANSATZ_CANONICAL,
    _MLX_INSTALLED,
    _kernels_available,
    mlx_backward,
    mlx_contract_backward,
    mlx_contract_forward,
    mlx_forward,
)

# Cheap import-time check (MLX is installed on macOS). Whether Metal kernels
# can actually launch is checked on first use, when MLX is imported.
_MLX_AVAILABLE = _MLX_INSTALLED

# Metal has no float64. The half and fp8 dtypes are computed in float32.
_SUPPORTED_MLX_DTYPES = (
    torch.complex64,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
)


_NO_SECOND_DERIVATIVES = (
    "the mlx solver has no second derivatives, so its gradients cannot be "
    "differentiated; use solver='exact' for higher-order gradients"
)


def _grad_like(grad, ref):
    """Cast an f32 kernel gradient back to its input's dtype."""
    if grad is None or grad.dtype == ref.dtype:
        return grad
    return grad.to(ref.dtype)


class _MLXFunction(torch.autograd.Function):
    """
    Custom autograd function: fused Metal forward and backward through MLX.

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
        ansatz,
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable
        ctx.ansatz = ansatz
        return mlx_forward(
            x,
            theta,
            preacts_w,
            preacts_b,
            reps,
            ansatz,
            preacts_trainable,
            fast_measure,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        # The first layer's input has no gradient: the kernel then skips the
        # per-sample grad_x write and its reduction.
        need_grad_x = ctx.needs_input_grad[0]
        grad_x, grad_theta, grad_pw, grad_pb = mlx_backward(
            x,
            theta,
            preacts_w,
            preacts_b,
            grad_output,
            ctx.reps,
            ctx.ansatz,
            ctx.preacts_trainable,
            ctx.fast_measure,
            need_grad_x=need_grad_x,
        )
        grads = _first_order_only(
            (
                _grad_like(grad_x, x),
                _grad_like(grad_theta, theta),
                _grad_like(grad_pw, preacts_w),
                _grad_like(grad_pb, preacts_b),
            ),
            (x, theta, preacts_w, preacts_b, grad_output),
            _NO_SECOND_DERIVATIVES,
        )
        return (
            *grads,
            None,  # reps
            None,  # fast_measure
            None,  # preacts_trainable
            None,  # ansatz
        )


class _MLXContractFunction(torch.autograd.Function):
    """
    Custom autograd function for the circuit fused with the layer's contraction
    ``sum_i postacts[b, o, i] * weight[o, i]``.

    Neither pass materialises a ``(batch, out_dim, in_dim)`` tensor: the
    forward kernel sums over ``i`` itself, and the backward kernel rebuilds
    each circuit's output gradient from ``grad_output[b, o] * weight[o, i]``.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        theta,
        preacts_w,
        preacts_b,
        weight,
        reps,
        fast_measure,
        preacts_trainable,
        ansatz,
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b, weight)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable
        ctx.ansatz = ansatz
        return mlx_contract_forward(
            x,
            theta,
            preacts_w,
            preacts_b,
            weight,
            reps,
            ansatz,
            preacts_trainable,
            fast_measure,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b, weight = ctx.saved_tensors
        # The first layer's input and a frozen postact weight need no
        # gradient: the kernel then skips their sums altogether.
        grad_x, grad_theta, grad_pw, grad_pb, grad_weight = mlx_contract_backward(
            x,
            theta,
            preacts_w,
            preacts_b,
            weight,
            grad_output,
            ctx.reps,
            ctx.ansatz,
            ctx.preacts_trainable,
            ctx.fast_measure,
            need_grad_x=ctx.needs_input_grad[0],
            need_grad_weight=ctx.needs_input_grad[4],
        )
        grads = _first_order_only(
            (
                _grad_like(grad_x, x),
                _grad_like(grad_theta, theta),
                _grad_like(grad_pw, preacts_w),
                _grad_like(grad_pb, preacts_b),
                _grad_like(grad_weight, weight),
            ),
            (x, theta, preacts_w, preacts_b, weight, grad_output),
            _NO_SECOND_DERIVATIVES,
        )
        return (
            *grads,
            None,  # reps
            None,  # fast_measure
            None,  # preacts_trainable
            None,  # ansatz
        )


def _is_fused(ansatz) -> bool:
    """Whether the Metal kernels implement this ansatz."""
    return isinstance(ansatz, str) and ansatz in _ANSATZ_CANONICAL


def _check_mlx(ansatz, c_dtype: torch.dtype) -> None:
    """Raise ``ValueError`` for an ansatz or c_dtype the Metal kernels do not cover."""
    check_ansatz(
        "mlx",
        ansatz,
        _ANSATZ_CANONICAL,
        "'pz_encoding', 'rpz_encoding', 'real' and 'mix'",
    )
    check_c_dtype(
        "mlx",
        c_dtype,
        _SUPPORTED_MLX_DTYPES,
        "torch.complex64 / torch.float32, and torch.bfloat16, torch.float16 and "
        "torch.float8_e4m3fn as float32 (Metal has no float64)",
    )


def _require_kernels() -> None:
    if not _kernels_available():
        raise ImportError(
            "MLX Metal kernels not available. The mlx solver needs an Apple "
            "silicon Mac and MLX: pip install qkan[mlx]"
        )


def _broadcast_params(theta, preacts_weight, preacts_bias, in_dim, out_dim, encoded):
    """Per-``(o, i)`` thetas and preacts (mirrors torch_exact_solver)."""
    if len(theta.shape) != 4:
        theta = theta.unsqueeze(0)
    if theta.shape[1] != in_dim:
        repeat_out = out_dim
        repeat_in = in_dim // theta.shape[1] + 1
        theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]

    if encoded:
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
        # The exact solver broadcasts a single row of thetas against per-output
        # preacts (or the reverse); the kernel indexes both per (o, i) circuit.
        n_out = max(theta.shape[0], preacts_weight.shape[0])
        if theta.shape[0] != n_out:
            theta = theta.expand(n_out, -1, -1, -1)
        if preacts_weight.shape[0] != n_out:
            preacts_weight = preacts_weight.expand(n_out, -1, -1)
            preacts_bias = preacts_bias.expand(n_out, -1, -1)
    return theta, preacts_weight, preacts_bias


def _needs_grad(*tensors) -> bool:
    """Whether to record autograd.

    Under torch.no_grad() / inference_mode the autograd.Function wrapper is
    skipped to avoid its overhead even when parameter tensors still have
    requires_grad=True.
    """
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def mlx_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    MLX-accelerated exact solver for Apple silicon GPUs.

    Drop-in replacement for the other fused solvers. Runs fused Metal kernels
    (through ``mlx.core.fast.metal_kernel``) for the ``pz``, ``rpz``, ``real``
    and ``mix`` ansatzes. Any other ansatz, or a ``dtype`` the kernels have no path for,
    raises ``ValueError``.

    Key properties:
      - The whole circuit is one kernel launch; theta trig is computed once
        per thread and reused across its batch chunk
      - Backward is state-free: gates are un-applied in reverse, so no
        per-layer state checkpoints are written to memory
      - On the ``mps`` device, tensors are shared with MLX through DLPack
        without a copy in either direction; on ``cpu`` the kernels still run
        on the GPU and the output is a zero-copy view of MLX's buffer

    QKANLayer normally bypasses this for :func:`mlx_contracted_solver`, which
    also folds the layer's sum over inputs into the kernel.

    Args:
        Same as torch_exact_solver.

    Returns:
        torch.Tensor, float32, shape: (batch_size, out_dim, in_dim)
    """
    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])
    c_dtype = kwargs.get("dtype", torch.complex64)
    batch, in_dim = x.shape

    _require_kernels()
    _check_mlx(ansatz, c_dtype)

    # rpz always needs encoded_x; others only when preacts_trainable
    encoded = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    theta, preacts_weight, preacts_bias = _broadcast_params(
        theta, preacts_weight, preacts_bias, in_dim, out_dim, encoded
    )
    params = (theta, preacts_weight, preacts_bias) if encoded else (theta,)
    if _needs_grad(x, *params):
        return _MLXFunction.apply(
            x,
            theta,
            preacts_weight,
            preacts_bias,
            reps,
            fast_measure,
            preacts_trainable,
            ansatz,
        )
    return mlx_forward(
        x,
        theta,
        preacts_weight,
        preacts_bias,
        reps,
        ansatz,
        preacts_trainable,
        fast_measure,
    )


def mlx_contracted_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    weight: torch.Tensor,
    reps: int,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    The MLX solver fused with the layer's contraction over inputs.

    Computes ``(postacts * weight).sum(dim=2)`` for the ``postacts`` that
    :func:`mlx_exact_solver` would return, without ever materialising the
    ``(batch, out_dim, in_dim)`` tensor in the forward or the backward pass.
    On a wide layer PyTorch's passes over that tensor cost about as much as
    the circuit itself.

    Args:
        weight: torch.Tensor, shape (out_dim, in_dim); QKANLayer passes its
            masked postact weights.
        The rest as for torch_exact_solver.

    Returns:
        torch.Tensor, float32, shape: (batch_size, out_dim), or None when this
        configuration is not fused and the caller should contract the output
        of the solver itself: an ansatz the kernels do not implement (which
        :func:`mlx_exact_solver` then refuses), or thetas shared across
        outputs (a ``group`` equal to ``in_dim``), which the unfused route
        evaluates once instead of once per output. An unsupported
        ``dtype`` raises ``ValueError``.
    """
    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    c_dtype = kwargs.get("dtype", torch.complex64)
    batch, in_dim = x.shape
    out_dim = weight.shape[0]

    if not _is_fused(ansatz) or weight.shape != (out_dim, in_dim):
        return None
    _require_kernels()
    _check_mlx(ansatz, c_dtype)

    encoded = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    theta, preacts_weight, preacts_bias = _broadcast_params(
        theta, preacts_weight, preacts_bias, in_dim, out_dim, encoded
    )
    if theta.shape[0] != out_dim:
        return None

    params = (theta, preacts_weight, preacts_bias) if encoded else (theta,)
    if _needs_grad(x, weight, *params):
        return _MLXContractFunction.apply(
            x,
            theta,
            preacts_weight,
            preacts_bias,
            weight,
            reps,
            fast_measure,
            preacts_trainable,
            ansatz,
        )
    return mlx_contract_forward(
        x,
        theta,
        preacts_weight,
        preacts_bias,
        weight,
        reps,
        ansatz,
        preacts_trainable,
        fast_measure,
    )


class MLXSolver(QKANSolver):
    """MLX Apple silicon solver (registered as ``"mlx"``)."""

    name = "mlx"

    def check(self, ansatz, c_dtype: torch.dtype, reps: int) -> None:
        _check_mlx(ansatz, c_dtype)

    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor:
        return mlx_exact_solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs)


register(MLXSolver())
