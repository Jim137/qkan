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
from .._utils import _cast_grads_to_dtype, check_ansatz, check_c_dtype
from .launch_policy import DEFAULT_POLICY_NAME

try:
    from .fused_ops import (
        resolve_pz_launch,
        triton_pz_backward,
        triton_pz_forward,
        triton_real_backward,
        triton_real_forward,
        triton_rpz_backward,
        triton_rpz_forward,
    )

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


_SUPPORTED_FLASH_ANSATZES = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}
_SUPPORTED_FLASH_DTYPES = (
    torch.complex64,
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
)


def _check_flash(ansatz, c_dtype: torch.dtype) -> None:
    """Raise ``ValueError`` for an ansatz or c_dtype the Triton kernels do not cover."""
    check_ansatz(
        "flash",
        ansatz,
        _SUPPORTED_FLASH_ANSATZES,
        "'pz_encoding', 'rpz_encoding' and 'real'",
    )
    # float16 buffers (float32 compute) exist in the pz and rpz kernels only.
    if ansatz == "real":
        check_c_dtype(
            "flash",
            c_dtype,
            _SUPPORTED_FLASH_DTYPES,
            "torch.complex64 / torch.float32, torch.bfloat16 and torch.float8_e4m3fn",
        )
    else:
        check_c_dtype(
            "flash",
            c_dtype,
            (*_SUPPORTED_FLASH_DTYPES, torch.float16),
            "torch.complex64 / torch.float32, torch.bfloat16, torch.float16 and "
            "torch.float8_e4m3fn",
        )


class _FlashFunction(torch.autograd.Function):
    """
    Custom autograd function: Triton forward and backward.

    Forward dispatches to the appropriate Triton kernel based on ansatz.
    Backward uses direct Triton kernels with forward recomputation.
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
        pz_launch,
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable
        ctx.out_dim = out_dim
        ctx.c_dtype = c_dtype
        ctx.ansatz = ansatz
        ctx.pz_launch = pz_launch

        if ansatz in ("pz_encoding", "pz"):
            return triton_pz_forward(
                x,
                theta,
                preacts_w,
                preacts_b,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            return triton_rpz_forward(
                x,
                theta,
                preacts_w,
                preacts_b,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz == "real":
            return triton_real_forward(
                x,
                theta,
                preacts_w,
                preacts_b,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        else:
            raise ValueError(f"Unsupported ansatz for flash: {ansatz}")

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        ansatz = ctx.ansatz

        if ansatz in ("pz_encoding", "pz"):
            grad_x, grad_theta, grad_pw, grad_pb = triton_pz_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.preacts_trainable,
                ctx.fast_measure,
                c_dtype=ctx.c_dtype,
                launch=ctx.pz_launch,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            grad_x, grad_theta, grad_pw, grad_pb = triton_rpz_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.fast_measure,
                c_dtype=ctx.c_dtype,
            )
        elif ansatz == "real":
            grad_x, grad_theta, grad_pw, grad_pb = triton_real_backward(
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
            raise ValueError(f"Unsupported ansatz for flash backward: {ansatz}")

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
            None,  # pz_launch
        )


def flash_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Triton-accelerated exact solver. Drop-in replacement for torch_exact_solver.

    Uses fused Triton kernels for pz_encoding, rpz_encoding, and real ansatzes.
    Any other ansatz, or a ``dtype`` the kernels have no path for, raises ``ValueError``.

    Args:
        Same as torch_exact_solver.

    Returns:
        torch.Tensor, shape: (batch_size, out_dim, in_dim)
    """
    if not _FLASH_AVAILABLE:
        raise ImportError(
            "Triton fused kernels not available. Install triton to use flash solver."
        )

    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])
    c_dtype = kwargs.get("dtype", torch.complex64)
    pz_launch_policy = kwargs.get("pz_launch_policy")
    batch, in_dim = x.shape

    _check_flash(ansatz, c_dtype)
    # The kernels are real-valued: complex64 selects their float32 path.
    if c_dtype == torch.complex64:
        c_dtype = torch.float32

    # "default" (or None) is the built-in launch, valid for every ansatz.  A
    # named policy only tunes the PZ backward launch.
    named_policy = pz_launch_policy not in (None, DEFAULT_POLICY_NAME)
    if named_policy and ansatz not in ("pz_encoding", "pz"):
        raise ValueError("pz_launch_policy is only valid for the PZ ansatz")

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
        if _needs_encoded_x or preacts_trainable:
            needs_grad = (
                needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad
            )

    if needs_grad:
        # Resolve the policy here rather than in backward: the backward then
        # receives plain constants, which torch.compile can trace.
        pz_launch = None
        if named_policy:
            pz_launch = resolve_pz_launch(
                pz_launch_policy,
                x.device,
                n_oi=out_dim * in_dim,
                batch=batch,
                reps=reps,
                c_dtype=c_dtype,
                preacts_trainable=preacts_trainable,
                fast_measure=fast_measure,
            )
        return _FlashFunction.apply(
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
            pz_launch,
        )
    else:
        if ansatz in ("pz_encoding", "pz"):
            return triton_pz_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            return triton_rpz_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                fast_measure,
                c_dtype=c_dtype,
            )
        elif ansatz == "real":
            return triton_real_forward(
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


# ---------------------------------------------------------------------------
# cuTile-accelerated solver
# ---------------------------------------------------------------------------

_SUPPORTED_CUTILE_ANSATZES = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}


class FlashSolver(QKANSolver):
    """Triton-fused solver (registered as ``"flash"``)."""

    name = "flash"

    def check(self, ansatz, c_dtype: torch.dtype, reps: int) -> None:
        _check_flash(ansatz, c_dtype)

    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor:
        if not _FLASH_AVAILABLE:
            raise ImportError(
                "Triton is required for solver='flash'. "
                "Install with: pip install triton"
            )
        if not x.is_cuda:
            raise RuntimeError(
                "solver='flash' runs Triton kernels, which need CUDA tensors: "
                f"got x on {x.device}. Move the layer and its input to a CUDA "
                "device, or use solver='exact'."
            )
        return flash_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )


register(FlashSolver())
