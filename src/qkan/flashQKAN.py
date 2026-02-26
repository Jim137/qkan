"""
Flash (Triton-accelerated) QKAN solver.

Provides a drop-in replacement for torch_exact_solver that uses Triton kernels
for the pz_encoding forward pass. Backward uses PyTorch autograd via the
original exact solver for correctness.

Usage:
    - solver="flash" in QKANLayer/QKAN for Triton-accelerated pz_encoding
    - Falls back to torch_exact_solver for unsupported ansatzes
"""

import torch

from .fused_ops import triton_pz_forward
from .solver import torch_exact_solver


class _FlashPZFunction(torch.autograd.Function):
    """
    Custom autograd function: Triton forward, PyTorch backward.

    Forward uses the fused Triton kernel for speed.
    Backward recomputes via torch_exact_solver for correct gradients.
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
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable
        ctx.out_dim = out_dim
        ctx.c_dtype = c_dtype
        return triton_pz_forward(
            x, theta, preacts_w, preacts_b, preacts_trainable, fast_measure
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors

        # Recompute forward with PyTorch autograd for gradient computation
        with torch.enable_grad():
            x2 = x.detach().requires_grad_(x.requires_grad)
            t2 = theta.detach().requires_grad_(theta.requires_grad)
            pw2 = preacts_w.detach().requires_grad_(preacts_w.requires_grad)
            pb2 = preacts_b.detach().requires_grad_(preacts_b.requires_grad)

            # theta is already expanded to (out_dim, in_dim, reps+1, 2)
            # so torch_exact_solver won't re-expand
            out = torch_exact_solver(
                x2,
                t2,
                pw2,
                pb2,
                ctx.reps,
                ansatz="pz_encoding",
                preacts_trainable=ctx.preacts_trainable,
                fast_measure=ctx.fast_measure,
                out_dim=ctx.out_dim,
                dtype=ctx.c_dtype,
            ).to(grad_output.dtype)
            out.backward(grad_output)

        return (
            x2.grad,
            t2.grad,
            pw2.grad if ctx.preacts_trainable else None,
            pb2.grad if ctx.preacts_trainable else None,
            None,  # reps
            None,  # fast_measure
            None,  # preacts_trainable
            None,  # out_dim
            None,  # c_dtype
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

    Uses a fused Triton kernel for the pz_encoding forward pass.
    Falls back to torch_exact_solver for other ansatzes.

    Args:
        Same as torch_exact_solver.

    Returns:
        torch.Tensor, shape: (batch_size, out_dim, in_dim)
    """
    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])
    c_dtype = kwargs.get("dtype", torch.complex64)
    batch, in_dim = x.shape

    # Only pz_encoding is supported with Triton
    if ansatz not in ("pz_encoding", "pz"):
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

    if preacts_trainable:
        if len(preacts_weight.shape) != 3:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]

    # Use autograd.Function when gradients are needed (training)
    needs_grad = theta.requires_grad or x.requires_grad
    if preacts_trainable:
        needs_grad = needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad

    if needs_grad:
        return _FlashPZFunction.apply(
            x,
            theta,
            preacts_weight,
            preacts_bias,
            reps,
            fast_measure,
            preacts_trainable,
            out_dim,
            c_dtype,
        )
    else:
        return triton_pz_forward(
            x, theta, preacts_weight, preacts_bias, preacts_trainable, fast_measure
        )
