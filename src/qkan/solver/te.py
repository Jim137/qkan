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


"""Fourier-GEMM solver (registered as ``"te"``).

With ``fast_measure=False`` and the preact weights fixed at 1, the circuit on
every edge is a trigonometric polynomial of degree ``reps`` in its input,
``a0 + sum_k a_k cos(kx) + b_k sin(kx)``. Sampling each edge's circuit at
``2*reps+1`` equispaced points gives the coefficients, so the whole layer is
one GEMM over the features ``[cos kx, sin kx, base_activation(x)]``.

Inside ``transformer_engine.pytorch.autocast`` the GEMM runs through
Transformer Engine with the active recipe (e.g. NVFP4); otherwise it is
``F.linear`` and follows ``torch.autocast``.
"""

import functools
import math
import sys
from typing import Optional

import torch
import torch.nn.functional as F

from ._base import QKANSolver, register
from .exact import torch_exact_solver
from .flash import _FLASH_AVAILABLE, flash_exact_solver
from .flash.flash import _SUPPORTED_FLASH_ANSATZES

# Ansatzes whose data gates are Rz(x) or Ry(x).
FOURIER_ANSATZES = ("pz_encoding", "pz", "rpz_encoding", "rpz", "real", "mix")

# Transformer Engine's NVFP4 and FP8 GEMMs need dims that are multiples of 32.
_TE_ALIGN = 32


def _sample(x, theta, preacts_weight, preacts_bias, reps, **kwargs) -> torch.Tensor:
    """Postacts in x's precision: flash for float32 on CUDA, else the exact solver."""
    if (
        x.is_cuda
        and _FLASH_AVAILABLE
        and x.dtype == torch.float32
        and kwargs.get("ansatz") in _SUPPORTED_FLASH_ANSATZES
    ):
        kwargs["dtype"] = torch.float32
        return flash_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )
    kwargs["dtype"] = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    return torch_exact_solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs)


def fourier_coefficients(
    theta, preacts_weight, preacts_bias, reps: int, in_dim: int, **kwargs
) -> torch.Tensor:
    """Coefficients ``[a0, a_1..a_R, b_1..b_R]`` of every edge, shape (O, 2R+1, I)."""
    n = 2 * reps + 1
    # Sample at full precision whatever c_dtype is: it is only n rows.
    dtype = torch.float64 if theta.dtype == torch.float64 else torch.float32
    xs = torch.arange(n, device=theta.device, dtype=dtype) * (2 * math.pi / n)
    samples = _sample(
        xs[:, None].expand(n, in_dim),
        theta,
        preacts_weight,
        preacts_bias,
        reps,
        **kwargs,
    )
    angles = torch.arange(1, reps + 1, device=theta.device, dtype=dtype)[:, None] * xs
    dft = torch.cat([angles.new_full((1, n), 0.5), angles.cos(), angles.sin()])
    return torch.einsum("fn,noi->ofi", dft * (2 / n), samples.to(dtype))


def fourier_features(x: torch.Tensor, reps: int, base_activation) -> torch.Tensor:
    """``[cos kx, sin kx, base_activation(x)]``, shape (B, (2R+1)*I).

    Frequency-major, so the concatenation copies whole rows of I.
    """
    k = torch.arange(1, reps + 1, device=x.device, dtype=x.dtype)
    kx = k[:, None] * x[:, None, :]
    features = [kx.cos(), kx.sin(), base_activation(x)[:, None]]
    return torch.cat(features, dim=1).flatten(1)


@functools.cache
def _fp8_state():
    """TE's FP8GlobalStateManager; _TEGemm relies on TE>=2.19 internals."""
    import transformer_engine  # type: ignore
    from packaging.version import Version

    if Version(transformer_engine.__version__) < Version("2.19"):
        raise ImportError(
            "solver='te' needs transformer_engine>=2.19, "
            f"found {transformer_engine.__version__}"
        )
    from transformer_engine.pytorch.quantization import (  # type: ignore
        FP8GlobalStateManager,
    )

    return FP8GlobalStateManager


def _te_recipe():
    """The active Transformer Engine recipe, or None outside ``te.autocast``."""
    # te.autocast can only be active once the caller has imported TE.
    if "transformer_engine.pytorch" not in sys.modules:
        return None
    state = _fp8_state()
    if not state.is_fp8_enabled():
        return None
    recipe = state.get_fp8_recipe()
    if recipe.delayed():
        raise ValueError(
            "solver='te' keeps no amax history between steps: use a current-"
            "scaling recipe such as Float8CurrentScaling or NVFP4BlockScaling"
        )
    return recipe


class _TEGemm(torch.autograd.Function):
    """``x @ w.T`` through Transformer Engine's quantized GEMM, as in ``te.ops.BasicLinear``."""

    @staticmethod
    def forward(ctx, x, w, recipe, dtype):
        from transformer_engine.pytorch.ops.basic.basic_linear import (  # type: ignore
            BasicLinear,
        )
        from transformer_engine.pytorch.quantization import RecipeState  # type: ignore

        x_q, w_q = RecipeState.create(
            recipe, mode="forward", num_quantizers=2
        ).make_quantizers()
        (grad_q,) = RecipeState.create(
            recipe, mode="backward", num_quantizers=1
        ).make_quantizers()
        override = recipe.backward_override
        # FP8 per-tensor scaling can't quantize grad_out for wgrad alone, so it
        # prepares dgrad too, as te.ops.BasicLinear always does.
        ctx.dgrad = ctx.needs_input_grad[0] or recipe.float8_per_tensor_scaling()
        out, x_saved, w_saved = BasicLinear._functional_forward(
            input=x,
            weight=w,
            dtype=dtype,
            with_quantized_compute=True,
            backward_override=override,
            input_quantizer=x_q,
            weight_quantizer=w_q,
            input_requires_grad=ctx.dgrad,
            weight_requires_grad=ctx.needs_input_grad[1],
        )
        if override == "high_precision":
            x_saved, w_saved = x, w
        ctx.save_for_backward(x_saved, w_saved)
        ctx.quantizers = (x_q, w_q, grad_q)
        ctx.quantized = override is None
        ctx.nvfp4 = recipe.nvfp4()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from transformer_engine.pytorch.ops.basic.basic_linear import (  # type: ignore
            BasicLinear,
        )

        x_saved, w_saved = ctx.saved_tensors
        x_q, w_q, grad_q = ctx.quantizers
        dtype = grad_out.dtype
        if ctx.nvfp4:  # its Hadamard transform only takes bf16
            grad_out = grad_out.to(torch.bfloat16)
        grad_x, grad_w = BasicLinear._functional_backward(
            grad_output=grad_out,
            input=x_saved,
            weight=w_saved,
            input_requires_grad=ctx.dgrad,
            weight_requires_grad=ctx.needs_input_grad[1],
            dtype=dtype,
            with_quantized_compute=ctx.quantized,
            input_quantizer=x_q,
            weight_quantizer=w_q,
            grad_output_quantizer=grad_q,
        )
        return grad_x, grad_w, None, None


def _pad(t: torch.Tensor) -> torch.Tensor:
    """Zero-pad both dims to multiples of 32; the product stays exact."""
    rows, cols = -t.shape[0] % _TE_ALIGN, -t.shape[1] % _TE_ALIGN
    return F.pad(t, (0, cols, 0, rows)) if rows or cols else t


def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    recipe = _te_recipe()
    if recipe is None:
        return F.linear(x, w, b)
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype("cuda")
    else:
        dtype = w.dtype
    if recipe.nvfp4():  # its Hadamard transform only takes bf16
        x, w = x.to(torch.bfloat16), w.to(torch.bfloat16)
    out = _TEGemm.apply(_pad(x), _pad(w), recipe, dtype)
    return out[: x.shape[0], : w.shape[0]] + b.to(dtype)


def fourier_layer(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    *,
    weight: torch.Tensor,
    base_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    base_activation,
    **kwargs,
) -> torch.Tensor:
    """``sum_i weight * postacts + base_activation(x) @ base_weight.T + bias`` as one GEMM."""
    c = fourier_coefficients(
        theta, preacts_weight, preacts_bias, reps, x.shape[1], **kwargs
    )
    dtype = base_weight.dtype
    w = torch.cat([c[:, 1:] * weight[:, None], base_weight[:, None]], dim=1)
    b = (c[:, 0] * weight).sum(1).to(dtype)
    if bias is not None:
        b = b + bias
    return _linear(
        fourier_features(x, reps, base_activation), w.flatten(1).to(dtype), b
    )


class TESolver(QKANSolver):
    """Fourier-GEMM solver (registered as ``"te"``).

    Calling it returns postacts like any solver, for ``forward_no_sum``,
    plotting and pruning. ``QKANLayer`` computes its output with
    ``fourier_layer`` instead, which never builds the (B, O, I) postacts.
    """

    name = "te"

    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor:
        return _sample(x, theta, preacts_weight, preacts_bias, reps, **kwargs)


register(TESolver())
