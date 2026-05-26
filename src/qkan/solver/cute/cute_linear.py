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

"""nn.Linear drop-in backed by a cuTe DSL CUDA kernel.

Mirrors ``torch.nn.Linear`` semantics (init, bias handling, leading-dim
preservation) but routes the matmul through the cuTe forward kernel in
``csrc/cute_linear.cu``.  Backward delegates to cuBLAS via torch APIs.

Use this when you want a CUDA-graph-friendly Linear that lives on the same
stream as QKAN's cuTe solver (``solver="cute"``).  For inference under
``torch.no_grad()`` / ``torch.inference_mode()`` we bypass autograd.Function
entirely to skip its overhead.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Extension loading (mirrors cute_ops.py: pre-built → JIT fallback)
# ---------------------------------------------------------------------------

# We probe availability eagerly (without compiling) by reusing cute_ops's
# detection logic, then load the extension lazily on first call.
try:
    from .cute_ops import _CUTE_KERNELS_AVAILABLE, _get_ext

    _CUTE_LINEAR_AVAILABLE: bool = bool(_CUTE_KERNELS_AVAILABLE)
except ImportError:
    _get_ext = None  # type: ignore[assignment]
    _CUTE_LINEAR_AVAILABLE = False


def _require_ext():
    """Resolve the CuTe extension, raising ImportError if unavailable."""
    if not _CUTE_LINEAR_AVAILABLE or _get_ext is None:
        raise ImportError(
            "CuTeLinear requires the qkan CuTe extension.  Build it with:\n"
            "  CUTLASS_PATH=/path/to/cutlass pip install --no-build-isolation -e .[cute]\n"
            "or pick a different Linear backend (Triton, cuTile, torch)."
        )
    ext = _get_ext()
    if not (hasattr(ext, "linear_forward") and hasattr(ext, "linear_backward")):
        raise ImportError(
            "qkan._C is loaded but does not expose linear_forward/linear_backward. "
            "Rebuild the extension after pulling the latest source."
        )
    return ext


# ---------------------------------------------------------------------------
# autograd.Function
# ---------------------------------------------------------------------------


class _CuTeLinearFunction(torch.autograd.Function):
    """y = x @ W^T + b, computed via the cuTe forward kernel."""

    @staticmethod
    def forward(
        ctx, x_2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ):
        ext = _require_ext()
        y = ext.linear_forward(x_2d, weight, bias)
        ctx.save_for_backward(x_2d, weight)
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        ext = _require_ext()
        x_2d, weight = ctx.saved_tensors
        grad_x, grad_w, grad_b = ext.linear_backward(
            grad_y.contiguous(), x_2d, weight, ctx.has_bias
        )
        if not ctx.has_bias:
            grad_b = None
        return grad_x, grad_w, grad_b


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------


class CuTeLinear(nn.Module):
    """``nn.Linear`` drop-in computing ``y = x @ W^T + b`` via a cuTe DSL kernel.

    Matches ``torch.nn.Linear`` initialization (``kaiming_uniform_(a=sqrt(5))``
    for the weight, ``uniform_`` within ``1/sqrt(fan_in)`` for the bias) and
    preserves leading batch dims.

    Supports float32 and bfloat16 I/O; compute is performed in f32 internally.
    Backward delegates to cuBLAS via ``torch.matmul`` for simplicity.
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
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match torch.nn.Linear.reset_parameters exactly.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten leading dims, run kernel, then restore.
        if x.dim() < 2:
            raise ValueError(
                f"CuTeLinear expects input with at least 2 dims, got {x.dim()}"
            )
        leading = x.shape[:-1]
        K = x.shape[-1]
        if K != self.in_features:
            raise ValueError(
                f"CuTeLinear: input last dim {K} != in_features {self.in_features}"
            )

        x_2d = x.reshape(-1, K).contiguous()

        # Inference fast path: skip autograd.Function overhead under no_grad /
        # inference_mode where backward is never called.
        if not torch.is_grad_enabled() or not (
            x_2d.requires_grad
            or self.weight.requires_grad
            or (self.bias is not None and self.bias.requires_grad)
        ):
            ext = _require_ext()
            y_2d = ext.linear_forward(x_2d, self.weight, self.bias)
        else:
            y_2d = _CuTeLinearFunction.apply(x_2d, self.weight, self.bias)

        return y_2d.reshape(*leading, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
