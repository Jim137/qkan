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
fp8 components for an end-to-end fp8 HQKAN workflow on Hopper / Blackwell.

The HQKAN block is ``Linear -> QKAN(base_activation=...) -> Linear`` —
the activation is QKAN's ``base_activation``, applied inside the layer.
The fp8 boundary in this workflow is at the **Linear GEMMs only**:
``Fp8Linear`` weights and inputs are e4m3, the GEMM runs through
cuBLASLt's fp8 path, and outputs are bf16. QKAN consumes bf16 forward-
side; its ``c_dtype=torch.float8_e4m3fn`` flag changes only the
**backward state checkpoints** (e4m3 with prescale=224 on
unitarity-bounded amplitudes), not forward compute. Pointwise ops
(silu / gelu / etc.) likewise stay in bf16 — fp8 brings no benefit
there and would cost precision.

Components
----------
- :class:`Fp8Linear` — drop-in for ``nn.Linear``. e4m3 weights, dynamic
  per-tensor input scaling, bf16 output. Falls back to bf16
  ``F.linear`` when either dimension isn't a multiple of 16 (cuBLASLt's
  fp8 alignment requirement) — preserves fp8-stored weights / reduced
  memory while keeping correctness on HQKAN's narrow inner projections
  (``els = ceil(log2(d_model))``).
- :class:`Fp8Activation` — optional wrapper for stand-alone fp8-in /
  bf16-out activation use cases outside the canonical HQKAN pipeline.
  Not needed for the standard HQKAN block since Fp8Linear's bf16 output
  flows directly into QKAN.
- :func:`quantize_to_fp8` / :func:`dequantize_from_fp8` — per-tensor
  amax-scaled e4m3 conversions.

Hardware: SM89+ (Ada / Hopper / Blackwell). Constructing ``Fp8Linear``
raises a clear error on older GPUs.

Canonical full-fp8 HQKAN block::

    import torch, torch.nn as nn
    from qkan import QKAN
    from qkan.fp8 import Fp8Linear

    block = nn.Sequential(
        Fp8Linear(768, 10),                   # fp8 GEMM, bf16 out
        QKAN([10, 10], reps=1, solver="cute",
             base_activation="silu",
             c_dtype=torch.float8_e4m3fn,     # fp8 state checkpoints (backward)
             p_dtype=torch.bfloat16),
        Fp8Linear(10, 768),                   # bf16 in (auto-quantises), bf16 out
    )
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import init

# fp8 e4m3 has a max representable magnitude of 448 (since the all-ones
# exponent is reserved for NaN). Quantising to "fill the range" means
# scaling so that |x|.max() lands at 448, which gives the best dynamic
# range for the typical activation distribution.
_E4M3_MAX = 448.0


def quantize_to_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``x`` to fp8 with a per-tensor amax scale.

    Returns ``(x_fp8, scale)`` where ``scale`` is the float32 scalar
    such that the original tensor is approximately ``x_fp8 * scale``.
    Use :func:`dequantize_from_fp8` to invert.
    """
    amax = x.detach().abs().amax().to(torch.float32).clamp_min(1e-12)
    scale = amax / _E4M3_MAX  # multiplicative scale: x_fp8 ≈ x / scale
    x_fp8 = (x.to(torch.float32) / scale).to(dtype)
    return x_fp8, scale


def dequantize_from_fp8(
    x_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantise ``x_fp8`` back to ``dtype`` using ``scale`` from :func:`quantize_to_fp8`."""
    return x_fp8.to(torch.float32).mul_(scale).to(dtype)


def _check_fp8_supported() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Fp8 layers require CUDA")
    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        raise RuntimeError(
            f"Fp8 GEMM via torch._scaled_mm requires SM89+ (Ada / Hopper / "
            f"Blackwell); got capability {cap}"
        )


class Fp8Linear(nn.Module):
    """``nn.Linear`` that runs the matmul in fp8 via ``torch._scaled_mm``.

    Weights are stored as e4m3 with a registered ``weight_scale`` buffer.
    Inputs may be either fp8 (with caller-supplied ``input_scale``) or
    bf16/f32 (auto-quantised per-call). Output dtype is configurable;
    bf16 is the default and matches what cuBLASLt's fp8 path produces.

    The class deliberately does NOT participate in autograd through the
    fp8 GEMM — fp8 training is out of scope for this drop-in. Use the
    bf16 backend Linears (``CuTeLinear`` / ``TritonLinear`` / ``CuTileLinear``)
    when you need backward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        _check_fp8_supported()
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_dtype = activation_dtype
        # cuBLASLt's fp8 GEMM requires both inner dims to be multiples of 16.
        # Cached once because the dims are immutable.
        self._fp8_capable = (in_features % 16 == 0) and (out_features % 16 == 0)

        # Initialise in f32 with the same Kaiming uniform nn.Linear uses,
        # then quantise to e4m3 once.
        w_f32 = torch.empty(
            (out_features, in_features), device=device, dtype=torch.float32
        )
        init.kaiming_uniform_(w_f32, a=math.sqrt(5))
        w_fp8, w_scale = quantize_to_fp8(w_f32)
        self.weight = nn.Parameter(w_fp8, requires_grad=False)
        self.register_buffer("weight_scale", w_scale)

        if bias:
            b_f32 = torch.empty(out_features, device=device, dtype=torch.float32)
            bound = 1 / math.sqrt(in_features)
            init.uniform_(b_f32, -bound, bound)
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                b_f32.to(activation_dtype), requires_grad=False
            )
        else:
            self.bias = None

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_scale: torch.Tensor = self.weight_scale  # type: ignore[assignment]
        leading_shape = x.shape[:-1]

        if not self._fp8_capable:
            # Dequantise weight to bf16 once and run a normal F.linear.
            # Inputs may be either fp8 (with scale) or bf16/f32.
            if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                if input_scale is None:
                    raise ValueError(
                        "Fp8Linear: when passing fp8 input, also pass input_scale"
                    )
                x = dequantize_from_fp8(x, input_scale, dtype=self.activation_dtype)
            elif x.dtype != self.activation_dtype:
                x = x.to(self.activation_dtype)
            w = dequantize_from_fp8(
                self.weight, weight_scale, dtype=self.activation_dtype
            )
            y = torch.nn.functional.linear(x, w, self.bias)
            return y.reshape(*leading_shape, self.out_features)

        if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            if input_scale is None:
                raise ValueError(
                    "Fp8Linear: when passing fp8 input, also pass input_scale"
                )
            x_fp8 = x
            x_scale = input_scale.to(torch.float32)
        else:
            x_fp8, x_scale = quantize_to_fp8(x)

        # _scaled_mm: y = (x_fp8 * x_scale) @ (W_fp8.t() * w_scale) + bias
        # Shapes: x_fp8 (M, K), W_fp8 (N, K) -> W_fp8.t() (K, N).
        x_flat = x_fp8.reshape(-1, self.in_features)
        y = torch._scaled_mm(
            x_flat,
            self.weight.t(),
            scale_a=x_scale,
            scale_b=weight_scale.to(torch.float32),
            bias=self.bias,
            out_dtype=self.activation_dtype,
        )
        return y.reshape(*leading_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, activation_dtype={self.activation_dtype}"
        )


class Fp8Activation(nn.Module):
    """Wraps a bf16/f32 activation module so it accepts fp8 input.

    The wrapped module runs in bf16 (which is where pointwise ops belong);
    fp8 inputs are dequantised first. Output is bf16 — the next fp8
    Linear in the chain will requantise it.

    ``activation`` may be any of:
      - :class:`qkan.solver.cute_activation.CuTeActivation`
      - :class:`qkan.solver.triton_activation.TritonActivation`
      - :class:`qkan.solver.cutile_activation.CuTileActivation`
      - any ``torch.nn`` activation (``nn.SiLU``, ``nn.GELU``, ...)
    """

    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            if input_scale is None:
                raise ValueError(
                    "Fp8Activation: when passing fp8 input, also pass input_scale"
                )
            x = dequantize_from_fp8(x, input_scale, dtype=torch.bfloat16)
        return self.activation(x)
