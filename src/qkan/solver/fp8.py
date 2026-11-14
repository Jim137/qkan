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
  per-tensor input scaling, bf16 output. Every size runs the fp8 GEMM:
  cuBLASLt needs both feature dimensions to be multiples of 16, so for
  other sizes — HQKAN's narrow inner projections,
  ``els = ceil(log2(d_model))`` — the weight and bias are stored
  zero-padded up to the next one and the output is sliced back.
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
import torch.nn.functional as F
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
    if x.dtype not in (torch.float16, torch.bfloat16):  # exact in float32
        x = x.to(torch.float32)
    amax = torch.linalg.vector_norm(x.detach(), ord=float("inf"), dtype=torch.float32)
    scale = amax.clamp_min(1e-12) / _E4M3_MAX  # x_fp8 ≈ x / scale
    # The division runs in float32: a half x is promoted by the kernel
    # itself, which a scale of x's shape makes it do.
    x_fp8 = (x / scale.expand_as(x)).to(dtype)
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

    ``in_features`` and ``out_features`` may be any size. cuBLASLt needs
    multiples of 16, so for other sizes ``weight`` and ``bias`` are stored
    zero-padded up to the next one: ``weight`` is
    ``(ceil16(out_features), ceil16(in_features))``. Each call then pads
    the quantised input, if ``in_features`` is not a multiple, and slices
    the output back, if ``out_features`` is not. It is the same fp8 GEMM.
    ``state_dict()`` keeps the ``(out_features, in_features)`` layout;
    operands given in that layout, through ``load_state_dict`` or
    ``torch.func.functional_call``, are padded.

    A dtype cast of the layer or of a parent (``.bfloat16()``, ``.half()``,
    ``.to(dtype)``) changes the bias and the output dtype. The e4m3 weight
    and its float32 scale follow device moves only. An output dtype that
    ``torch._scaled_mm`` does not support raises in ``forward`` (torch 2.11
    supports bf16 and float16, and float32 without a bias).

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
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        _check_fp8_supported()
        super().__init__()
        # ``dtype`` is accepted for ``nn.Linear`` drop-in compatibility; it
        # selects the bias / output dtype (same role as ``activation_dtype``).
        # Pass either, not both — they refer to the same thing.
        if dtype is not None:
            if activation_dtype is not torch.bfloat16 and activation_dtype is not dtype:
                raise ValueError(
                    "Fp8Linear: pass either 'dtype' or 'activation_dtype', not both"
                )
            activation_dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.activation_dtype = activation_dtype

        # Initialise in f32 with the same Kaiming uniform nn.Linear uses,
        # then quantise to e4m3 once.
        w_f32 = torch.empty(
            (out_features, in_features), device=device, dtype=torch.float32
        )
        init.kaiming_uniform_(w_f32, a=math.sqrt(5))
        w_fp8, w_scale = quantize_to_fp8(self._pad(w_f32))
        self.weight = nn.Parameter(w_fp8, requires_grad=False)
        self.register_buffer("weight_scale", w_scale)

        if bias:
            b_f32 = torch.empty(out_features, device=device, dtype=torch.float32)
            bound = 1 / math.sqrt(in_features)
            init.uniform_(b_f32, -bound, bound)
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                self._pad(b_f32.to(activation_dtype)), requires_grad=False
            )
        else:
            self.bias = None

    def _pad(self, t: torch.Tensor) -> torch.Tensor:
        """Zero-pad a weight ``(out, in)`` or a bias ``(out,)`` to multiples of 16.

        cuBLASLt's fp8 GEMM takes nothing else. The extra products are zero
        and ``forward`` slices the extra outputs off.
        """
        pad: Tuple[int, ...] = (0, -self.out_features % 16)
        if t.dim() == 2:
            pad = (0, -self.in_features % 16, *pad)
        return F.pad(t, pad) if any(pad) else t

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight_scale: torch.Tensor = self.weight_scale  # type: ignore[assignment]
        leading_shape = x.shape[:-1]

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
        weight: torch.Tensor = self.weight
        bias: Optional[torch.Tensor] = self.bias
        # Operands in the (out, in) layout of a state dict, as
        # torch.func.functional_call installs them.
        if weight.shape == (self.out_features, self.in_features):
            weight = self._pad(weight)
        if bias is not None and bias.shape == (self.out_features,):
            bias = self._pad(bias)
        in_pad, out_pad = -self.in_features % 16, -self.out_features % 16
        if in_pad:
            x_flat = F.pad(x_flat, (0, in_pad))
        mm_out = torch._scaled_mm(
            x_flat,
            weight.t(),
            scale_a=x_scale,
            scale_b=weight_scale.to(torch.float32),
            bias=bias,
            out_dtype=self.activation_dtype,
        )
        # torch < 2.5 returns (out, amax); newer versions return the tensor.
        y = mm_out[0] if isinstance(mm_out, tuple) else mm_out
        if out_pad:
            y = y[:, : self.out_features].contiguous()
        return y.reshape(*leading_shape, self.out_features)

    def _unpadded_shapes(self) -> dict:
        return {
            "weight": (self.out_features, self.in_features),
            "bias": (self.out_features,),
        }

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # A state dict holds the (out, in) layout, as views of the operands:
        # the padded shapes would not tell this layer from another of the
        # same ceil16 sizes.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name, shape in self._unpadded_shapes().items():
            t = destination.get(prefix + name)
            if t is not None:
                destination[prefix + name] = t[tuple(slice(n) for n in shape)]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name, shape in self._unpadded_shapes().items():
            key = prefix + name
            t = state_dict.get(key)
            if self._parameters.get(name) is None or not isinstance(t, torch.Tensor):
                continue  # torch reports these
            if t.shape == shape:
                state_dict[key] = self._pad(t)
            else:
                # The padded shape too: it is what another layer stores.
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{t.shape} from checkpoint, the shape in current model is "
                    f"{torch.Size(shape)}."
                )
                # Reported; given its own parameter torch adds nothing.
                state_dict[key] = self._parameters[name]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __setstate__(self, state):
        super().__setstate__(state)
        # A module pickled before the operands were stored padded.
        for name, shape in self._unpadded_shapes().items():
            p = self._parameters.get(name)
            if p is not None and p.shape == shape:
                p.data = self._pad(p.data)

    def _apply(self, fn, recurse=True):
        # torch._scaled_mm takes an e4m3 weight with a float32 scale and
        # nothing else, so a dtype cast (.bfloat16(), .to(dtype), ...) must
        # not convert them. fn is opaque: ask it what it does to an empty
        # probe, and when that is a cast give the two tensors its device
        # move only. The cast lands on the bias and on the output dtype.
        fp8_state = (self.weight, self.weight_scale)

        def keep_dtype(t):
            if all(t is not s for s in fp8_state):
                return fn(t)
            probe = fn(t.new_empty(0))
            return fn(t) if probe.dtype == t.dtype else t.to(probe.device)

        super()._apply(keep_dtype, recurse)
        probe = torch.empty(0, dtype=self.activation_dtype, device=self.weight.device)
        self.activation_dtype = fn(probe).dtype
        return self

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
