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

"""``nn.Linear`` drop-in on NVIDIA Transformer Engine, for NVFP4 / FP8 training.

Low precision is switched on with TE's own context, around a forward pass::

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    with te.autocast(recipe=NVFP4BlockScaling()):
        y = layer(x)

Outside ``te.autocast`` the layer is an ordinary linear layer that follows
``torch.autocast``.  NVFP4 only pays off for large GEMMs; for small ones TE's
quantization overhead makes it slower than bf16.

TE's PyTorch extension is built from source against the installed torch::

    pip install --no-build-isolation "transformer_engine[pytorch,core-cu13]>=2.19"
"""

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import transformer_engine  # type: ignore
import transformer_engine.pytorch as te  # type: ignore
from packaging.version import Version
from transformer_engine.pytorch.quantization import (  # type: ignore
    FP8GlobalStateManager,
)

from ._utils import autocast_dtype

# Before 2.19, NVFP4 training on sm_120 ran on all-zero gradients (TE #3281).
if Version(transformer_engine.__version__) < Version("2.19"):
    _msg = f"TELinear needs transformer_engine>=2.19, found {transformer_engine.__version__}"
    warnings.warn(_msg, stacklevel=2)
    raise ImportError(_msg)

# TE's NVFP4 GEMMs need the token count and both feature sizes divisible by 32.
_ALIGN = 32


class TELinear(te.Linear):
    """``te.Linear`` with ``nn.Linear``'s initialization and checkpoints.

    Under a ``te.autocast`` recipe, shapes TE can't quantize run unquantized
    instead of raising, so the layer can replace any ``nn.Linear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            init_method=partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
            params_dtype=dtype,
            device=device or "cuda",
        )
        self._aligned = in_features % _ALIGN == 0 and out_features % _ALIGN == 0

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "TELinear":
        """A ``TELinear`` with ``linear``'s parameters, e.g. to swap a model's
        projections after its own initialization or checkpoint load."""
        new = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        new.load_state_dict(linear.state_dict())
        return new

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)
        # te.Linear zeroes its bias; nn.Linear draws it like this.
        if self.use_bias and not defer_init:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not FP8GlobalStateManager.is_fp8_enabled():
            return super().forward(x)  # follows torch.autocast by itself
        # TE quantizes the input as given, and NVFP4's Hadamard transform only
        # takes bf16, so cast to the autocast dtype first, as nn.Linear would.
        x = x.to(autocast_dtype(x.dtype))
        if self._aligned and (x.numel() // x.shape[-1]) % _ALIGN == 0:
            return super().forward(x)
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        with te.autocast(enabled=False, recipe=recipe):
            return super().forward(x)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # nn.Linear checkpoints carry no TE metadata; TE ignores None.
        state_dict.setdefault(prefix + "_extra_state", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
