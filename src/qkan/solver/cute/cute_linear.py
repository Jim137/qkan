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

"""nn.Linear drop-in for QKAN's CuTe solver family.

The forward in ``src/qkan/csrc/cute_linear.cu`` deliberately delegates to
``at::addmm`` / ``at::matmul`` (cuBLAS) rather than reimplementing a tuned
bf16/f32 GEMM with MMA atoms.  That means the cuTe kernel wrapper offers
zero kernel-level benefit over ``torch.nn.functional.linear`` — and on tiny
HQKAN shapes (out ~ els ~ 10) the Python wrapper overhead around it
(extension lookup, autograd.Function dispatch, reshape branch) makes the
"backend" 1.3-1.4x slower than ``nn.Linear``.

So this class is now a thin wrapper that calls ``F.linear`` directly.  It
exists for API symmetry with ``TritonLinear`` / ``CuTileLinear`` (so users
of the ``cute`` solver can write ``CuTeLinear(...)`` next to their solver
choice without thinking) and to match ``nn.Linear`` initialization exactly.
Numerically and graph-wise it is identical to ``nn.Linear``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CuTeLinear(nn.Module):
    """``nn.Linear`` drop-in for the CuTe solver family.

    Matches ``torch.nn.Linear`` initialization (``kaiming_uniform_(a=sqrt(5))``
    for the weight, ``uniform_`` within ``1/sqrt(fan_in)`` for the bias) and
    delegates forward + backward to ``F.linear`` (cuBLAS).
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
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
