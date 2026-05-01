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

"""Resolve a string base-activation kind to a backend-matched ``nn.Module``.

When QKANLayer's ``base_activation`` is a string (e.g. ``"silu"``), this
helper picks the activation kernel matching the chosen ``solver``:

  cute   -> qkan.solver.cute_activation.CuTeActivation
  flash  -> qkan.solver.triton_activation.TritonActivation
  cutile -> qkan.solver.cutile_activation.CuTileActivation
  other  -> torch.nn equivalent (SiLU, GELU, ReLU, Tanh, Sigmoid)

Backend imports are guarded; if the matching backend isn't installed we
fall back to the torch.nn equivalent so the layer still works.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

# Canonical kinds — what the backend kernels actually implement.
_CANONICAL_KINDS = {"silu", "gelu_exact", "gelu_tanh", "relu", "tanh", "sigmoid"}

# Aliases the user may pass.  ``"gelu"`` defaults to the EXACT (erf-based)
# variant, matching ``torch.nn.functional.gelu(approximate="none")``.
_KIND_ALIASES = {
    "swish": "silu",
    "gelu": "gelu_exact",
}

_TORCH_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "gelu_exact": lambda: nn.GELU(approximate="none"),
    "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _normalize_kind(kind: str) -> str:
    k = kind.lower()
    k = _KIND_ALIASES.get(k, k)
    if k not in _CANONICAL_KINDS:
        raise ValueError(
            f"Unknown activation kind {kind!r}. "
            f"Supported: {sorted(_CANONICAL_KINDS | _KIND_ALIASES.keys())}"
        )
    return k


def make_base_activation(kind: str, solver: object) -> nn.Module:
    """Return an nn.Module computing the named activation, matched to the solver."""
    k = _normalize_kind(kind)

    if solver == "cute":
        try:
            from .cute_activation import CuTeActivation

            return CuTeActivation(k)
        except ImportError:
            pass
    elif solver == "flash":
        try:
            from .triton_activation import TritonActivation

            return TritonActivation(k)
        except ImportError:
            pass
    elif solver == "cutile":
        try:
            from .cutile_activation import CuTileActivation

            return CuTileActivation(k)
        except ImportError:
            pass

    return _TORCH_ACTIVATIONS[k]()
