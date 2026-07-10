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


"""QKANSolver ABC and a string-keyed registry.

Each backend module (``cute``, ``flash``, ``cutile``, ``cutn``, ``exact``,
``cudaq``, ``qiskit``, ``qml``) defines a thin ``QKANSolver`` subclass and
registers an instance at import time. ``QKANLayer.forward`` then dispatches
through ``get_solver(name)`` instead of an if/elif chain.
"""

from abc import ABC, abstractmethod

import torch


class QKANSolver(ABC):
    """Common interface for QKAN exact solvers.

    All solvers compute postacts of shape (batch, out_dim, in_dim) given:
      - x: (batch, in_dim) input tensor
      - theta: trainable circuit parameters (shape varies by ansatz)
      - preacts_weight, preacts_bias: pre-activation linear params
      - reps: number of circuit repetitions
      - kwargs: ansatz, group, preacts_trainable, fast_measure, out_dim, dtype
    """

    name: str  # registry key (e.g., "cute", "flash")

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor: ...


_SOLVER_REGISTRY: dict[str, QKANSolver] = {}


def register(solver: QKANSolver) -> QKANSolver:
    """Register a solver instance under its ``name`` attribute."""
    _SOLVER_REGISTRY[solver.name] = solver
    return solver


def get_solver(name: str) -> QKANSolver:
    """Look up a registered solver by name. Raises KeyError if missing."""
    return _SOLVER_REGISTRY[name]


def get_registry() -> dict[str, QKANSolver]:
    """Return the live registry dict (read-only by convention)."""
    return _SOLVER_REGISTRY
