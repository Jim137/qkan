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
QKAN solver backends.

Each solver is implemented in its own module:
- exact: Pure PyTorch reference implementation
- flash: Triton fused kernels
- cute: CuTe DSL CUDA fused kernels
- cutile: cuTile (NVIDIA Tile Language) fused kernels
- cutn: cuQuantum tensor network contraction
- qml: PennyLane quantum circuits
- qiskit: IBM Quantum backends via Qiskit Runtime
- cudaq: NVIDIA CUDA-Q backends (GPU-accelerated simulation or QPU)
"""

from ._activation import make_base_activation
from ._base import QKANSolver, get_registry, get_solver, register
from .cudaq import cudaq_solver
from .cute import _CUTE_AVAILABLE, cute_exact_solver
from .cutile import _CUTILE_AVAILABLE, cutile_flash_exact_solver
from .cutn import cutn_solver
from .exact import torch_exact_solver
from .flash import _FLASH_AVAILABLE, flash_exact_solver
from .qiskit import qiskit_solver
from .qml import qml_solver

__all__ = [
    "_CUTE_AVAILABLE",
    "_CUTILE_AVAILABLE",
    "_FLASH_AVAILABLE",
    "QKANSolver",
    "cudaq_solver",
    "cute_exact_solver",
    "cutile_flash_exact_solver",
    "cutn_solver",
    "flash_exact_solver",
    "get_registry",
    "get_solver",
    "make_base_activation",
    "qiskit_solver",
    "qml_solver",
    "register",
    "torch_exact_solver",
]
