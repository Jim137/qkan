# Copyright (c) 2024, Jiun-Cheng Jiang. All rights reserved.
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
- torch_exact: Pure PyTorch reference implementation
- flash: Triton fused kernels
- cutile: cuTile (NVIDIA Tile Language) fused kernels
- cutn: cuQuantum tensor network contraction
- qml: PennyLane quantum circuits
- qiskit_solver: IBM Quantum backends via Qiskit Runtime
- cudaq_solver: NVIDIA CUDA-Q backends (GPU-accelerated simulation or QPU)
"""

from .cudaq_solver import cudaq_solver
from .cutile import _CUTILE_AVAILABLE, cutile_flash_exact_solver
from .cutn import cutn_solver
from .flash import _FLASH_AVAILABLE, flash_exact_solver
from .qiskit_solver import qiskit_solver
from .qml import qml_solver
from .torch_exact import torch_exact_solver

__all__ = [
    "_CUTILE_AVAILABLE",
    "_FLASH_AVAILABLE",
    "cudaq_solver",
    "cutile_flash_exact_solver",
    "cutn_solver",
    "flash_exact_solver",
    "qiskit_solver",
    "qml_solver",
    "torch_exact_solver",
]
