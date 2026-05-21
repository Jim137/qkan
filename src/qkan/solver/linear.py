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

"""Backend-specific drop-in replacements for ``torch.nn.Linear``.

Each backend mirrors ``torch.nn.Linear`` (init, bias handling, leading-dim
preservation) but routes the matmul through a backend-matched kernel:

- :class:`CuTeLinear`   — cuTe DSL CUDA kernel (``csrc/cute_linear.cu``)
- :class:`TritonLinear` — Triton kernel (``triton_linear.py``)
- :class:`CuTileLinear` — cuTile kernel (``cutile_linear.py``)
- :class:`Fp8Linear`    — cuBLASLt fp8 GEMM (Hopper / Blackwell only)
"""

from __future__ import annotations

import importlib

__all__: list[str] = []

# (module_name, class_name, exception_to_swallow). Backends whose runtime
# isn't installed get skipped silently — the user only sees an error when
# they construct the missing class through some other code path.
_BACKENDS: list[tuple[str, str, tuple[type[BaseException], ...]]] = [
    (".cute_linear", "CuTeLinear", (ImportError,)),
    (".triton_linear", "TritonLinear", (ImportError,)),
    (".cutile_linear", "CuTileLinear", (ImportError,)),
    (".fp8", "Fp8Linear", (ImportError, RuntimeError)),
    (".fp8", "Fp8Activation", (ImportError, RuntimeError)),
]

for _mod, _cls, _exc in _BACKENDS:
    try:
        globals()[_cls] = getattr(importlib.import_module(_mod, __package__), _cls)
        __all__.append(_cls)
    except _exc:
        pass

del _mod, _cls, _exc, _BACKENDS, importlib
