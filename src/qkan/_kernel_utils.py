# mypy: ignore-errors
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
"""Shared utilities for GPU kernel launch functions (cuTile and Triton)."""


def _select_block_b(n_oi: int, batch: int, base: int = 32) -> int:
    """Choose BLOCK_B adaptively based on grid size.

    For large grids, bigger blocks reduce total program count, state memory
    allocation, and per-block indexing overhead.
    """
    if base != 32:
        return base  # non-standard base (e.g. BLOCK_B=1 for f32 real)
    # Only scale up when n_oi is large enough that reducing batch blocks helps
    if n_oi >= 256 and batch >= 128:
        return 128
    if n_oi >= 256 and batch >= 64:
        return 64
    return 32
