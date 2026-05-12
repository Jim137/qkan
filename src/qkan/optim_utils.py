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

"""Optimizer-side helpers for cross-``p_dim`` checkpoint portability.

The QKAN ``p_dim`` knob changes the *storage rank* of theta / preacts /
(O,I) parameters. Model state dicts are auto-reshaped on load via the
per-module ``_load_from_state_dict`` hook, but optimizer state
(``exp_avg``, ``exp_avg_sq``, ``momentum_buffer`` …) is held by the
optimizer and follows the parameter shape at the time it was first seen.

Call :func:`reshape_optimizer_state` immediately after
``optimizer.load_state_dict(...)`` when the model's current ``p_dim``
differs from the checkpoint's.
"""

from __future__ import annotations

import torch

__all__ = ["reshape_optimizer_state"]


def reshape_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
    """Reshape state tensors to match their parameters' current shapes.

    Walks every per-parameter state entry and, when a tensor's stored
    shape doesn't match its parameter's current shape but the element
    count agrees, reshapes it in place. Mismatches with differing
    ``numel`` are left untouched (the optimizer will raise on its next
    step — that's the right behaviour for a genuine mismatch).

    Returns the number of tensors that were reshaped.
    """
    reshaped = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if value.shape == param.shape:
                    continue
                if value.numel() == param.numel():
                    state[key] = value.reshape(param.shape)
                    reshaped += 1
    return reshaped
