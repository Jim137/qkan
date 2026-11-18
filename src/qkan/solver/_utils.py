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

import sys
from typing import Optional

import torch


def te_recipe():
    """The recipe of the active ``te.autocast``, or None outside one."""
    quantization = sys.modules.get("transformer_engine.pytorch.quantization")
    if quantization is None:  # te.autocast lives there, so nobody entered one
        return None
    state = quantization.FP8GlobalStateManager
    return state.get_fp8_recipe() if state.is_fp8_enabled() else None


def te_recipe_key() -> Optional[str]:
    """``repr`` of the active recipe, as Transformer Engine keys it itself.

    A layer that follows ``te.autocast`` picks its GEMM from that state when it
    is called, so a captured CUDA graph is valid for one value of this key.
    """
    recipe = te_recipe()
    return None if recipe is None else repr(recipe)


def autocast_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype ``F.linear`` computes ``dtype`` operands in: the CUDA autocast
    dtype under autocast, which leaves float64 alone, else ``dtype``."""
    if dtype == torch.float64 or not torch.is_autocast_enabled():
        return dtype
    if hasattr(torch, "get_autocast_dtype"):  # torch >= 2.4
        return torch.get_autocast_dtype("cuda")
    return torch.get_autocast_gpu_dtype()


def _cast_grads_to_dtype(grad_x, grad_theta, grad_pw, grad_pb, dtype):
    """Cast f32 gradients back to parameter dtype (used after bf16 backward)."""
    grad_x = grad_x.to(dtype)
    grad_theta = grad_theta.to(dtype)
    if grad_pw is not None:
        grad_pw = grad_pw.to(dtype)
    if grad_pb is not None:
        grad_pb = grad_pb.to(dtype)
    return grad_x, grad_theta, grad_pw, grad_pb
