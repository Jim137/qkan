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

"""``TritonAdaBelief`` — single-kernel AdaBelief step via Triton.

PyTorch's ``AdamW(fused=True)`` is ~2x faster than eager AdamW because
``_fused_adam_`` collapses the m/v update + bias-correction + addcdiv
into a single CUDA kernel. There is no analogous ``_fused_adabelief_``
in PyTorch — this module provides one via Triton.

Per-param the step becomes ONE kernel launch instead of ~7 in the eager
path (lerp + sub + mul + addcmul + sqrt + add + addcdiv). The total
kernel-launch count drops by ~7x, which is the dominant cost for many
small parameter tensors on GPU.

CPU and non-Triton fallback: when ``p`` is on CPU or Triton isn't
available, the optimizer transparently falls back to the eager
AdaBelief path (same algorithm, just per-op).

Speed on a GPT-2-small-shaped parameter stack (50257×768 embed + 96
of 768×768 + 48 of 768; RTX 5090):

    AdamW (fused=True)      :  2.43 ms/step
    AdaBelief (eager)       :  4.79 ms/step
    TritonAdaBelief         :  ~2.6 ms/step   (-46% vs eager)
    TritonAdaBelief bf16    :  ~2.0 ms/step   (-58% vs eager, +50% memory savings)

Same algorithm as ``qkan.optim.AdaBelief`` — drop-in.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

from .adabelief import adabelief_step_

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

__all__ = ["TritonAdaBelief"]


if _TRITON_AVAILABLE:

    @triton.jit
    def _adabelief_kernel(
        p_ptr,
        g_ptr,
        m_ptr,
        s_ptr,
        N,
        lr_eff,  # = -lr * sqrt(bc2) / bc1
        b1,
        b2,
        one_minus_b1,
        one_minus_b2,
        eps_sb,  # = eps * sqrt(bc2)
        wd_factor,  # = 1 - lr * wd (1.0 if no wd)
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        # Load (state dtypes may differ from param; we always compute in f32).
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # m ← β₁·m + (1-β₁)·g  (lerp form)
        m_new = b1 * m + one_minus_b1 * g
        # resid = g − m_new
        resid = g - m_new
        # s ← β₂·s + (1-β₂)·resid²
        s_new = b2 * s + one_minus_b2 * resid * resid
        # denom = √s + ε·√bc2
        denom = tl.sqrt(s_new) + eps_sb
        # p ← (1 − lr·wd)·p − lr·√bc2/bc1 · m / denom
        p_new = p * wd_factor + lr_eff * m_new / denom

        tl.store(p_ptr + offs, p_new, mask=mask)
        tl.store(m_ptr + offs, m_new, mask=mask)
        tl.store(s_ptr + offs, s_new, mask=mask)


def _adabelief_kernel_launch(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    s: torch.Tensor,
    *,
    b1: float,
    b2: float,
    lr_eff: float,
    eps_sb: float,
    wd_factor: float,
) -> None:
    N = p.numel()
    if N == 0:
        return
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _adabelief_kernel[grid](
        p,
        g,
        m,
        s,
        N,
        lr_eff,
        b1,
        b2,
        1.0 - b1,
        1.0 - b2,
        eps_sb,
        wd_factor,
        BLOCK=BLOCK,
    )


class TritonAdaBelief(Optimizer):
    """Single-kernel AdaBelief step via Triton — same algorithm as ``AdaBelief``.

    Args:
        params: iterable of parameters.
        lr: learning rate.
        betas: ``(β₁, β₂)``.
        eps: numerical floor on the denominator.
        weight_decay: decoupled (AdamW-style) weight decay.
        state_dtype: dtype for ``m`` and ``s``. ``None`` (default)
            inherits the param dtype. Pass ``torch.bfloat16`` to halve
            state memory; the kernel always computes in fp32 via
            implicit upcasts on load. Note: when the params themselves
            are bf16 and ``state_dtype=None``, ``s`` will accumulate
            squared residuals in bf16 and may underflow on long runs —
            pass ``state_dtype=torch.float32`` explicitly to be safe.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        state_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                state_dtype=state_dtype,
            ),
        )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            state_dtype = group["state_dtype"]
            # Group-level step counter — bc1/sqrt_bc2 shared across params.
            group["_step"] = group.get("_step", 0) + 1
            step = group["_step"]
            bc1 = 1.0 - b1**step
            sqrt_bc2 = math.sqrt(1.0 - b2**step)
            lr_eff = -lr * sqrt_bc2 / bc1
            eps_sb = eps * sqrt_bc2
            wd_factor = 1.0 - lr * wd

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("TritonAdaBelief does not support sparse grads")

                state = self.state[p]
                if not state:
                    sd = state_dtype if state_dtype is not None else p.dtype
                    state["m"] = torch.zeros_like(
                        p, dtype=sd, memory_format=torch.preserve_format
                    )
                    state["s"] = torch.zeros_like(
                        p, dtype=sd, memory_format=torch.preserve_format
                    )

                m = state["m"]
                s = state["s"]
                if _TRITON_AVAILABLE and p.is_cuda:
                    _adabelief_kernel_launch(
                        p,
                        p.grad,
                        m,
                        s,
                        b1=b1,
                        b2=b2,
                        lr_eff=lr_eff,
                        eps_sb=eps_sb,
                        wd_factor=wd_factor,
                    )
                else:
                    adabelief_step_(
                        p,
                        p.grad,
                        m,
                        s,
                        lr=lr,
                        b1=b1,
                        b2=b2,
                        eps=eps,
                        wd=wd,
                        bc1=bc1,
                        sqrt_bc2=sqrt_bc2,
                    )

        return loss
