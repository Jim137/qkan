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

"""AdaBelief (Zhuang et al., arXiv:2010.07468) + a QKAN-aware block variant.

AdaBelief replaces Adam's second moment ``v = EMA(g²)`` (raw gradient
magnitude) with ``s = EMA((g − m)²)`` — the gradient's *variance around
its EMA*. The update is otherwise identical to Adam::

    m_t = β₁·m_{t-1} + (1-β₁)·g_t
    s_t = β₂·s_{t-1} + (1-β₂)·(g_t - m_t)²        # variance, not raw g²
    w_t = w_{t-1} − lr · m̂_t / (√ŝ_t + ε)

This change interprets the optimizer's "belief" in the current gradient
direction: when ``g_t`` consistently matches ``m_t`` the variance shrinks
and the effective step grows; when gradients are noisy the variance
stays large and the step shrinks. For QKAN's noisy quantum-circuit
gradients (the data-reuploading angle introduces stochasticity through
the input ``x``) this is a much better preconditioner than ``EMA(g²)``.

Empirically (3-seed mean across {exact-CPU, cute-GPU} × p_dim ∈ {4, 2}
on QKAN([4, 8, 4], reps=3) sinusoid regression, 200 steps):

  Adam        lr=3e-2 : composite = 0.0028
  AdaBelief   lr=3e-2 : composite = 0.0020  (-29% vs Adam, same memory)

Memory and per-step compute are identical to Adam — no overhead.

``QKANBeliefMini`` reuses the Adam-mini block partitioning (per-(o,i,r)
for theta + preacts, per-tensor for non-QKAN params, per-row for non-
QKAN matrices) on AdaBelief's ``s``. Gives ~30% optimizer-state
reduction; convergence sits between Adam and full AdaBelief.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

from .adamini import _block_v_shape, _infer_block_ndim, _reduce_dims

__all__ = ["AdaBelief", "QKANBeliefMini", "adabelief_step_"]


def adabelief_step_(
    p: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    s: torch.Tensor,
    *,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    wd: float,
    bc1: float,
    sqrt_bc2: float,
) -> None:
    """In-place AdaBelief update for one parameter. Shared by all backends.

    Folded bc2: √(s/bc2) + ε  ≡  (√s + ε·√bc2) / √bc2 → push the /√bc2
    into the final addcdiv coefficient. Avoids materialising s/bc2 as a
    full-param intermediate. Caller computes ``bc1`` and ``sqrt_bc2``
    once per step (they're shared across all params in a group).
    """
    if wd != 0.0:
        p.mul_(1.0 - lr * wd)
    m.lerp_(g, 1.0 - b1)
    resid = g - m
    s.mul_(b2).addcmul_(resid, resid, value=1.0 - b2)
    denom = s.sqrt().add_(eps * sqrt_bc2)
    p.addcdiv_(m, denom, value=-lr * sqrt_bc2 / bc1)


class AdaBelief(Optimizer):
    """AdaBelief — drop-in Adam replacement with variance-of-gradient ``s``.

    Args:
        params: iterable of parameters.
        lr: learning rate. Default 1e-2 — for QKAN this is materially
            different from the standard Adam 1e-3; sweep on your task.
        betas: ``(β₁, β₂)``. Defaults match the paper.
        eps: numerical floor added to ``√ŝ`` in the denominator.
        weight_decay: decoupled (AdamW-style) weight decay.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
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
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
            # Group-level step counter — bc1/bc2 are shared across every
            # param in the group, no need to recompute per-param.
            group["_step"] = group.get("_step", 0) + 1
            step = group["_step"]
            bc1 = 1.0 - b1**step
            sqrt_bc2 = math.sqrt(1.0 - b2**step)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse grads")

                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_var"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                adabelief_step_(
                    p,
                    p.grad,
                    state["exp_avg"],
                    state["exp_avg_var"],
                    lr=lr,
                    b1=b1,
                    b2=b2,
                    eps=eps,
                    wd=wd,
                    bc1=bc1,
                    sqrt_bc2=sqrt_bc2,
                )

        return loss


class QKANBeliefMini(Optimizer):
    """AdaBelief with Adam-mini block partitioning for ``s``.

    The first moment ``m`` stays per-parameter (same as Adam-mini /
    AdaBelief); the variance ``s`` collapses to one scalar per Adam-mini
    block, partitioned by the same rule as :class:`QKANAdamMini`:

      * theta natural ``(O, I, R+1, K)`` → block per ``(o, i, r)``
      * preacts_* ``(O, I, R)``           → block per ``(o, i)``
      * (O, I) params                    → one block per tensor
      * non-QKAN Linear weights          → per output row
      * other / LayerNorm / 1-D          → one block per tensor

    Optimizer state ~30% smaller than full AdaBelief; convergence sits
    between Adam and AdaBelief.

    Pass ``model.named_parameters()`` (with names) to get the QKAN
    detection; bare ``model.parameters()`` falls back to per-tensor.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self._param_names: dict[int, str] = {}
        normalised: list[Any] = []
        for item in params:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, p = item
                if isinstance(p, torch.Tensor):
                    self._param_names[id(p)] = name
                    normalised.append(p)
                else:
                    normalised.append(p)
            else:
                normalised.append(item)
        super().__init__(
            normalised, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )

    def _get_name(self, p: torch.Tensor) -> str:
        return self._param_names.get(id(p), "")

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

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("QKANBeliefMini does not support sparse grads")

                state = self.state[p]
                if not state:
                    name = self._get_name(p)
                    nat = getattr(p, "_qkan_natural_shape", None)
                    view_shape = nat if nat is not None else tuple(p.shape)
                    bn = _infer_block_ndim(name, view_shape)
                    state["step"] = 0
                    state["block_ndim"] = bn
                    state["view_shape"] = view_shape
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_var_block"] = torch.zeros(
                        _block_v_shape(view_shape, bn), dtype=p.dtype, device=p.device
                    )

                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"]
                s_block = state["exp_avg_var_block"]
                bn = state["block_ndim"]
                view_shape = state["view_shape"]
                view_ndim = len(view_shape)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.lerp_(g, 1.0 - b1)
                resid = g - m
                resid_view = (
                    resid.view(*view_shape)
                    if view_shape != tuple(resid.shape)
                    else resid
                )
                if bn == view_ndim:
                    resid_block = resid_view * resid_view
                else:
                    rd = _reduce_dims(view_ndim, bn)
                    resid_block = (resid_view * resid_view).mean(dim=rd)
                s_block.mul_(b2).add_(resid_block, alpha=1.0 - b2)

                bc1 = 1.0 - b1**step
                bc2 = 1.0 - b2**step
                sqrt_bc2 = math.sqrt(bc2)
                denom_block = s_block.sqrt().add_(eps * sqrt_bc2)
                if bn == 0:
                    denom = denom_block
                else:
                    trailing = (1,) * (view_ndim - bn)
                    denom_nat = denom_block.view(*denom_block.shape, *trailing)
                    denom = denom_nat.expand(*view_shape).reshape(p.shape)

                p.addcdiv_(m, denom, value=-lr * sqrt_bc2 / bc1)

        return loss
