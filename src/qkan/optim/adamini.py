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

"""QKAN-aware Adam-mini optimizer (ICLR 2025 — Zhang et al., arXiv:2406.16793).

Adam-mini's **Principle 1** says: blocks should match the smallest
dense Hessian sub-block. For an MLP weight that's one row (per output
neuron). We computed the QKAN loss Hessian (see
``analyze_qkan_hessian.py``) and confirmed: per-``o`` blocking captures
**100 %** of the within-tensor |H| mass.

**However**, the per-``o`` Hessian block has structure
(rank-1 Gauss-Newton outer product + a diagonal) that a single shared
LR cannot precondition well — empirically per-``o`` blocking is **2×
worse** than Adam on a width-[4, 8, 4] QKAN regression task. Adam's
per-parameter ``v`` adapts the diagonal contribution per-edge which a
mean-LR throws away. The paper itself flags this as a limitation in its
conclusion ("we believe there is great room to improve the learning
rate design if we utilize more information in the dense block, e.g.
eigenvalues").

The empirical sweet spot for QKAN is **one level finer** than the
strict Hessian principle:

* ``theta`` natural ``(O, I, R+1, K)`` → block per ``(o, i, r)``,
  collapse only ``K`` (2 elements for pz / mix; 1 elsewhere). ``v``
  shape ``(O, I, R+1)``. Keeps per-edge per-rep adaptive LRs.
* ``preacts_*`` natural ``(O, I, R)`` → block per ``(o, i)``, collapse
  ``R``. ``v`` shape ``(O, I)``. Per-edge LR.
* ``base_weight`` / ``postact_weights`` / ``postact_bias`` ``(O, I)``
  → one block per tensor. (Per-edge would degenerate to Adam; per-``o``
  hurts — see bench.)
* Other params — one block per tensor.

This matches the paper's intent (collapse the redundant tail dims that
share Hessian curvature) without over-flattening QKAN's per-edge
preconditioning regime. Achieves ~30 % state reduction vs Adam while
matching or beating Adam's convergence on the toy regression bench
(``bench_adamini.py``).

Under ``p_dim=2`` storage the natural rank is collapsed to 2-D
``(O*I, ·)``; the optimiser falls back to per-row blocking on the
stored shape (= per-``(o, i)`` for theta/preacts).

Memory: Adam stores ``2·N`` floats; this stores ``N + #blocks``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["QKANAdamMini"]


def _infer_block_ndim(name: str, shape: tuple[int, ...]) -> int:
    """Return number of leading dims that index *blocks* for this parameter.

    Empirically tuned partition (see module docstring + bench_adamini.py
    for the comparison vs the strict Adam-mini per-output rule):

    * ``theta`` (4-D natural ``(O, I, R+1, K)`` or 2-D ``(O*I, T*K)``):
      block per ``(o, i, r)``. ``block_ndim = ndim - 1`` — collapse only
      the trailing ``K`` axis. Preserves per-edge per-rep adaptive LR.
    * ``preacts_*`` (3-D natural ``(O, I, R)`` or 2-D ``(O*I, R)``):
      block per ``(o, i)``. ``block_ndim = ndim - 1``.
    * ``(O, I)`` params (``base_weight``, ``postact_*``): one block per
      tensor (``block_ndim = 0``). Per-edge would just be Adam.
    * Anything else (BN, scalar bias, mask, custom): one block per
      tensor.
    """
    n = name.lower()
    nd = len(shape)

    if n.endswith("theta") or ".theta" in n:
        # 4-D natural: reduce only the trailing K → block per (o,i,r).
        # 2-D storage (p_dim=2): reduce only the trailing T·K → block per
        # (o,i) row, i.e. per-edge.
        return max(nd - 1, 0)
    if "preacts_weight" in n or "preacts_bias" in n:
        # 3-D natural: reduce only R → block per (o, i).
        # 2-D storage: reduce only R → block per (o,i) row.
        return max(nd - 1, 0)
    if (
        "base_weight" in n
        or "postact_weights" in n
        or "postact_bias" in n
        or "mask" in n
    ):
        # (O, I) params: per-edge ≡ Adam (no savings). One block per
        # tensor is the right point on the granularity / preconditioning
        # tradeoff per the bench.
        return 0

    # ── Non-QKAN parameters (HQKANsformer / mixed models) ────────────────
    # Apply the original Adam-mini partition rules (paper Sec. 2.3):
    #   - LayerNorm / scalar bias / 1-D → single block
    #   - Q/K projections             → per attention head (handled below
    #                                    only with explicit n_heads info;
    #                                    without it, fall back to per-row)
    #   - V / attn_proj / MLP weights → per output neuron (row)
    #   - Embeddings / output / lm_head → per token (row)
    # Per-row is the safe Adam-mini default for any 2-D weight matrix;
    # the per-head refinement requires layer metadata and is provided by
    # passing ``n_heads_per_param`` to the constructor.
    if any(k in n for k in ("ln", "norm", "bias")) and nd <= 1:
        return 0
    if nd >= 2:
        # Per-row blocking (block_ndim=1). For Linear weights this is
        # one block per output neuron, matching the paper's MLP rule.
        # Embeddings (vocab, d) get one block per token row.
        return 1
    # 1-D anything else (scale params, etc.) → one block per tensor.
    return 0


def _block_v_shape(shape: tuple[int, ...], block_ndim: int) -> tuple[int, ...]:
    """Shape of the second-moment tensor ``v``.

    With ``block_ndim`` leading axes indexing blocks and the rest reduced,
    ``v`` keeps the leading axes and is *scalar* (shape ``()``) when
    ``block_ndim == 0``.
    """
    return shape[:block_ndim]


def _reduce_dims(ndim: int, block_ndim: int) -> tuple[int, ...]:
    """The trailing dims of ``g*g`` to mean-reduce over."""
    return tuple(range(block_ndim, ndim))


class QKANAdamMini(Optimizer):
    """Adam-mini optimizer with QKAN-aware block partitioning.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize, or dicts defining parameter
        groups. Parameters passed as ``(name, parameter)`` tuples (via
        ``model.named_parameters()``) get block layouts inferred from the
        name; otherwise all parameters get per-tensor blocking.
    lr : float
        Learning rate.
    betas : (float, float)
        Coefficients for first and second moment running averages.
    eps : float
        Numerical stability term added to ``sqrt(v)``.
    weight_decay : float
        Decoupled weight decay coefficient (AdamW-style).

    Notes
    -----
    For best results, build with ``QKANAdamMini(model.named_parameters(),
    ...)`` — passing names lets the optimizer pick per-edge blocks for
    theta / preacts. Without names, it falls back to per-tensor blocking
    (still correct, just with less memory savings).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        # Normalise input: accept either an iterable of tensors / param
        # groups, or an iterable of (name, tensor) tuples from
        # ``model.named_parameters()``. We stash names on a side dict keyed
        # by id(param) so they survive param_group reshuffling.
        self._param_names: dict[int, str] = {}
        normalised: list[Any] = []
        for item in params:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, p = item
                if isinstance(p, torch.Tensor):
                    self._param_names[id(p)] = name
                    normalised.append(p)
                else:
                    # (name, param_group_dict)
                    normalised.append(p)
            else:
                normalised.append(item)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(normalised, defaults)

    def _get_name(self, p: torch.Tensor) -> str:
        return self._param_names.get(id(p), "")

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("QKANAdamMini does not support sparse grads")

                state = self.state[p]
                if len(state) == 0:
                    name = self._get_name(p)
                    # Prefer natural-rank shape if QKANLayer stashed one on
                    # the parameter (so p_dim=2 storage gets per-(o,i,r)
                    # blocking via the natural 4-D shape, not per-(o,i) via
                    # the collapsed 2-D storage shape).
                    nat = getattr(p, "_qkan_natural_shape", None)
                    shape_for_partition = nat if nat is not None else tuple(p.shape)
                    block_ndim = _infer_block_ndim(name, shape_for_partition)
                    v_shape = _block_v_shape(shape_for_partition, block_ndim)
                    state["step"] = 0
                    state["block_ndim"] = block_ndim
                    state["v_view_shape"] = shape_for_partition
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # ``v`` lives at the block resolution. Use the parameter
                    # dtype / device so mixed-precision setups stay
                    # consistent with Adam's behaviour.
                    state["exp_avg_sq_block"] = torch.zeros(
                        v_shape, dtype=p.dtype, device=p.device
                    )

                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"]
                v_block = state["exp_avg_sq_block"]
                block_ndim = state["block_ndim"]
                view_shape = state["v_view_shape"]
                # Reduce / broadcast at the natural rank — this is what
                # decouples block_ndim from the storage rank (p_dim).
                # When p_dim=4 view_shape == p.shape, so the view is a no-op.
                view_ndim = len(view_shape)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.lerp_(g, 1.0 - beta1)

                # Block-reduced 2nd moment: g·g averaged over the trailing
                # `view_ndim - block_ndim` axes of the natural-rank view.
                g_view = g.view(*view_shape) if g.shape != view_shape else g
                if block_ndim == view_ndim:
                    gg_block = g_view * g_view
                else:
                    reduce_dims = _reduce_dims(view_ndim, block_ndim)
                    gg_block = (g_view * g_view).mean(dim=reduce_dims)
                v_block.mul_(beta2).add_(gg_block, alpha=1.0 - beta2)

                bc1 = 1.0 - beta1**step
                bc2 = 1.0 - beta2**step
                m_hat = m / bc1
                v_hat_block = v_block / bc2

                denom_block = v_hat_block.sqrt().add_(eps)
                if block_ndim == 0:
                    denom = denom_block
                else:
                    trailing = (1,) * (view_ndim - block_ndim)
                    denom_nat = denom_block.view(*denom_block.shape, *trailing)
                    # broadcast back; storage shape ⊇ natural shape via reshape.
                    denom = denom_nat.expand(*view_shape).reshape(p.shape)

                p.addcdiv_(m_hat, denom, value=-lr)

        return loss
