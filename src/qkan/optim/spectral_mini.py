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

"""QKANSpectralMini — eigenvalue-aware Adam-mini variant.

Adam-mini collapses each dense per-output Hessian block to a single
shared LR. The Adam-mini paper itself flags this as a limitation:

    "we can reach much faster convergence if we utilize more information
    in the dense block to design the learning rate (e.g., using
    eigenvalues of each block)"

The empirical QKAN Hessian (see ``analyze_qkan_hessian.py``) is
essentially a rank-1 Gauss-Newton outer product + a diagonal. We
exploit this by tracking the rank-1 GN direction explicitly:

* Per block ``b`` (one per output neuron), store a single vector
  ``c_b`` flattened to the block's parameter count ``M_b``.
* Update ``c_b ← beta2 * c_b + (1 - beta2) * g_b`` — an EMA of the
  gradient direction within the block. The dominant eigvec of
  ``E[g_b g_b^T]`` is well-approximated by the EMA of ``g_b``.
* Precondition with the inverse of a rank-1-plus-diag operator:
  ``(c_b c_b^T + lambda I)^{-1} g_b``. Sherman-Morrison gives a
  closed-form, no eigendecomp needed:

      (1/lambda) * (g_b - (<c_b, g_b> / (lambda + |c_b|^2)) * c_b)

  Concretely: subtract the c_b-direction component (the GN direction)
  with a smaller effective step, and leave the residual scaled by
  ``1/lambda``.

State: ``m`` (per-param, like Adam) + ``c_b`` (per-block vector,
storage = ``num_blocks * M_b``). Total bytes ≈ 2N for theta (same as
Adam) but the second-moment lives on a sub-manifold — and we get
true second-order info on the dominant GN direction.

Block partition: same as QKANAdamMini — per ``(o, i, r)`` for theta,
per ``(o, i)`` for preacts, per-tensor for (O, I) shapes.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

from .adamini import _infer_block_ndim

__all__ = ["QKANSpectralMini"]


class QKANSpectralMini(Optimizer):
    """Eigenvalue-aware Adam-mini using a rank-1 GN preconditioner.

    Parameters
    ----------
    params : iterable
        Iterable of parameters or ``(name, parameter)`` tuples from
        ``model.named_parameters()``. Names enable per-block layout.
    lr : float
        Learning rate.
    betas : (float, float)
        ``beta1`` for momentum, ``beta2`` for the GN-direction EMA.
    eps : float
        Damping ``lambda`` for the Sherman-Morrison inverse. Acts like
        the diagonal floor of the curvature estimate. Default 1e-3
        empirically stable on QKAN.
    weight_decay : float
        AdamW-style decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(normalised, defaults)

    def _get_name(self, p: torch.Tensor) -> str:
        return self._param_names.get(id(p), "")

    def describe_layout(self) -> list[tuple[str, tuple[int, ...], int, int]]:
        """Return (name, shape, block_ndim, num_blocks) per parameter.

        Useful for sanity-checking the block partition at construction.
        """
        out: list[tuple[str, tuple[int, ...], int, int]] = []
        for group in self.param_groups:
            for p in group["params"]:
                name = self._get_name(p)
                bn = _infer_block_ndim(name, tuple(p.shape))
                if bn == 0:
                    n_blocks = 1
                else:
                    n_blocks = 1
                    for d in p.shape[:bn]:
                        n_blocks *= d
                out.append((name, tuple(p.shape), bn, n_blocks))
        return out

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
                    raise RuntimeError("QKANSpectralMini does not support sparse grads")

                state = self.state[p]
                if len(state) == 0:
                    name = self._get_name(p)
                    # Use natural-rank shape if QKANLayer stashed one (so
                    # the partition decouples from p_dim storage).
                    nat = getattr(p, "_qkan_natural_shape", None)
                    view_shape = nat if nat is not None else tuple(p.shape)
                    block_ndim = _infer_block_ndim(name, view_shape)
                    state["step"] = 0
                    state["block_ndim"] = block_ndim
                    state["view_shape"] = view_shape
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["c"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"]
                c = state["c"]
                block_ndim = state["block_ndim"]
                view_shape = state["view_shape"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.lerp_(g, 1.0 - beta1)
                c.lerp_(g, 1.0 - beta2)

                bc1 = 1.0 - beta1**step
                bc2 = 1.0 - beta2**step
                m_hat = m / bc1
                c_hat = c / bc2

                # Sherman-Morrison: (cc^T + ε I)^{-1} g = (g - (c·g/(ε+|c|²))·c) / ε.
                if block_ndim == 0:
                    c_flat = c_hat.flatten()
                    m_flat = m_hat.flatten()
                    c_norm_sq = (c_flat * c_flat).sum()
                    dot = (c_flat * m_flat).sum()
                    scale = dot / (eps + c_norm_sq)
                    upd = (m_flat - scale * c_flat).div_(eps).view_as(p)
                else:
                    # Reshape to natural rank, then collapse trailing dims
                    # into a single block-internal axis.
                    num_blocks = 1
                    for d in view_shape[:block_ndim]:
                        num_blocks *= d
                    c_blk = c_hat.view(*view_shape).reshape(num_blocks, -1)
                    m_blk = m_hat.view(*view_shape).reshape(num_blocks, -1)
                    c_norm_sq = (c_blk * c_blk).sum(dim=1, keepdim=True)
                    dot = (c_blk * m_blk).sum(dim=1, keepdim=True)
                    scale = dot / (eps + c_norm_sq)
                    upd = (m_blk - scale * c_blk).div_(eps).reshape(p.shape)

                # SM scales like 1/eps on the residual direction; clip on
                # near-zero-curvature blocks at startup to keep magnitudes sane.
                upd = upd.clamp_(min=-1.0, max=1.0)

                p.add_(upd, alpha=-lr)

        return loss
