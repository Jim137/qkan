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

"""Adam -> L-BFGS finishing schedule.

The original KAN paper (Liu et al., arXiv:2404.19756) and pykan use a
two-phase schedule for symbolic-regression style fits: first-order
optimizer (Adam) for the bulk of training to find a good basin, then
L-BFGS to polish the minimum. The BFGS Hessian approximation captures
curvature once parameters are near a minimum and typically reduces final
loss by 2-10x on KAN-style tasks.

:class:`LBFGSFinisher` is a thin wrapper that holds an "early" optimizer
(any ``torch.optim.Optimizer``, including :class:`~qkan.QKANAdamMini`)
for the first ``pct_early`` fraction of total steps, then auto-switches
to ``torch.optim.LBFGS``. It exposes the standard ``step(closure)``
interface.

Note: L-BFGS *requires* a closure that re-evaluates the loss and zeros
gradients itself. The early optimizer phase also accepts and uses a
closure (zero_grad / forward / backward) — so user code can be uniform
across the swap point.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["LBFGSFinisher", "adam_then_lbfgs"]


class LBFGSFinisher:
    """Composite optimizer: ``early`` for ``pct_early * total_steps``, then LBFGS.

    Use ``step(closure)`` exactly like a regular ``torch.optim.Optimizer``.
    The closure must zero grads, run forward+backward, and return loss::

        def closure():
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            return loss

        for _ in range(total_steps):
            loss = opt.step(closure)

    Parameters
    ----------
    early : torch.optim.Optimizer
        Any standard optimizer (Adam, QKANAdamMini, ...). Will be used
        for the first ``pct_early`` of steps.
    params : iterable of nn.Parameter
        The same params passed to ``early``. Needed to construct the
        L-BFGS optimizer at swap time.
    total_steps : int
        Total budget. After ``int(pct_early * total_steps)`` calls to
        ``step``, the wrapper switches to L-BFGS for the remainder.
    pct_early : float
        Fraction of ``total_steps`` to run with ``early``. Default 0.7
        matches the pykan recipe.
    lbfgs_kwargs : dict, optional
        Override defaults for ``torch.optim.LBFGS``. Sensible defaults
        for KAN-style fits: ``lr=1.0, max_iter=20, history_size=100,
        tolerance_grad=1e-7, line_search_fn='strong_wolfe'``.
    """

    _LBFGS_DEFAULTS = dict(
        lr=1.0,
        max_iter=20,
        history_size=100,
        tolerance_grad=1e-7,
        line_search_fn="strong_wolfe",
    )

    def __init__(
        self,
        early: Optimizer,
        params: Iterable[torch.nn.Parameter],
        total_steps: int,
        pct_early: float = 0.7,
        lbfgs_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if not 0.0 < pct_early <= 1.0:
            raise ValueError(f"pct_early must be in (0, 1], got {pct_early}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        # Materialise params so we can pass them to LBFGS later (generators
        # would be exhausted by the early optimizer's constructor).
        self._params: list[torch.nn.Parameter] = list(params)
        self.early = early
        self.total_steps = total_steps
        self.pct_early = pct_early
        self.switch_at = int(pct_early * total_steps)

        self._lbfgs_kwargs = dict(self._LBFGS_DEFAULTS)
        if lbfgs_kwargs:
            self._lbfgs_kwargs.update(lbfgs_kwargs)

        self._step_count = 0
        self._lbfgs: Optional[torch.optim.LBFGS] = None

    @property
    def using_lbfgs(self) -> bool:
        """True if the next ``step`` will use L-BFGS."""
        return self._step_count >= self.switch_at

    @property
    def current(self) -> Optimizer:
        """The optimizer that will handle the next ``step`` call."""
        if self.using_lbfgs:
            if self._lbfgs is None:
                self._lbfgs = torch.optim.LBFGS(self._params, **self._lbfgs_kwargs)  # type: ignore[arg-type]
            return self._lbfgs
        return self.early

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on the active optimizer.

        Most users won't call this directly — the closure does it. Provided
        for parity with ``torch.optim.Optimizer``.
        """
        self.current.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run one optimization step. ``closure`` is required.

        L-BFGS *requires* a closure that zeros grads, runs forward+backward,
        and returns the loss tensor. The early optimizer is also driven via
        the same closure so user code stays uniform across the swap.
        """
        opt = self.current
        # Both LBFGS and Adam-family accept tensor-returning closures at
        # runtime; the typed stubs declare ``Callable[[], float]`` only.
        loss = opt.step(closure)  # type: ignore[arg-type]
        self._step_count += 1
        return loss  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # state_dict / load_state_dict aren't strictly needed for the bench
    # but keep the API surface plausible for future checkpoint use.
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "total_steps": self.total_steps,
            "pct_early": self.pct_early,
            "early": self.early.state_dict(),
            "lbfgs": self._lbfgs.state_dict() if self._lbfgs is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step_count = state["step_count"]
        self.total_steps = state["total_steps"]
        self.pct_early = state["pct_early"]
        self.switch_at = int(self.pct_early * self.total_steps)
        self.early.load_state_dict(state["early"])
        if state.get("lbfgs") is not None:
            # Materialise the LBFGS opt before loading.
            self._lbfgs = torch.optim.LBFGS(self._params, **self._lbfgs_kwargs)  # type: ignore[arg-type]
            self._lbfgs.load_state_dict(state["lbfgs"])


def adam_then_lbfgs(
    model: torch.nn.Module,
    total_steps: int,
    lr_adam: float = 1e-2,
    pct_adam: float = 0.7,
    use_adam_mini: bool = True,
    lbfgs_kwargs: Optional[dict[str, Any]] = None,
) -> LBFGSFinisher:
    """Convenience factory: build an Adam/AdamMini -> LBFGS schedule.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimized.
    total_steps : int
        Total budget. Adam runs for ``int(pct_adam * total_steps)`` steps,
        L-BFGS for the rest.
    lr_adam : float
        Learning rate for the Adam (or QKANAdamMini) phase.
    pct_adam : float
        Fraction of ``total_steps`` to spend in Adam. Default 0.7.
    use_adam_mini : bool
        If True (default), use :class:`~qkan.QKANAdamMini` as the early
        optimizer — block-aware, ~26% less optimizer state. If False,
        fall back to plain ``torch.optim.Adam``.
    lbfgs_kwargs : dict, optional
        Override L-BFGS defaults.

    Returns
    -------
    LBFGSFinisher
        Ready to drive with ``opt.step(closure)``.
    """
    if use_adam_mini:
        # Local import to avoid a hard cycle at module-load time.
        from .adamini import QKANAdamMini

        early: Optimizer = QKANAdamMini(model.named_parameters(), lr=lr_adam)
    else:
        early = torch.optim.Adam(model.parameters(), lr=lr_adam)

    return LBFGSFinisher(
        early=early,
        params=model.parameters(),
        total_steps=total_steps,
        pct_early=pct_adam,
        lbfgs_kwargs=lbfgs_kwargs,
    )
