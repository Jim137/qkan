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

"""QKANMuon — Muon for QKAN-bearing transformers (HQKANsformer).

Muon (MomentUm Orthogonalized by Newton-schulz) orthogonalizes the momentum
update of 2-D hidden weight matrices via a Newton-Schulz iteration, and falls
back to AdamW for everything else. In an HQKANsformer the Muon path handles the
genuine transformer matrices (attention qkv/proj, MLP in/out projections) while
the AdamW path handles QKAN's own parameters (``theta``/``base_weight``/
``postact`` — which Muon provably hurts), embeddings, the LM head, biases and
norms.

Recipe (Moonlight/Kimi "Muon is Scalable"): momentum + Newton-Schulz +
update-RMS matching (scale the orthogonalized update so its per-element RMS
matches AdamW's, ~0.2, letting Muon share AdamW's LR/WD) + decoupled weight
decay. Vanilla Muon (Keller Jordan) is recovered with ``match_adamw_rms=False``
and ``weight_decay=0``.

State: Muon params store only a momentum buffer (1x param); AdamW params store
``exp_avg`` + ``exp_avg_sq`` (2x). So vs AdamW-on-everything, QKANMuon saves
~1x param-numel of optimizer state on the matrix params.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["QKANMuon"]

# QKAN-internal parameter name markers — never on the Muon path.
_QKAN_MARKERS = ("theta", "preacts", "base_weight", "postact", "mask")
# Embedding / output-head markers — kept on AdamW (standard Muon practice).
_EMBED_MARKERS = ("wte", "wpe", "embed", "lm_head")


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize ``g`` via the quintic Newton-Schulz iteration.

    Returns a matrix the same shape as ``g`` whose singular values are pushed
    toward 1 (semi-orthogonal). Computed in bfloat16 for speed; coefficients
    ``(a, b, c)`` are Keller Jordan's tuned quintic. The matrix is transposed so
    the iteration runs on the smaller dimension, and Frobenius-normalized first.
    """
    assert g.ndim == 2, "Newton-Schulz expects a 2-D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class QKANMuon(Optimizer):
    """Muon for the 2-D matrices of a QKAN-bearing transformer; AdamW elsewhere.

    Parameters
    ----------
    params : iterable
        Parameters or ``(name, parameter)`` tuples from ``named_parameters()``.
        Names drive the Muon/AdamW routing.
    lr : float
        Learning rate for the Muon (matrix) path.
    adamw_lr : float
        Learning rate for the AdamW (fallback) path.
    momentum : float
        Muon momentum.
    nesterov : bool
        Nesterov-style lookahead on the Muon momentum.
    ns_steps : int
        Newton-Schulz iterations.
    weight_decay : float
        Decoupled weight decay (applied on both paths).
    match_adamw_rms : bool
        If True, scale the orthogonalized update by ``0.2*sqrt(max(out,in))`` so
        its per-element RMS matches AdamW's (Moonlight). If False, use Keller's
        ``max(1, out/in)**0.5`` aspect-ratio scale (vanilla Muon).
    adamw_betas, adamw_eps :
        AdamW hyperparameters (defaults match ``torch.optim.AdamW``).
    muon_filter : optional callable ``(name, shape) -> bool``
        Overrides the default routing (used for ablations and tests).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.02,
        adamw_lr: float = 3e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        match_adamw_rms: bool = True,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        muon_filter: Optional[Callable[[str, tuple[int, ...]], bool]] = None,
    ) -> None:
        if lr < 0.0 or adamw_lr < 0.0:
            raise ValueError(f"Invalid lr={lr}, adamw_lr={adamw_lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not (0.0 <= adamw_betas[0] < 1.0 and 0.0 <= adamw_betas[1] < 1.0):
            raise ValueError(f"Invalid adamw_betas: {adamw_betas}")
        if adamw_eps <= 0.0:
            raise ValueError(f"Invalid adamw_eps: {adamw_eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self._param_names: dict[int, str] = {}
        self._muon_filter = muon_filter
        normalised: list[Any] = []
        for item in params:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, p = item
                if isinstance(p, torch.Tensor):
                    self._param_names[id(p)] = name
                normalised.append(p)
            else:
                normalised.append(item)

        defaults = dict(
            lr=lr,
            adamw_lr=adamw_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            match_adamw_rms=match_adamw_rms,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(normalised, defaults)

    def _get_name(self, p: torch.Tensor) -> str:
        return self._param_names.get(id(p), "")

    @staticmethod
    def is_muon_param(name: str, shape: tuple[int, ...]) -> bool:
        """Default routing: True -> Muon path, False -> AdamW path.

        Decided purely from the parameter name and rank, which are invariant to
        QKAN's ``p_dim`` storage — so (unlike the Adam-mini-family siblings) this
        needs no ``_qkan_natural_shape``: any 2-D QKAN param (``base_weight``,
        ``postact``, or a ``p_dim=2``-collapsed ``theta``) is routed to AdamW by
        its name marker regardless of its stored shape.
        """
        if len(shape) != 2:
            return False
        if any(m in name for m in _QKAN_MARKERS):
            return False
        if any(m in name for m in _EMBED_MARKERS):
            return False
        return True

    def _use_muon(self, p: torch.Tensor) -> bool:
        decide = self._muon_filter or self.is_muon_param
        return decide(self._get_name(p), tuple(p.shape))

    def describe_layout(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Return ``(name, shape, "muon"|"adamw")`` per parameter."""
        out: list[tuple[str, tuple[int, ...], str]] = []
        for group in self.param_groups:
            for p in group["params"]:
                path = "muon" if self._use_muon(p) else "adamw"
                out.append((self._get_name(p), tuple(p.shape), path))
        return out

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            adamw_lr = group["adamw_lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            match_rms = group["match_adamw_rms"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("QKANMuon does not support sparse grads")
                state = self.state[p]

                # First touch: decide the (immutable) route once and init state.
                # Routing + RMS scale depend only on name/shape, so caching them
                # here avoids re-deriving them every step for every parameter.
                if len(state) == 0:
                    state["is_muon"] = self._use_muon(p)
                    if state["is_muon"]:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["scale"] = (
                            0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
                            if match_rms
                            else max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                        )
                    else:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                if state["is_muon"]:
                    # ----- Muon path: state = momentum buffer only -----
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g_eff = g.add(buf, alpha=momentum) if nesterov else buf
                    o = _zeropower_via_newtonschulz5(g_eff, steps=ns_steps)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(o, alpha=-lr * state["scale"])
                else:
                    # ----- AdamW path: replicates torch.optim.AdamW exactly -----
                    state["step"] += 1
                    t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    if wd != 0.0:
                        p.mul_(1.0 - adamw_lr * wd)
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bc1 = 1.0 - beta1**t
                    bc2 = 1.0 - beta2**t
                    step_size = adamw_lr / bc1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
