# Copyright (c) 2024, Jiun-Cheng Jiang. All rights reserved.
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

"""
Quantum-inspired Kolmogorov Arnold Networks (QKANs) implementation in PyTorch.
Paper: Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks: https://arxiv.org/abs/2509.14026

Supported solvers:
    - PennyLane
    - Exact solver implemented in PyTorch (faster)
    - Custom solvers api
"""

import math
import os
import random
import warnings
from copy import deepcopy
from glob import glob
from typing import Callable, Literal, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # type: ignore

from .info import get_dist_info, print0, print_version
from .optim.lbfgs import LBFGSFinisher
from .qkan_epilogue import QKANEpilogue, qkan_epilogue_forward
from .solver import get_registry, get_solver, make_base_activation, qml_solver


class QKANLayer(nn.Module):
    """
    QKANLayer Class

    Attributes
    ----------
        in_dim : int
            Input dimension
        out_dim : int
            Output dimension
        reps : int
            Repetitions of quantum layers
        group : int
            Group of neurons
        device :
            Device to use
        solver : Union[str, Callable]
            Solver to use, currently supports "qml", "exact", "flash", "cutn" or custom callable
        ansatz : Union[str, Callable]
            Ansatz to use, "pz_encoding", "px_encoding", "rpz_encoding" or custom
        qml_device : str
            PennyLane device to use
        theta : nn.Parameter
            Learnable parameter of quantum circuit
        base_weight : nn.Parameter
            Learnable parameter of base activation
        preact_trainable : bool
            Whether preact weights are trainable
        preacts_weight : nn.Parameter
            Learnable parameter of preact weights
        preacts_bias : nn.Parameter
            Learnable parameter of preact bias
        postact_weight_trainable : bool
            Whether postact weights are trainable
        postact_weights : nn.Parameter
            Learnable parameter of postact weights
        postact_bias_trainable : bool
            Whether postact bias are trainable
        postact_bias : nn.Parameter
            Learnable parameter of postact bias
        mask : nn.Parameter
            Mask for pruning
        is_batchnorm : bool
            Whether to use batch normalization
        fast_measure : bool
            Enable to use fast measurement in exact solver. Which would be quantum-inspired method.
            When False, the exact solver simulates the exact measurement process of quantum circuit.
        c_dtype : torch.dtype
            Compute dtype for quantum simulation. Supported values:

            - ``torch.complex64`` / ``torch.float32``: full-precision f32 (default)
            - ``torch.bfloat16``: mixed-precision bf16 I/O, f32 compute, bf16 state checkpoints
            - ``torch.float8_e4m3fn``: bf16 I/O, f32 compute, fp8 prescaled state checkpoints
        p_dtype : torch.dtype
            Parameter dtype (``torch.float32`` or ``torch.bfloat16``).
            Use ``torch.bfloat16`` with bf16/fp8 ``c_dtype`` for full mixed-precision pipeline.
        _x0 : Optional[torch.Tensor]
            Leave for ResQKANLayer
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        reps: int = 3,
        group: Union[int, tuple] = -1,
        device="cpu",
        solver: Union[
            Literal[
                "qml",
                "exact",
                "flash",
                "cutn",
                "tn",
                "cute",
                "cutile",
                "qiskit",
                "cudaq",
            ],
            Callable,
        ] = "exact",
        qml_device="default.qubit",
        ansatz: Union[str, Callable] = "pz_encoding",
        theta_size: Optional[list[int]] = None,
        preact_trainable: bool = False,
        preact_init: bool = False,
        postact_weight_trainable: bool = False,
        postact_bias_trainable: bool = False,
        base_activation: Union[str, nn.Module] = "silu",
        ba_trainable: bool = True,
        is_batchnorm: bool = False,
        fast_measure: bool = True,
        c_dtype: torch.dtype = torch.complex64,
        p_dtype: torch.dtype = torch.float32,
        seed=None,
        solver_kwargs: Optional[dict] = None,
        p_dim: Literal[2, 4] = 4,
        checkpoint_reps: bool = False,
        theta_init: Literal["small_gaussian", "xavier"] = "xavier",
    ):
        super(QKANLayer, self).__init__()
        if p_dim not in (2, 4):
            raise ValueError(f"p_dim must be 2 or 4, got {p_dim}")
        self.p_dim: Literal[2, 4] = p_dim
        self.solver_kwargs = solver_kwargs or {}
        # Activation checkpointing for the rep loop (exact solver only).
        # Trades ~1 extra forward pass for reps× less rep-state memory.
        # Non-exact solvers ignore this kwarg.
        self.checkpoint_reps = checkpoint_reps
        # If base_activation is a string ("silu", "gelu", ...), pick the
        # backend implementation matching the chosen solver — cute/flash/
        # cutile get their fused kernel, others fall back to torch.nn.
        # ``"gelu"`` resolves to gelu_exact (matches torch's default).
        if isinstance(base_activation, str):
            base_activation = make_base_activation(base_activation, solver)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if isinstance(group, int):
            if group == -1:
                group = (out_dim, in_dim)
            else:
                group = tuple([group])

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reps = reps
        self.group = group
        self.device = device
        self.solver: Union[
            Literal[
                "qml",
                "exact",
                "flash",
                "cutn",
                "tn",
                "cute",
                "cutile",
                "qiskit",
                "cudaq",
            ],
            Callable,
        ] = solver
        self.qml_device = qml_device
        self.ansatz = ansatz
        self.theta_size = theta_size
        self.base_activation = base_activation
        self.ba_trainable = ba_trainable
        self.is_batchnorm = is_batchnorm
        self.fast_measure = fast_measure
        self.seed = seed
        # Flash / cutile / cute solvers use real-valued Triton/CUDA kernels and
        # cannot operate on complex dtypes.  Transparently map complex → real so
        # users can keep the default ``c_dtype=torch.complex64`` without error.
        if solver in ("flash", "cutile", "cute") and c_dtype in (
            torch.complex64,
            torch.complex128,
        ):
            _complex_to_real = {
                torch.complex64: torch.float32,
                torch.complex128: torch.float64,
            }
            c_dtype = _complex_to_real[c_dtype]

        self.c_dtype = c_dtype
        self.p_dtype = p_dtype

        self.preact_trainable = preact_trainable
        self.preact_init = preact_init
        self.postact_weight_trainable = postact_weight_trainable
        self.postact_bias_trainable = postact_bias_trainable
        self.theta_init = theta_init

        # Fused-epilogue feature flag. When True the (postacts+pb)*eff_pw sum
        # and the base linear term are collapsed into a single Triton kernel.
        # Default OFF for parity; opt-in via constructor arg, env var, or
        # ``layer.set_fused_epilogue(True)`` post-construction.
        self._use_fused_epilogue: bool = (
            os.environ.get("QKAN_FUSED_EPILOGUE", "0") == "1"
        )

        self.init_parameters()

    def init_parameters(self):
        """Create all learnable parameters.

        Called once from ``__init__`` to allocate ``nn.Parameter`` objects.
        Reads configuration from ``self.*`` attributes.
        If ``self.seed`` is set, the RNG is seeded for reproducibility.

        Calls ``xavier_init()`` at the end to apply Xavier normal
        initialization to theta (and preacts when ``preact_init`` is set).
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        group = self.group
        reps = self.reps
        device = self.device
        p_dtype = self.p_dtype

        # group_count = prod(group_tuple): unifies (O,I) and (g,) cases.
        self.group_count = 1
        for _g in group:
            self.group_count *= _g

        # -- theta --
        # Determine natural-rank shape (what solvers / internal code see).
        # Storage may be 2D under p_dim=2; we expose natural via _theta_natural().
        if callable(self.solver) or callable(self.ansatz):
            if not self.theta_size:
                raise ValueError("theta_size is required for custom ansatz")
            # Custom theta: respect literally; p_dim does NOT reshape it.
            self._theta_natural_shape = tuple(self.theta_size)
            self._theta_collapsible = False
            storage_theta_shape: tuple[int, ...] = tuple(self.theta_size)
        else:
            if self.ansatz in ("pz_encoding", "pz", "mix"):
                self._theta_natural_shape = (*group, reps + 1, 2)
            elif self.ansatz in ("rpz_encoding", "rpz", "px_encoding", "px"):
                self._theta_natural_shape = (*group, reps + 1, 1)
            elif self.ansatz == "real":
                self._theta_natural_shape = (*group, reps, 1)
                self.c_dtype = torch.bfloat16
            else:
                raise NotImplementedError()
            self._theta_collapsible = True
            if self.p_dim == 2:
                storage_theta_shape = (
                    self.group_count,
                    self._theta_natural_shape[-2] * self._theta_natural_shape[-1],
                )
            else:
                storage_theta_shape = self._theta_natural_shape

        self.theta = nn.Parameter(
            torch.empty(*storage_theta_shape, device=device, dtype=p_dtype)
        )

        # -- preacts_weight / preacts_bias --
        # rpz_encoding always needs trainable bias (even when preact_trainable=False)
        _bias_trainable = self.preact_trainable or self.ansatz in (
            "rpz_encoding",
            "rpz",
        )
        self._preacts_natural_shape: tuple[int, ...] = (*group, reps)
        if self.p_dim == 2:
            storage_preacts_shape: tuple[int, ...] = (self.group_count, reps)
        else:
            storage_preacts_shape = self._preacts_natural_shape
        self.preacts_weight = nn.Parameter(
            torch.ones(*storage_preacts_shape, device=device, dtype=p_dtype),
            requires_grad=self.preact_trainable,
        )
        self.preacts_bias = nn.Parameter(
            torch.zeros(*storage_preacts_shape, device=device, dtype=p_dtype),
            requires_grad=_bias_trainable,
        )

        # -- (O, I) params: base_weight, postact_weights, postact_bias, mask --
        if self.p_dim == 2:
            oi_shape: tuple[int, ...] = (self.out_dim * self.in_dim,)
        else:
            oi_shape = (self.out_dim, self.in_dim)

        # -- base_weight --
        if self.ba_trainable:
            self.base_weight = torch.nn.Parameter(
                0.5 * torch.ones(*oi_shape, device=device, dtype=p_dtype),
                requires_grad=True,
            )
        else:
            self.base_weight = torch.nn.Parameter(
                torch.zeros(*oi_shape, device=device, dtype=p_dtype),
                requires_grad=False,
            )

        # -- postact_weights / postact_bias --
        self.postact_weights = nn.Parameter(
            torch.ones(*oi_shape, device=device, dtype=p_dtype),
            requires_grad=self.postact_weight_trainable,
        )
        self.postact_bias = nn.Parameter(
            torch.zeros(*oi_shape, device=device, dtype=p_dtype),
            requires_grad=self.postact_bias_trainable,
        )
        # Track whether a *frozen* postact_bias holds nonzero values (e.g.
        # loaded from a checkpoint trained with postact_bias_trainable=True)
        # so the bias-skip fast paths can't drop stored values. Refreshed on
        # checkpoint load and by ``reset_parameters``.
        self._pb_is_nonzero: bool = False

        # -- mask --
        self.mask = nn.Parameter(
            torch.ones(*oi_shape, device=device, dtype=p_dtype),
            requires_grad=False,
        )
        # Track whether mask is still the all-ones identity so the forward
        # hot path can skip the `* mask` multiplies. Flipped to False by
        # ``prune_edge``; ``reset_parameters`` restores it to True and
        # checkpoint loads recompute it from the loaded tensor.
        self._mask_is_identity: bool = True

        # -- batchnorm --
        if self.is_batchnorm:
            self.bn = nn.BatchNorm1d(self.in_dim, device=device, dtype=p_dtype)

        # -- inference-mode cached effective weights --
        # Populated when the module enters eval() mode; invalidated in train().
        # Stored as plain attributes (not buffers) so state_dict isn't polluted.
        self._eff_pw: Optional[torch.Tensor] = None
        self._eff_base_w: Optional[torch.Tensor] = None
        self._bias_sum: Optional[torch.Tensor] = None
        # Tensor aliases for the forward hot path. Under p_dim=2 they are
        # ``.view()`` results (plain Tensor, not Parameter — won't leak
        # into state_dict); under p_dim=4 they alias the Parameters
        # themselves via object.__setattr__ (also kept out of state_dict).
        # ``_init_view_caches`` populates all seven attributes before any
        # forward — they are non-None on every code path. ``.to()`` rebuilds
        # them if param storage moved.
        self._theta_natural_v: torch.Tensor = None  # type: ignore[assignment]
        self._preacts_w_natural_v: torch.Tensor = None  # type: ignore[assignment]
        self._preacts_b_natural_v: torch.Tensor = None  # type: ignore[assignment]
        self._pw_oi: torch.Tensor = None  # type: ignore[assignment]
        self._pb_oi: torch.Tensor = None  # type: ignore[assignment]
        self._bw_oi: torch.Tensor = None  # type: ignore[assignment]
        self._mask_oi: torch.Tensor = None  # type: ignore[assignment]
        self._init_view_caches()
        self._resolve_solver_fn()

        self._x0: Optional[torch.Tensor] = None

        try:
            self.xavier_init()
        except Exception:
            warnings.warn("xavier_init failed, using default initialization")

    # ------------------------------------------------------------------
    # View helpers: bridge between storage rank (p_dim) and the natural
    # rank consumed by solvers and the contraction epilogue.
    #
    # Under ``p_dim=2`` the heavy-use views (natural-rank theta/preacts,
    # 2D (O,I) views of postact_*, base_weight, mask) are pre-materialised
    # by ``_init_view_caches`` so the hot-path forward avoids per-call
    # ``.view()`` overhead. The cached tensors share storage with their
    # parent Parameter — in-place optimizer updates remain visible.
    # ------------------------------------------------------------------
    def _init_view_caches(self) -> None:
        """Build / rebuild the natural-rank and (O,I) tensor aliases.

        Under ``p_dim=2`` constructs proper ``.view()`` aliases that
        share storage with the parent Parameter. Under ``p_dim=4`` the
        parameters already match the natural rank, so the aliases are
        the parameters themselves. Either way the aliases are written
        via ``object.__setattr__`` to bypass ``nn.Module.__setattr__``'s
        Parameter detection — keeps them out of ``state_dict`` and the
        forward hot path can read a single attribute regardless of
        ``p_dim`` with no Python-side dispatch.
        """
        if self.p_dim == 2:
            theta_v = (
                self.theta.view(*self._theta_natural_shape)
                if self._theta_collapsible
                else self.theta
            )
            pw_v = self.preacts_weight.view(*self._preacts_natural_shape)
            pb_v = self.preacts_bias.view(*self._preacts_natural_shape)
            oi = (self.out_dim, self.in_dim)
            pw_oi = self.postact_weights.view(*oi)
            pb_oi = self.postact_bias.view(*oi)
            bw_oi = self.base_weight.view(*oi)
            mask_oi = self.mask.view(*oi)
        else:
            theta_v = self.theta
            pw_v = self.preacts_weight
            pb_v = self.preacts_bias
            pw_oi = self.postact_weights
            pb_oi = self.postact_bias
            bw_oi = self.base_weight
            mask_oi = self.mask
        # Bypass Module's Parameter detection so aliases don't show up
        # in state_dict / named_parameters.
        object.__setattr__(self, "_theta_natural_v", theta_v)
        object.__setattr__(self, "_preacts_w_natural_v", pw_v)
        object.__setattr__(self, "_preacts_b_natural_v", pb_v)
        object.__setattr__(self, "_pw_oi", pw_oi)
        object.__setattr__(self, "_pb_oi", pb_oi)
        object.__setattr__(self, "_bw_oi", bw_oi)
        object.__setattr__(self, "_mask_oi", mask_oi)
        # Stash natural-rank shape on each Parameter so optimisers (e.g.
        # QKANAdamMini) can partition by the natural shape even when
        # storage is p_dim=2-collapsed.
        oi = (self.out_dim, self.in_dim)
        object.__setattr__(self.theta, "_qkan_natural_shape", self._theta_natural_shape)
        object.__setattr__(
            self.preacts_weight, "_qkan_natural_shape", self._preacts_natural_shape
        )
        object.__setattr__(
            self.preacts_bias, "_qkan_natural_shape", self._preacts_natural_shape
        )
        object.__setattr__(self.postact_weights, "_qkan_natural_shape", oi)
        object.__setattr__(self.postact_bias, "_qkan_natural_shape", oi)
        object.__setattr__(self.base_weight, "_qkan_natural_shape", oi)
        object.__setattr__(self.mask, "_qkan_natural_shape", oi)

    def _theta_natural(self) -> torch.Tensor:
        return self._theta_natural_v  # type: ignore[return-value]

    def _preacts_w_natural(self) -> torch.Tensor:
        return self._preacts_w_natural_v  # type: ignore[return-value]

    def _preacts_b_natural(self) -> torch.Tensor:
        return self._preacts_b_natural_v  # type: ignore[return-value]

    def _as_oi(self, t: torch.Tensor) -> torch.Tensor:
        """Return a 2D ``(O, I)`` view of an (O,I) param (no-op when p_dim=4).

        Cold-path helper for index/copy ops on arbitrary tensors. Hot-path
        forward uses the pre-cached ``self._pw_oi`` etc. directly.
        """
        if self.p_dim == 4:
            return t
        return t.view(self.out_dim, self.in_dim)

    def xavier_init(self):
        """Initialise theta and (optionally) preacts.

        Theta init follows ``self.theta_init``:

        - ``"xavier"`` (default): Xavier-normal init (fan-in/fan-out aware).
        - ``"small_gaussian"``: ``theta ~ N(0, sigma^2)`` with
          ``sigma = 1/sqrt(reps)``. Mitigates the barren-plateau regime in
          deep parametric quantum circuits (Zhang+ 2022 arXiv:2203.09376).
          Opt-in for ``reps >= 8``; underperforms xavier at small reps.

        Preacts (when ``preact_init`` is True) keep Xavier — they are
        amplitude-scaled so fan-in/out are meaningful.
        """
        if self.theta_init == "small_gaussian":
            nn.init.normal_(
                self._theta_natural().data,
                mean=0.0,
                std=1.0 / math.sqrt(max(self.reps, 1)),
            )
        elif self.theta_init == "xavier":
            nn.init.xavier_normal_(self._theta_natural().data)
        else:
            raise ValueError(f"Unknown theta_init: {self.theta_init!r}")
        if self.preact_init:
            nn.init.xavier_normal_(self._preacts_w_natural().data)
            nn.init.xavier_normal_(self._preacts_b_natural().data)

    # Tensor names whose shape may differ between p_dim=2 and p_dim=4 storage.
    # Restricting the load-time reshape to this set avoids masking genuine
    # shape mismatches elsewhere in the state_dict.
    _PDIM_RESHAPABLE_NAMES = (
        "theta",
        "preacts_weight",
        "preacts_bias",
        "base_weight",
        "postact_weights",
        "postact_bias",
        "mask",
    )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Allow cross-p_dim checkpoint loading: when a known param's saved
        # shape doesn't match the current storage but the element count
        # agrees, reshape in place. Anything else falls through to PyTorch's
        # default behaviour (which will error on mismatch as expected).
        for name in self._PDIM_RESHAPABLE_NAMES:
            key = prefix + name
            if key not in state_dict:
                continue
            target = getattr(self, name, None)
            if not isinstance(target, torch.Tensor):
                continue
            source = state_dict[key]
            if source.shape != target.shape and source.numel() == target.numel():
                state_dict[key] = source.reshape(target.shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # The checkpoint may carry a pruned (non-identity) mask or a nonzero
        # frozen bias — recompute the fast-path flags so the forward routes
        # don't silently skip the `* mask` / `+ postact_bias` terms.
        with torch.no_grad():
            self._mask_is_identity = bool((self.mask == 1).all())
            self._pb_is_nonzero = bool(self.postact_bias.data.any())

    def to(self, *args, **kwargs):
        """
        Move the layer to the specified device.

        Args
        ----
            device : str | torch.device
                Device to move the layer to, default: "cpu"
        """
        device = None
        for arg in args:
            if isinstance(arg, str) or isinstance(arg, torch.device):
                device = arg
                break
            elif isinstance(arg, torch.dtype):
                self.p_dtype = arg
        if isinstance(kwargs.get("dtype"), torch.dtype):
            self.p_dtype = kwargs["dtype"]
        if "device" in kwargs:
            device = kwargs["device"]
        if device:
            self.device = device
            for param in self.parameters():
                param.data = param.to(device)
        ret = super(QKANLayer, self).to(*args, **kwargs)
        # Param storage may have moved — rebuild cached views over the new
        # storage. No-op under p_dim=4.
        self._init_view_caches()
        # p_dtype may have changed — refresh the cached dtype-cast flag so
        # solver outputs are cast to the live parameter dtype.
        self._needs_dtype_cast = self.c_dtype != self.p_dtype
        return ret

    def train(self, mode: bool = True):
        ret = super().train(mode)
        if mode:
            # Invalidate inference caches (stale once params start updating).
            self._eff_pw = None
            self._eff_base_w = None
            self._bias_sum = None
        else:
            # Fold mask into postact_weights/base_weight and collapse the
            # postact_bias contribution into a (O,) bias term that rides
            # along with F.linear's bias arg. Cache is always materialised
            # as (O, I)/(O,) regardless of p_dim. Skip bias_sum when bias
            # is untrainable (identically zero) — F.linear(bias=None) is
            # cheaper than F.linear with a zero bias. When mask is the
            # all-ones identity (the default), skip the `* mask` multiplies
            # — the cached weights then alias the underlying params.
            with torch.no_grad():
                pw_2d = self._as_oi(self.postact_weights)
                bw_2d = self._as_oi(self.base_weight)
                if self._mask_is_identity:
                    # Alias the params (detach so nn.Module.__setattr__
                    # doesn't re-register them as duplicate Parameters).
                    self._eff_pw = pw_2d.detach()
                    self._eff_base_w = bw_2d.detach()
                else:
                    m_2d = self._as_oi(self.mask)
                    self._eff_pw = pw_2d * m_2d
                    self._eff_base_w = bw_2d * m_2d
                if self.postact_bias_trainable or self._pb_is_nonzero:
                    pb_2d = self._as_oi(self.postact_bias)
                    self._bias_sum = (pb_2d * self._eff_pw).sum(dim=1)
                else:
                    self._bias_sum = None
        return ret

    @property
    def param_size(self):
        if hasattr(self, "_param_size"):
            return self._param_size
        count = 0
        for param in self.parameters():
            if param.requires_grad:
                count += param.numel()
        self._param_size = count
        return self._param_size

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x: torch.Tensor):
        self._x0 = None

    def forward(self, x: torch.Tensor):
        # Two-route dispatch: eval (cached) vs train (live). The eval route
        # is the inference hot path and uses pre-folded weights + cached
        # natural-view solver inputs. The train route reads parameters
        # directly so autograd records grads on the live storage.
        if self._eff_pw is not None:
            return self._forward_eval(x)
        return self._forward_train(x)

    # ------------------------------------------------------------------
    # Solver dispatch (shared by both routes)
    # ------------------------------------------------------------------
    def _resolve_solver_fn(self) -> None:
        """Pre-resolve the solver callable + dtype-cast need.

        Skips per-forward registry lookups and a no-op ``.to(p_dtype)``
        cast when ``c_dtype == p_dtype``.
        """
        fn: Optional[Callable] = None
        if self.solver == "qml":
            fn = None  # qml has its own loop in _run_solver
        elif isinstance(self.solver, str) and self.solver in get_registry():
            fn = get_solver(self.solver)
        elif callable(self.solver):
            fn = self.solver
        self._solver_fn: Optional[Callable] = fn
        # Cast is a no-op when dtypes match — skip the `.to()` Python call.
        # (Also refreshed by ``to()`` when a dtype move changes p_dtype.)
        self._needs_dtype_cast = self.c_dtype != self.p_dtype

    def _run_solver(
        self,
        x: torch.Tensor,
        theta_v: torch.Tensor,
        pw_v: torch.Tensor,
        pb_v: torch.Tensor,
    ) -> torch.Tensor:
        if self.solver == "qml":
            batch = x.shape[0]
            postacts = torch.zeros(
                batch, self.out_dim, self.in_dim, dtype=self.p_dtype
            ).to(self.device)
            for j in range(self.out_dim):
                for i in range(self.in_dim):
                    postacts[:, j, i] = qml_solver(
                        x=x[:, i],
                        theta=theta_v[i, j],
                        reps=self.reps,
                        device=self.device,
                        qml_device=self.qml_device,
                    ).to(self.p_dtype)
            return postacts
        fn = self._solver_fn
        if fn is None:
            raise NotImplementedError()
        if isinstance(self.solver, str):
            postacts = fn(
                x,
                theta_v,
                pw_v,
                pb_v,
                self.reps,
                device=self.device,
                ansatz=self.ansatz,
                group=self.group,
                preacts_trainable=self.preact_trainable,
                fast_measure=self.fast_measure,
                out_dim=self.out_dim,
                dtype=self.c_dtype,
                checkpoint_reps=self.checkpoint_reps,
                **self.solver_kwargs,
            )
            return postacts.to(self.p_dtype) if self._needs_dtype_cast else postacts
        return fn(
            x,
            theta_v,
            pw_v,
            pb_v,
            self.reps,
            device=self.device,
            ansatz=self.ansatz,
            **self.solver_kwargs,
        )

    def _forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path — uses pre-folded weights + aliased solver inputs."""
        assert x.shape[1] == self.in_dim, "Invalid input dimension"
        if self.is_batchnorm:
            x = self.bn(x)
        base_input = self.base_activation(x)
        postacts = self._run_solver(
            x,
            self._theta_natural_v,
            self._preacts_w_natural_v,
            self._preacts_b_natural_v,
        )
        eff_pw = self._eff_pw
        eff_base_w = self._eff_base_w
        assert eff_pw is not None and eff_base_w is not None
        if self._use_fused_epilogue and postacts.is_cuda:
            return qkan_epilogue_forward(
                postacts, eff_pw, None, base_input, eff_base_w, self._bias_sum
            )
        main = (postacts * eff_pw).sum(dim=2)
        base = F.linear(base_input, eff_base_w, bias=self._bias_sum)
        return main + base

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Training path — reads live params via aliased views, no eval cache."""
        assert x.shape[1] == self.in_dim, "Invalid input dimension"
        if self.is_batchnorm:
            x = self.bn(x)
        base_input = self.base_activation(x)
        postacts = self._run_solver(
            x,
            self._theta_natural_v,
            self._preacts_w_natural_v,
            self._preacts_b_natural_v,
        )
        pw_oi = self._pw_oi
        bw_oi = self._bw_oi
        if self._mask_is_identity:
            eff_pw = pw_oi
            base_w = bw_oi
        else:
            mask_oi = self._mask_oi
            eff_pw = pw_oi * mask_oi
            base_w = bw_oi * mask_oi
        # Fused-epilogue train path: forward is a single Triton kernel; the
        # backward fuses grad_postacts / grad_eff_pw / grad_pb into one Triton
        # kernel and keeps the two matmul-shaped gradients on cuBLAS (~4
        # launches vs ~7 eager dispatches).

        use_pb = self.postact_bias_trainable or self._pb_is_nonzero
        if self._use_fused_epilogue and postacts.is_cuda:
            pb = self._pb_oi if use_pb else None
            return QKANEpilogue.apply(postacts, eff_pw, pb, base_input, base_w)
        if use_pb:
            main = torch.sum((postacts + self._pb_oi) * eff_pw, dim=2)
        else:
            main = (postacts * eff_pw).sum(dim=2)
        base = F.linear(base_input, base_w)
        return main + base

    def set_fused_epilogue(self, enabled: bool = True) -> None:
        """Toggle the fused Triton epilogue path (forward + backward).

        Off by default. When on, ``_forward_train`` / ``_forward_eval`` collapse
        the (postacts + pb)·eff_pw sum and the base linear term into a single
        kernel launch; the backward fuses grad_postacts / grad_eff_pw / grad_pb
        into one Triton kernel and keeps the matmul-shaped gradients on cuBLAS.
        CPU tensors and non-CUDA paths transparently fall back to the eager
        chain.
        """
        self._use_fused_epilogue = bool(enabled)

    def reset_parameters(self):
        """
        Reset all learnable parameters to default values in-place.

        Note: The thetas are set to zero to do layer extension.
        If you wish to re-init the parameters, please use `init_parameters` instead.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.theta.data.zero_()
        if self.ansatz == "real":
            self.c_dtype = torch.bfloat16

        if self.ba_trainable:
            self.base_weight.data.fill_(0.5)
        else:
            self.base_weight.data.zero_()

        self.preacts_weight.data.fill_(1)
        self.preacts_bias.data.zero_()

        self.postact_weights.data.fill_(1)
        self.postact_bias.data.zero_()
        self._pb_is_nonzero = False

        self.mask.data.fill_(1)
        self._mask_is_identity = True

        if self.is_batchnorm:
            self.bn.reset_parameters()

        self._x0 = None

    @torch.no_grad()
    def forward_no_sum(self, x: torch.Tensor):
        assert x.shape[1] == self.in_dim, "Invalid input dimension"

        bw_2d = self._as_oi(self.base_weight)
        pw_2d = self._as_oi(self.postact_weights)
        pb_2d = self._as_oi(self.postact_bias)
        m_2d = self._as_oi(self.mask)

        base_output = torch.einsum("oi,bi->boi", bw_2d, self.base_activation(x))

        theta_v = self._theta_natural()
        pw_solver_v = self._preacts_w_natural()
        pb_solver_v = self._preacts_b_natural()
        if self.solver == "qml":
            postacts = torch.cat(
                [
                    torch.stack(
                        [
                            qml_solver(
                                x=x[:, i],
                                theta=theta_v[i, j],
                                reps=self.reps,
                                device=self.device,
                                qml_device=self.qml_device,
                            )
                            for i in range(self.in_dim)
                        ],
                    )
                    .unsqueeze(-1)
                    .permute(1, 2, 0)
                    for j in range(self.out_dim)
                ],
                dim=1,
            ).to(self.p_dtype)
        elif isinstance(self.solver, str) and self.solver in get_registry():
            postacts = get_solver(self.solver)(
                x,
                theta_v,
                pw_solver_v,
                pb_solver_v,
                self.reps,
                device=self.device,
                ansatz=self.ansatz,
                group=self.group,
                preacts_trainable=self.preact_trainable,
                fast_measure=self.fast_measure,
                out_dim=self.out_dim,
                dtype=self.c_dtype,
                **self.solver_kwargs,
            ).to(self.p_dtype)
        else:
            raise NotImplementedError()
        x_new = ((postacts + pb_2d) * pw_2d[None, :, :] + base_output) * m_2d[
            None, :, :
        ]
        return x_new

    def get_subset(self, in_id, out_id):
        """
        Get a smaller QKANLayer from a larger QKANLayer (used for pruning).

        Args
        ----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns
        -------
            QKANLayer
                New QKANLayer with selected neurons
        """
        spb = QKANLayer(
            in_dim=len(in_id),
            out_dim=len(out_id),
            reps=self.reps,
            device=self.device,
            solver=self.solver,
            qml_device=self.qml_device,
            ansatz=self.ansatz,
            preact_trainable=self.preact_trainable,
            postact_weight_trainable=self.postact_weight_trainable,
            postact_bias_trainable=self.postact_bias_trainable,
            base_activation=self.base_activation,
            ba_trainable=self.ba_trainable,
            seed=self.seed,
            p_dim=self.p_dim,
        )
        # Slice through natural-rank views so the indexing semantics match
        # today's behavior regardless of storage rank. For group != -1 the
        # theta/preacts natural rank is < 4, and the (O,I)-style slice has
        # always been ill-defined there — preserve that pre-existing behavior
        # by copying as-is when the natural rank is too low to index by edge.
        src_t = self._theta_natural()
        dst_t = spb._theta_natural()
        if src_t.dim() == 4:
            dst_t.data.copy_(src_t.data[out_id][:, in_id])
        else:
            dst_t.data.copy_(src_t.data)
        src_pw = self._preacts_w_natural()
        dst_pw = spb._preacts_w_natural()
        src_pb = self._preacts_b_natural()
        dst_pb = spb._preacts_b_natural()
        if src_pw.dim() == 3:
            dst_pw.data.copy_(src_pw.data[out_id][:, in_id])
            dst_pb.data.copy_(src_pb.data[out_id][:, in_id])
        else:
            dst_pw.data.copy_(src_pw.data)
            dst_pb.data.copy_(src_pb.data)
        # (O, I) params: slice in 2D, write back through the 2D view.
        spb._as_oi(spb.base_weight).data.copy_(
            self._as_oi(self.base_weight)[out_id][:, in_id]
        )
        spb._as_oi(spb.postact_weights).data.copy_(
            self._as_oi(self.postact_weights)[out_id][:, in_id]
        )
        spb._as_oi(spb.postact_bias).data.copy_(
            self._as_oi(self.postact_bias)[out_id][:, in_id]
        )
        spb._as_oi(spb.mask).data.copy_(self._as_oi(self.mask)[out_id][:, in_id])
        return spb


class QKANModuleList(nn.ModuleList):
    def __init__(self):
        super(QKANModuleList, self).__init__()

    # make type hint for getitem method
    def __getitem__(self, idx) -> Union[QKANLayer, nn.Linear, "QKANModuleList"]:  # type: ignore
        return super(QKANModuleList, self).__getitem__(idx)


class QKAN(nn.Module):
    """
    Quantum-inspired Kolmogorov Arnold Network (QKAN) Class

    A quantum-inspired neural network that uses DatA Re-Uploading ActivatioN (DARUAN)
    as its learnable variation activation function.

    References:
        Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks: https://arxiv.org/abs/2509.14026

    Attributes
    ----------
        width : list[int]
            List of width of each layer
        reps : int
            Repetitions of quantum layers
        group : int
            Group of neurons
        device : Literal["cpu", "cuda"]
            Device to use
        solver : Union[str, Callable]
            Solver to use, currently supports "qml", "exact", "flash", "cutn" or custom callable
        qml_device : str
            PennyLane device to use
        layers : QKANModuleList
            List of layers
        is_map : bool
            Whether to use map layer
        is_batchnorm : bool
            Whether to use batch normalization
        reps : int
            Repetitions of quantum layers
        norm_out : int
            Normalize output
        postact_weight_trainable : bool
            Whether postact weights are trainable
        postact_bias_trainable : bool
            Whether postact bias are trainable
        preact_trainable : bool
            Whether preact weights are trainable
        base_activation : torch.nn.Module or lambda function
            Base activation function
        ba_trainable : bool
            Whether base activation weights are trainable
        fast_measure : bool
            Enable to use fast measurement in exact solver. Which would be quantum-inspired method.
            When False, the exact solver simulates the exact measurement process of quantum circuit.
        save_act : bool
            Whether to save activations
        seed : int
            Random seed
    """

    def __init__(
        self,
        width: list[int],
        reps: int = 3,
        group: int = -1,
        is_map: bool = False,
        is_batchnorm: bool = False,
        hidden: int = 0,
        device="cpu",
        solver: Union[
            Literal[
                "qml",
                "exact",
                "flash",
                "cutn",
                "tn",
                "cute",
                "cutile",
                "qiskit",
                "cudaq",
            ],
            Callable,
        ] = "exact",
        qml_device: str = "default.qubit",
        ansatz: Union[str, Callable] = "pz_encoding",
        theta_size: Optional[list[int]] = None,
        norm_out: int = 0,
        preact_trainable: bool = False,
        preact_init: bool = False,
        postact_weight_trainable: bool = False,
        postact_bias_trainable: bool = False,
        base_activation: Union[str, nn.Module] = "silu",
        ba_trainable: bool = False,
        fast_measure: bool = True,
        save_act: bool = False,
        c_dtype: torch.dtype = torch.complex64,
        p_dtype: torch.dtype = torch.float32,
        seed=None,
        solver_kwargs: Optional[dict] = None,
        p_dim: Literal[2, 4] = 4,
        checkpoint_reps: bool = False,
        theta_init: Literal["small_gaussian", "xavier"] = "xavier",
        **kwargs,
    ):
        """
        Initialize QKAN model

        Args
        ----
            width : list[int]
                List of width of each layer
            reps : int
                Repetitions of quantum layers, default: 3
            group : int
                Group of neurons, default: -1
            is_map : bool
                Whether to use map layer, default: False
            is_batchnorm: bool
                Whether to add a batchnorm layer before QKANLayer, default: False
            hidden : int
                Number of hidden units in map layer, default: 0
            device :
                Device to use, default: "cpu"
            solver : Union[str, Callable]
                Solver to use, currently supports "qml", "exact", "flash", "cutn" or custom callable, default: "exact"
            ansatz : Union[str, Callable]
                Ansatz to use, "pz_encoding" ("pz"), "px_encoding" ("px"), "rpz_encoding" ("rpz", reduced pz encoding) or custom
            qml_device : str
                PennyLane device to use, default: "default.qubit"
            ansatz : str | Callable
                Ansatz to use, default: "pz_encoding"
            norm_out : int
                Normalize output, default: 0
            postact_weight_trainable : bool
                Whether postact weights are trainable, default: False
            postact_bias_trainable : bool
                Whether postact bias are trainable, default: False
            base_activation : torch.nn.Module | lambda function
                Base activation function, default: torch.nn.SiLU()
            ba_trainable : bool
                Whether base activation weights are trainable, default: False
            save_act : bool
                Whether to save activations, default: False
            fast_measure : bool
                Enable to use fast measurement in exact solver. Which would be quantum-inspired method.
                When False, the exact solver simulates the exact measurement process of quantum circuit.
            p_dtype : torch.dtype
                Parameter dtype (``torch.float32`` or ``torch.bfloat16``).
                Use ``torch.bfloat16`` with bf16/fp8 ``c_dtype`` for full mixed-precision pipeline.
            c_dtype : torch.dtype
                Compute dtype for quantum simulation. Supported values:

                - ``torch.complex64`` / ``torch.float32``: full-precision f32 (default)
                - ``torch.bfloat16``: mixed-precision bf16 I/O, f32 compute, bf16 state checkpoints
                - ``torch.float8_e4m3fn``: bf16 I/O, f32 compute, fp8 prescaled state checkpoints
            seed : Any
                Random seed, default: None
        """
        super(QKAN, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.depth = len(width) - (2 if is_map else 1)
        self.width = width
        self.reps = reps
        self.group = group
        self.device = device
        self.solver: Union[
            Literal[
                "qml",
                "exact",
                "flash",
                "cutn",
                "tn",
                "cute",
                "cutile",
                "qiskit",
                "cudaq",
            ],
            Callable,
        ] = solver
        self.ansatz = ansatz
        self.qml_device = qml_device
        self.norm_out = norm_out
        self.postact_weight_trainable = postact_weight_trainable
        self.postact_bias_trainable = postact_bias_trainable
        self.preact_trainable = preact_trainable
        self.preact_init = preact_init
        self.theta_size = theta_size
        if isinstance(base_activation, str):
            base_activation = make_base_activation(base_activation, solver)
        self.base_activation = base_activation
        self.ba_trainable = ba_trainable
        self.fast_measure = fast_measure
        self.save_act = save_act
        self.c_dtype = c_dtype
        self.p_dtype = p_dtype
        self.seed = seed
        self.solver_kwargs = solver_kwargs or {}
        if p_dim not in (2, 4):
            raise ValueError(f"p_dim must be 2 or 4, got {p_dim}")
        self.p_dim: Literal[2, 4] = p_dim
        self.checkpoint_reps = checkpoint_reps
        self.theta_init = theta_init

        self.layers = QKANModuleList()
        for l in range(self.depth):
            self.layers.append(
                QKANLayer(
                    in_dim=width[l],
                    out_dim=width[l + 1],
                    reps=reps,
                    group=group,
                    device=self.device,
                    solver=self.solver,
                    qml_device=self.qml_device,
                    ansatz=self.ansatz,
                    theta_size=theta_size,
                    preact_trainable=preact_trainable,
                    preact_init=preact_init,
                    postact_weight_trainable=postact_weight_trainable,
                    postact_bias_trainable=postact_bias_trainable,
                    base_activation=base_activation,
                    ba_trainable=ba_trainable,
                    is_batchnorm=is_batchnorm,
                    fast_measure=fast_measure,
                    c_dtype=c_dtype,
                    p_dtype=p_dtype,
                    seed=seed,
                    solver_kwargs=self.solver_kwargs,
                    p_dim=p_dim,
                    checkpoint_reps=checkpoint_reps,
                    theta_init=theta_init,
                )
            )

        self.is_batchnorm = is_batchnorm
        self.is_map = is_map
        self.hidden = hidden
        if is_map:
            self.layers.append(
                nn.Linear(width[-2], hidden, device=self.device, dtype=self.p_dtype)
            )
            self.layers.append(nn.SiLU())
            self.layers.append(
                nn.Linear(hidden, width[-1], device=self.device, dtype=self.p_dtype)
            )
        self.input_id: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        """
        Move the model to the specified device.

        Args
        ----
            device : str | torch.device
                Device to move the model to, default: "cpu"
        """
        device = None
        for arg in args:
            if isinstance(arg, str) or isinstance(arg, torch.device):
                device = arg
                break
            elif isinstance(arg, torch.dtype):
                self.p_dtype = arg
        if isinstance(kwargs.get("dtype"), torch.dtype):
            self.p_dtype = kwargs["dtype"]
        if "device" in kwargs:
            device = kwargs["device"]
        if device:
            self.device = device
        for layer in self.layers:
            layer.to(*args, **kwargs)
        return super(QKAN, self).to(*args, **kwargs)

    @property
    def param_size(self):
        if hasattr(self, "_param_size"):
            return self._param_size
        count = 0
        for layer in self.layers:
            if not isinstance(layer, QKANLayer):
                count += sum(p.numel() for p in layer.parameters())
                continue
            count += layer.param_size
        self._param_size = count
        return self._param_size

    def forward(self, x: torch.Tensor):
        shape_size = len(x.shape)

        if shape_size == 3:
            B, C, T = x.shape
        elif shape_size == 2:
            B, T = x.shape
        else:
            raise NotImplementedError()

        x = x.view(-1, T)
        B_flat = x.shape[0]  # Flattened batch size (e.g., (B,C,T) -> (B*C,T))
        if self.input_id is not None:
            x = x[:, self.input_id.long()]

        if self.save_act:
            self.cache_data = x
            self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
            self.subnode_actscale = []
            self.dr_preacts = []
            self.dr_postacts = []
            self.acts_scale = []
            self.acts_scale_dr = []
            self.edge_actscale = []
            self.acts.append(x.detach())

        for layer in self.layers:
            if self.save_act and isinstance(layer, QKANLayer):
                self.subnode_actscale.append(torch.std(x, dim=0).detach())
                # Use the flattened batch size to match x after view(-1, T)
                preacts = x[:, None, :].expand(B_flat, layer.out_dim, layer.in_dim)
                postacts = layer.forward_no_sum(x)  # shape: (batch, out_dim, in_dim)

            x = layer(x)

            if self.save_act and isinstance(layer, QKANLayer):
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_dr = torch.std(
                    postacts, dim=0
                )  # for training, only penalize the dr part
                output_range = torch.std(
                    postacts, dim=0
                )  # leave for symbolic (Not implemented yet)
                # save edge_scale
                self.edge_actscale.append(output_range)
                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_dr.append(output_range_dr / input_range)
                self.dr_preacts.append(preacts.detach())
                self.dr_postacts.append(postacts.detach())
                self.acts.append(x.detach())

        if self.norm_out:
            x = F.normalize(x, p=self.norm_out, dim=1)

        U = x.shape[1]

        if shape_size == 3:
            x = x.view(B, C, U)
        elif shape_size == 2:
            assert x.shape == (B, U)
        return x

    def initialize_from_another_model(self, another_model: "QKAN"):
        """
        Initialize from another model.
        Used for layer extension to refine the model.

        Args
        ----
            another_model : QKAN
                Another model to initialize from
        """
        assert all(x == y for x, y in zip(self.width, another_model.width)), (
            "Cannot initialize from another model with different width"
        )
        count = -2
        for l, layer in enumerate(self.layers):
            if isinstance(layer, QKANLayer):
                src_layer = another_model.layers[l]
                assert isinstance(src_layer, QKANLayer)
                layer.reset_parameters()
                # Use natural-rank views so the slice indexing is identical
                # regardless of p_dim on either side.
                src_t = src_layer._theta_natural().data
                dst_t = layer._theta_natural().data
                src_pw = src_layer._preacts_w_natural().data
                dst_pw = layer._preacts_w_natural().data
                src_pb = src_layer._preacts_b_natural().data
                dst_pb = layer._preacts_b_natural().data
                for i in range(src_layer.reps):
                    dst_t[..., i, :].copy_(src_t[..., i, :])
                    dst_pw[..., i].copy_(src_pw[..., i])
                    dst_pb[..., i].copy_(src_pb[..., i])
                dst_t[..., src_layer.reps, :].copy_(src_t[..., src_layer.reps, :])
                # Added reps stay at zero from reset_parameters; for
                # pz/rpz/px this is identity, so refine() is function-
                # preserving at step 0. For "real" ansatz, theta=0 gives
                # X·Z (not identity), so the refine boundary is
                # discontinuous for that ansatz.
                # (O, I) params: copy via 2D view to bridge p_dim differences.
                layer._as_oi(layer.postact_weights).data.copy_(
                    src_layer._as_oi(src_layer.postact_weights).data
                )
                layer._as_oi(layer.postact_bias).data.copy_(
                    src_layer._as_oi(src_layer.postact_bias).data
                )
                layer._as_oi(layer.base_weight).data.copy_(
                    src_layer._as_oi(src_layer.base_weight).data
                )
            if isinstance(layer, nn.Linear):
                layer.weight.data.copy_(another_model.layers[count - 1].weight.data)  # type: ignore
                layer.bias.data.copy_(another_model.layers[count - 1].bias.data)  # type: ignore
                count += 2
        return self

    def initialize_parameters(self):
        """Reinitialize parameters of all QKANLayer layers in-place."""
        for layer in self.layers:
            if isinstance(layer, QKANLayer):
                layer.reset_parameters()
        return self

    def xavier_init(self):
        """Apply Xavier normal initialization to all QKANLayer layers."""
        for layer in self.layers:
            if isinstance(layer, QKANLayer):
                layer.xavier_init()
        return self

    def refine(self, new_reps: int) -> "QKAN":
        """
        Refine the model by layer extension, increasing the number of repetitions of quantum layers.

        Args
        ----
            new_reps : int
                New number of repetitions of quantum layers

        Returns
        -------
            QKAN
                New QKAN model with increased repetitions
        """
        assert new_reps > self.reps, (
            "New repetitions must be greater than current repetitions"
        )
        new_model = QKAN(
            width=self.width,
            reps=new_reps,
            group=self.group,
            device=self.device,
            solver=self.solver,
            qml_device=self.qml_device,
            ansatz=self.ansatz,
            theta_size=self.theta_size,
            norm_out=self.norm_out,
            preact_trainable=self.preact_trainable,
            preact_init=self.preact_init,
            postact_weight_trainable=self.postact_weight_trainable,
            postact_bias_trainable=self.postact_bias_trainable,
            base_activation=self.base_activation,
            ba_trainable=self.ba_trainable,
            is_batchnorm=self.is_batchnorm,
            is_map=self.is_map,
            hidden=self.hidden,
            fast_measure=self.fast_measure,
            save_act=self.save_act,
            c_dtype=self.c_dtype,
            p_dtype=self.p_dtype,
            seed=self.seed,
            p_dim=self.p_dim,
            theta_init=self.theta_init,
        )
        new_model.initialize_from_another_model(self)
        return new_model

    def layer_extension(self, new_reps: int) -> "QKAN":
        """
        Refine the model by layer extension, increasing the number of repetitions of quantum layers.

        Args
        ----
            new_reps : int
                New number of repetitions of quantum layers

        Returns
        -------
            QKAN
                New QKAN model with increased repetitions
        """
        return self.refine(new_reps)

    def _reg(
        self,
        reg_metric: str,
        lamb_l1: float,
        lamb_entropy: float,
        lamb_coef: float,
        lamb_coefdiff: float,
    ):
        """
        Get regularization.

        Adapted from "pykan".

        Args
        ----
            reg_metric : the regularization metric
                'edge_forward_dr_n', 'edge_forward_dr_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient penalty strength
            lamb_coefdiff : float
                coefficient smoothness strength

        Returns
        -------
            torch.Tensor
        """
        if reg_metric == "edge_forward_dr_n":
            acts_scale = self.acts_scale_dr
        elif reg_metric == "edge_forward_sum":
            acts_scale = self.acts_scale
        elif reg_metric == "edge_forward_dr_u":
            acts_scale = self.edge_actscale
        elif reg_metric == "edge_backward":
            acts_scale = self.edge_scores
        elif reg_metric == "node_backward":
            acts_scale = self.node_attribute_scores
        else:
            raise RuntimeError(f"reg_metric = {reg_metric} not recognized!")

        reg_: torch.Tensor = torch.tensor(0.0, device=self.device)
        for i in range(len(acts_scale)):
            vec = acts_scale[i]

            l1 = torch.sum(vec)
            # Optimize: reuse sum computations for normalization
            sum_row = torch.sum(vec, dim=1, keepdim=True) + 1
            sum_col = torch.sum(vec, dim=0, keepdim=True) + 1
            p_row = vec / sum_row
            p_col = vec / sum_col
            entropy_row = -torch.mean(
                torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1)
            )
            entropy_col = -torch.mean(
                torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0)
            )
            reg_ += lamb_l1 * l1 + lamb_entropy * (
                entropy_row + entropy_col
            )  # both l1 and entropy

        # regularize coefficient to encourage activation to be zero
        for layer in self.layers:
            if not isinstance(layer, QKANLayer):
                continue
            pw_2d = layer._as_oi(layer.postact_weights)
            coeff_l1 = torch.sum(torch.mean(torch.abs(pw_2d), dim=1))
            coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(pw_2d)), dim=1))
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    def get_reg(
        self,
        reg_metric: str,
        lamb_l1: float,
        lamb_entropy: float,
        lamb_coef: float,
        lamb_coefdiff: float,
    ):
        """
        Get regularization from the model.

        Adapted from "pykan".

        Args
        ----
            reg_metric : str
                Regularization metric.
                'edge_forward_dr_n', 'edge_forward_dr_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            lamb_l1 : float
                L1 Regularization parameter
            lamb_entropy : float
                Entropy Regularization parameter
            lamb_coef : float
                Coefficient Regularization parameter
            lamb_coefdiff : float
                Coefficient Smoothness Regularization parameter

        Returns
        -------
            torch.Tensor
        """
        return self._reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)

    def attribute(self, l=None, i=None, out_score=None, plot=True):
        """
        Get attribution scores

        Adapted from "pykan".

        Args
        ----
            l : None | int
                layer index
            i : None | int
                neuron index
            out_score : None | torch.Tensor
                specify output scores
            plot : bool
                when plot = True, display the bar show

        Returns
        -------
            torch.Tensor
                attribution scores
        """
        if not self.save_act:
            warnings.warn(
                "Activations are not saved, cannot get attribution scores",
                RuntimeWarning,
            )
            return None

        if l is not None:
            self.attribute()
            out_score = self.node_scores[l]

        node_scores = []
        subnode_scores = []
        edge_scores = []

        l_query = l
        if l is None:
            l_end = self.depth
        else:
            l_end = l

        # back propagate from the queried layer
        out_dim = self.width[l_end]
        if out_score is None:
            node_score = torch.eye(out_dim).requires_grad_(True)
        else:
            node_score = torch.diag(out_score).requires_grad_(True)
        node_scores.append(node_score)

        # Pre-compute inverse subnode_actscale for efficiency
        inv_subnode_actscale = [1 / (scale + 1e-4) for scale in self.subnode_actscale]

        for l in range(l_end, 0, -1):
            subnode_score = node_score[:, : self.width[l]]

            subnode_scores.append(subnode_score)
            # subnode to edge
            edge_score = torch.einsum(
                "oi,ko,i->koi",
                self.edge_actscale[l - 1],
                subnode_score.to(self.device),
                inv_subnode_actscale[l - 1],
            )
            edge_scores.append(edge_score)

            # edge to node
            node_score = torch.sum(edge_score, dim=1)
            node_scores.append(node_score)

        self.node_scores_all = list(reversed(node_scores))
        self.edge_scores_all = list(reversed(edge_scores))
        self.subnode_scores_all = list(reversed(subnode_scores))

        self.node_scores = [torch.mean(l, dim=0) for l in self.node_scores_all]
        self.edge_scores = [torch.mean(l, dim=0) for l in self.edge_scores_all]
        self.subnode_scores = [torch.mean(l, dim=0) for l in self.subnode_scores_all]

        # return: (out_dim, in_dim)
        if l_query is not None:
            if i is None:
                return self.node_scores_all[0]
            else:
                # plot
                if plot:
                    in_dim = self.width[0]
                    plt.figure(figsize=(1 * in_dim, 3))
                    plt.bar(
                        range(in_dim), self.node_scores_all[0][i].cpu().detach().numpy()
                    )
                    plt.xticks(range(in_dim))

                return self.node_scores_all[0][i]

    def node_attribute(self):
        """
        Get node attribution scores.

        Adapted from "pykan".
        """
        self.node_attribute_scores = []
        for l in range(1, self.depth + 1):
            node_attr = self.attribute(l)
            self.node_attribute_scores.append(node_attr)

    def train_(
        self,
        dataset,
        optimizer=None,
        closure=None,
        scheduler=None,
        steps: int = 10,
        log: int = 1,
        loss_fn=None,
        batch=-1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        reg_metric="edge_forward_dr_n",
        verbose=True,
        clip_theta_grad_norm: Optional[float] = None,
        lamb_theta_l1: float = 0.0,
        prune_every: int = 0,
        prune_threshold: float = 1e-3,
    ):
        """
        Train the model

        Args
        ----
            dataset : dict
                Dictionary containing train_input, train_label, test_input, test_label
            optimizer : torch.optim.Optimizer | None
                Optimizer to use, default: None
            closure : Callable | None
                Closure function for optimizer, default: None
            scheduler : torch.optim.lr_scheduler | None
                Scheduler to use, default: None
            steps : int
                Number of steps, default: 10
            log : int
                Logging frequency, default: 1
            loss_fn : torch.nn.Module | Callable |None
                Loss function to use, default: None
            batch : int
                batch size, if -1 then full., default: -1
            lamb : float
                L1 Regularization parameter. If 0, no regularization.
            lamb_l1 : float
                L1 Regularization parameter
            lamb_entropy : float
                Entropy Regularization parameter
            lamb_coef : float
                Coefficient Regularization parameter
            lamb_coefdiff : float
                Coefficient Smoothness Regularization parameter
            reg_metric : str
                Regularization metric.
                'edge_forward_dr_n', 'edge_forward_dr_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            verbose : bool
                Verbose mode, default: True
            clip_theta_grad_norm : float | None
                If not None, apply ``torch.nn.utils.clip_grad_norm_`` to the
                concatenation of all ``QKANLayer.theta`` parameters (only)
                with this max_norm. Useful at high reps / bf16 where theta
                gradients occasionally explode near saddle points. Default
                None (no clipping).
            lamb_theta_l1 : float
                Weight for an additional ``sum(|theta|)`` penalty added to
                the training objective. Pushes low-contribution rotations
                toward zero so periodic mask pruning can zero them out.
                Default 0.0 (no penalty).
            prune_every : int
                If > 0, every ``prune_every`` steps, set ``mask[o,i]=0``
                wherever ``theta[o,i].abs().sum() < prune_threshold`` and
                flip ``_mask_is_identity=False`` on affected layers. Layers
                with grouped (shared) theta are skipped — no per-edge
                magnitude exists. Default 0 (no auto-pruning).
            prune_threshold : float
                Threshold for the per-edge theta-magnitude check used by
                ``prune_every``. Default 1e-3.

        Returns
        -------
            dict
                Dictionary containing train_loss and test_loss
        """
        if verbose:
            print_version()

        if lamb > 0.0 and not self.save_act:
            lamb = 0.0
            warnings.warn(
                "Regularization is not supported without saving activations",
                RuntimeWarning,
            )

        pbar = tqdm(range(steps), ncols=100)

        if loss_fn is None:
            loss_fn = loss_fn_eval = torch.nn.MSELoss()
        else:
            loss_fn = loss_fn_eval = loss_fn

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        else:
            optimizer = optimizer

        results: dict = {}
        results["train_loss"] = []
        results["test_loss"] = []
        results["reg"] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size = dataset["train_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        qkan_layers = [layer for layer in self.layers if isinstance(layer, QKANLayer)]
        theta_params = [layer.theta for layer in qkan_layers]

        def _theta_l1_term() -> torch.Tensor:
            """Sum of |theta| across all QKANLayer's (zero if no layers)."""
            term = torch.tensor(0.0, device=self.device)
            for t in theta_params:
                term = term + torch.sum(torch.abs(t))
            return term

        def _clip_theta_grads() -> None:
            if clip_theta_grad_norm is None or not theta_params:
                return
            torch.nn.utils.clip_grad_norm_(theta_params, max_norm=clip_theta_grad_norm)

        def _closure():
            nonlocal train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset["train_input"][train_id].to(self.device))
            train_loss = loss_fn(pred, dataset["train_label"][train_id].to(self.device))
            if self.save_act:
                if reg_metric == "edge_backward":
                    self.attribute()
                if reg_metric == "node_backward":
                    self.node_attribute()
                reg_ = self.get_reg(
                    reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                )
            else:
                reg_ = torch.tensor(0.0, device=self.device)
            objective = train_loss + lamb * reg_
            if lamb_theta_l1 > 0.0:
                objective = objective + lamb_theta_l1 * _theta_l1_term()
            objective.backward()
            _clip_theta_grads()
            return objective

        if closure is None and isinstance(
            optimizer, (torch.optim.LBFGS, LBFGSFinisher)
        ):
            closure = _closure

        for _ in pbar:
            self.train()
            train_id = np.random.choice(
                dataset["train_input"].shape[0], batch_size, replace=False
            )
            test_id = np.random.choice(
                dataset["test_input"].shape[0], batch_size_test, replace=False
            )

            if isinstance(optimizer, (torch.optim.LBFGS, LBFGSFinisher)):
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                pred = self.forward(dataset["train_input"][train_id].to(self.device))
                train_loss = loss_fn(
                    pred, dataset["train_label"][train_id].to(self.device)
                )
                if self.save_act:
                    if reg_metric == "edge_backward":
                        self.attribute()
                    if reg_metric == "node_backward":
                        self.node_attribute()
                    reg_ = self.get_reg(
                        reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                    )
                else:
                    reg_ = torch.tensor(0.0, device=self.device)
                loss = train_loss + lamb * reg_
                if lamb_theta_l1 > 0.0:
                    loss = loss + lamb_theta_l1 * _theta_l1_term()
                loss.backward()
                _clip_theta_grads()
                optimizer.step(closure)

            if prune_every > 0 and ((_ + 1) % prune_every == 0):
                with torch.no_grad():
                    for layer in qkan_layers:
                        # Natural shape is (*group, reps+1, k); reduce
                        # trailing (reps+1, k) axes to per-edge magnitude.
                        edge_mag = torch.sum(
                            torch.abs(layer._theta_natural()), dim=(-2, -1)
                        )
                        kill = edge_mag < prune_threshold
                        if not bool(kill.any()):
                            continue
                        mask_2d = layer._as_oi(layer.mask).data
                        if kill.numel() != mask_2d.numel():
                            # Grouped theta (group != -1) is shared across
                            # edges — no per-edge magnitude exists; skip.
                            continue
                        if kill.shape != mask_2d.shape:
                            kill = kill.reshape(mask_2d.shape)
                        mask_2d[kill] = 0.0
                        layer._mask_is_identity = False

            self.eval()
            test_loss = loss_fn_eval(
                self.forward(dataset["test_input"][test_id].to(self.device)),
                dataset["test_label"][test_id].to(self.device),
            )
            if scheduler is not None:
                scheduler.step(test_loss)

            if _ % log == 0:
                pbar.set_postfix(
                    {
                        "train loss": train_loss.cpu().detach().numpy(),
                        "test loss": test_loss.cpu().detach().numpy(),
                    }
                )

            results["train_loss"].append(train_loss.cpu().detach().numpy())
            results["test_loss"].append(test_loss.cpu().detach().numpy())
            results["reg"].append(reg_.cpu().detach().numpy())

        return results

    def plot(
        self,
        x0=None,
        sampling=1000,
        from_acts=False,
        scale=0.5,
        beta=3,
        metric="forward_n",
        mask=False,
        in_vars=None,
        out_vars=None,
        title=None,
    ):
        """
        Plot the model.

        Adapted from "pykan".

        Arguments
        ---------
            x0 : torch.Tensor | None
                Input tensor to plot, if None, plot from saved activations
            sampling : int
                Sampling frequency
            from_acts : bool
                Plot from saved activations
            scale : float
                Scale of the plot
            beta : float
                Beta value
            metric : str
                Metric to use. 'forward_n', 'forward_u', 'backward'
            in_vars : list[int] | None
                Input variables to plot
            out_vars : list[int] | None
                Output variables to plot
            title : str | None
                Title of the plot
        """
        if self.is_map:
            warnings.warn("Not supported for map layer", RuntimeWarning)
            return None
        if self.is_batchnorm:
            warnings.warn("Not supported for batchnorm layer", RuntimeWarning)
            return None
        if x0 is None and not from_acts:
            warnings.warn(
                "x0 is not provided, try plot from saved activations.", RuntimeWarning
            )
            from_acts = True
        if from_acts and not self.acts:
            warnings.warn(
                "Activations are not saved, cannot plot from activations",
                RuntimeWarning,
            )
            return None
        if mask and not hasattr(self, "mask"):
            warnings.warn(
                "Make sure to run model.prune_node() first to compute mask. Continue without mask.",
                RuntimeWarning,
            )
            mask = False

        if not os.path.exists("./figures"):
            os.makedirs("./figures")

        if metric == "backward":
            self.attribute()

        save_act = self.save_act
        self.save_act = False
        self.eval()
        for idx, qkan_layer in enumerate(self.layers):
            assert isinstance(qkan_layer, QKANLayer)

            if idx == 0:
                x = x0
            else:
                ymin = torch.min(ynew.cpu().detach(), dim=0).values  # noqa: F821
                ymax = torch.max(ynew.cpu().detach(), dim=0).values  # noqa: F821
                x = torch.stack(
                    [
                        torch.linspace(
                            ymin[i],
                            ymax[i],
                            steps=sampling,
                            device=self.device,
                        )
                        for i in range(qkan_layer.in_dim)
                    ]
                ).permute(1, 0)  # x.shape = (sampling, in_dim)
            if from_acts:
                x = self.acts[idx]

            y = qkan_layer.forward_no_sum(x).transpose(
                1, 2
            )  # y.shape = (sampling, in_dim, out_dim)
            for i in range(self.width[idx]):
                for j in range(self.width[idx + 1]):
                    fig, ax = plt.subplots(figsize=(2, 2))
                    plt.xticks([])
                    plt.yticks([])
                    plt.gca().patch.set_edgecolor("black")
                    plt.gca().patch.set_linewidth(1.5)
                    plt.scatter(
                        x[:, i].detach().cpu().numpy(),
                        y[:, i, j].detach().cpu().numpy(),
                        color="black",
                        s=40,
                    )
                    plt.gca().spines[:].set_color("black")
                    plt.savefig(
                        f"./figures/dr_{idx}_{i}_{j}.png", bbox_inches="tight", dpi=400
                    )
                    plt.close()
            with torch.no_grad():
                ynew = qkan_layer.forward(x)  # noqa: F841

        def score2alpha(score):
            return np.tanh(beta * score)

        alpha = []
        try:
            if save_act and metric is not None:
                if metric == "forward_n":
                    scores = self.acts_scale
                elif metric == "forward_u":
                    scores = self.edge_actscale
                elif metric == "backward":
                    scores = self.edge_scores
                else:
                    raise RuntimeError(f"metric = '{metric}' cannot be recognized")

                alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]
        except RuntimeError:
            warnings.warn(f"metric = '{metric}' cannot be recognized", RuntimeWarning)
        finally:
            if not alpha:
                alpha = [
                    torch.ones(layer.out_dim, layer.in_dim).detach().numpy()
                    for layer in self.layers
                ]

        # draw skeleton
        width = np.array(self.width)
        A = 1
        y0 = 0.4

        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width), 5)

        # max_neuron = np.max(width)
        max_num_weights = np.max(width[:-1] * width[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 3)

        fig, ax = plt.subplots(
            figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * y0)
        )
        # plot scatters and lines
        for l in range(neuron_depth):
            n = width[l]
            # spacing = A / n
            for i in range(n):
                plt.scatter(
                    1 / (2 * n) + i / n,
                    l * y0,
                    s=min_spacing**2 * 10000 * scale**2,
                    color="black",
                )

                if l < neuron_depth - 1:
                    # plot connections
                    n_next = width[l + 1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j
                        if mask:
                            plt.plot(
                                [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N],
                                [l * y0, (l + 1 / 2) * y0 - y1],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i]
                                * self.mask[l][i].item()
                                * self.mask[l + 1][j].item(),
                            )
                            plt.plot(
                                [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                [(l + 1 / 2) * y0 + y1, (l + 1) * y0],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i]
                                * self.mask[l][i].item()
                                * self.mask[l + 1][j].item(),
                            )
                        else:
                            plt.plot(
                                [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N],
                                [l * y0, (l + 1 / 2) * y0 - y1],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i],
                            )
                            plt.plot(
                                [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                [(l + 1 / 2) * y0 + y1, (l + 1) * y0],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i],
                            )

            plt.xlim(0, 1)
            plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

        plt.axis("off")

        # plot splines
        for l in range(neuron_depth - 1):
            n = width[l]
            for i in range(n):
                n_next = width[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f"./figures/dr_{l}_{i}_{j}.png")
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
                    up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    if mask:
                        newax.imshow(
                            im,
                            alpha=alpha[l][j][i]
                            * self.mask[l][i].item()
                            * self.mask[l + 1][j].item(),
                        )
                    else:
                        newax.imshow(im, alpha=alpha[l][j][i])
                    newax.axis("off")

        if in_vars is not None:
            n = self.width[0]
            for i in range(n):
                plt.gcf().get_axes()[0].text(
                    1 / (2 * (n)) + i / (n),
                    -0.1,
                    in_vars[i],
                    fontsize=40 * scale,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if out_vars is not None:
            n = self.width[-1]
            for i in range(n):
                plt.gcf().get_axes()[0].text(
                    1 / (2 * (n)) + i / (n),
                    y0 * (len(self.width) - 1) + 0.1,
                    out_vars[i],
                    fontsize=40 * scale,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if title is not None:
            plt.gcf().get_axes()[0].text(
                0.5,
                y0 * (len(self.width) - 1) + 0.2,
                title,
                fontsize=40 * scale,
                horizontalalignment="center",
                verticalalignment="center",
            )
        self.save_act = save_act

    def prune_node(
        self,
        threshold: float = 1e-2,
        mode: str = "auto",
        active_neurons_id: Optional[list] = None,
    ):
        """
        Pruning nodes.

        Adapted from "pykan".

        Args
        ----
            threshold : float
                if the attribution score of a neuron is below the threshold, it is considered dead and will be removed
            mode : str
                "auto" or "manual". with "auto", nodes are automatically pruned using threshold.
                With "manual", active_neurons_id should be passed in.

        Returns
        -------
            QKAN
                pruned network
        """
        if not hasattr(self, "acts"):
            warnings.warn("No activations, cannot prune nodes", RuntimeWarning)
            return None
        if mode == "manual" and active_neurons_id is None:
            warnings.warn(
                "active_neurons_id is not provided. Continue with auto mode.",
                RuntimeWarning,
            )
            mode = "auto"

        mask = [
            torch.ones(
                self.width[0],
            )
        ]
        active_neurons = [list(range(self.width[0]))]
        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold
                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold
                overall_important = in_important * out_important
            elif mode == "manual":
                assert active_neurons_id is not None
                overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                overall_important[active_neurons_id[i + 1]] = True
            mask.append(overall_important.float())
            active_neurons.append(
                torch.where(overall_important == True)[0].tolist()  # noqa: E712
            )
        active_neurons.append(list(range(self.width[-1])))
        mask.append(
            torch.ones(
                self.width[-1],
            )
        )

        self.mask = mask  # for plot

        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width[l + 1]):
                if i not in active_neurons[l + 1]:
                    self.remove_node(l + 1, i)

        model2 = QKAN(
            deepcopy(self.width),
            reps=self.reps,
            is_map=self.is_map,
            is_batchnorm=self.is_batchnorm,
            hidden=self.hidden,
            device=self.device,
            solver=self.solver,
            qml_device=self.qml_device,
            ansatz=self.ansatz,
            norm_out=self.norm_out,
            preact_trainable=self.preact_trainable,
            postact_weight_trainable=self.postact_weight_trainable,
            postact_bias_trainable=self.postact_bias_trainable,
            base_activation=self.base_activation,
            ba_trainable=self.ba_trainable,
            save_act=self.save_act,
            seed=self.seed,
            p_dim=self.p_dim,
        )
        model2.load_state_dict(self.state_dict())
        for i, layer in enumerate(model2.layers):
            if not isinstance(layer, QKANLayer):
                continue
            model2.layers[i] = layer.get_subset(
                active_neurons[i], active_neurons[i + 1]
            )
            model2.width[i] = len(active_neurons[i])
        model2.cache_data = self.cache_data

        return model2

    def prune_edge(self, threshold: float = 3e-2):
        """
        Pruning edges.

        Adapted from "pykan".

        Args:
            threshold: float
                if the attribution score of an edge is below the threshold, it is considered dead and will be set to zero.
        """
        if not hasattr(self, "acts"):
            warnings.warn("No activations, cannot prune edges", RuntimeWarning)
            return None

        for i in range(len(self.width) - 1):
            layer = self.layers[i]
            assert isinstance(layer, QKANLayer)
            mask_2d = layer._as_oi(layer.mask)
            new_2d = (self.edge_scores[i] > threshold).to(mask_2d.dtype) * mask_2d.data
            layer.mask.data.copy_(new_2d.reshape(layer.mask.shape))
            layer._mask_is_identity = False

    def prune(self, node_th: float = 1e-2, edge_th: float = 3e-2):
        """
        Prune (both nodes and edges).

        Adapted from "pykan".

        Args
        ----
            node_th : float
                if the attribution score of a node is below node_th, it is considered dead and will be set to zero.
            edge_th : float
                if the attribution score of an edge is below node_th, it is considered dead and will be set to zero.

        Returns
        -------
            QKAN
                pruned network
        """
        if not hasattr(self, "acts"):
            warnings.warn("No activations, cannot prune.", RuntimeWarning)
            return None

        self = self.prune_node(node_th)
        self.forward(self.cache_data)
        self.attribute()
        self.prune_edge(edge_th)
        return self

    def prune_input(
        self, threshold: float = 1e-2, active_inputs: Optional[list] = None
    ):
        """
        Prune inputs.

        Adapted from "pykan".

        Args
        ----
            threshold : float
                if the attribution score of the input feature is below threshold, it is considered irrelevant.
            active_inputs : list | None
                if a list is passed, the manual mode will disregard attribution score and prune as instructed.

        Returns
        -------
            QKAN
                pruned network
        """
        if active_inputs is None:
            self.attribute()
            input_score = self.node_scores[0]
            input_mask = input_score > threshold
            print0("keep:", input_mask.tolist())
            input_id = torch.where(input_mask == True)[0]  # noqa: E712

        else:
            input_id = torch.tensor(active_inputs, dtype=torch.long).to(self.device)

        model2 = QKAN(
            deepcopy(self.width),
            reps=self.reps,
            is_map=self.is_map,
            is_batchnorm=self.is_batchnorm,
            hidden=self.hidden,
            device=self.device,
            solver=self.solver,
            qml_device=self.qml_device,
            ansatz=self.ansatz,
            norm_out=self.norm_out,
            preact_trainable=self.preact_trainable,
            postact_weight_trainable=self.postact_weight_trainable,
            postact_bias_trainable=self.postact_bias_trainable,
            base_activation=self.base_activation,
            ba_trainable=self.ba_trainable,
            save_act=self.save_act,
            seed=self.seed,
            p_dim=self.p_dim,
        )
        model2.load_state_dict(self.state_dict())

        model2.layers[0] = model2.layers[0].get_subset(
            input_id,
            torch.arange(self.width[1]),  # type: ignore
        )

        model2.cache_data = self.cache_data

        model2.width[0] = len(input_id)
        model2.input_id = input_id

        return model2

    def remove_edge(self, layer_idx, in_idx, out_idx):
        """
        Remove activtion phi(layer_idx, in_idx, out_idx) (set its mask to zero)

        Args
        ----
            layer_idx : int
                Layer index
            in_idx : int
                Input node index
            out_idx : int
                Output node index
        """
        if not isinstance(self.layers[layer_idx], QKAN):
            return
        self.layers[layer_idx].mask[out_idx, in_idx] = 0.0

    def remove_node(self, layer_idx, in_idx, mode="all"):
        """
        remove neuron (layer_idx, in_idx) (set the masks of all incoming and outgoing activation functions to zero)

        Args
        ----
            layer_idx : int
                Layer index
            in_idx : int
                Input node index
            mode : str
                Mode to remove. "all" or "up" or "down", default: "all"
        """
        if mode == "down":
            if not isinstance(self.layers[layer_idx - 1], QKAN):
                return
            self.layers[layer_idx - 1].mask[in_idx, :] = 0.0
        elif mode == "up":
            if not isinstance(self.layers[layer_idx], QKAN):
                return
            self.layers[layer_idx].mask[:, in_idx] = 0.0
        else:
            self.remove_node(layer_idx, in_idx, mode="up")
            self.remove_node(layer_idx, in_idx, mode="down")

    @staticmethod
    def clear_ckpts(folder="./model_ckpt"):
        """
        Clear all checkpoints.

        Args
        ----
            folder : str
                Folder containing checkpoints, default: "./model_ckpt"
        """
        is_ddp, ddp_rank, _, _ = get_dist_info()
        if not is_ddp or ddp_rank == 0:
            if os.path.exists(folder):
                files = glob(folder + "/*")
                for f in files:
                    os.remove(f)
            else:
                os.makedirs(folder)

    def save_ckpt(self, name, folder="./model_ckpt"):
        """
        Save the current model as checkpoint.

        Args
        ----
            name : str
                Name of the checkpoint
            folder : str
                Folder to save the checkpoint, default: "./model_ckpt"
        """
        is_ddp, ddp_rank, _, _ = get_dist_info()
        if not is_ddp or ddp_rank == 0:
            if not os.path.exists(folder):
                os.makedirs(folder)

            torch.save(self.state_dict(), folder + "/" + name)
            print0("save this model to", folder + "/" + name)

    def load_ckpt(self, name, folder="./model_ckpt"):
        """
        Load a checkpoint to the current model.

        Args
        ----
            name : str
                Name of the checkpoint
            folder : str
                Folder containing the checkpoint, default: "./model_ckpt"
        """
        self.load_state_dict(torch.load(folder + "/" + name))
