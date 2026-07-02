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


"""
QKAN solver for CUDA-Q GPU-accelerated simulation or QPU execution.
"""

import math
import warnings
from typing import Optional

import torch

from ._base import QKANSolver, register

try:
    import cudaq  # type: ignore

    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False


from ._mitigation import _apply_mitigation
from .layout import DeviceProfile, rank_qubits

# ---------------------------------------------------------------------------
# CUDA-Q gate-folded kernel builders (for ZNE)
# ---------------------------------------------------------------------------


def _build_cudaq_pz_folded_kernel(reps: int, scale_factor: int):
    """Build a gate-folded pz_encoding kernel for ZNE: U . (U_dag . U)^n."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(x_val: float, thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            rz(thetas[2 * l], q)
            ry(thetas[2 * l + 1], q)
            rz(x_val, q)
        rz(thetas[2 * reps], q)
        ry(thetas[2 * reps + 1], q)
        for _f in range(n_folds):
            ry(-thetas[2 * reps + 1], q)
            rz(-thetas[2 * reps], q)
            for l in range(reps - 1, -1, -1):
                rz(-x_val, q)
                ry(-thetas[2 * l + 1], q)
                rz(-thetas[2 * l], q)
            h(q)
            h(q)
            for l in range(reps):
                rz(thetas[2 * l], q)
                ry(thetas[2 * l + 1], q)
                rz(x_val, q)
            rz(thetas[2 * reps], q)
            ry(thetas[2 * reps + 1], q)

    return kernel


def _build_cudaq_pz_preact_folded_kernel(reps: int, scale_factor: int):
    """Build a gate-folded pz_encoding preact kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(encoded_x: list[float], thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            rz(thetas[2 * l], q)
            ry(thetas[2 * l + 1], q)
            rz(encoded_x[l], q)
        rz(thetas[2 * reps], q)
        ry(thetas[2 * reps + 1], q)
        for _f in range(n_folds):
            ry(-thetas[2 * reps + 1], q)
            rz(-thetas[2 * reps], q)
            for l in range(reps - 1, -1, -1):
                rz(-encoded_x[l], q)
                ry(-thetas[2 * l + 1], q)
                rz(-thetas[2 * l], q)
            h(q)
            h(q)
            for l in range(reps):
                rz(thetas[2 * l], q)
                ry(thetas[2 * l + 1], q)
                rz(encoded_x[l], q)
            rz(thetas[2 * reps], q)
            ry(thetas[2 * reps + 1], q)

    return kernel


def _build_cudaq_rpz_folded_kernel(reps: int, scale_factor: int):
    """Build a gate-folded rpz_encoding kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(encoded_x: list[float], thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            ry(thetas[l], q)
            rz(encoded_x[l], q)
        ry(thetas[reps], q)
        for _f in range(n_folds):
            ry(-thetas[reps], q)
            for l in range(reps - 1, -1, -1):
                rz(-encoded_x[l], q)
                ry(-thetas[l], q)
            h(q)
            h(q)
            for l in range(reps):
                ry(thetas[l], q)
                rz(encoded_x[l], q)
            ry(thetas[reps], q)

    return kernel


def _build_cudaq_real_folded_kernel(reps: int, scale_factor: int):
    """Build a gate-folded real ansatz kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(x_val: float, thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            x(q)
            ry(thetas[l], q)
            z(q)
            ry(x_val, q)
        for _f in range(n_folds):
            for l in range(reps - 1, -1, -1):
                ry(-x_val, q)
                z(q)
                ry(-thetas[l], q)
                x(q)
            h(q)
            h(q)
            for l in range(reps):
                x(q)
                ry(thetas[l], q)
                z(q)
                ry(x_val, q)

    return kernel


def _build_cudaq_real_preact_folded_kernel(reps: int, scale_factor: int):
    """Build a gate-folded real preact kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(encoded_x: list[float], thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            x(q)
            ry(thetas[l], q)
            z(q)
            ry(encoded_x[l], q)
        for _f in range(n_folds):
            for l in range(reps - 1, -1, -1):
                ry(-encoded_x[l], q)
                z(q)
                ry(-thetas[l], q)
                x(q)
            h(q)
            h(q)
            for l in range(reps):
                x(q)
                ry(thetas[l], q)
                z(q)
                ry(encoded_x[l], q)

    return kernel


def _build_cudaq_parallel_pz_folded_kernel(
    n_qubits: int,
    reps: int,
    scale_factor: int,
    width: Optional[int] = None,
    layout: Optional[list[int]] = None,
):
    """Build a gate-folded parallel pz kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2
    if layout is None:
        layout = list(range(n_qubits))
    if width is None:
        width = max(layout) + 1

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(width)
        params_per = 2 * (reps + 1)
        for q_idx in range(n_qubits):
            offset = q_idx * params_per
            h(qubits[layout[q_idx]])
            for l in range(reps):
                rz(all_thetas[offset + 2 * l], qubits[layout[q_idx]])
                ry(all_thetas[offset + 2 * l + 1], qubits[layout[q_idx]])
                rz(x_vals[q_idx], qubits[layout[q_idx]])
            rz(all_thetas[offset + 2 * reps], qubits[layout[q_idx]])
            ry(all_thetas[offset + 2 * reps + 1], qubits[layout[q_idx]])
            for _f in range(n_folds):
                ry(-all_thetas[offset + 2 * reps + 1], qubits[layout[q_idx]])
                rz(-all_thetas[offset + 2 * reps], qubits[layout[q_idx]])
                for l in range(reps - 1, -1, -1):
                    rz(-x_vals[q_idx], qubits[layout[q_idx]])
                    ry(-all_thetas[offset + 2 * l + 1], qubits[layout[q_idx]])
                    rz(-all_thetas[offset + 2 * l], qubits[layout[q_idx]])
                h(qubits[layout[q_idx]])
                h(qubits[layout[q_idx]])
                for l in range(reps):
                    rz(all_thetas[offset + 2 * l], qubits[layout[q_idx]])
                    ry(all_thetas[offset + 2 * l + 1], qubits[layout[q_idx]])
                    rz(x_vals[q_idx], qubits[layout[q_idx]])
                rz(all_thetas[offset + 2 * reps], qubits[layout[q_idx]])
                ry(all_thetas[offset + 2 * reps + 1], qubits[layout[q_idx]])

    return kernel


def _build_cudaq_parallel_real_folded_kernel(
    n_qubits: int,
    reps: int,
    scale_factor: int,
    width: Optional[int] = None,
    layout: Optional[list[int]] = None,
):
    """Build a gate-folded parallel real kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2
    if layout is None:
        layout = list(range(n_qubits))
    if width is None:
        width = max(layout) + 1

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(width)
        for q_idx in range(n_qubits):
            offset = q_idx * reps
            h(qubits[layout[q_idx]])
            for l in range(reps):
                x(qubits[layout[q_idx]])
                ry(all_thetas[offset + l], qubits[layout[q_idx]])
                z(qubits[layout[q_idx]])
                ry(x_vals[q_idx], qubits[layout[q_idx]])
            for _f in range(n_folds):
                for l in range(reps - 1, -1, -1):
                    ry(-x_vals[q_idx], qubits[layout[q_idx]])
                    z(qubits[layout[q_idx]])
                    ry(-all_thetas[offset + l], qubits[layout[q_idx]])
                    x(qubits[layout[q_idx]])
                h(qubits[layout[q_idx]])
                h(qubits[layout[q_idx]])
                for l in range(reps):
                    x(qubits[layout[q_idx]])
                    ry(all_thetas[offset + l], qubits[layout[q_idx]])
                    z(qubits[layout[q_idx]])
                    ry(x_vals[q_idx], qubits[layout[q_idx]])

    return kernel


def _build_cudaq_parallel_rpz_folded_kernel(
    n_qubits: int,
    reps: int,
    scale_factor: int,
    width: Optional[int] = None,
    layout: Optional[list[int]] = None,
):
    """Build a gate-folded parallel rpz kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2
    if layout is None:
        layout = list(range(n_qubits))
    if width is None:
        width = max(layout) + 1

    @cudaq.kernel
    def kernel(encoded_xs: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(width)
        t_per = reps + 1
        for q_idx in range(n_qubits):
            t_off = q_idx * t_per
            x_off = q_idx * reps
            h(qubits[layout[q_idx]])
            for l in range(reps):
                ry(all_thetas[t_off + l], qubits[layout[q_idx]])
                rz(encoded_xs[x_off + l], qubits[layout[q_idx]])
            ry(all_thetas[t_off + reps], qubits[layout[q_idx]])
            for _f in range(n_folds):
                ry(-all_thetas[t_off + reps], qubits[layout[q_idx]])
                for l in range(reps - 1, -1, -1):
                    rz(-encoded_xs[x_off + l], qubits[layout[q_idx]])
                    ry(-all_thetas[t_off + l], qubits[layout[q_idx]])
                h(qubits[layout[q_idx]])
                h(qubits[layout[q_idx]])
                for l in range(reps):
                    ry(all_thetas[t_off + l], qubits[layout[q_idx]])
                    rz(encoded_xs[x_off + l], qubits[layout[q_idx]])
                ry(all_thetas[t_off + reps], qubits[layout[q_idx]])

    return kernel


# ---------------------------------------------------------------------------
# CUDA-Q solver
# ---------------------------------------------------------------------------


def _build_cudaq_pz_kernel(reps: int):
    """Build a CUDA-Q kernel for pz_encoding ansatz."""

    @cudaq.kernel
    def kernel(x_val: float, thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            rz(thetas[2 * l], q)
            ry(thetas[2 * l + 1], q)
            rz(x_val, q)
        rz(thetas[2 * reps], q)
        ry(thetas[2 * reps + 1], q)

    return kernel


def _build_cudaq_pz_preact_kernel(reps: int):
    """Build a CUDA-Q kernel for pz_encoding with trainable preactivation."""

    @cudaq.kernel
    def kernel(encoded_x: list[float], thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            rz(thetas[2 * l], q)
            ry(thetas[2 * l + 1], q)
            rz(encoded_x[l], q)
        rz(thetas[2 * reps], q)
        ry(thetas[2 * reps + 1], q)

    return kernel


def _build_cudaq_rpz_kernel(reps: int):
    """Build a CUDA-Q kernel for rpz_encoding ansatz."""

    @cudaq.kernel
    def kernel(encoded_x: list[float], thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            ry(thetas[l], q)
            rz(encoded_x[l], q)
        ry(thetas[reps], q)

    return kernel


def _build_cudaq_real_kernel(reps: int):
    """Build a CUDA-Q kernel for real ansatz."""

    @cudaq.kernel
    def kernel(x_val: float, thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            x(q)
            ry(thetas[l], q)
            z(q)
            ry(x_val, q)

    return kernel


def _build_cudaq_real_preact_kernel(reps: int):
    """Build a CUDA-Q kernel for real ansatz with trainable preactivation."""

    @cudaq.kernel
    def kernel(encoded_x: list[float], thetas: list[float]):
        q = cudaq.qubit()
        h(q)
        for l in range(reps):
            x(q)
            ry(thetas[l], q)
            z(q)
            ry(encoded_x[l], q)

    return kernel


# Cache: {(ansatz, reps, preacts_trainable, scale_factor): kernel}
_CUDAQ_KERNEL_CACHE: dict = {}


def _get_cudaq_kernel(
    ansatz: str, reps: int, preacts_trainable: bool, scale_factor: int = 1
):
    """Get or build a cached CUDA-Q kernel (optionally gate-folded for ZNE)."""
    key = (ansatz, reps, preacts_trainable, scale_factor)
    if key not in _CUDAQ_KERNEL_CACHE:
        sf = scale_factor
        if ansatz in ("pz_encoding", "pz"):
            if preacts_trainable:
                _CUDAQ_KERNEL_CACHE[key] = (
                    _build_cudaq_pz_preact_folded_kernel(reps, sf)
                    if sf > 1
                    else _build_cudaq_pz_preact_kernel(reps)
                )
            else:
                _CUDAQ_KERNEL_CACHE[key] = (
                    _build_cudaq_pz_folded_kernel(reps, sf)
                    if sf > 1
                    else _build_cudaq_pz_kernel(reps)
                )
        elif ansatz in ("rpz_encoding", "rpz"):
            _CUDAQ_KERNEL_CACHE[key] = (
                _build_cudaq_rpz_folded_kernel(reps, sf)
                if sf > 1
                else _build_cudaq_rpz_kernel(reps)
            )
        elif ansatz == "real":
            if preacts_trainable:
                _CUDAQ_KERNEL_CACHE[key] = (
                    _build_cudaq_real_preact_folded_kernel(reps, sf)
                    if sf > 1
                    else _build_cudaq_real_preact_kernel(reps)
                )
            else:
                _CUDAQ_KERNEL_CACHE[key] = (
                    _build_cudaq_real_folded_kernel(reps, sf)
                    if sf > 1
                    else _build_cudaq_real_kernel(reps)
                )
        else:
            raise NotImplementedError(
                f"Ansatz '{ansatz}' not supported by cudaq_solver"
            )
    return _CUDAQ_KERNEL_CACHE[key]


def _build_cudaq_parallel_pz_kernel(
    n_qubits: int,
    reps: int,
    width: Optional[int] = None,
    layout: Optional[list[int]] = None,
):
    """Build a CUDA-Q kernel that runs N independent pz circuits in parallel."""
    if layout is None:
        layout = list(range(n_qubits))
    if width is None:
        width = max(layout) + 1

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(width)
        params_per = 2 * (reps + 1)
        for q_idx in range(n_qubits):
            h(qubits[layout[q_idx]])
            offset = q_idx * params_per
            for l in range(reps):
                rz(all_thetas[offset + 2 * l], qubits[layout[q_idx]])
                ry(all_thetas[offset + 2 * l + 1], qubits[layout[q_idx]])
                rz(x_vals[q_idx], qubits[layout[q_idx]])
            rz(all_thetas[offset + 2 * reps], qubits[layout[q_idx]])
            ry(all_thetas[offset + 2 * reps + 1], qubits[layout[q_idx]])

    return kernel


def _build_cudaq_parallel_real_kernel(
    n_qubits: int,
    reps: int,
    width: Optional[int] = None,
    layout: Optional[list[int]] = None,
):
    """Build a CUDA-Q kernel that runs N independent real circuits in parallel."""
    if layout is None:
        layout = list(range(n_qubits))
    if width is None:
        width = max(layout) + 1

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(width)
        for q_idx in range(n_qubits):
            h(qubits[layout[q_idx]])
            offset = q_idx * reps
            for l in range(reps):
                x(qubits[layout[q_idx]])
                ry(all_thetas[offset + l], qubits[layout[q_idx]])
                z(qubits[layout[q_idx]])
                ry(x_vals[q_idx], qubits[layout[q_idx]])

    return kernel


def _build_cudaq_parallel_rpz_kernel(
    n_qubits: int,
    reps: int,
    width: Optional[int] = None,
    layout: Optional[list[int]] = None,
):
    """Build a CUDA-Q kernel that runs N independent rpz circuits in parallel."""
    if layout is None:
        layout = list(range(n_qubits))
    if width is None:
        width = max(layout) + 1

    @cudaq.kernel
    def kernel(encoded_xs: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(width)
        t_per = reps + 1
        for q_idx in range(n_qubits):
            h(qubits[layout[q_idx]])
            t_off = q_idx * t_per
            x_off = q_idx * reps
            for l in range(reps):
                ry(all_thetas[t_off + l], qubits[layout[q_idx]])
                rz(encoded_xs[x_off + l], qubits[layout[q_idx]])
            ry(all_thetas[t_off + reps], qubits[layout[q_idx]])

    return kernel


def _sample_z_marginals(result, positions, shots_count):
    """Per-qubit <Z> from sampled bitstring counts at the given positions."""
    acc = [0.0] * len(positions)
    total = 0
    for bits, count in result.items():
        total += count
        for i, pos in enumerate(positions):
            acc[i] += count if bits[pos] == "0" else -count
    total = total or shots_count
    return [a / total for a in acc]


def _cudaq_run_parallel(
    all_args,
    ansatz,
    reps,
    preacts_trainable,
    n_qubits,
    shots_count,
    scale_factor=1,
    initial_layout=None,
    expectation_via_sample=False,
):
    """
    Pack independent single-qubit circuits onto an N-qubit QPU.

    Runs ceil(total / n_qubits) multi-qubit jobs instead of `total` single-qubit jobs.

    When `initial_layout` is a list of physical qubit indices, circuit slot j
    is applied to register index `initial_layout[j]`, the register is
    widened to max(layout)+1 qubits, and every idle register qubit receives
    zero-angle padding gates so that compiler dead-code elimination cannot
    renumber the indices (the iqm/oqc/anyon pipelines compact untouched
    qubits). A layout longer than `n_qubits` keeps its first (best-ranked)
    entries; whether the physical indices are honored end-to-end depends on
    the target (see the solver docs).
    """
    total = len(all_args)
    expvals = []

    if initial_layout is not None:
        if len(initial_layout) < n_qubits:
            raise ValueError(
                "initial_layout has fewer qubits than the packing width: "
                f"{len(initial_layout)} < {n_qubits}"
            )
        layout = [int(q) for q in initial_layout[:n_qubits]]
        if len(set(layout)) != len(layout):
            raise ValueError("initial_layout assigns the same physical qubit twice")
    else:
        layout = list(range(n_qubits))
    width = max(layout) + 1
    # Pad every register qubit below `width` with a zero-angle circuit slot:
    # runtime-argument gates defeat dead-code elimination, which would
    # otherwise compact the qubit indices on iqm/oqc/anyon targets.
    used = set(layout)
    slot_layout = layout + [q for q in range(width) if q not in used]
    n_slots = width

    # Build kernel once — all chunks pad to n_qubits
    if ansatz in ("pz_encoding", "pz") and not preacts_trainable:
        par_kernel = (
            _build_cudaq_parallel_pz_folded_kernel(
                n_slots, reps, scale_factor, width=width, layout=slot_layout
            )
            if scale_factor > 1
            else _build_cudaq_parallel_pz_kernel(
                n_slots, reps, width=width, layout=slot_layout
            )
        )
    elif ansatz == "real" and not preacts_trainable:
        par_kernel = (
            _build_cudaq_parallel_real_folded_kernel(
                n_slots, reps, scale_factor, width=width, layout=slot_layout
            )
            if scale_factor > 1
            else _build_cudaq_parallel_real_kernel(
                n_slots, reps, width=width, layout=slot_layout
            )
        )
    elif ansatz in ("rpz_encoding", "rpz") or preacts_trainable:
        par_kernel = (
            _build_cudaq_parallel_rpz_folded_kernel(
                n_slots, reps, scale_factor, width=width, layout=slot_layout
            )
            if scale_factor > 1
            else _build_cudaq_parallel_rpz_kernel(
                n_slots, reps, width=width, layout=slot_layout
            )
        )
    else:
        raise NotImplementedError(f"Parallel not supported for ansatz '{ansatz}'")

    spin_sum = cudaq.spin.z(layout[0])
    for q_idx in range(1, n_qubits):
        spin_sum += cudaq.spin.z(layout[q_idx])

    for start in range(0, total, n_qubits):
        end = min(start + n_qubits, total)
        chunk = all_args[start:end]
        chunk_size = end - start

        # Flatten args and pad to n_qubits
        if ansatz in ("pz_encoding", "pz") and not preacts_trainable:
            x_vals = [a[0] for a in chunk]
            all_thetas = []
            for a in chunk:
                all_thetas.extend(a[1])
            if chunk_size < n_slots:
                x_vals.extend([0.0] * (n_slots - chunk_size))
                pad_thetas = [0.0] * (2 * (reps + 1))
                for _ in range(n_slots - chunk_size):
                    all_thetas.extend(pad_thetas)
            args = (x_vals, all_thetas)

        elif ansatz == "real" and not preacts_trainable:
            x_vals = [a[0] for a in chunk]
            all_thetas = []
            for a in chunk:
                all_thetas.extend(a[1])
            if chunk_size < n_slots:
                x_vals.extend([0.0] * (n_slots - chunk_size))
                for _ in range(n_slots - chunk_size):
                    all_thetas.extend([0.0] * reps)
            args = (x_vals, all_thetas)

        else:  # rpz or preacts_trainable
            encoded_xs = []
            all_thetas = []
            for a in chunk:
                enc, t = a
                if isinstance(enc, list):
                    encoded_xs.extend(enc)
                else:
                    encoded_xs.extend([enc] * reps)
                all_thetas.extend(t)
            if chunk_size < n_slots:
                for _ in range(n_slots - chunk_size):
                    encoded_xs.extend([0.0] * reps)
                    all_thetas.extend([0.0] * (reps + 1))
            args = (encoded_xs, all_thetas)

        if expectation_via_sample:
            # Some REST targets (quantum_machines) mis-handle multi-term
            # observe; sample once and read per-qubit <Z> marginals instead.
            result = cudaq.sample(par_kernel, *args, shots_count=shots_count)
            expvals.extend(
                _sample_z_marginals(result, layout[:chunk_size], shots_count)
            )
            continue
        if shots_count is not None:
            result = cudaq.observe(par_kernel, spin_sum, *args, shots_count=shots_count)
        else:
            result = cudaq.observe(par_kernel, spin_sum, *args)

        for q_idx in range(chunk_size):
            expvals.append(result.expectation(cudaq.spin.z(layout[q_idx])))

    return expvals


def _cudaq_evaluate(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    config: dict,
) -> torch.Tensor:
    """
    Evaluate all circuits on CUDA-Q and return expectation values.

    Returns shape: (batch_size, out_dim, in_dim)
    """
    batch, in_dim = x.shape
    ansatz = config["ansatz"]
    preacts_trainable = config["preacts_trainable"]
    out_dim = config["out_dim"]
    shots_count = config["shots"]
    parallel_qubits = config.get("parallel_qubits", None)
    initial_layout = config.get("initial_layout", None)

    # Broadcasting (same as other solvers)
    if len(theta.shape) != 4:
        theta = theta.unsqueeze(0)
    if theta.shape[1] != in_dim:
        repeat_out = out_dim
        repeat_in = in_dim // theta.shape[1] + 1
        theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]

    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    encoded_x = None
    if _needs_encoded_x:
        if len(preacts_weight.shape) != 3:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]
        encoded_x = torch.einsum("oir,bi->boir", preacts_weight, x).add(preacts_bias)

    x_np = x.detach().cpu()
    theta_np = theta.detach().cpu()
    encoded_x_np = encoded_x.detach().cpu() if encoded_x is not None else None

    # Pre-convert theta (batch-independent) and x to Python lists
    theta_lists = {
        (o, i): theta_np[o, i].reshape(-1).tolist()
        for o in range(out_dim)
        for i in range(in_dim)
    }
    x_py = x_np.tolist() if not _needs_encoded_x else None
    enc_py = encoded_x_np.tolist() if encoded_x_np is not None else None

    all_args = []
    for b in range(batch):
        for o in range(out_dim):
            for i in range(in_dim):
                t = theta_lists[(o, i)]
                if _needs_encoded_x:
                    all_args.append((enc_py[b][o][i], t))
                else:
                    all_args.append((x_py[b][i], t))

    mitigation = config.get("mitigation", {})
    expectation_via_sample = config.get("expectation_via_sample", False)

    def _run_cudaq(scale_factor=1):
        if parallel_qubits and parallel_qubits > 1:
            return _cudaq_run_parallel(
                all_args,
                ansatz,
                reps,
                preacts_trainable,
                parallel_qubits,
                shots_count,
                scale_factor=scale_factor,
                initial_layout=initial_layout,
                expectation_via_sample=expectation_via_sample,
            )
        else:
            kernel = _get_cudaq_kernel(ansatz, reps, preacts_trainable, scale_factor)
            spin_z = cudaq.spin.z(0)
            ev = []
            for args in all_args:
                if expectation_via_sample:
                    result = cudaq.sample(kernel, *args, shots_count=shots_count)
                    ev.extend(_sample_z_marginals(result, [0], shots_count))
                    continue
                if shots_count is not None:
                    result = cudaq.observe(
                        kernel, spin_z, *args, shots_count=shots_count
                    )
                else:
                    result = cudaq.observe(kernel, spin_z, *args)
                ev.append(result.expectation())
            return ev

    if mitigation:
        expvals = _apply_mitigation(_run_cudaq, mitigation)
    else:
        expvals = _run_cudaq(1)

    output = torch.tensor(expvals, dtype=x.dtype, device=x.device)
    return output.reshape(batch, out_dim, in_dim)


class _CudaqParamShift(torch.autograd.Function):
    """Autograd function using parameter-shift rule for CUDA-Q circuits."""

    @staticmethod
    def forward(ctx, x, theta, preacts_w, preacts_b, reps, config):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.config = config
        return _cudaq_evaluate(x, theta, preacts_w, preacts_b, reps, config)

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        reps = ctx.reps
        config = ctx.config
        shift = math.pi / 2

        grad_theta = torch.zeros_like(theta)
        flat_theta = theta.reshape(-1)
        for k in range(flat_theta.numel()):
            theta_plus = flat_theta.clone()
            theta_plus[k] += shift
            theta_minus = flat_theta.clone()
            theta_minus[k] -= shift

            f_plus = _cudaq_evaluate(
                x, theta_plus.reshape(theta.shape), preacts_w, preacts_b, reps, config
            )
            f_minus = _cudaq_evaluate(
                x, theta_minus.reshape(theta.shape), preacts_w, preacts_b, reps, config
            )
            grad_theta.reshape(-1)[k] = (
                grad_output * (f_plus - f_minus) / (2 * math.sin(shift))
            ).sum()

        grad_pw = None
        if preacts_w.requires_grad:
            grad_pw = torch.zeros_like(preacts_w)
            flat_pw = preacts_w.reshape(-1)
            for k in range(flat_pw.numel()):
                pw_plus = flat_pw.clone()
                pw_plus[k] += shift
                pw_minus = flat_pw.clone()
                pw_minus[k] -= shift
                f_plus = _cudaq_evaluate(
                    x, theta, pw_plus.reshape(preacts_w.shape), preacts_b, reps, config
                )
                f_minus = _cudaq_evaluate(
                    x, theta, pw_minus.reshape(preacts_w.shape), preacts_b, reps, config
                )
                grad_pw.reshape(-1)[k] = (
                    grad_output * (f_plus - f_minus) / (2 * math.sin(shift))
                ).sum()

        grad_pb = None
        if preacts_b.requires_grad:
            grad_pb = torch.zeros_like(preacts_b)
            flat_pb = preacts_b.reshape(-1)
            for k in range(flat_pb.numel()):
                pb_plus = flat_pb.clone()
                pb_plus[k] += shift
                pb_minus = flat_pb.clone()
                pb_minus[k] -= shift
                f_plus = _cudaq_evaluate(
                    x, theta, preacts_w, pb_plus.reshape(preacts_b.shape), reps, config
                )
                f_minus = _cudaq_evaluate(
                    x, theta, preacts_w, pb_minus.reshape(preacts_b.shape), reps, config
                )
                grad_pb.reshape(-1)[k] = (
                    grad_output * (f_plus - f_minus) / (2 * math.sin(shift))
                ).sum()

        return None, grad_theta, grad_pw, grad_pb, None, None


# Calibration snapshots fetched per machine ARN; cached for the process
# lifetime so 'auto' does not issue one AWS GetDevice call per forward pass.
_BRAKET_PROFILE_CACHE: dict = {}


def _braket_device_profile(machine: str) -> Optional[DeviceProfile]:
    """Fetch a calibration profile for a Braket machine ARN (best effort)."""
    if machine in _BRAKET_PROFILE_CACHE:
        return _BRAKET_PROFILE_CACHE[machine]
    try:
        from braket.aws import AwsDevice  # type: ignore
    except ImportError:
        warnings.warn(
            "initial_layout='auto' on the braket target needs "
            "amazon-braket-sdk for calibration data; proceeding without a "
            "layout.",
            stacklevel=3,
        )
        return None
    try:
        profile = DeviceProfile.from_braket(AwsDevice(machine))
        _BRAKET_PROFILE_CACHE[machine] = profile
        return profile
    except Exception as exc:
        warnings.warn(
            f"could not fetch Braket calibration for {machine}: {exc}; "
            "proceeding without a layout.",
            stacklevel=3,
        )
        return None


def cudaq_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Execute QKAN circuits via NVIDIA CUDA-Q.

    Drop-in replacement for torch_exact_solver using CUDA-Q's GPU-accelerated
    quantum simulation or QPU backends. Circuits are built as CUDA-Q kernels
    and expectation values are computed via cudaq.observe().

    Supports training via the parameter-shift rule when gradients are needed.

    Args
    ----
        x : torch.Tensor
            shape: (batch_size, in_dim)
        theta : torch.Tensor
            shape: (\\*group, reps+1, n_params) or (\\*group, reps, 1) for real
        preacts_weight : torch.Tensor
            shape: (\\*group, reps)
        preacts_bias : torch.Tensor
            shape: (\\*group, reps)
        reps : int
        ansatz : str
            "pz_encoding", "pz", "rpz_encoding", "rpz", or "real"
        preacts_trainable : bool
        out_dim : int
        target : str, optional
            CUDA-Q target (e.g., "nvidia", "nvidia-mqpu", "qpp-cpu",
            "braket", "iqm", "quantum_machines").
            Set before calling via cudaq.set_target().
        url / executor / api_key : optional
            Forwarded to cudaq.set_target for REST hardware targets. For
            "quantum_machines" these select the Qoperator server URL, the
            executor name, and the X-API-Key credential (the
            QUANTUM_MACHINES_API_KEY env var also works).
        shots : int, optional
            Number of shots. None for exact statevector expectation.
            Required for the quantum_machines target.
        expectation_via_sample : bool, optional
            Compute <Z> from sampled bitstring marginals instead of
            cudaq.observe. Defaults to True on the quantum_machines target,
            whose REST helper mis-handles multi-term observe.
        initial_layout : optional
            None (default), "auto", or a list of physical qubit indices for
            the packed `parallel_qubits` path on hardware targets. "auto"
            ranks qubits from calibration data: pass `device_profile` (a
            qkan.solver.layout.DeviceProfile) or use the braket target with
            a machine ARN and amazon-braket-sdk installed. Applied by
            construction (circuit slot j runs on register index layout[j]);
            native iqm/oqc/anyon targets preserve these indices, while
            Braket's vendor compiler may remap them.
        device_profile : DeviceProfile, optional
            Calibration snapshot used by initial_layout="auto" and for
            layout bounds checking (see qkan.solver.layout).
        max_readout_error / qubit_error_threshold : float, optional
            Calibration thresholds forwarded to the "auto" qubit ranking.

    Returns
    -------
        torch.Tensor
            shape: (batch_size, out_dim, in_dim)
    """
    if not _CUDAQ_AVAILABLE:
        raise ImportError(
            "CUDA-Q is required for cudaq_solver. "
            "Install from: https://nvidia.github.io/cuda-quantum/"
        )

    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    out_dim = kwargs.get("out_dim", x.shape[1])
    shots = kwargs.get("shots", None)
    target = kwargs.get("target", None)
    machine = kwargs.get("machine", None)
    parallel_qubits = kwargs.get("parallel_qubits", None)

    if target is not None:
        target_kwargs = {}
        # Forward the target options CUDA-Q hardware backends accept:
        # machine (braket/ionq/...), url/executor/api_key (quantum_machines).
        for key in ("machine", "url", "executor", "api_key"):
            if kwargs.get(key) is not None:
                target_kwargs[key] = kwargs[key]
        cudaq.set_target(target, **target_kwargs)

    # Resolve initial_layout (device-independent):
    #   None (default) -> default placement
    #   "auto"         -> rank qubits from a DeviceProfile calibration snapshot
    #   list[int]      -> caller-supplied physical qubit indices
    initial_layout = kwargs.get("initial_layout", None)
    device_profile = kwargs.get("device_profile", None)
    if isinstance(initial_layout, tuple):
        initial_layout = list(initial_layout)
    elif initial_layout is not None and hasattr(initial_layout, "tolist"):
        # Normalize numpy arrays / torch tensors to a plain list.
        initial_layout = list(initial_layout.tolist())
    if isinstance(initial_layout, list) and not initial_layout:
        initial_layout = None
    if isinstance(initial_layout, str) and initial_layout != "auto":
        raise ValueError(
            "initial_layout must be None, 'auto', or a list of physical "
            f"qubit indices; got {initial_layout!r}"
        )
    if isinstance(initial_layout, list):
        entries = []
        for q in initial_layout:
            if int(q) != q or int(q) < 0:
                raise ValueError(
                    f"initial_layout entries must be non-negative integers; got {q!r}"
                )
            entries.append(int(q))
        if len(set(entries)) != len(entries):
            raise ValueError("initial_layout assigns the same physical qubit twice")
        initial_layout = entries
    if parallel_qubits == "auto":
        # CUDA-Q exposes no device qubit count; resolve from the profile.
        if device_profile is None:
            raise ValueError(
                "parallel_qubits='auto' with the cudaq solver requires "
                "device_profile=DeviceProfile(...) to know the device size."
            )
        parallel_qubits = device_profile.num_qubits
    if initial_layout is not None and not (
        isinstance(parallel_qubits, int) and parallel_qubits > 1
    ):
        warnings.warn(
            "the cudaq solver applies initial_layout only on the "
            "parallel_qubits packing path; ignoring it.",
            stacklevel=2,
        )
        initial_layout = None
    if initial_layout is not None:
        current = cudaq.get_target()
        braket_simulator = machine is not None and "/quantum-simulator/" in str(machine)
        if not current.is_remote() or current.is_emulated() or braket_simulator:
            warnings.warn(
                "initial_layout has no effect on simulator or emulated cudaq "
                "targets; ignoring it.",
                stacklevel=2,
            )
            initial_layout = None
    if isinstance(initial_layout, str):
        profile = device_profile
        if profile is None and target == "braket" and machine is not None:
            profile = _braket_device_profile(machine)
        elif profile is None:
            raise ValueError(
                "initial_layout='auto' for the cudaq solver needs calibration "
                "data: pass device_profile=DeviceProfile(...) (see "
                "qkan.solver.layout) — required when the target was "
                "configured globally — or use the braket target with a "
                "machine ARN and amazon-braket-sdk installed."
            )
        if profile is None:
            # Braket fetch failed; a warning was already emitted.
            initial_layout = None
        else:
            initial_layout = rank_qubits(
                profile,
                parallel_qubits,
                max_readout_error=kwargs.get("max_readout_error", None),
                qubit_error_threshold=kwargs.get("qubit_error_threshold", None),
            )
            if not initial_layout:
                warnings.warn(
                    "initial_layout='auto': profile carries no calibration "
                    "data; falling back to the default placement.",
                    stacklevel=2,
                )
                initial_layout = None
            device_profile = profile
    if initial_layout is not None:
        if (
            device_profile is not None
            and max(initial_layout) >= device_profile.num_qubits
        ):
            raise ValueError(
                f"initial_layout index {max(initial_layout)} exceeds the "
                f"device's {device_profile.num_qubits} qubits"
            )
        if target == "braket":
            warnings.warn(
                "CUDA-Q does not disable Braket's qubit rewiring: the vendor "
                "compiler may remap the requested layout on braket targets "
                "(verified on Rigetti). Native iqm/oqc/anyon targets "
                "preserve it.",
                stacklevel=2,
            )
        elif target == "quantum_machines":
            warnings.warn(
                "the requested physical indices are baked into the submitted "
                "OpenQASM, but qubit mapping on quantum_machines is decided "
                "by the Qoperator server; verify with your operator "
                "configuration.",
                stacklevel=2,
            )

    # quantum_machines executes via physical sampling and its REST helper
    # mis-handles multi-term observe (first Z term returns full-register
    # parity); route expectations through sample() marginals instead.
    expectation_via_sample = kwargs.get(
        "expectation_via_sample", target == "quantum_machines"
    )
    if target == "quantum_machines" and shots is None:
        raise ValueError(
            "the quantum_machines target requires explicit shots (results "
            "come from physical sampling); pass shots=... in solver_kwargs."
        )

    config = {
        "ansatz": ansatz,
        "preacts_trainable": preacts_trainable,
        "out_dim": out_dim,
        "shots": shots,
        "parallel_qubits": parallel_qubits,
        "initial_layout": initial_layout,
        "expectation_via_sample": expectation_via_sample,
        "mitigation": kwargs.get("mitigation", {}),
    }

    needs_grad = theta.requires_grad or x.requires_grad
    if preacts_trainable:
        needs_grad = (
            needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad
        )

    if needs_grad:
        return _CudaqParamShift.apply(
            x, theta, preacts_weight, preacts_bias, reps, config
        )
    else:
        return _cudaq_evaluate(x, theta, preacts_weight, preacts_bias, reps, config)


class CudaqSolver(QKANSolver):
    """NVIDIA CUDA-Q solver (registered as ``"cudaq"``)."""

    name = "cudaq"

    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor:
        return cudaq_solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs)


register(CudaqSolver())
