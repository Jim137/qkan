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

import torch

try:
    import cudaq  # type: ignore

    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False


from ._mitigation import _apply_mitigation


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
    n_qubits: int, reps: int, scale_factor: int
):
    """Build a gate-folded parallel pz kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        params_per = 2 * (reps + 1)
        for q_idx in range(n_qubits):
            offset = q_idx * params_per
            h(qubits[q_idx])
            for l in range(reps):
                rz(all_thetas[offset + 2 * l], qubits[q_idx])
                ry(all_thetas[offset + 2 * l + 1], qubits[q_idx])
                rz(x_vals[q_idx], qubits[q_idx])
            rz(all_thetas[offset + 2 * reps], qubits[q_idx])
            ry(all_thetas[offset + 2 * reps + 1], qubits[q_idx])
            for _f in range(n_folds):
                ry(-all_thetas[offset + 2 * reps + 1], qubits[q_idx])
                rz(-all_thetas[offset + 2 * reps], qubits[q_idx])
                for l in range(reps - 1, -1, -1):
                    rz(-x_vals[q_idx], qubits[q_idx])
                    ry(-all_thetas[offset + 2 * l + 1], qubits[q_idx])
                    rz(-all_thetas[offset + 2 * l], qubits[q_idx])
                h(qubits[q_idx])
                h(qubits[q_idx])
                for l in range(reps):
                    rz(all_thetas[offset + 2 * l], qubits[q_idx])
                    ry(all_thetas[offset + 2 * l + 1], qubits[q_idx])
                    rz(x_vals[q_idx], qubits[q_idx])
                rz(all_thetas[offset + 2 * reps], qubits[q_idx])
                ry(all_thetas[offset + 2 * reps + 1], qubits[q_idx])

    return kernel


def _build_cudaq_parallel_real_folded_kernel(
    n_qubits: int, reps: int, scale_factor: int
):
    """Build a gate-folded parallel real kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        for q_idx in range(n_qubits):
            offset = q_idx * reps
            h(qubits[q_idx])
            for l in range(reps):
                x(qubits[q_idx])
                ry(all_thetas[offset + l], qubits[q_idx])
                z(qubits[q_idx])
                ry(x_vals[q_idx], qubits[q_idx])
            for _f in range(n_folds):
                for l in range(reps - 1, -1, -1):
                    ry(-x_vals[q_idx], qubits[q_idx])
                    z(qubits[q_idx])
                    ry(-all_thetas[offset + l], qubits[q_idx])
                    x(qubits[q_idx])
                h(qubits[q_idx])
                h(qubits[q_idx])
                for l in range(reps):
                    x(qubits[q_idx])
                    ry(all_thetas[offset + l], qubits[q_idx])
                    z(qubits[q_idx])
                    ry(x_vals[q_idx], qubits[q_idx])

    return kernel


def _build_cudaq_parallel_rpz_folded_kernel(
    n_qubits: int, reps: int, scale_factor: int
):
    """Build a gate-folded parallel rpz kernel for ZNE."""
    n_folds = (scale_factor - 1) // 2

    @cudaq.kernel
    def kernel(encoded_xs: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        t_per = reps + 1
        for q_idx in range(n_qubits):
            t_off = q_idx * t_per
            x_off = q_idx * reps
            h(qubits[q_idx])
            for l in range(reps):
                ry(all_thetas[t_off + l], qubits[q_idx])
                rz(encoded_xs[x_off + l], qubits[q_idx])
            ry(all_thetas[t_off + reps], qubits[q_idx])
            for _f in range(n_folds):
                ry(-all_thetas[t_off + reps], qubits[q_idx])
                for l in range(reps - 1, -1, -1):
                    rz(-encoded_xs[x_off + l], qubits[q_idx])
                    ry(-all_thetas[t_off + l], qubits[q_idx])
                h(qubits[q_idx])
                h(qubits[q_idx])
                for l in range(reps):
                    ry(all_thetas[t_off + l], qubits[q_idx])
                    rz(encoded_xs[x_off + l], qubits[q_idx])
                ry(all_thetas[t_off + reps], qubits[q_idx])

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


def _build_cudaq_parallel_pz_kernel(n_qubits: int, reps: int):
    """Build a CUDA-Q kernel that runs N independent pz circuits in parallel."""

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        params_per = 2 * (reps + 1)
        for q_idx in range(n_qubits):
            h(qubits[q_idx])
            offset = q_idx * params_per
            for l in range(reps):
                rz(all_thetas[offset + 2 * l], qubits[q_idx])
                ry(all_thetas[offset + 2 * l + 1], qubits[q_idx])
                rz(x_vals[q_idx], qubits[q_idx])
            rz(all_thetas[offset + 2 * reps], qubits[q_idx])
            ry(all_thetas[offset + 2 * reps + 1], qubits[q_idx])

    return kernel


def _build_cudaq_parallel_real_kernel(n_qubits: int, reps: int):
    """Build a CUDA-Q kernel that runs N independent real circuits in parallel."""

    @cudaq.kernel
    def kernel(x_vals: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        for q_idx in range(n_qubits):
            h(qubits[q_idx])
            offset = q_idx * reps
            for l in range(reps):
                x(qubits[q_idx])
                ry(all_thetas[offset + l], qubits[q_idx])
                z(qubits[q_idx])
                ry(x_vals[q_idx], qubits[q_idx])

    return kernel


def _build_cudaq_parallel_rpz_kernel(n_qubits: int, reps: int):
    """Build a CUDA-Q kernel that runs N independent rpz circuits in parallel."""

    @cudaq.kernel
    def kernel(encoded_xs: list[float], all_thetas: list[float]):
        qubits = cudaq.qvector(n_qubits)
        t_per = reps + 1
        for q_idx in range(n_qubits):
            h(qubits[q_idx])
            t_off = q_idx * t_per
            x_off = q_idx * reps
            for l in range(reps):
                ry(all_thetas[t_off + l], qubits[q_idx])
                rz(encoded_xs[x_off + l], qubits[q_idx])
            ry(all_thetas[t_off + reps], qubits[q_idx])

    return kernel


def _cudaq_run_parallel(
    all_args, ansatz, reps, preacts_trainable, n_qubits, shots_count,
    scale_factor=1,
):
    """
    Pack independent single-qubit circuits onto an N-qubit QPU.

    Runs ceil(total / n_qubits) multi-qubit jobs instead of `total` single-qubit jobs.
    """
    total = len(all_args)
    expvals = []

    for start in range(0, total, n_qubits):
        chunk = all_args[start : start + n_qubits]
        chunk_size = len(chunk)

        # Flatten args into parallel kernel format
        if ansatz in ("pz_encoding", "pz") and not preacts_trainable:
            x_vals = [a[0] for a in chunk]
            all_thetas = []
            for a in chunk:
                all_thetas.extend(a[1])
            # Pad if chunk < n_qubits
            actual_n = chunk_size
            if actual_n < n_qubits:
                x_vals.extend([0.0] * (n_qubits - actual_n))
                pad_thetas = [0.0] * (2 * (reps + 1))
                for _ in range(n_qubits - actual_n):
                    all_thetas.extend(pad_thetas)
                actual_n = n_qubits

            par_kernel = (
                _build_cudaq_parallel_pz_folded_kernel(actual_n, reps, scale_factor)
                if scale_factor > 1
                else _build_cudaq_parallel_pz_kernel(actual_n, reps)
            )
            args = (x_vals, all_thetas)

        elif ansatz == "real" and not preacts_trainable:
            x_vals = [a[0] for a in chunk]
            all_thetas = []
            for a in chunk:
                all_thetas.extend(a[1])
            actual_n = chunk_size
            if actual_n < n_qubits:
                x_vals.extend([0.0] * (n_qubits - actual_n))
                for _ in range(n_qubits - actual_n):
                    all_thetas.extend([0.0] * reps)
                actual_n = n_qubits

            par_kernel = (
                _build_cudaq_parallel_real_folded_kernel(actual_n, reps, scale_factor)
                if scale_factor > 1
                else _build_cudaq_parallel_real_kernel(actual_n, reps)
            )
            args = (x_vals, all_thetas)

        elif ansatz in ("rpz_encoding", "rpz") or preacts_trainable:
            encoded_xs = []
            all_thetas = []
            for a in chunk:
                enc, t = a
                if isinstance(enc, list):
                    encoded_xs.extend(enc)
                else:
                    encoded_xs.extend([enc] * reps)
                all_thetas.extend(t)
            actual_n = chunk_size
            if actual_n < n_qubits:
                for _ in range(n_qubits - actual_n):
                    encoded_xs.extend([0.0] * reps)
                    all_thetas.extend([0.0] * (reps + 1))
                actual_n = n_qubits

            par_kernel = (
                _build_cudaq_parallel_rpz_folded_kernel(actual_n, reps, scale_factor)
                if scale_factor > 1
                else _build_cudaq_parallel_rpz_kernel(actual_n, reps)
            )
            args = (encoded_xs, all_thetas)
        else:
            raise NotImplementedError(f"Parallel not supported for ansatz '{ansatz}'")

        # Single observe call with Z0 + Z1 + ... + Z_{N-1} Hamiltonian,
        # then extract per-qubit <Z_k> from the result
        spin_sum = cudaq.spin.z(0)
        for q_idx in range(1, actual_n):
            spin_sum += cudaq.spin.z(q_idx)

        if shots_count is not None:
            result = cudaq.observe(par_kernel, spin_sum, *args, shots_count=shots_count)
        else:
            result = cudaq.observe(par_kernel, spin_sum, *args)

        for q_idx in range(chunk_size):
            expvals.append(result.expectation(cudaq.spin.z(q_idx)))

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
    target = config.get("target", None)
    parallel_qubits = config.get("parallel_qubits", None)

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

    # Collect all circuit args first
    all_args = []
    for b in range(batch):
        for o in range(out_dim):
            for i in range(in_dim):
                t = theta_np[o, i].reshape(-1).tolist()

                if ansatz in ("pz_encoding", "pz"):
                    if preacts_trainable:
                        all_args.append((encoded_x_np[b, o, i].tolist(), t))
                    else:
                        all_args.append((float(x_np[b, i]), t))
                elif ansatz in ("rpz_encoding", "rpz"):
                    all_args.append((encoded_x_np[b, o, i].tolist(), t))
                elif ansatz == "real":
                    if preacts_trainable:
                        all_args.append((encoded_x_np[b, o, i].tolist(), t))
                    else:
                        all_args.append((float(x_np[b, i]), t))
                else:
                    raise NotImplementedError

    mitigation = config.get("mitigation", {})

    def _run_cudaq(scale_factor=1):
        if parallel_qubits and parallel_qubits > 1:
            return _cudaq_run_parallel(
                all_args, ansatz, reps, preacts_trainable,
                parallel_qubits, shots_count, scale_factor=scale_factor,
            )
        else:
            kernel = _get_cudaq_kernel(ansatz, reps, preacts_trainable, scale_factor)
            spin_z = cudaq.spin.z(0)
            ev = []
            for args in all_args:
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
            shape: (*group, reps+1, n_params) or (*group, reps, 1) for real
        preacts_weight : torch.Tensor
            shape: (*group, reps)
        preacts_bias : torch.Tensor
            shape: (*group, reps)
        reps : int
        ansatz : str
            "pz_encoding", "pz", "rpz_encoding", "rpz", or "real"
        preacts_trainable : bool
        out_dim : int
        target : str, optional
            CUDA-Q target (e.g., "nvidia", "nvidia-mqpu", "qpp-cpu").
            Set before calling via cudaq.set_target().
        shots : int, optional
            Number of shots. None for exact statevector expectation.

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
        if machine is not None:
            target_kwargs["machine"] = machine
        cudaq.set_target(target, **target_kwargs)

    config = {
        "ansatz": ansatz,
        "preacts_trainable": preacts_trainable,
        "out_dim": out_dim,
        "shots": shots,
        "target": target,
        "parallel_qubits": parallel_qubits,
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
