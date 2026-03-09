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
QKAN solver for real quantum device execution via Qiskit and CUDA-Q.

Provides drop-in replacements for torch_exact_solver that execute circuits
on real quantum hardware or high-fidelity simulators:

  - qiskit_solver: IBM Quantum backends via Qiskit Runtime
  - cudaq_solver:  NVIDIA CUDA-Q backends (GPU-accelerated simulation or QPU)

These solvers follow the standard QKAN solver interface:
    solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs) -> Tensor

Gradient computation uses the parameter-shift rule, making these solvers
compatible with torch.autograd when training is needed.
"""

import math
from typing import Optional

import torch

# Qiskit availability
try:
    from qiskit import QuantumCircuit, transpile  # type: ignore
    from qiskit.quantum_info import SparsePauliOp  # type: ignore
    from qiskit.transpiler.preset_passmanagers import (  # type: ignore
        generate_preset_pass_manager,
    )

    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False

# Qiskit Runtime (Estimator V2) availability
try:
    from qiskit_ibm_runtime import EstimatorV2 as Estimator  # type: ignore
    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

    _QISKIT_RUNTIME_AVAILABLE = True
except ImportError:
    _QISKIT_RUNTIME_AVAILABLE = False

# Qiskit built-in StatevectorEstimator (Qiskit >= 1.0, no external backend needed)
try:
    from qiskit.primitives import StatevectorEstimator  # type: ignore

    _SV_ESTIMATOR_AVAILABLE = True
except ImportError:
    _SV_ESTIMATOR_AVAILABLE = False

# Qiskit Aer availability
try:
    from qiskit_aer import AerSimulator  # type: ignore

    _AER_AVAILABLE = True
except ImportError:
    _AER_AVAILABLE = False

# CUDA-Q availability
try:
    import cudaq  # type: ignore

    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Qiskit circuit builders
# ---------------------------------------------------------------------------


def _build_qiskit_pz_circuit(
    x_val: float,
    theta_vals: list[float],
    reps: int,
    encoded_x_vals: Optional[list[float]] = None,
) -> "QuantumCircuit":
    """
    Build pz_encoding circuit: H -> [RZ(θ₀) RY(θ₁) RZ(x)]^reps -> RZ(θ_f₀) RY(θ_f₁)

    theta_vals layout: [θ₀₀, θ₀₁, θ₁₀, θ₁₁, ..., θ_f₀, θ_f₁]  (2 params per layer + 2 final)
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    for l in range(reps):
        qc.rz(theta_vals[2 * l], 0)
        qc.ry(theta_vals[2 * l + 1], 0)
        enc = encoded_x_vals[l] if encoded_x_vals is not None else x_val
        qc.rz(enc, 0)
    qc.rz(theta_vals[2 * reps], 0)
    qc.ry(theta_vals[2 * reps + 1], 0)
    return qc


def _build_qiskit_rpz_circuit(
    encoded_x_vals: list[float],
    theta_vals: list[float],
    reps: int,
) -> "QuantumCircuit":
    """
    Build rpz_encoding circuit: H -> [RY(θ) RZ(encoded_x)]^reps -> RY(θ_final)

    theta_vals layout: [θ₀, θ₁, ..., θ_final]  (1 param per layer + 1 final)
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    for l in range(reps):
        qc.ry(theta_vals[l], 0)
        qc.rz(encoded_x_vals[l], 0)
    qc.ry(theta_vals[reps], 0)
    return qc


def _build_qiskit_real_circuit(
    x_val: float,
    theta_vals: list[float],
    reps: int,
    encoded_x_vals: Optional[list[float]] = None,
) -> "QuantumCircuit":
    """
    Build real ansatz circuit: H -> [X RY(θ) Z RY(x)]^reps

    theta_vals layout: [θ₀, θ₁, ...]  (1 param per layer, no final gate)
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    for l in range(reps):
        qc.x(0)
        qc.ry(theta_vals[l], 0)
        qc.z(0)
        enc = encoded_x_vals[l] if encoded_x_vals is not None else x_val
        qc.ry(enc, 0)
    return qc


# ---------------------------------------------------------------------------
# Parallel multi-qubit packing (Qiskit)
# ---------------------------------------------------------------------------


def _build_qiskit_parallel_circuit(
    single_circuits: list["QuantumCircuit"],
) -> "QuantumCircuit":
    """
    Pack N independent single-qubit circuits into one N-qubit circuit.

    Each single-qubit circuit is applied to a separate qubit, enabling
    parallel execution on a multi-qubit QPU.
    """
    n = len(single_circuits)
    qc = QuantumCircuit(n)
    for qubit_idx, sc in enumerate(single_circuits):
        for instruction in sc.data:
            gate = instruction.operation
            params = gate.params
            name = gate.name
            if name == "h":
                qc.h(qubit_idx)
            elif name == "x":
                qc.x(qubit_idx)
            elif name == "z":
                qc.z(qubit_idx)
            elif name == "rx":
                qc.rx(params[0], qubit_idx)
            elif name == "ry":
                qc.ry(params[0], qubit_idx)
            elif name == "rz":
                qc.rz(params[0], qubit_idx)
            else:
                raise ValueError(f"Unsupported gate '{name}' in parallel packing")
    return qc


def _make_parallel_observables(n_qubits: int) -> list["SparsePauliOp"]:
    """
    Create Z observables for each qubit in an N-qubit circuit.

    Returns a list of N SparsePauliOp, each measuring Z on one qubit.
    Qiskit uses little-endian ordering: qubit 0 is the rightmost character.
    E.g. for 3 qubits: [IIZ, IZI, ZII] for qubits 0, 1, 2 respectively.
    """
    observables = []
    for k in range(n_qubits):
        # Qiskit little-endian: qubit k is at string position (n-1-k) from the left
        pauli_str = "I" * (n_qubits - 1 - k) + "Z" + "I" * k
        observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
    return observables


# ---------------------------------------------------------------------------
# Qiskit solver
# ---------------------------------------------------------------------------


class _QiskitParamShift(torch.autograd.Function):
    """Autograd function using parameter-shift rule for Qiskit circuits."""

    @staticmethod
    def forward(ctx, x, theta, preacts_w, preacts_b, reps, config):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.config = config
        return _qiskit_evaluate(x, theta, preacts_w, preacts_b, reps, config)

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        reps = ctx.reps
        config = ctx.config
        shift = math.pi / 2

        # Gradient w.r.t. theta via parameter-shift rule
        grad_theta = torch.zeros_like(theta)
        flat_theta = theta.reshape(-1)
        for k in range(flat_theta.numel()):
            theta_plus = flat_theta.clone()
            theta_plus[k] += shift
            theta_minus = flat_theta.clone()
            theta_minus[k] -= shift

            f_plus = _qiskit_evaluate(
                x, theta_plus.reshape(theta.shape), preacts_w, preacts_b, reps, config
            )
            f_minus = _qiskit_evaluate(
                x, theta_minus.reshape(theta.shape), preacts_w, preacts_b, reps, config
            )
            grad_k = (f_plus - f_minus) / (2 * math.sin(shift))
            grad_theta.reshape(-1)[k] = (grad_output * grad_k).sum()

        # Gradient w.r.t. preacts_weight
        grad_pw = None
        if preacts_w.requires_grad:
            grad_pw = torch.zeros_like(preacts_w)
            flat_pw = preacts_w.reshape(-1)
            for k in range(flat_pw.numel()):
                pw_plus = flat_pw.clone()
                pw_plus[k] += shift
                pw_minus = flat_pw.clone()
                pw_minus[k] -= shift
                f_plus = _qiskit_evaluate(
                    x, theta, pw_plus.reshape(preacts_w.shape), preacts_b, reps, config
                )
                f_minus = _qiskit_evaluate(
                    x, theta, pw_minus.reshape(preacts_w.shape), preacts_b, reps, config
                )
                grad_pw.reshape(-1)[k] = (
                    grad_output * (f_plus - f_minus) / (2 * math.sin(shift))
                ).sum()

        # Gradient w.r.t. preacts_bias
        grad_pb = None
        if preacts_b.requires_grad:
            grad_pb = torch.zeros_like(preacts_b)
            flat_pb = preacts_b.reshape(-1)
            for k in range(flat_pb.numel()):
                pb_plus = flat_pb.clone()
                pb_plus[k] += shift
                pb_minus = flat_pb.clone()
                pb_minus[k] -= shift
                f_plus = _qiskit_evaluate(
                    x, theta, preacts_w, pb_plus.reshape(preacts_b.shape), reps, config
                )
                f_minus = _qiskit_evaluate(
                    x, theta, preacts_w, pb_minus.reshape(preacts_b.shape), reps, config
                )
                grad_pb.reshape(-1)[k] = (
                    grad_output * (f_plus - f_minus) / (2 * math.sin(shift))
                ).sum()

        return None, grad_theta, grad_pw, grad_pb, None, None


def _probe_max_pubs(est, probe_pubs, max_pubs):
    """
    Binary-search for the largest PUB batch the QPU accepts.

    Submits `probe_pubs[:max_pubs]` synchronously. On memory error (6073),
    halves and retries until a working size is found. Returns (result, max_pubs)
    where result is the successful job result for the probe batch.
    """
    while max_pubs >= 1:
        batch = probe_pubs[:max_pubs]
        try:
            job = est.run(batch)
            result = job.result()
            return result, max_pubs
        except Exception as e:
            err_str = str(e)
            if "6073" in err_str or "memory" in err_str.lower():
                old_max = max_pubs
                max_pubs = max(1, max_pubs // 2)
                if max_pubs == old_max:
                    raise  # can't go smaller than 1
                print(
                    f"  [qsolver] Job memory limit hit at {old_max} PUBs/job, "
                    f"trying {max_pubs}"
                )
            else:
                raise
    raise RuntimeError("Could not find a working PUB batch size")


def _submit_and_collect(est, all_pubs, all_chunk_sizes, max_pubs):
    """
    Submit PUBs with the largest batch size the QPU can handle.

    1. Probes with max_pubs (all PUBs if 0) synchronously to find the
       largest accepted batch size via binary search on memory errors.
    2. Submits all remaining batches asynchronously for max throughput.
    3. Collects results in order.

    Returns (expvals, actual_max_pubs) so callers can cache the working size.
    """
    n_total = len(all_pubs)
    if max_pubs <= 0:
        max_pubs = n_total
    expvals = [None] * n_total

    # Step 1: Probe with first batch to discover working max_pubs
    first_batch_size = min(max_pubs, n_total)
    first_batch = all_pubs[:first_batch_size]
    probe_result, max_pubs = _probe_max_pubs(est, first_batch, first_batch_size)

    # Collect probe results (first max_pubs PUBs)
    probed_count = min(max_pubs, n_total)
    for i in range(probed_count):
        evs = probe_result[i].data.evs
        expvals[i] = [float(v) for v in evs]

    # Step 2: Submit remaining batches asynchronously
    remaining_start = probed_count
    if remaining_start < n_total:
        jobs = []
        job_ranges = []
        for batch_start in range(remaining_start, n_total, max_pubs):
            batch_end = min(batch_start + max_pubs, n_total)
            job_pubs = all_pubs[batch_start:batch_end]
            jobs.append(est.run(job_pubs))
            job_ranges.append((batch_start, batch_end))

        n_jobs = len(jobs)
        print(f"  [qsolver] Submitting {n_jobs} async job(s), {max_pubs} PUBs/job")

        # Collect all async results
        for job, (batch_start, batch_end) in zip(jobs, job_ranges):
            result = job.result()
            for i, global_idx in enumerate(range(batch_start, batch_end)):
                evs = result[i].data.evs
                expvals[global_idx] = [float(v) for v in evs]

    # Flatten
    flat = []
    for ev_list in expvals:
        flat.extend(ev_list)
    return flat, max_pubs


# Module-level cache for the discovered max PUBs per backend
_MAX_PUBS_CACHE: dict = {}


def _qiskit_run_parallel(
    circuits,
    n_qubits,
    estimator,
    backend,
    optimization_level,
    shots,
    max_pubs_per_job=0,
):
    """
    Pack single-qubit circuits into multi-qubit batches and submit async.

    Groups `circuits` into chunks of `n_qubits`, packs each chunk into one
    multi-qubit circuit. Jobs are submitted asynchronously with automatic
    PUB batch sizing:

    - If `max_pubs_per_job` > 0, uses that as the initial batch size.
    - If `max_pubs_per_job` == 0 (default), starts with all PUBs in one job.
    - On memory error (6073), automatically halves and retries.
    - The discovered working batch size is cached per backend.
    """
    total = len(circuits)

    # Build all PUBs first
    all_pubs = []
    all_chunk_sizes = []

    if estimator is not None:
        for start in range(0, total, n_qubits):
            batch_circuits = circuits[start : start + n_qubits]
            chunk_size = len(batch_circuits)
            all_chunk_sizes.append(chunk_size)
            packed_qc = _build_qiskit_parallel_circuit(batch_circuits)
            chunk_obs = _make_parallel_observables(chunk_size)
            all_pubs.append((packed_qc, chunk_obs))

        initial_max = max_pubs_per_job if max_pubs_per_job > 0 else len(all_pubs)
        expvals, _ = _submit_and_collect(
            estimator, all_pubs, all_chunk_sizes, initial_max
        )
        return expvals

    elif backend is not None:
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=optimization_level
        )
        rt_estimator = Estimator(mode=backend)
        if shots is not None:
            rt_estimator.options.default_shots = shots

        for start in range(0, total, n_qubits):
            batch_circuits = circuits[start : start + n_qubits]
            chunk_size = len(batch_circuits)
            all_chunk_sizes.append(chunk_size)
            packed_qc = _build_qiskit_parallel_circuit(batch_circuits)
            isa_qc = pm.run(packed_qc)
            chunk_obs = _make_parallel_observables(chunk_size)
            isa_obs = [obs.apply_layout(isa_qc.layout) for obs in chunk_obs]
            all_pubs.append((isa_qc, isa_obs))

        # Use cached max or start with all PUBs
        cache_key = getattr(backend, "name", str(backend))
        initial_max = (
            max_pubs_per_job
            if max_pubs_per_job > 0
            else _MAX_PUBS_CACHE.get(cache_key, len(all_pubs))
        )
        expvals, actual_max = _submit_and_collect(
            rt_estimator, all_pubs, all_chunk_sizes, initial_max
        )
        _MAX_PUBS_CACHE[cache_key] = actual_max
        return expvals

    return []


def _qiskit_evaluate(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    config: dict,
) -> torch.Tensor:
    """
    Evaluate all circuits on the Qiskit backend and return expectation values.

    Returns shape: (batch_size, out_dim, in_dim)
    """
    batch, in_dim = x.shape
    ansatz = config["ansatz"]
    preacts_trainable = config["preacts_trainable"]
    out_dim = config["out_dim"]
    backend = config.get("backend", None)
    estimator = config.get("estimator", None)
    shots = config["shots"]
    optimization_level = config.get("optimization_level", 1)
    parallel_qubits = config.get("parallel_qubits", None)

    # Broadcast theta/preacts to (out_dim, in_dim, ...)
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

    # Move to CPU for circuit parameter extraction
    x_np = x.detach().cpu()
    theta_np = theta.detach().cpu()
    encoded_x_np = encoded_x.detach().cpu() if encoded_x is not None else None

    # Build circuits and observables
    circuits = []
    observables = []
    pauli_z = SparsePauliOp.from_list([("Z", 1.0)])

    for b in range(batch):
        for o in range(out_dim):
            for i in range(in_dim):
                if ansatz in ("pz_encoding", "pz"):
                    t = theta_np[o, i].reshape(-1).tolist()
                    enc_vals = None
                    if encoded_x_np is not None:
                        enc_vals = encoded_x_np[b, o, i].tolist()
                    qc = _build_qiskit_pz_circuit(float(x_np[b, i]), t, reps, enc_vals)
                elif ansatz in ("rpz_encoding", "rpz"):
                    t = theta_np[o, i].reshape(-1).tolist()
                    enc_vals = encoded_x_np[b, o, i].tolist()  # type: ignore
                    qc = _build_qiskit_rpz_circuit(enc_vals, t, reps)
                elif ansatz == "real":
                    t = theta_np[o, i].reshape(-1).tolist()
                    enc_vals = None
                    if encoded_x_np is not None:
                        enc_vals = encoded_x_np[b, o, i].tolist()
                    qc = _build_qiskit_real_circuit(
                        float(x_np[b, i]), t, reps, enc_vals
                    )
                else:
                    raise NotImplementedError(
                        f"Ansatz '{ansatz}' not supported by qiskit_solver"
                    )
                circuits.append(qc)
                observables.append(pauli_z)

    # Execute via the appropriate Estimator
    max_pubs = config.get("max_pubs_per_job", 0)
    if parallel_qubits and parallel_qubits > 1:
        # Pack independent 1-qubit circuits onto multi-qubit QPU
        expvals = _qiskit_run_parallel(
            circuits,
            parallel_qubits,
            estimator,
            backend,
            optimization_level,
            shots,
            max_pubs_per_job=max_pubs,
        )
    elif estimator is not None:
        pubs = list(zip(circuits, observables))
        job = estimator.run(pubs)
        result = job.result()
        expvals = [float(r.data.evs) for r in result]
    elif backend is not None:
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=optimization_level
        )
        isa_circuits = pm.run(circuits)
        isa_observables = [
            obs.apply_layout(qc.layout) for obs, qc in zip(observables, isa_circuits)
        ]
        rt_estimator = Estimator(mode=backend)
        if shots is not None:
            rt_estimator.options.default_shots = shots
        pubs = list(zip(isa_circuits, isa_observables))
        job = rt_estimator.run(pubs)
        result = job.result()
        expvals = [float(r.data.evs) for r in result]
    else:
        raise ValueError("No estimator or backend provided.")

    output = torch.tensor(expvals, dtype=x.dtype, device=x.device)
    return output.reshape(batch, out_dim, in_dim)


def qiskit_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Execute QKAN circuits on IBM Quantum backends via Qiskit Runtime.

    Drop-in replacement for torch_exact_solver. Circuits are built to match
    the exact gate sequences of each ansatz, then executed on the specified
    backend using Qiskit's Estimator primitive.

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
        backend : qiskit Backend
            Qiskit backend instance (e.g., AerSimulator(), or from QiskitRuntimeService)
        shots : int, optional
            Number of shots per circuit. None for exact expectation (statevector).
        optimization_level : int
            Transpiler optimization level (0-3), default: 1

    Returns
    -------
        torch.Tensor
            shape: (batch_size, out_dim, in_dim)
    """
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is required for qiskit_solver. "
            "Install with: pip install qiskit qiskit-ibm-runtime"
        )

    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    out_dim = kwargs.get("out_dim", x.shape[1])
    shots = kwargs.get("shots", None)
    optimization_level = kwargs.get("optimization_level", 1)
    parallel_qubits = kwargs.get("parallel_qubits", None)

    backend = kwargs.get("backend", None)
    estimator = kwargs.get("estimator", None)

    # Resolve execution mode: estimator > backend > StatevectorEstimator > AerSimulator
    if estimator is None and backend is None:
        if _SV_ESTIMATOR_AVAILABLE:
            estimator = StatevectorEstimator()
        elif _AER_AVAILABLE:
            backend = AerSimulator(method="statevector")
        else:
            raise ValueError(
                "No backend or estimator specified. Install qiskit >= 1.0 "
                "(for StatevectorEstimator), qiskit-aer, or qiskit-ibm-runtime."
            )

    # Auto-detect QPU size from backend if parallel_qubits="auto"
    if parallel_qubits == "auto" and backend is not None:
        parallel_qubits = backend.num_qubits

    max_pubs_per_job = kwargs.get("max_pubs_per_job", 0)

    config = {
        "ansatz": ansatz,
        "preacts_trainable": preacts_trainable,
        "out_dim": out_dim,
        "backend": backend,
        "estimator": estimator,
        "shots": shots,
        "optimization_level": optimization_level,
        "parallel_qubits": parallel_qubits,
        "max_pubs_per_job": max_pubs_per_job,
    }

    needs_grad = theta.requires_grad or x.requires_grad
    if preacts_trainable:
        needs_grad = (
            needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad
        )

    if needs_grad:
        return _QiskitParamShift.apply(
            x, theta, preacts_weight, preacts_bias, reps, config
        )
    else:
        return _qiskit_evaluate(x, theta, preacts_weight, preacts_bias, reps, config)


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


# Cache for compiled CUDA-Q kernels: {(ansatz, reps, preacts_trainable): kernel}
_CUDAQ_KERNEL_CACHE: dict = {}


def _get_cudaq_kernel(ansatz: str, reps: int, preacts_trainable: bool):
    """Get or build a cached CUDA-Q kernel."""
    key = (ansatz, reps, preacts_trainable)
    if key not in _CUDAQ_KERNEL_CACHE:
        if ansatz in ("pz_encoding", "pz"):
            if preacts_trainable:
                _CUDAQ_KERNEL_CACHE[key] = _build_cudaq_pz_preact_kernel(reps)
            else:
                _CUDAQ_KERNEL_CACHE[key] = _build_cudaq_pz_kernel(reps)
        elif ansatz in ("rpz_encoding", "rpz"):
            _CUDAQ_KERNEL_CACHE[key] = _build_cudaq_rpz_kernel(reps)
        elif ansatz == "real":
            if preacts_trainable:
                _CUDAQ_KERNEL_CACHE[key] = _build_cudaq_real_preact_kernel(reps)
            else:
                _CUDAQ_KERNEL_CACHE[key] = _build_cudaq_real_kernel(reps)
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
    all_args, ansatz, reps, preacts_trainable, n_qubits, shots_count
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

            par_kernel = _build_cudaq_parallel_pz_kernel(actual_n, reps)
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

            par_kernel = _build_cudaq_parallel_real_kernel(actual_n, reps)
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

            par_kernel = _build_cudaq_parallel_rpz_kernel(actual_n, reps)
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

    if parallel_qubits and parallel_qubits > 1:
        expvals = _cudaq_run_parallel(
            all_args,
            ansatz,
            reps,
            preacts_trainable,
            parallel_qubits,
            shots_count,
        )
    else:
        kernel = _get_cudaq_kernel(ansatz, reps, preacts_trainable)
        spin_z = cudaq.spin.z(0)
        expvals = []
        for args in all_args:
            if shots_count is not None:
                result = cudaq.observe(kernel, spin_z, *args, shots_count=shots_count)
            else:
                result = cudaq.observe(kernel, spin_z, *args)
            expvals.append(result.expectation())

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
