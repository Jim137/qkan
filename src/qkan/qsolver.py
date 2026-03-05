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
                    enc_vals = encoded_x_np[b, o, i].tolist()
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
    if estimator is not None:
        # User-provided estimator (StatevectorEstimator, Runtime EstimatorV2, etc.)
        pubs = list(zip(circuits, observables))
        job = estimator.run(pubs)
        result = job.result()
    elif backend is not None:
        # IBM Runtime backend: pass manager + EstimatorV2
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
    else:
        raise ValueError("No estimator or backend provided.")

    # Extract expectation values
    expvals = []
    for pub_result in result:
        expvals.append(float(pub_result.data.evs))

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

    config = {
        "ansatz": ansatz,
        "preacts_trainable": preacts_trainable,
        "out_dim": out_dim,
        "backend": backend,
        "estimator": estimator,
        "shots": shots,
        "optimization_level": optimization_level,
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

    kernel = _get_cudaq_kernel(ansatz, reps, preacts_trainable)
    spin_z = cudaq.spin.z(0)

    expvals = []
    for b in range(batch):
        for o in range(out_dim):
            for i in range(in_dim):
                t = theta_np[o, i].reshape(-1).tolist()

                if ansatz in ("pz_encoding", "pz"):
                    if preacts_trainable:
                        enc_vals = encoded_x_np[b, o, i].tolist()
                        args = (enc_vals, t)
                    else:
                        args = (float(x_np[b, i]), t)
                elif ansatz in ("rpz_encoding", "rpz"):
                    enc_vals = encoded_x_np[b, o, i].tolist()
                    args = (enc_vals, t)
                elif ansatz == "real":
                    if preacts_trainable:
                        enc_vals = encoded_x_np[b, o, i].tolist()
                        args = (enc_vals, t)
                    else:
                        args = (float(x_np[b, i]), t)
                else:
                    raise NotImplementedError

                if shots_count is not None:
                    result = cudaq.observe(
                        kernel, spin_z, *args, shots_count=shots_count
                    )
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
