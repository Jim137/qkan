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


import math

import torch

from ..torch_qc import TorchGates
from .torch_exact import torch_exact_solver

# cuQuantum / opt_einsum availability
try:
    from cuquantum.tensornet import contract_path as _cutn_contract_path  # type: ignore

    _CUTN_AVAILABLE = True
except ImportError:
    _CUTN_AVAILABLE = False

try:
    from opt_einsum import contract_path as _oe_contract_path  # type: ignore

    _OE_AVAILABLE = True
except ImportError:
    _OE_AVAILABLE = False

_INV_SQRT2 = math.sqrt(2.0) / 2.0


def _combined_xryz_gate(theta, dtype=torch.complex64):
    """
    Analytically compute X @ RY(theta) @ Z as a single 2x2 gate (real ansatz).

    X @ RY(θ) @ Z = [[sin(θ/2), -cos(θ/2)],
                      [cos(θ/2),  sin(θ/2)]]
    """
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    return torch.stack(
        [
            torch.stack([sin, -cos]),
            torch.stack([cos, sin]),
        ]
    ).to(dtype)


def _combined_rz_ry_gate(alpha, beta, dtype=torch.complex64):
    """
    Fused gate for the pz ansatz sequence: first RZ(alpha), then RY(beta).

    Matrix product RY(β) @ RZ(α) (rightmost acts first on the state):

        [[cos(β/2)·e^{-iα/2}, -sin(β/2)·e^{+iα/2}],
         [sin(β/2)·e^{-iα/2},  cos(β/2)·e^{+iα/2}]]
    """
    cos = torch.cos(beta / 2)
    sin = torch.sin(beta / 2)
    exp_neg = torch.exp(-0.5j * alpha)
    exp_pos = torch.exp(0.5j * alpha)
    return torch.stack(
        [
            torch.stack([cos * exp_neg, -sin * exp_pos]),
            torch.stack([sin * exp_neg, cos * exp_pos]),
        ]
    ).to(dtype)


def _find_contraction_path(expression, operands):
    """Find optimal contraction path using cuQuantum or opt_einsum."""
    if _CUTN_AVAILABLE:
        path, _ = _cutn_contract_path(expression, *operands)
        return path
    if _OE_AVAILABLE:
        path, _ = _oe_contract_path(expression, *operands)
        return path
    return None


def _build_real_expression(reps, preacts_trainable):
    """
    Build einsum expression for the real-ansatz circuit.
    Circuit: |0> -> H -> [XRyZ(theta) -> RY(x)]^reps -> measure
    Operands per rep: 2 (fused gate + data encoding). No final gate.
    """
    chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_needed = 2 + 2 * reps
    if n_needed > 26:
        return None
    ci = 0
    q = chain[ci]
    ci += 1
    subs = [f"boi{q}"]
    q_new = chain[ci]
    ci += 1
    subs.append(f"{q_new}{q}")
    q = q_new
    for _ in range(reps):
        q_new = chain[ci]
        ci += 1
        subs.append(f"{q_new}{q}oi")
        q = q_new
        q_new = chain[ci]
        ci += 1
        subs.append(f"{q_new}{q}boi" if preacts_trainable else f"{q_new}{q}bi")
        q = q_new
    return ",".join(subs) + "->" + f"boi{q}"


def _build_pz_expression(reps, preacts_trainable):
    """
    Build einsum expression for the pz_encoding circuit.
    Circuit: |0> -> H -> [RzRy_fused(theta) -> RZ(x)]^reps -> RzRy_fused(theta_final) -> measure
    Operands per rep: 2 (fused gate + data encoding). +1 final gate.
    """
    chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_needed = 3 + 2 * reps  # H(2) + reps*2 + final(1)
    if n_needed > 26:
        return None
    ci = 0
    q = chain[ci]
    ci += 1
    subs = [f"boi{q}"]
    q_new = chain[ci]
    ci += 1
    subs.append(f"{q_new}{q}")
    q = q_new
    for _ in range(reps):
        q_new = chain[ci]
        ci += 1
        subs.append(f"{q_new}{q}oi")
        q = q_new
        q_new = chain[ci]
        ci += 1
        subs.append(f"{q_new}{q}boi" if preacts_trainable else f"{q_new}{q}bi")
        q = q_new
    # Final RzRy gate
    q_new = chain[ci]
    ci += 1
    subs.append(f"{q_new}{q}oi")
    q = q_new
    return ",".join(subs) + "->" + f"boi{q}"


def _build_rpz_expression(reps):
    """
    Build einsum expression for the rpz_encoding circuit.
    Circuit: |0> -> H -> [RY(theta) -> RZ(encoded_x)]^reps -> RY(theta_final) -> measure
    rpz always uses encoded_x so data gates are (batch, out, in).
    Operands per rep: 2 (RY + RZ_data). +1 final RY.
    """
    chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_needed = 3 + 2 * reps
    if n_needed > 26:
        return None
    ci = 0
    q = chain[ci]
    ci += 1
    subs = [f"boi{q}"]
    q_new = chain[ci]
    ci += 1
    subs.append(f"{q_new}{q}")
    q = q_new
    for _ in range(reps):
        q_new = chain[ci]
        ci += 1
        subs.append(f"{q_new}{q}oi")
        q = q_new
        q_new = chain[ci]
        ci += 1
        subs.append(f"{q_new}{q}boi")
        q = q_new
    # Final RY gate
    q_new = chain[ci]
    ci += 1
    subs.append(f"{q_new}{q}oi")
    q = q_new
    return ",".join(subs) + "->" + f"boi{q}"


# Cache for precompiled contraction plans: {(expression, shapes_tuple): plan}
_CUTN_PLAN_CACHE: dict = {}


def _precompile_plan(equation, operand_shapes):
    """
    Precompile a contraction plan: find the optimal path once and convert it
    into a list of pairwise einsum strings that can be executed without any
    string parsing in the hot path.

    Returns (steps, permute_str) or None if no path optimizer is available.
        steps: list of (idx1, idx2, einsum_str)
        permute_str: final transposition einsum or None
    """
    dummy_ops = [torch.empty(*s) for s in operand_shapes]
    path = _find_contraction_path(equation, dummy_ops)
    if path is None:
        return None

    input_str, output_str = equation.split("->")
    subscripts = input_str.split(",")
    final_indices = set(output_str)

    steps = []
    for i, j in path:
        idx1, idx2 = sorted((i, j))
        sub1, sub2 = subscripts[idx1], subscripts[idx2]

        remaining = [s for k, s in enumerate(subscripts) if k != idx1 and k != idx2]
        needed = set("".join(remaining)) | final_indices

        out_chars = [
            c for c in (sub1 + sub2) if c in (set(sub1) | set(sub2)) and c in needed
        ]
        out_sub = "".join(dict.fromkeys(out_chars))

        steps.append((idx1, idx2, f"{sub1},{sub2}->{out_sub}"))

        subscripts.pop(idx2)
        subscripts.pop(idx1)
        subscripts.append(out_sub)

    permute = f"{subscripts[0]}->{output_str}" if subscripts[0] != output_str else None
    return steps, permute


def _execute_plan(plan, operands):
    """Execute a precompiled contraction plan (hot path, no string parsing)."""
    steps, permute = plan
    ops = list(operands)
    for idx1, idx2, einsum_str in steps:
        new_op = torch.einsum(einsum_str, ops[idx1], ops[idx2])
        ops.pop(idx2)
        ops.pop(idx1)
        ops.append(new_op)
    if permute:
        return torch.einsum(permute, ops[0])
    return ops[0]


def _get_plan(expression, operands):
    """Get (or compute and cache) a contraction plan for the given expression."""
    key = (expression, tuple(op.shape for op in operands))
    if key not in _CUTN_PLAN_CACHE:
        _CUTN_PLAN_CACHE[key] = _precompile_plan(
            expression, [op.shape for op in operands]
        )
    return _CUTN_PLAN_CACHE[key]


def cutn_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Tensor network contraction solver using optimal contraction paths.

    Expresses the entire quantum circuit as a single tensor network and
    contracts it using an optimal path from cuQuantum or opt_einsum.
    The contraction plan is precompiled and cached so repeated forward
    calls pay no path-finding overhead.

    Supports ``pz_encoding`` (``pz``), ``rpz_encoding`` (``rpz``), and ``real`` ansatzes.
    Falls back to torch_exact_solver for unsupported ansatzes or reps > 11.

    Args
    ----
        x : torch.Tensor
            shape: (batch_size, in_dim)
        theta : torch.Tensor
            shape: (*group, reps+1, n_params_per_gate)
        preacts_weight : torch.Tensor
            shape: (*group, reps)
        preacts_bias : torch.Tensor
            shape: (*group, reps)
        reps : int
        ansatz : str
            options: "pz_encoding", "pz", "rpz_encoding", "rpz", "real"
        preacts_trainable : bool
        fast_measure : bool
        out_dim : int
        dtype : torch.dtype

    Returns
    -------
        torch.Tensor
            shape: (batch_size, out_dim, in_dim)
    """
    batch, in_dim = x.shape
    device = x.device
    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", in_dim)
    dtype = kwargs.get("dtype", torch.complex64)

    _SUPPORTED = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}
    if ansatz not in _SUPPORTED:
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )

    # Build whole-circuit expression based on ansatz
    if ansatz in ("pz_encoding", "pz"):
        expression = _build_pz_expression(reps, preacts_trainable)
    elif ansatz in ("rpz_encoding", "rpz"):
        expression = _build_rpz_expression(reps)
    else:  # real
        expression = _build_real_expression(reps, preacts_trainable)

    if expression is None:  # reps too large for single-char indices
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )

    # Broadcasting logic (same as torch_exact_solver)
    if len(theta.shape) != 4:
        theta = theta.unsqueeze(0)
    if theta.shape[1] != in_dim:
        repeat_out = out_dim
        repeat_in = in_dim // theta.shape[1] + 1
        theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]

    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
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

    # Build 2x2 H gate
    inv_sqrt2 = torch.tensor(_INV_SQRT2, device=device, dtype=dtype)
    h_gate = torch.stack(
        [
            torch.stack([inv_sqrt2, inv_sqrt2]),
            torch.stack([inv_sqrt2, -inv_sqrt2]),
        ]
    )

    # -- Build initial state --
    psi = torch.zeros(batch, out_dim, in_dim, 2, dtype=dtype, device=device)
    psi[:, :, :, 0] = 1.0

    # -- Build operands based on ansatz --
    operands = [psi, h_gate]

    if ansatz in ("pz_encoding", "pz"):
        # Circuit: H -> [RZ(θ₀)·RY(θ₁) -> RZ(x)]^reps -> RZ(θ_f₀)·RY(θ_f₁)
        if not preacts_trainable:
            rz_data = TorchGates.rz_gate(x, dtype=dtype)  # (2, 2, batch, in_dim)
        for l in range(reps):
            fused_l = _combined_rz_ry_gate(
                theta[:, :, l, 0], theta[:, :, l, 1], dtype=dtype
            )
            operands.append(fused_l)
            if not preacts_trainable:
                operands.append(rz_data)
            else:
                operands.append(TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype))
        # Final fused gate
        operands.append(
            _combined_rz_ry_gate(
                theta[:, :, reps, 0], theta[:, :, reps, 1], dtype=dtype
            )
        )

    elif ansatz in ("rpz_encoding", "rpz"):
        # Circuit: H -> [RY(θ) -> RZ(encoded_x)]^reps -> RY(θ_final)
        for l in range(reps):
            operands.append(TorchGates.ry_gate(theta[:, :, l, 0], dtype=dtype))
            operands.append(TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype))
        # Final RY gate
        operands.append(TorchGates.ry_gate(theta[:, :, reps, 0], dtype=dtype))

    else:  # real
        # Circuit: H -> [X·RY(θ)·Z -> RY(x)]^reps
        if not preacts_trainable:
            ry_data = TorchGates.ry_gate(x, dtype=dtype)
        for l in range(reps):
            operands.append(_combined_xryz_gate(theta[:, :, l, 0], dtype=dtype))
            if not preacts_trainable:
                operands.append(ry_data)
            else:
                operands.append(TorchGates.ry_gate(encoded_x[:, :, :, l], dtype=dtype))

    # Get cached contraction plan (path computed only once per shape config)
    plan = _get_plan(expression, operands)

    if plan is not None:
        psi = _execute_plan(plan, operands)
    else:
        psi = torch.einsum(expression, *operands)

    # Measurement (Z basis)
    return (
        psi[:, :, :, 0].abs() - psi[:, :, :, 1].abs()
        if fast_measure
        else torch.square(psi[:, :, :, 0].abs()) - torch.square(psi[:, :, :, 1].abs())
    )


# ---------------------------------------------------------------------------
# Flash (Triton-accelerated) solver
# ---------------------------------------------------------------------------

_SUPPORTED_FLASH_ANSATZES = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}
