# Copyright (c) 2026, Jiun-Cheng Jiang and the QKAN contributors. All rights reserved.
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
MLX fused Metal kernels for the QKAN solver on Apple silicon.

Same computation as the FlyDSL, Triton and CuTe kernels, written in the Metal
Shading Language and launched through ``mlx.core.fast.metal_kernel``:

  - One kernel evaluates the whole data re-uploading circuit. Each GPU thread
    owns one ``(o, i)`` circuit, keeps the trig of its thetas in registers and
    walks a chunk of the batch.
  - The source is generated per ``(ansatz, reps, ...)`` configuration, so the
    gate sequence is straight-line code with no branches on the hot path.
  - The backward kernel recomputes the forward sweep and walks the adjoint
    back through the gates, so no per-layer state is checkpointed in memory:
    the intermediate states live in registers, or for deep circuits are
    recovered by un-applying the gates. Each thread accumulates
    its parameter gradients over its batch chunk and writes one partial sum,
    which the host reduces; there are no atomics, so the result is
    deterministic.

A second pair of kernels (``mlx_contract_forward`` / ``mlx_contract_backward``)
fuses the layer's epilogue into the circuit: it returns
``sum_i postacts[b, o, i] * weight[o, i]`` and never materialises the
``(batch, out_dim, in_dim)`` tensor. The circuit kernels are compute-bound
either way; what this saves is PyTorch's own passes over that tensor
(multiply, reduce, and their backward), which on a wide layer cost about as
much as the circuit.

Tensors cross between PyTorch and MLX through DLPack. On the ``mps`` device
both frameworks address the same Metal buffer, so inputs, the
``(batch, out_dim, in_dim)`` output and the gradients are all shared without a
copy. The two frameworks submit work on separate command queues, so pending
PyTorch work is flushed before a launch (:func:`_to_mx`).

All math runs in float32; Metal has no float64.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
import threading
from typing import Any, NamedTuple

import torch

# MLX is imported on first use: ``import mlx.core`` costs over 100 ms and brings
# up Metal, which a process that never selects this solver should not pay for.
_MLX_INSTALLED = (
    sys.platform == "darwin" and importlib.util.find_spec("mlx") is not None
)

# Batch rows walked by one thread. A thread reads its thetas once and reuses
# them for the whole chunk, so a longer chunk amortises that setup (and, in
# the backward pass, means fewer partial sums for the host to reduce); a
# shorter one gives the GPU more threads. Measured on an M3, throughput is
# flat from 8 to 64 rows forward and from 16 to 128 backward, and drops by
# 1.5-3x at one row per thread.
_B_CHUNK_FWD = 16
_B_CHUNK_BWD = 32
# Below this many threads the launch is too small to fill the GPU, so the
# batch chunk shrinks until the grid is at least this large.
_MIN_THREADS = 4096
_THREADGROUP = 64
# The contraction kernels keep one running sum per batch row of their chunk in
# a thread-local array of this length (QK_MAX_ROWS in the Metal source).
_MAX_ROWS = 64
_B_CHUNK_CONTRACT_FWD = 32
_B_CHUNK_CONTRACT_BWD = 32
# Their threads also loop over a block of inputs (forward) or outputs
# (backward), so each does far more work than a thread of the plain kernels
# and a grid that only just fills the GPU leaves it waiting on the slowest
# ones. Measured on an M3: 4x the plain kernels' minimum is 5-15% faster on
# wide layers, and a single block per thread is up to 8x slower.
_MIN_THREADS_CONTRACT = 16384
# The backward kernels need the state after every gate. While they fit in
# registers the kernel keeps the states of its forward sweep; beyond this many
# live floats it un-applies each gate to the state instead, which costs one
# more gate application per gate but holds a single state. Measured on an M3
# with pz: keeping is 10-17% faster at reps=3 (44 floats), even at reps=5
# (68) and 45% slower at reps=8 (104).
_STATE_REGISTER_BUDGET = 56

_ANSATZ_CANONICAL = {
    "pz_encoding": "pz",
    "pz": "pz",
    "rpz_encoding": "rpz",
    "rpz": "rpz",
    "real": "real",
    "mix": "mix",
}

# (theta layers, thetas per layer) each ansatz reads, given ``reps``.
_THETA_LAYOUT = {
    "pz": (lambda reps: reps + 1, 2),
    "mix": (lambda reps: reps + 1, 2),
    "rpz": (lambda reps: reps + 1, 1),
    "real": (lambda reps: reps, 1),
}


class _Config(NamedTuple):
    """Everything that changes the generated kernel source."""

    ansatz: str  # canonical: pz / rpz / real / mix
    reps: int
    th_l: int  # theta.shape[-2]
    th_p: int  # theta.shape[-1]
    preacts: bool  # data angle is w * x + b instead of x
    fast: bool  # |a| - |b| instead of |a|^2 - |b|^2


# ---------------------------------------------------------------------------
# Metal source generation
#
# A complex state is a float4 (Re a, Im a, Re b, Im b) for a|0> + b|1>; the
# ``real`` ansatz never leaves the reals and uses a float2 (a, b). A rotation
# is passed as a float2 (cos(t/2), sin(t/2)).
# ---------------------------------------------------------------------------

_HEADER = """
#define QK_INV_SQRT2 0.70710678118654752f
#define QK_MAX_ROWS 64

inline float2 qk_cs(float t) {
    float c;
    float s = sincos(0.5f * t, c);
    return float2(c, s);
}

// Rz(t) = diag(e^{-it/2}, e^{+it/2})
inline float4 qk_rz(float4 p, float2 g) {
    return float4(p.x * g.x + p.y * g.y, p.y * g.x - p.x * g.y,
                  p.z * g.x - p.w * g.y, p.w * g.x + p.z * g.y);
}
inline float4 qk_rz_inv(float4 p, float2 g) {
    return float4(p.x * g.x - p.y * g.y, p.y * g.x + p.x * g.y,
                  p.z * g.x + p.w * g.y, p.w * g.x - p.z * g.y);
}
// Ry(t) = [[c, -s], [s, c]]
inline float4 qk_ry(float4 p, float2 g) {
    return float4(g.x * p.x - g.y * p.z, g.x * p.y - g.y * p.w,
                  g.y * p.x + g.x * p.z, g.y * p.y + g.x * p.w);
}
inline float4 qk_ry_inv(float4 p, float2 g) {
    return float4(g.x * p.x + g.y * p.z, g.x * p.y + g.y * p.w,
                  g.x * p.z - g.y * p.x, g.x * p.w - g.y * p.y);
}
// Rx(t) = [[c, -is], [-is, c]]
inline float4 qk_rx(float4 p, float2 g) {
    return float4(g.x * p.x + g.y * p.w, g.x * p.y - g.y * p.z,
                  g.x * p.z + g.y * p.y, g.x * p.w - g.y * p.x);
}
inline float4 qk_rx_inv(float4 p, float2 g) {
    return float4(g.x * p.x - g.y * p.w, g.x * p.y + g.y * p.z,
                  g.x * p.z - g.y * p.y, g.x * p.w + g.y * p.x);
}
// dL/dt of a rotation from the adjoint a and the state p *after* the gate:
// dR/dt = (-i/2) P R for the Pauli P, so dL/dt = Re<a, (-i/2) P p>.
inline float qk_rz_grad(float4 a, float4 p) {
    return 0.5f * ((a.x * p.y - a.y * p.x) - (a.z * p.w - a.w * p.z));
}
inline float qk_ry_grad(float4 a, float4 p) {
    return 0.5f * ((a.z * p.x + a.w * p.y) - (a.x * p.z + a.y * p.w));
}
inline float qk_rx_grad(float4 a, float4 p) {
    return 0.5f * ((a.x * p.w - a.y * p.z) + (a.z * p.y - a.w * p.x));
}
// 1/n, or 0 where the amplitude vanishes (the subgradient torch.abs uses).
inline float qk_abs_inv(float n) {
    return n > 1e-30f ? 1.0f / n : 0.0f;
}

// The real ansatz: H, X, Z and Ry are all real.
inline float2 qk_ryr(float2 p, float2 g) {
    return float2(g.x * p.x - g.y * p.y, g.y * p.x + g.x * p.y);
}
inline float2 qk_ryr_inv(float2 p, float2 g) {
    return float2(g.x * p.x + g.y * p.y, g.x * p.y - g.y * p.x);
}
inline float qk_ryr_grad(float2 a, float2 p) {
    return 0.5f * (a.y * p.x - a.x * p.y);
}
"""


def _circuit(cfg: _Config) -> list[tuple[str, tuple | None]]:
    """The gate sequence after the initial Hadamard.

    Each entry is ``(gate, angle)`` where the angle is ``("th", layer, index)``
    for a trainable theta, ``("enc", layer)`` for the data-encoding angle, or
    ``None`` for the fixed X and Z gates.
    """
    ops: list[tuple[str, tuple | None]] = []
    reps = cfg.reps
    for l in range(reps):
        if cfg.ansatz == "pz":
            ops += [("rz", ("th", l, 0)), ("ry", ("th", l, 1)), ("rz", ("enc", l))]
        elif cfg.ansatz == "rpz":
            ops += [("ry", ("th", l, 0)), ("rz", ("enc", l))]
        elif cfg.ansatz == "mix":
            ops += [("rz", ("th", l, 0)), ("rx", ("th", l, 1)), ("ry", ("enc", l))]
        else:  # real
            ops += [("x", None), ("ry", ("th", l, 0)), ("z", None), ("ry", ("enc", l))]
    if cfg.ansatz == "pz":
        ops += [("rz", ("th", reps, 0)), ("ry", ("th", reps, 1))]
    elif cfg.ansatz == "rpz":
        ops += [("ry", ("th", reps, 0))]
    elif cfg.ansatz == "mix":
        ops += [("rz", ("th", reps, 0)), ("rx", ("th", reps, 1))]
    return ops


def _angle_var(cfg: _Config, angle: tuple) -> str:
    if angle[0] == "th":
        return f"t{angle[1]}_{angle[2]}"
    return f"e{angle[1]}" if cfg.preacts else "e"


def _gate_fn(cfg: _Config, gate: str) -> str:
    return "qk_ryr" if cfg.ansatz == "real" else f"qk_{gate}"


def _used_thetas(cfg: _Config) -> list[tuple[int, int]]:
    return sorted(
        {(a[1], a[2]) for _, a in _circuit(cfg) if a is not None and a[0] == "th"}
    )


def _src_setup(cfg: _Config) -> str:
    """Thread indices and the per-circuit parameters, loaded once per thread."""
    lines = [
        "uint t = thread_position_in_grid.x;",
        "uint batch = (uint)x_shape[0];",
        "uint in_dim = (uint)x_shape[1];",
        "uint b0 = thread_position_in_grid.y * b_chunk;",
        "uint b1 = min(b0 + b_chunk, batch);",
        "uint i = t % in_dim;",
        _src_params(cfg),
    ]
    return "\n".join(lines)


def _keep_states(cfg: _Config) -> bool:
    """Whether the backward kernel keeps the forward sweep's states."""
    width = 2 if cfg.ansatz == "real" else 4
    rotations = sum(1 for _, angle in _circuit(cfg) if angle is not None)
    floats = rotations * width + (4 * cfg.reps if cfg.preacts else 0)
    return floats <= _STATE_REGISTER_BUDGET


def _src_params(cfg: _Config) -> str:
    """Trig of the thetas and the preacts of circuit ``t = o * in_dim + i``."""
    lines = []
    n_th = cfg.th_l * cfg.th_p
    for l, p in _used_thetas(cfg):
        lines.append(
            f"float2 t{l}_{p} = qk_cs(theta[t * {n_th}u + {l * cfg.th_p + p}u]);"
        )
    if cfg.preacts:
        for l in range(cfg.reps):
            lines.append(f"float w{l} = pw[t * {cfg.reps}u + {l}u];")
            lines.append(f"float c{l} = pb[t * {cfg.reps}u + {l}u];")
    return "\n".join(lines)


def _src_encode(cfg: _Config) -> str:
    """Data-gate trig for batch row ``b``."""
    if cfg.preacts:
        lines = ["float xv = x[b * in_dim + i];"]
        lines += [f"float2 e{l} = qk_cs(w{l} * xv + c{l});" for l in range(cfg.reps)]
        return "\n".join(lines)
    # Without preacts the angle is x itself: every circuit of column i shares
    # it, so the host passes cos(x/2), sin(x/2) once per (b, i).
    return "uint xe = 2u * (b * in_dim + i);\nfloat2 e = float2(x[xe], x[xe + 1u]);"


def _src_simulate(cfg: _Config, keep: bool = False) -> str:
    """Run the circuit for one sample, leaving the final state in ``p``.

    With ``keep`` every intermediate state stays alive as ``s0 .. sN`` (state
    ``s<k+1>`` follows gate ``k``) for the backward sweep to read.
    """
    vec = "float2" if cfg.ansatz == "real" else "float4"
    init = (
        "float2(QK_INV_SQRT2, QK_INV_SQRT2)"
        if cfg.ansatz == "real"
        else "float4(QK_INV_SQRT2, 0.0f, QK_INV_SQRT2, 0.0f)"
    )
    circuit = _circuit(cfg)
    if not keep:
        lines = [f"{vec} p = {init};"]
        for gate, angle in circuit:
            if gate == "x":
                lines.append("p = p.yx;")
            elif gate == "z":
                lines.append("p.y = -p.y;")
            else:
                assert angle is not None
                lines.append(f"p = {_gate_fn(cfg, gate)}(p, {_angle_var(cfg, angle)});")
        return "\n".join(lines)

    lines = [f"{vec} s0 = {init};"]
    for k, (gate, angle) in enumerate(circuit):
        if gate == "x":
            lines.append(f"{vec} s{k + 1} = s{k}.yx;")
        elif gate == "z":
            lines.append(f"{vec} s{k + 1} = float2(s{k}.x, -s{k}.y);")
        else:
            assert angle is not None
            lines.append(
                f"{vec} s{k + 1} = {_gate_fn(cfg, gate)}(s{k}, {_angle_var(cfg, angle)});"
            )
    lines.append(f"{vec} p = s{len(circuit)};")
    return "\n".join(lines)


def _src_measure(cfg: _Config) -> str:
    if cfg.ansatz == "real":
        return "fabs(p.x) - fabs(p.y)" if cfg.fast else "p.x * p.x - p.y * p.y"
    if cfg.fast:
        return "sqrt(p.x * p.x + p.y * p.y) - sqrt(p.z * p.z + p.w * p.w)"
    return "(p.x * p.x + p.y * p.y) - (p.z * p.z + p.w * p.w)"


def _src_measure_adjoint(cfg: _Config) -> str:
    """``a = dL/dp`` of the Z measurement, given ``go = dL/dout``."""
    if cfg.ansatz == "real":
        if cfg.fast:
            return "float2 a = float2(go * sign(p.x), -go * sign(p.y));"
        return "float2 a = float2(2.0f * go * p.x, -2.0f * go * p.y);"
    if cfg.fast:
        return (
            "float g0 = go * qk_abs_inv(sqrt(p.x * p.x + p.y * p.y));\n"
            "float g1 = -go * qk_abs_inv(sqrt(p.z * p.z + p.w * p.w));\n"
            "float4 a = float4(p.x * g0, p.y * g0, p.z * g1, p.w * g1);"
        )
    return "float4 a = 2.0f * go * float4(p.x, p.y, -p.z, -p.w);"


def _forward_source(cfg: _Config) -> str:
    return f"""
{_src_setup(cfg)}
for (uint b = b0; b < b1; ++b) {{
{_src_encode(cfg)}
{_src_simulate(cfg)}
out[(ulong)b * n_oi + t] = {_src_measure(cfg)};
}}
"""


def _src_grad_accumulators(cfg: _Config) -> str:
    """Zeroed per-circuit sums of the theta and preacts gradients."""
    acc = [f"float gt{l}_{p} = 0.0f;" for l, p in _used_thetas(cfg)]
    if cfg.preacts:
        acc += [f"float gw{l} = 0.0f;" for l in range(cfg.reps)]
        acc += [f"float gb{l} = 0.0f;" for l in range(cfg.reps)]
    return "\n".join(acc)


def _src_sweep(cfg: _Config, need_gx: bool) -> str:
    """Reverse sweep of the adjoint ``a`` from the final state.

    At each gate take the parameter gradient from the adjoint and the
    post-gate state, then un-apply the gate to the adjoint. The post-gate
    states are the ones the forward sweep kept (``_src_simulate(cfg,
    keep=True)``) when :func:`_keep_states` says they fit in registers;
    otherwise the gate is un-applied to the state ``p`` as well. Adds to the accumulators of
    :func:`_src_grad_accumulators` and leaves this sample's dL/dx in ``gx``.
    """
    keep = _keep_states(cfg)
    sweep = ["float ge;"] if cfg.preacts else []
    if need_gx:
        sweep.append("float gx = 0.0f;")
    circuit = _circuit(cfg)
    for k in range(len(circuit) - 1, -1, -1):
        gate, angle = circuit[k]
        post = f"s{k + 1}" if keep else "p"
        if gate == "x":
            sweep += ["a = a.yx;"] if keep else ["p = p.yx;", "a = a.yx;"]
            continue
        if gate == "z":
            sweep += ["a.y = -a.y;"] if keep else ["p.y = -p.y;", "a.y = -a.y;"]
            continue
        assert angle is not None
        fn = _gate_fn(cfg, gate)
        var = _angle_var(cfg, angle)
        if angle[0] == "th":
            sweep.append(f"gt{angle[1]}_{angle[2]} += {fn}_grad(a, {post});")
        elif cfg.preacts:
            l = angle[1]
            sweep += [
                f"ge = {fn}_grad(a, {post});",
                f"gw{l} += ge * xv;",
                f"gb{l} += ge;",
            ]
            if need_gx:
                sweep.append(f"gx += ge * w{l};")
        elif need_gx:
            sweep.append(f"gx += {fn}_grad(a, {post});")
        if not keep:
            sweep.append(f"p = {fn}_inv(p, {var});")
        if k > 0:
            sweep.append(f"a = {fn}_inv(a, {var});")
    return "\n".join(sweep)


def _src_store_grads(cfg: _Config) -> str:
    """Write the accumulators to partial-sum row ``row`` (a ulong).

    Thetas no gate reads get an explicit zero, so the buffer needs no
    zero-fill.
    """
    n_th = cfg.th_l * cfg.th_p
    used = set(_used_thetas(cfg))
    store = []
    for l in range(cfg.th_l):
        for p in range(cfg.th_p):
            val = f"gt{l}_{p}" if (l, p) in used else "0.0f"
            store.append(f"gtheta[row * {n_th}u + {l * cfg.th_p + p}u] = {val};")
    if cfg.preacts:
        for l in range(cfg.reps):
            store.append(f"gpw[row * {cfg.reps}u + {l}u] = gw{l};")
            store.append(f"gpb[row * {cfg.reps}u + {l}u] = gb{l};")
    return "\n".join(store)


def _backward_source(cfg: _Config, need_gx: bool) -> str:
    # Each batch chunk writes its partial sums to its own row.
    return f"""
{_src_setup(cfg)}
{_src_grad_accumulators(cfg)}
for (uint b = b0; b < b1; ++b) {{
{_src_encode(cfg)}
{_src_simulate(cfg, _keep_states(cfg))}
ulong oidx = (ulong)b * n_oi + t;
float go = gout[oidx];
{_src_measure_adjoint(cfg)}
{_src_sweep(cfg, need_gx)}
{"gxp[oidx] = gx;" if need_gx else ""}
}}
ulong row = (ulong)thread_position_in_grid.y * n_oi + t;
{_src_store_grads(cfg)}
"""


# The contraction kernels fold the layer's ``sum_i postacts * weight`` into the
# circuit. Summing over ``i`` and keeping a circuit's thetas in registers pull
# in different directions, so a thread owns one output ``o`` (forward) or one
# input ``i`` (backward), a chunk of batch rows and a block of the other
# index: it loops over the block, loading each circuit's parameters once, and
# keeps one running sum per batch row in a local array.


def _contract_forward_source(cfg: _Config) -> str:
    return f"""
uint o = thread_position_in_grid.x;
uint batch = (uint)x_shape[0];
uint in_dim = (uint)x_shape[1];
uint out_dim = (uint)weight_shape[0];
uint b0 = thread_position_in_grid.y * b_chunk;
uint rows = min(b_chunk, batch - b0);
uint i0 = thread_position_in_grid.z * i_block;
uint i1 = min(i0 + i_block, in_dim);
float acc[QK_MAX_ROWS];
for (uint r = 0; r < rows; ++r) {{ acc[r] = 0.0f; }}
for (uint i = i0; i < i1; ++i) {{
uint t = o * in_dim + i;
{_src_params(cfg)}
float wv = weight[t];
for (uint r = 0; r < rows; ++r) {{
uint b = b0 + r;
{_src_encode(cfg)}
{_src_simulate(cfg)}
acc[r] += wv * ({_src_measure(cfg)});
}}
}}
// One row of partial sums per block of inputs; the host adds the blocks.
ulong base = ((ulong)thread_position_in_grid.z * batch + b0) * out_dim + o;
for (uint r = 0; r < rows; ++r) {{ out[base + (ulong)r * out_dim] = acc[r]; }}
"""


def _contract_backward_source(cfg: _Config, need_gx: bool, need_gw: bool) -> str:
    gx_init = (
        "float gxa[QK_MAX_ROWS];\nfor (uint r = 0; r < rows; ++r) { gxa[r] = 0.0f; }"
        if need_gx
        else ""
    )
    gx_store = (
        "ulong xb = ((ulong)thread_position_in_grid.z * batch + b0) * in_dim + i;\n"
        "for (uint r = 0; r < rows; ++r) { gxp[xb + (ulong)r * in_dim] = gxa[r]; }"
        if need_gx
        else ""
    )
    return f"""
uint i = thread_position_in_grid.x;
uint batch = (uint)x_shape[0];
uint in_dim = (uint)x_shape[1];
uint out_dim = (uint)weight_shape[0];
uint b0 = thread_position_in_grid.y * b_chunk;
uint rows = min(b_chunk, batch - b0);
uint o0 = thread_position_in_grid.z * o_block;
uint o1 = min(o0 + o_block, out_dim);
{gx_init}
for (uint o = o0; o < o1; ++o) {{
uint t = o * in_dim + i;
{_src_params(cfg)}
float wv = weight[t];
{_src_grad_accumulators(cfg)}
{"float gwa = 0.0f;" if need_gw else ""}
for (uint r = 0; r < rows; ++r) {{
uint b = b0 + r;
{_src_encode(cfg)}
{_src_simulate(cfg, _keep_states(cfg))}
float gs = gout[(ulong)b * out_dim + o];
{f"gwa += gs * ({_src_measure(cfg)});" if need_gw else ""}
float go = gs * wv;
{_src_measure_adjoint(cfg)}
{_src_sweep(cfg, need_gx)}
{"gxa[r] += gx;" if need_gx else ""}
}}
// Parameter gradients: one partial sum per batch chunk and circuit.
ulong row = ((ulong)thread_position_in_grid.y * out_dim + o) * in_dim + i;
{_src_store_grads(cfg)}
{"gweight[row] = gwa;" if need_gw else ""}
}}
// dL/dx: one row of partial sums per block of outputs.
{gx_store}
"""


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

_COMPILE_LOCK = threading.Lock()


def _mx() -> Any:
    """``mlx.core``, imported on first use."""
    import mlx.core as mx  # type: ignore[import-not-found, import-untyped]

    return mx


@functools.lru_cache(maxsize=None)
def _kernels_available() -> bool:
    """Whether MLX can launch custom Metal kernels on this machine."""
    if not _MLX_INSTALLED:
        return False
    try:
        mx = _mx()
        return bool(mx.metal.is_available()) and hasattr(mx.fast, "metal_kernel")
    except Exception:
        return False


def _kernel_name(kind: str, cfg: _Config, need_gx: bool = False) -> str:
    return (
        f"qkan_{kind}_{cfg.ansatz}_r{cfg.reps}_l{cfg.th_l}p{cfg.th_p}"
        f"_{'pre' if cfg.preacts else 'raw'}_{'fast' if cfg.fast else 'born'}"
        f"{'_gx' if need_gx else ''}"
    )


@functools.lru_cache(maxsize=None)
def _forward_kernel(cfg: _Config):
    mx = _mx()
    inputs = ["x", "theta"] + (["pw", "pb"] if cfg.preacts else [])
    with _COMPILE_LOCK:
        return mx.fast.metal_kernel(
            name=_kernel_name("fwd", cfg),
            input_names=[*inputs, "n_oi", "b_chunk"],
            output_names=["out"],
            source=_forward_source(cfg),
            header=_HEADER,
        )


@functools.lru_cache(maxsize=None)
def _backward_kernel(cfg: _Config, need_gx: bool):
    mx = _mx()
    inputs = ["x", "theta"] + (["pw", "pb"] if cfg.preacts else [])
    outputs = ["gtheta"] + (["gpw", "gpb"] if cfg.preacts else [])
    if need_gx:
        outputs.append("gxp")
    with _COMPILE_LOCK:
        return mx.fast.metal_kernel(
            name=_kernel_name("bwd", cfg, need_gx),
            input_names=[*inputs, "gout", "n_oi", "b_chunk"],
            output_names=outputs,
            source=_backward_source(cfg, need_gx),
            header=_HEADER,
        )


@functools.lru_cache(maxsize=None)
def _contract_forward_kernel(cfg: _Config):
    mx = _mx()
    inputs = ["x", "theta"] + (["pw", "pb"] if cfg.preacts else [])
    with _COMPILE_LOCK:
        return mx.fast.metal_kernel(
            name=_kernel_name("cfwd", cfg),
            input_names=[*inputs, "weight", "b_chunk", "i_block"],
            output_names=["out"],
            source=_contract_forward_source(cfg),
            header=_HEADER,
        )


@functools.lru_cache(maxsize=None)
def _contract_backward_kernel(cfg: _Config, need_gx: bool, need_gw: bool):
    mx = _mx()
    inputs = ["x", "theta"] + (["pw", "pb"] if cfg.preacts else [])
    outputs = ["gtheta"] + (["gpw", "gpb"] if cfg.preacts else [])
    if need_gw:
        outputs.append("gweight")
    if need_gx:
        outputs.append("gxp")
    with _COMPILE_LOCK:
        return mx.fast.metal_kernel(
            name=_kernel_name("cbwd", cfg, need_gx) + ("_gw" if need_gw else ""),
            input_names=[*inputs, "weight", "gout", "b_chunk", "o_block"],
            output_names=outputs,
            source=_contract_backward_source(cfg, need_gx, need_gw),
            header=_HEADER,
        )


def _batch_chunk(batch: int, n_oi: int, target: int) -> int:
    """Batch rows per thread: ``target``, shrunk while the grid is too small."""
    chunk = max(1, min(target, batch))
    while chunk > 1 and n_oi * -(-batch // chunk) < _MIN_THREADS:
        chunk //= 2
    return chunk


# ---------------------------------------------------------------------------
# MLX-array level forward / backward
# ---------------------------------------------------------------------------


def _data_input(mx, x, cfg: _Config):
    """The kernel's ``x`` input: x itself with preacts, else (cos, sin)(x/2)."""
    if cfg.preacts:
        return x
    half = x * 0.5
    return mx.stack([mx.cos(half), mx.sin(half)], axis=-1)


def _forward_arrays(mx, x, theta, pw, pb, cfg: _Config):
    """Lazy ``(batch, out_dim, in_dim)`` MLX array of the circuit outputs."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    n_oi = out_dim * in_dim
    chunk = _batch_chunk(batch, n_oi, _B_CHUNK_FWD)
    inputs = [_data_input(mx, x, cfg), theta]
    if cfg.preacts:
        inputs += [pw, pb]
    inputs += [mx.array(n_oi, dtype=mx.uint32), mx.array(chunk, dtype=mx.uint32)]
    (out,) = _forward_kernel(cfg)(
        inputs=inputs,
        grid=(n_oi, -(-batch // chunk), 1),
        threadgroup=(min(_THREADGROUP, n_oi), 1, 1),
        output_shapes=[(batch, out_dim, in_dim)],
        output_dtypes=[mx.float32],
    )
    return out


def _backward_arrays(mx, x, theta, pw, pb, gout, cfg: _Config, need_gx: bool):
    """Lazy MLX gradients ``(grad_x, grad_theta, grad_pw, grad_pb)``.

    ``grad_x`` is None unless ``need_gx``; the preacts gradients are None
    unless the configuration encodes the data through them.
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    n_oi = out_dim * in_dim
    chunk = _batch_chunk(batch, n_oi, _B_CHUNK_BWD)
    n_chunks = -(-batch // chunk)
    inputs = [_data_input(mx, x, cfg), theta]
    if cfg.preacts:
        inputs += [pw, pb]
    inputs += [
        gout,
        mx.array(n_oi, dtype=mx.uint32),
        mx.array(chunk, dtype=mx.uint32),
    ]
    shapes: list[tuple[int, ...]] = [(n_chunks, out_dim, in_dim, cfg.th_l, cfg.th_p)]
    if cfg.preacts:
        shapes += [(n_chunks, out_dim, in_dim, cfg.reps)] * 2
    if need_gx:
        shapes.append((batch, out_dim, in_dim))
    outs = _backward_kernel(cfg, need_gx)(
        inputs=inputs,
        grid=(n_oi, n_chunks, 1),
        threadgroup=(min(_THREADGROUP, n_oi), 1, 1),
        output_shapes=shapes,
        output_dtypes=[mx.float32] * len(shapes),
    )
    grad_theta = outs[0].sum(axis=0)
    grad_pw = outs[1].sum(axis=0) if cfg.preacts else None
    grad_pb = outs[2].sum(axis=0) if cfg.preacts else None
    grad_x = outs[-1].sum(axis=1) if need_gx else None
    return grad_x, grad_theta, grad_pw, grad_pb


def _contract_launch(owned: int, batch: int, looped: int, target: int):
    """``(chunk, block)`` of a contraction launch.

    A thread owns one of ``owned`` indices and ``chunk`` batch rows and loops
    over a block of ``block`` of the ``looped`` indices. The loop is split
    into more blocks, and then the chunk shortened, while the grid is too
    small to fill the GPU: a GPU thread is slow on its own, so a few threads
    with long loops lose to many threads plus the extra partial sums the
    host then has to add, even on a tiny layer.
    """
    chunk = max(1, min(target, batch, _MAX_ROWS))
    threads = owned * -(-batch // chunk)
    n_blocks = min(looped, max(1, -(-_MIN_THREADS_CONTRACT // threads)))
    block = -(-looped // n_blocks)
    n_blocks = -(-looped // block)
    while chunk > 1 and owned * -(-batch // chunk) * n_blocks < _MIN_THREADS_CONTRACT:
        chunk //= 2
    return chunk, block


def _sum_blocks(part, n_blocks: int):
    """Reduce the leading per-block axis of a partial-sum output."""
    return part.reshape(part.shape[1:]) if n_blocks == 1 else part.sum(axis=0)


def _contract_forward_arrays(mx, x, theta, pw, pb, weight, cfg: _Config):
    """Lazy ``(batch, out_dim)`` array of ``sum_i postacts * weight``."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    chunk, block = _contract_launch(out_dim, batch, in_dim, _B_CHUNK_CONTRACT_FWD)
    n_blocks = -(-in_dim // block)
    inputs = [_data_input(mx, x, cfg), theta]
    if cfg.preacts:
        inputs += [pw, pb]
    inputs += [
        weight,
        mx.array(chunk, dtype=mx.uint32),
        mx.array(block, dtype=mx.uint32),
    ]
    (out,) = _contract_forward_kernel(cfg)(
        inputs=inputs,
        grid=(out_dim, -(-batch // chunk), n_blocks),
        threadgroup=(min(_THREADGROUP, out_dim), 1, 1),
        output_shapes=[(n_blocks, batch, out_dim)],
        output_dtypes=[mx.float32],
    )
    return _sum_blocks(out, n_blocks)


def _contract_backward_arrays(
    mx, x, theta, pw, pb, weight, gout, cfg: _Config, need_gx: bool, need_gw: bool
):
    """Lazy gradients ``(grad_x, grad_theta, grad_pw, grad_pb, grad_weight)``.

    ``gout`` is ``(batch, out_dim)``. ``grad_x`` and ``grad_weight`` are None
    unless asked for; the preacts gradients are None unless the configuration
    encodes the data through them.
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    chunk, block = _contract_launch(in_dim, batch, out_dim, _B_CHUNK_CONTRACT_BWD)
    n_chunks = -(-batch // chunk)
    n_blocks = -(-out_dim // block)
    inputs = [_data_input(mx, x, cfg), theta]
    if cfg.preacts:
        inputs += [pw, pb]
    inputs += [
        weight,
        gout,
        mx.array(chunk, dtype=mx.uint32),
        mx.array(block, dtype=mx.uint32),
    ]
    shapes: list[tuple[int, ...]] = [(n_chunks, out_dim, in_dim, cfg.th_l, cfg.th_p)]
    if cfg.preacts:
        shapes += [(n_chunks, out_dim, in_dim, cfg.reps)] * 2
    if need_gw:
        shapes.append((n_chunks, out_dim, in_dim))
    if need_gx:
        shapes.append((n_blocks, batch, in_dim))
    outs = list(
        _contract_backward_kernel(cfg, need_gx, need_gw)(
            inputs=inputs,
            grid=(in_dim, n_chunks, n_blocks),
            threadgroup=(min(_THREADGROUP, in_dim), 1, 1),
            output_shapes=shapes,
            output_dtypes=[mx.float32] * len(shapes),
        )
    )
    grad_x = _sum_blocks(outs.pop(), n_blocks) if need_gx else None
    grad_weight = outs.pop().sum(axis=0) if need_gw else None
    grad_theta = outs[0].sum(axis=0)
    grad_pw = outs[1].sum(axis=0) if cfg.preacts else None
    grad_pb = outs[2].sum(axis=0) if cfg.preacts else None
    return grad_x, grad_theta, grad_pw, grad_pb, grad_weight


# ---------------------------------------------------------------------------
# PyTorch <-> MLX
# ---------------------------------------------------------------------------


def _to_mx(mx, *tensors: torch.Tensor | None) -> list:
    """Float32 MLX arrays over ``tensors``, sharing their memory where possible.

    PyTorch and MLX submit to separate Metal command queues, which Metal does
    not order against each other, so a kernel launched by MLX could read a
    shared buffer before PyTorch's queued writes to it have run. All PyTorch
    work is therefore flushed after the last tensor has been prepared and
    before any of them is handed over: the casts and copies below queue MPS
    work of their own.
    """
    prepared: list[torch.Tensor | None] = []
    for t in tensors:
        if t is not None:
            t = t.detach()
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            if t.device.type not in ("mps", "cpu"):
                t = t.cpu()
            if not t.is_contiguous():
                t = t.contiguous()
        prepared.append(t)
    if any(t is not None and t.device.type == "mps" for t in prepared):
        torch.mps.synchronize()
    return [None if t is None else _import(mx, t) for t in prepared]


def _import(mx, t: torch.Tensor):
    if hasattr(mx, "from_dlpack"):
        try:
            return mx.from_dlpack(t)
        except Exception:
            pass
    return mx.array(t.cpu().numpy())


def _to_torch(mx, a, device: torch.device) -> torch.Tensor:
    """A torch tensor over the evaluated MLX array ``a`` on ``device``."""
    if device.type == "mps":
        try:
            # Zero-copy: the tensor addresses MLX's Metal buffer and keeps the
            # array alive through the DLPack capsule.
            return torch.from_dlpack(a)
        except Exception:
            pass
    import numpy as np

    out = torch.from_numpy(np.array(a, copy=False))
    return out if device.type == "cpu" else out.to(device)


def _config(theta: torch.Tensor, reps: int, ansatz: str, preacts: bool, fast: bool):
    canonical = _ANSATZ_CANONICAL[ansatz]
    layers, per_layer = _THETA_LAYOUT[canonical]
    if theta.dim() != 4 or theta.shape[2] < layers(reps) or theta.shape[3] < per_layer:
        raise ValueError(
            f"the mlx solver expects theta of shape (out_dim, in_dim, "
            f">={layers(reps)}, >={per_layer}) for ansatz {ansatz!r} with "
            f"reps={reps}, got {tuple(theta.shape)}"
        )
    # rpz always encodes through the preacts (its bias is trainable).
    return _Config(
        ansatz=canonical,
        reps=int(reps),
        th_l=int(theta.shape[2]),
        th_p=int(theta.shape[3]),
        preacts=bool(preacts) or canonical == "rpz",
        fast=bool(fast),
    )


def mlx_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    reps: int,
    ansatz: str,
    preacts_trainable: bool,
    fast_measure: bool,
) -> torch.Tensor:
    """Circuit outputs, float32, shape ``(batch, theta.shape[0], in_dim)``.

    ``theta`` is ``(out_dim, in_dim, layers, per_layer)`` and the preacts are
    ``(out_dim, in_dim, reps)``, already broadcast by the caller.
    """
    cfg = _config(theta, reps, ansatz, preacts_trainable, fast_measure)
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    if batch == 0 or out_dim * in_dim == 0:
        return x.new_zeros((batch, out_dim, in_dim), dtype=torch.float32)
    mx = _mx()
    x_a, theta_a, pw_a, pb_a = _to_mx(
        mx,
        x,
        theta,
        preacts_w if cfg.preacts else None,
        preacts_b if cfg.preacts else None,
    )
    out = _forward_arrays(mx, x_a, theta_a, pw_a, pb_a, cfg)
    mx.eval(out)
    return _to_torch(mx, out, x.device)


def mlx_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    grad_output: torch.Tensor,
    reps: int,
    ansatz: str,
    preacts_trainable: bool,
    fast_measure: bool,
    need_grad_x: bool = True,
):
    """Float32 gradients ``(grad_x, grad_theta, grad_pw, grad_pb)``.

    ``grad_x`` is None when ``need_grad_x`` is False (the kernel then skips
    it); the preacts gradients are None when the data is not encoded through
    them.
    """
    cfg = _config(theta, reps, ansatz, preacts_trainable, fast_measure)
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    if batch == 0 or out_dim * in_dim == 0:
        zeros = lambda t: torch.zeros_like(t, dtype=torch.float32)  # noqa: E731
        return (
            zeros(x) if need_grad_x else None,
            zeros(theta),
            zeros(preacts_w) if cfg.preacts else None,
            zeros(preacts_b) if cfg.preacts else None,
        )
    mx = _mx()
    x_a, theta_a, pw_a, pb_a, gout_a = _to_mx(
        mx,
        x,
        theta,
        preacts_w if cfg.preacts else None,
        preacts_b if cfg.preacts else None,
        grad_output,
    )
    grads = _backward_arrays(mx, x_a, theta_a, pw_a, pb_a, gout_a, cfg, need_grad_x)
    mx.eval(*[g for g in grads if g is not None])
    return tuple(None if g is None else _to_torch(mx, g, x.device) for g in grads)


def mlx_contract_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    weight: torch.Tensor,
    reps: int,
    ansatz: str,
    preacts_trainable: bool,
    fast_measure: bool,
) -> torch.Tensor:
    """``sum_i postacts[b, o, i] * weight[o, i]``, float32, ``(batch, out_dim)``.

    The circuit outputs are summed inside the kernel and never written out.
    ``theta`` is ``(out_dim, in_dim, layers, per_layer)``, the preacts are
    ``(out_dim, in_dim, reps)`` and ``weight`` is ``(out_dim, in_dim)``.
    """
    cfg = _config(theta, reps, ansatz, preacts_trainable, fast_measure)
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    if batch == 0 or out_dim * in_dim == 0:
        return x.new_zeros((batch, out_dim), dtype=torch.float32)
    mx = _mx()
    x_a, theta_a, pw_a, pb_a, weight_a = _to_mx(
        mx,
        x,
        theta,
        preacts_w if cfg.preacts else None,
        preacts_b if cfg.preacts else None,
        weight,
    )
    out = _contract_forward_arrays(mx, x_a, theta_a, pw_a, pb_a, weight_a, cfg)
    mx.eval(out)
    return _to_torch(mx, out, x.device)


def mlx_contract_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    reps: int,
    ansatz: str,
    preacts_trainable: bool,
    fast_measure: bool,
    need_grad_x: bool = True,
    need_grad_weight: bool = True,
):
    """Float32 ``(grad_x, grad_theta, grad_pw, grad_pb, grad_weight)``.

    ``grad_output`` is ``(batch, out_dim)``. ``grad_x`` and ``grad_weight``
    are None when not needed (the kernel then skips them); the preacts
    gradients are None when the data is not encoded through them.
    """
    cfg = _config(theta, reps, ansatz, preacts_trainable, fast_measure)
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    if batch == 0 or out_dim * in_dim == 0:
        zeros = lambda t: torch.zeros_like(t, dtype=torch.float32)  # noqa: E731
        return (
            zeros(x) if need_grad_x else None,
            zeros(theta),
            zeros(preacts_w) if cfg.preacts else None,
            zeros(preacts_b) if cfg.preacts else None,
            zeros(weight) if need_grad_weight else None,
        )
    mx = _mx()
    x_a, theta_a, pw_a, pb_a, weight_a, gout_a = _to_mx(
        mx,
        x,
        theta,
        preacts_w if cfg.preacts else None,
        preacts_b if cfg.preacts else None,
        weight,
        grad_output,
    )
    grads = _contract_backward_arrays(
        mx,
        x_a,
        theta_a,
        pw_a,
        pb_a,
        weight_a,
        gout_a,
        cfg,
        need_grad_x,
        need_grad_weight,
    )
    mx.eval(*[g for g in grads if g is not None])
    return tuple(None if g is None else _to_torch(mx, g, x.device) for g in grads)
