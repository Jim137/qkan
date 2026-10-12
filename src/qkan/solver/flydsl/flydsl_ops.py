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
FlyDSL (ROCm) kernels for QKAN quantum circuit simulation.

FlyDSL is AMD's Python DSL that compiles through the MLIR ``fly`` dialect to
ROCDL, so this backend targets AMD Instinct / Radeon GPUs running a ROCm build
of PyTorch (where ``torch.cuda`` is backed by HIP).

Kernel design
-------------
Each thread owns one ``(o, i)`` circuit and a contiguous chunk of the batch:

  * ``reps`` is a compile-time constant, so the circuit is fully unrolled and
    the sin/cos of every theta (and of the preacts weight/bias) is computed
    once per thread and kept in registers for the whole batch chunk — the
    FlyDSL counterpart of CuTe's shared-memory trig cache.
  * Adjacent threads own adjacent ``(o, i)``, so the per-sample output
    ``out[b, o, i]`` and ``grad_output[b, o, i]`` accesses are coalesced.
  * The backward pass needs no state checkpoints. Every gate is unitary, so
    after recomputing the final state the kernel walks the circuit in reverse,
    un-applying each gate to recover the state it saw (and applying the same
    inverse to the adjoint). All of it stays in registers.
  * theta / preacts gradients are summed in registers over the batch chunk and
    flushed with one atomic add per parameter per thread. ``grad_x`` is written
    per ``(b, o, i)`` and reduced over ``o`` with a single ``torch.sum``,
    which keeps it deterministic.

All math runs in float32. The forward output is float32 or bfloat16 (bf16 I/O
halves the dominant ``(batch, out_dim, in_dim)`` write); the backward pass
always runs in float32.
"""

from __future__ import annotations

import math

import torch

try:
    import flydsl.compiler as flyc  # type: ignore[import-not-found, import-untyped]
    import flydsl.expr as fx  # type: ignore[import-not-found, import-untyped]
    from flydsl.expr import (  # type: ignore[import-not-found, import-untyped]
        const_expr,
        range_constexpr,
    )

    # FlyDSL emits AMDGPU code only; a CUDA build of torch cannot run it.
    _FLYDSL_KERNELS_AVAILABLE = torch.version.hip is not None
except ImportError:
    _FLYDSL_KERNELS_AVAILABLE = False


_INV_SQRT2 = 0.7071067811865476

# Threads per workgroup: one CDNA wavefront (wave64).
_BLOCK = 64

# Threads to keep in flight per compute unit before batch chunks grow longer.
_THREADS_PER_CU = 2048

# Canonical ansatz names used as the kernels' compile-time switch.
_ANSATZ_CANONICAL = {
    "pz_encoding": "pz",
    "pz": "pz",
    "rpz_encoding": "rpz",
    "rpz": "rpz",
    "real": "real",
}


# ---------------------------------------------------------------------------
# Single-qubit gate algebra (traced inline into the kernels)
#
# A state is (r0, i0, r1, i1) = (Re a, Im a, Re b, Im b) for a|0> + b|1>.
# (c, s) = (cos(t/2), sin(t/2)) of the rotation angle t.
# ---------------------------------------------------------------------------


def _rz(r0, i0, r1, i1, c, s):
    """Rz(t) = diag(e^{-it/2}, e^{+it/2})."""
    return r0 * c + i0 * s, i0 * c - r0 * s, r1 * c - i1 * s, i1 * c + r1 * s


def _rz_inv(r0, i0, r1, i1, c, s):
    return r0 * c - i0 * s, i0 * c + r0 * s, r1 * c + i1 * s, i1 * c - r1 * s


def _ry(r0, i0, r1, i1, c, s):
    """Ry(t) = [[c, -s], [s, c]]."""
    return c * r0 - s * r1, c * i0 - s * i1, s * r0 + c * r1, s * i0 + c * i1


def _ry_inv(r0, i0, r1, i1, c, s):
    return c * r0 + s * r1, c * i0 + s * i1, c * r1 - s * r0, c * i1 - s * i0


def _rz_grad(adj, psi):
    """dL/dt of an Rz gate, given the adjoint and the state *after* the gate.

    dRz/dt = (-i/2) Z Rz, so dL/dt = Re<adj, (-i/2) Z psi>.
    """
    ar0, ai0, ar1, ai1 = adj
    pr0, pi0, pr1, pi1 = psi
    return 0.5 * ((ar0 * pi0 - ai0 * pr0) - (ar1 * pi1 - ai1 * pr1))


def _ry_grad(adj, psi):
    """dL/dt of an Ry gate, given the adjoint and the state *after* the gate.

    dRy/dt = (-i/2) Y Ry = 0.5 [[0, -1], [1, 0]] Ry.
    """
    ar0, ai0, ar1, ai1 = adj
    pr0, pi0, pr1, pi1 = psi
    return 0.5 * ((ar1 * pr0 + ai1 * pi0) - (ar0 * pr1 + ai0 * pi1))


# The ``real`` ansatz never leaves the reals: H, X, Z and Ry are all real.


def _ry_real(r0, r1, c, s):
    return c * r0 - s * r1, s * r0 + c * r1


def _ry_real_inv(r0, r1, c, s):
    return c * r0 + s * r1, c * r1 - s * r0


def _ry_real_grad(a0, a1, p0, p1):
    return 0.5 * (a1 * p0 - a0 * p1)


def _half_angle(t):
    """(cos(t/2), sin(t/2)).

    A single ``sincos`` shares the range reduction and is several times
    faster than separate ``sin``/``cos`` on CDNA (which lower to the precise
    OCML routines); the extra error stays within the rounding of the f32
    angle itself, comparable to CuTe's ``__sincosf``.
    """
    s, c = fx.math.sincos(t * 0.5)
    return c, s


def _abs_inv(n):
    """1/n, or 0 where the amplitude vanishes (matches the Triton backward)."""
    nonzero = n > 1e-30
    return nonzero.select(1.0 / nonzero.select(n, 1.0), 0.0)


def _load_f32(T, idx, bf16):
    v = T[idx]
    if const_expr(bf16):
        v = v.to(fx.Float32)
    return v


if _FLYDSL_KERNELS_AVAILABLE:

    def _load_circuit_params(TH, PW, PB, oi, REPS, N_TH, TH_P, ANSATZ, PREACTS):
        """Per-(o, i) trig of every theta plus preacts weight/bias, in registers."""
        th_base = fx.Int64(oi) * N_TH
        n_th_layers = REPS + int(ANSATZ != "real")
        n_th_per_layer = 1 + int(ANSATZ == "pz")
        trig = []
        for l in range_constexpr(n_th_layers):
            layer = []
            for p in range_constexpr(n_th_per_layer):
                layer.append(_half_angle(TH[th_base + (l * TH_P + p)]))
            trig.append(layer)

        ws, bs = [], []
        if const_expr(PREACTS):
            pw_base = fx.Int64(oi) * REPS
            for l in range_constexpr(REPS):
                ws.append(PW[pw_base + l])
                bs.append(PB[pw_base + l])
        return trig, ws, bs

    def _encodings(x, ws, bs, REPS, PREACTS):
        """(cos, sin) of half the data angle per layer: w*x+b, or x itself."""
        if const_expr(PREACTS):
            return [_half_angle(ws[l] * x + bs[l]) for l in range_constexpr(REPS)]
        shared = _half_angle(x)
        return [shared for _ in range_constexpr(REPS)]

    def _simulate(enc, trig, REPS, ANSATZ):
        """Run the circuit for one sample; returns the final state.

        ``enc`` holds the per-layer data-gate trig from :func:`_encodings`.
        """
        if const_expr(ANSATZ == "real"):
            r0 = fx.Float32(_INV_SQRT2)
            r1 = fx.Float32(_INV_SQRT2)
            for l in range_constexpr(REPS):
                r0, r1 = r1, r0  # X
                r0, r1 = _ry_real(r0, r1, *trig[l][0])
                r1 = 0.0 - r1  # Z
                r0, r1 = _ry_real(r0, r1, *enc[l])
            return r0, r1

        state = (
            fx.Float32(_INV_SQRT2),
            fx.Float32(0.0),
            fx.Float32(_INV_SQRT2),
            fx.Float32(0.0),
        )
        if const_expr(ANSATZ == "pz"):
            for l in range_constexpr(REPS):
                state = _rz(*state, *trig[l][0])
                state = _ry(*state, *trig[l][1])
                state = _rz(*state, *enc[l])
            state = _rz(*state, *trig[REPS][0])
            state = _ry(*state, *trig[REPS][1])
        else:  # rpz
            for l in range_constexpr(REPS):
                state = _ry(*state, *trig[l][0])
                state = _rz(*state, *enc[l])
            state = _ry(*state, *trig[REPS][0])
        return state

    def _measure(state, ANSATZ, FAST):
        if const_expr(ANSATZ == "real"):
            r0, r1 = state
            if const_expr(FAST):
                return fx.math.absf(r0) - fx.math.absf(r1)
            return r0 * r0 - r1 * r1
        r0, i0, r1, i1 = state
        p0 = r0 * r0 + i0 * i0
        p1 = r1 * r1 + i1 * i1
        if const_expr(FAST):
            return fx.math.sqrt(p0) - fx.math.sqrt(p1)
        return p0 - p1

    def _measure_adjoint(state, go, ANSATZ, FAST):
        """dL/dstate of the Z measurement, given dL/dout = go."""
        if const_expr(ANSATZ == "real"):
            r0, r1 = state
            if const_expr(FAST):
                g0 = go * _abs_inv(fx.math.absf(r0))
                g1 = go * _abs_inv(fx.math.absf(r1))
                return r0 * g0, 0.0 - r1 * g1
            return 2.0 * go * r0, -2.0 * go * r1
        r0, i0, r1, i1 = state
        if const_expr(FAST):
            g0 = go * _abs_inv(fx.math.sqrt(r0 * r0 + i0 * i0))
            g1 = 0.0 - go * _abs_inv(fx.math.sqrt(r1 * r1 + i1 * i1))
        else:
            g0 = 2.0 * go
            g1 = -2.0 * go
        return r0 * g0, i0 * g0, r1 * g1, i1 * g1

    def _batch_range(batch, b_chunk):
        b_start = fx.block_idx.y * b_chunk
        b_end = fx.min(b_start + b_chunk, batch)
        return b_start, b_end

    @flyc.kernel
    def _qkan_forward_kernel(
        X: fx.Tensor,
        TH: fx.Tensor,
        PW: fx.Tensor,
        PB: fx.Tensor,
        OUT: fx.Tensor,
        batch: fx.Int32,
        in_dim: fx.Int32,
        n_oi: fx.Int32,
        b_chunk: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
        X_BF16: fx.Constexpr[bool],
        OUT_BF16: fx.Constexpr[bool],
    ):
        oi = fx.block_idx.x * _BLOCK + fx.thread_idx.x
        if oi < n_oi:
            i = oi % in_dim
            trig, ws, bs = _load_circuit_params(
                TH, PW, PB, oi, REPS, N_TH, TH_P, ANSATZ, PREACTS
            )
            b_start, b_end = _batch_range(batch, b_chunk)
            for b in range(b_start, b_end):
                x = _load_f32(X, fx.Int64(b) * in_dim + i, X_BF16)
                enc = _encodings(x, ws, bs, REPS, PREACTS)
                state = _simulate(enc, trig, REPS, ANSATZ)
                result = _measure(state, ANSATZ, FAST)
                if const_expr(OUT_BF16):
                    result = result.to(fx.BFloat16)
                OUT[fx.Int64(b) * n_oi + oi] = result

    @flyc.kernel
    def _qkan_backward_kernel(
        X: fx.Tensor,
        TH: fx.Tensor,
        PW: fx.Tensor,
        PB: fx.Tensor,
        GO: fx.Tensor,
        GT: fx.Tensor,
        GW: fx.Tensor,
        GB: fx.Tensor,
        GXP: fx.Tensor,
        batch: fx.Int32,
        in_dim: fx.Int32,
        n_oi: fx.Int32,
        b_chunk: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
    ):
        oi = fx.block_idx.x * _BLOCK + fx.thread_idx.x
        if oi < n_oi:
            i = oi % in_dim
            trig, ws, bs = _load_circuit_params(
                TH, PW, PB, oi, REPS, N_TH, TH_P, ANSATZ, PREACTS
            )
            zero = fx.Float32(0.0)
            g_th = [[zero for _ in layer] for layer in trig]
            g_w = [zero for _ in range_constexpr(len(ws))]
            g_b = [zero for _ in range_constexpr(len(ws))]

            b_start, b_end = _batch_range(batch, b_chunk)
            for b in range(b_start, b_end):
                x = X[fx.Int64(b) * in_dim + i]
                out_idx = fx.Int64(b) * n_oi + oi
                enc = _encodings(x, ws, bs, REPS, PREACTS)
                psi = _simulate(enc, trig, REPS, ANSATZ)
                adj = _measure_adjoint(psi, GO[out_idx], ANSATZ, FAST)
                g_x = zero

                # Reverse sweep: at each gate take the parameter gradient from
                # the post-gate state, then un-apply the gate to state and adjoint.
                if const_expr(ANSATZ == "real"):
                    p0, p1 = psi
                    a0, a1 = adj
                    for l in range_constexpr(REPS - 1, -1, -1):
                        g_enc = _ry_real_grad(a0, a1, p0, p1)
                        p0, p1 = _ry_real_inv(p0, p1, *enc[l])
                        a0, a1 = _ry_real_inv(a0, a1, *enc[l])
                        if const_expr(PREACTS):
                            g_w[l] = g_w[l] + g_enc * x
                            g_b[l] = g_b[l] + g_enc
                            g_x = g_x + g_enc * ws[l]
                        else:
                            g_x = g_x + g_enc
                        p1 = 0.0 - p1  # Z
                        a1 = 0.0 - a1
                        g_th[l][0] = g_th[l][0] + _ry_real_grad(a0, a1, p0, p1)
                        p0, p1 = _ry_real_inv(p0, p1, *trig[l][0])
                        a0, a1 = _ry_real_inv(a0, a1, *trig[l][0])
                        p0, p1 = p1, p0  # X
                        a0, a1 = a1, a0
                else:
                    # The final layer is indexed as [-1]: a name inside a
                    # subscript-assignment target (g_th[REPS]) would be
                    # treated by the AST rewriter as loop state and lose its
                    # compile-time value.
                    if const_expr(ANSATZ == "pz"):
                        g_th[-1][1] = g_th[-1][1] + _ry_grad(adj, psi)
                        psi = _ry_inv(*psi, *trig[-1][1])
                        adj = _ry_inv(*adj, *trig[-1][1])
                        g_th[-1][0] = g_th[-1][0] + _rz_grad(adj, psi)
                        psi = _rz_inv(*psi, *trig[-1][0])
                        adj = _rz_inv(*adj, *trig[-1][0])
                    else:  # rpz
                        g_th[-1][0] = g_th[-1][0] + _ry_grad(adj, psi)
                        psi = _ry_inv(*psi, *trig[-1][0])
                        adj = _ry_inv(*adj, *trig[-1][0])

                    for l in range_constexpr(REPS - 1, -1, -1):
                        g_enc = _rz_grad(adj, psi)
                        psi = _rz_inv(*psi, *enc[l])
                        adj = _rz_inv(*adj, *enc[l])
                        if const_expr(PREACTS):
                            g_w[l] = g_w[l] + g_enc * x
                            g_b[l] = g_b[l] + g_enc
                            g_x = g_x + g_enc * ws[l]
                        else:
                            g_x = g_x + g_enc

                        if const_expr(ANSATZ == "pz"):
                            g_th[l][1] = g_th[l][1] + _ry_grad(adj, psi)
                            psi = _ry_inv(*psi, *trig[l][1])
                            adj = _ry_inv(*adj, *trig[l][1])
                            g_th[l][0] = g_th[l][0] + _rz_grad(adj, psi)
                            psi = _rz_inv(*psi, *trig[l][0])
                            adj = _rz_inv(*adj, *trig[l][0])
                        else:  # rpz
                            g_th[l][0] = g_th[l][0] + _ry_grad(adj, psi)
                            psi = _ry_inv(*psi, *trig[l][0])
                            adj = _ry_inv(*adj, *trig[l][0])

                GXP[out_idx] = g_x

            th_base = fx.Int64(oi) * N_TH
            for l in range_constexpr(len(g_th)):
                for p in range_constexpr(len(g_th[l])):
                    fx.atomic_add(
                        fx.get_iter(GT) + (th_base + (l * TH_P + p)), g_th[l][p]
                    )
            if const_expr(PREACTS):
                pw_base = fx.Int64(oi) * REPS
                for l in range_constexpr(REPS):
                    fx.atomic_add(fx.get_iter(GW) + (pw_base + l), g_w[l])
                    fx.atomic_add(fx.get_iter(GB) + (pw_base + l), g_b[l])

    @flyc.jit
    def _launch_forward(
        X: fx.Tensor,
        TH: fx.Tensor,
        PW: fx.Tensor,
        PB: fx.Tensor,
        OUT: fx.Tensor,
        batch: fx.Int32,
        in_dim: fx.Int32,
        n_oi: fx.Int32,
        b_chunk: fx.Int32,
        n_b_chunks: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
        X_BF16: fx.Constexpr[bool],
        OUT_BF16: fx.Constexpr[bool],
        stream: fx.Stream = fx.Stream(None),
    ):
        _qkan_forward_kernel(
            X,
            TH,
            PW,
            PB,
            OUT,
            batch,
            in_dim,
            n_oi,
            b_chunk,
            REPS,
            N_TH,
            TH_P,
            ANSATZ,
            PREACTS,
            FAST,
            X_BF16,
            OUT_BF16,
        ).launch(
            grid=((n_oi + _BLOCK - 1) // _BLOCK, n_b_chunks, 1),
            block=(_BLOCK, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def _launch_backward(
        X: fx.Tensor,
        TH: fx.Tensor,
        PW: fx.Tensor,
        PB: fx.Tensor,
        GO: fx.Tensor,
        GT: fx.Tensor,
        GW: fx.Tensor,
        GB: fx.Tensor,
        GXP: fx.Tensor,
        batch: fx.Int32,
        in_dim: fx.Int32,
        n_oi: fx.Int32,
        b_chunk: fx.Int32,
        n_b_chunks: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
        stream: fx.Stream = fx.Stream(None),
    ):
        _qkan_backward_kernel(
            X,
            TH,
            PW,
            PB,
            GO,
            GT,
            GW,
            GB,
            GXP,
            batch,
            in_dim,
            n_oi,
            b_chunk,
            REPS,
            N_TH,
            TH_P,
            ANSATZ,
            PREACTS,
            FAST,
        ).launch(
            grid=((n_oi + _BLOCK - 1) // _BLOCK, n_b_chunks, 1),
            block=(_BLOCK, 1, 1),
            stream=stream,
        )


# ---------------------------------------------------------------------------
# Host-side wrappers
# ---------------------------------------------------------------------------


def _use_bf16(c_dtype: torch.dtype) -> bool:
    """bf16 output for bf16 / fp8 compute dtypes (as in the CuTe backend)."""
    return c_dtype in (torch.bfloat16, torch.float8_e4m3fn)


def _batch_chunks(n_oi: int, batch: int, device: torch.device) -> tuple[int, int]:
    """Split the batch so roughly ``_THREADS_PER_CU`` threads run per CU.

    Fewer, longer chunks reuse the per-thread trig registers across more
    samples and issue fewer gradient atomics; more chunks fill the GPU when
    ``out_dim * in_dim`` alone cannot.
    """
    n_cu = torch.cuda.get_device_properties(device).multi_processor_count
    want = max(1, math.ceil(n_cu * _THREADS_PER_CU / max(n_oi, 1)))
    n_chunks = max(1, min(batch, want))
    b_chunk = math.ceil(batch / n_chunks)
    return b_chunk, math.ceil(batch / b_chunk)


def _check_shapes(x, theta, preacts_w, preacts_b, reps, ansatz, preacts):
    if x.dim() != 2 or theta.dim() != 4:
        raise ValueError(
            "flydsl kernels expect x of shape (batch, in_dim) and theta of shape "
            f"(out_dim, in_dim, layers, params); got {tuple(x.shape)} and "
            f"{tuple(theta.shape)}"
        )
    layers = reps if ansatz == "real" else reps + 1
    min_params = 2 if ansatz == "pz" else 1
    if theta.shape[1] != x.shape[1] or theta.shape[2] != layers:
        raise ValueError(
            f"theta shape {tuple(theta.shape)} does not match in_dim={x.shape[1]}, "
            f"reps={reps} for ansatz {ansatz!r}"
        )
    if theta.shape[3] < min_params:
        raise ValueError(
            f"ansatz {ansatz!r} needs {min_params} theta entries per layer, "
            f"got {theta.shape[3]}"
        )
    if preacts:
        expected = (theta.shape[0], x.shape[1], reps)
        if tuple(preacts_w.shape) != expected or tuple(preacts_b.shape) != expected:
            raise ValueError(
                f"preacts weight/bias must have shape {expected}, got "
                f"{tuple(preacts_w.shape)} and {tuple(preacts_b.shape)}"
            )


def _flat_f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).contiguous().view(-1)


def _prepare(x, theta, preacts_w, preacts_b, reps, ansatz, preacts_trainable):
    """Validate and flatten the circuit parameters for a kernel launch."""
    if not _FLYDSL_KERNELS_AVAILABLE:
        raise ImportError(
            "FlyDSL kernels need the flydsl package and a ROCm build of PyTorch."
        )
    if ansatz not in _ANSATZ_CANONICAL:
        raise ValueError(f"Unsupported ansatz for flydsl: {ansatz}")
    kind = _ANSATZ_CANONICAL[ansatz]
    # rpz always encodes w*x+b; pz / real only when the preacts are trainable.
    preacts = preacts_trainable or kind == "rpz"
    _check_shapes(x, theta, preacts_w, preacts_b, reps, kind, preacts)

    th = _flat_f32(theta)
    if preacts:
        pw, pb = _flat_f32(preacts_w), _flat_f32(preacts_b)
    else:
        # Unused by the kernel, but every tensor argument needs real storage.
        pw = pb = th
    n_th = theta.shape[2] * theta.shape[3]
    return kind, preacts, th, pw, pb, n_th, theta.shape[3]


def flydsl_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    reps: int,
    ansatz: str,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FlyDSL QKAN forward.

    Args:
        x: (batch, in_dim)
        theta: (out_dim, in_dim, reps+1, P), or (out_dim, in_dim, reps, 1)
            for the ``real`` ansatz
        preacts_w, preacts_b: (out_dim, in_dim, reps); read only by ``rpz``
            or when ``preacts_trainable`` is set
        reps: number of data re-uploading layers
        ansatz: ``"pz_encoding"``/``"pz"``, ``"rpz_encoding"``/``"rpz"`` or
            ``"real"``
        fast_measure: True for |alpha|-|beta|, False for |alpha|^2-|beta|^2
        c_dtype: bfloat16/float8 selects bf16 output, anything else float32

    Returns:
        (batch, out_dim, in_dim) in float32 or bfloat16.
    """
    kind, preacts, th, pw, pb, n_th, th_p = _prepare(
        x, theta, preacts_w, preacts_b, reps, ansatz, preacts_trainable
    )
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    n_oi = out_dim * in_dim
    out_bf16 = _use_bf16(c_dtype)
    x_bf16 = x.dtype == torch.bfloat16
    xf = x.contiguous().view(-1) if x_bf16 else _flat_f32(x)

    out = torch.empty(
        batch,
        out_dim,
        in_dim,
        device=x.device,
        dtype=torch.bfloat16 if out_bf16 else torch.float32,
    )
    if out.numel() == 0:
        return out
    b_chunk, n_b_chunks = _batch_chunks(n_oi, batch, x.device)
    _launch_forward(
        xf,
        th,
        pw,
        pb,
        out.view(-1),
        batch,
        in_dim,
        n_oi,
        b_chunk,
        n_b_chunks,
        reps,
        n_th,
        th_p,
        kind,
        preacts,
        fast_measure,
        x_bf16,
        out_bf16,
        stream=torch.cuda.current_stream(x.device),
    )
    return out


def flydsl_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    grad_output: torch.Tensor,
    reps: int,
    ansatz: str,
    preacts_trainable: bool,
    fast_measure: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """FlyDSL QKAN backward (float32).

    Returns ``(grad_x, grad_theta, grad_preacts_w, grad_preacts_b)``; the
    preacts gradients are ``None`` when the kernel does not read the preacts.
    """
    kind, preacts, th, pw, pb, n_th, th_p = _prepare(
        x, theta, preacts_w, preacts_b, reps, ansatz, preacts_trainable
    )
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    n_oi = out_dim * in_dim
    device = x.device

    grad_theta = torch.zeros(theta.shape, device=device, dtype=torch.float32)
    grad_pw: torch.Tensor | None = None
    grad_pb: torch.Tensor | None = None
    if preacts:
        grad_pw = torch.zeros(preacts_w.shape, device=device, dtype=torch.float32)
        grad_pb = torch.zeros(preacts_b.shape, device=device, dtype=torch.float32)
    if batch == 0 or n_oi == 0:
        grad_x = torch.zeros(batch, in_dim, device=device, dtype=torch.float32)
        return grad_x, grad_theta, grad_pw, grad_pb

    # Per-(b, o, i) contributions to grad_x, summed over o below.
    grad_x_part = torch.empty(
        batch, out_dim, in_dim, device=device, dtype=torch.float32
    )
    b_chunk, n_b_chunks = _batch_chunks(n_oi, batch, device)
    _launch_backward(
        _flat_f32(x),
        th,
        pw,
        pb,
        _flat_f32(grad_output),
        grad_theta.view(-1),
        grad_pw.view(-1) if grad_pw is not None else grad_theta.view(-1),
        grad_pb.view(-1) if grad_pb is not None else grad_theta.view(-1),
        grad_x_part.view(-1),
        batch,
        in_dim,
        n_oi,
        b_chunk,
        n_b_chunks,
        reps,
        n_th,
        th_p,
        kind,
        preacts,
        fast_measure,
        stream=torch.cuda.current_stream(device),
    )
    return grad_x_part.sum(dim=1), grad_theta, grad_pw, grad_pb
