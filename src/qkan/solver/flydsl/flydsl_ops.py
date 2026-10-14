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
    written to that chunk's own row of a partial-sum buffer, which is reduced
    over chunks with a single ``torch.sum``. ``grad_x`` is written per
    ``(b, o, i)`` and reduced over ``o`` the same way. No atomics are used, so
    the backward pass is deterministic.

All math runs in float32. bf16 inputs, parameters and ``grad_output`` are
read as bf16 and widened in registers, so no host-side casts run before a
launch. The forward output is float32 or bfloat16 (bf16 I/O halves the
dominant ``(batch, out_dim, in_dim)`` write); the backward pass always
accumulates and returns float32.

Launchers are compiled once per compile-time configuration and then called
through FlyDSL's compiled-function fast path (see :func:`_launch`).
"""

from __future__ import annotations

import functools
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

# Threads to keep in flight per compute unit before batch chunks grow longer,
# for the forward and backward kernels. Tuned on MI300X (sweep over 256-8192
# for QKAN shapes from 16x16 to 256x256, real / pz / rpz, reps 1-3).
_THREADS_PER_CU_FWD = 1024
_THREADS_PER_CU_BWD = 1024

# Most circuits (outputs o) one thread runs for its input index i when the
# encoding depends on x alone, so they share one sincos and one x load. The
# backward pass keeps a gradient accumulator per theta, so its block is also
# capped at _O_BLOCK_BWD_THETAS thetas per thread to avoid register spills.
_MAX_O_BLOCK_FWD = 4
_MAX_O_BLOCK_BWD = 8
_O_BLOCK_BWD_THETAS = 8

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

    def _load_circuit_params(TH, PW, PB, oi, REPS, N_TH, TH_P, ANSATZ, PREACTS, P_BF16):
        """Per-(o, i) trig of every theta plus preacts weight/bias, in registers."""
        th_base = fx.Int64(oi) * N_TH
        n_th_layers = REPS + int(ANSATZ != "real")
        n_th_per_layer = 1 + int(ANSATZ == "pz")
        trig = []
        for l in range_constexpr(n_th_layers):
            layer = []
            for p in range_constexpr(n_th_per_layer):
                layer.append(
                    _half_angle(_load_f32(TH, th_base + (l * TH_P + p), P_BF16))
                )
            trig.append(layer)

        ws, bs = [], []
        if const_expr(PREACTS):
            pw_base = fx.Int64(oi) * REPS
            for l in range_constexpr(REPS):
                ws.append(_load_f32(PW, pw_base + l, P_BF16))
                bs.append(_load_f32(PB, pw_base + l, P_BF16))
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

    def _thread_circuits(in_dim, O_BLK):
        """This thread's input index ``i`` and the flat ``(o, i)`` index of each
        of its ``O_BLK`` circuits (outputs ``o0 .. o0 + O_BLK - 1``)."""
        t = fx.block_idx.x * _BLOCK + fx.thread_idx.x
        i = t % in_dim
        o0 = (t // in_dim) * O_BLK
        return t, i, [(o0 + k) * in_dim + i for k in range_constexpr(O_BLK)]

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
        n_threads: fx.Int32,
        b_chunk: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
        X_BF16: fx.Constexpr[bool],
        P_BF16: fx.Constexpr[bool],
        OUT_BF16: fx.Constexpr[bool],
        O_BLK: fx.Constexpr[int],
    ):
        t, i, ois = _thread_circuits(in_dim, O_BLK)
        if t < n_threads:
            params = [
                _load_circuit_params(
                    TH, PW, PB, ois[k], REPS, N_TH, TH_P, ANSATZ, PREACTS, P_BF16
                )
                for k in range_constexpr(O_BLK)
            ]
            b_start, b_end = _batch_range(batch, b_chunk)
            for b in range(b_start, b_end):
                x = _load_f32(X, fx.Int64(b) * in_dim + i, X_BF16)
                # O_BLK > 1 only without preacts, where the encoding depends on
                # x alone: one sincos serves all of this thread's circuits.
                enc = _encodings(x, params[0][1], params[0][2], REPS, PREACTS)
                out_base = fx.Int64(b) * n_oi
                for k in range_constexpr(O_BLK):
                    state = _simulate(enc, params[k][0], REPS, ANSATZ)
                    result = _measure(state, ANSATZ, FAST)
                    if const_expr(OUT_BF16):
                        result = result.to(fx.BFloat16)
                    OUT[out_base + ois[k]] = result

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
        n_threads: fx.Int32,
        b_chunk: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
        X_BF16: fx.Constexpr[bool],
        P_BF16: fx.Constexpr[bool],
        GO_BF16: fx.Constexpr[bool],
        O_BLK: fx.Constexpr[int],
    ):
        t, i, ois = _thread_circuits(in_dim, O_BLK)
        if t < n_threads:
            params = [
                _load_circuit_params(
                    TH, PW, PB, ois[k], REPS, N_TH, TH_P, ANSATZ, PREACTS, P_BF16
                )
                for k in range_constexpr(O_BLK)
            ]
            zero = fx.Float32(0.0)
            g_th = [
                [[zero for _ in layer] for layer in params[k][0]]
                for k in range_constexpr(O_BLK)
            ]
            g_w = [
                [zero for _ in range_constexpr(len(params[k][1]))]
                for k in range_constexpr(O_BLK)
            ]
            g_b = [
                [zero for _ in range_constexpr(len(params[k][1]))]
                for k in range_constexpr(O_BLK)
            ]

            b_start, b_end = _batch_range(batch, b_chunk)
            for b in range(b_start, b_end):
                x = _load_f32(X, fx.Int64(b) * in_dim + i, X_BF16)
                out_base = fx.Int64(b) * n_oi
                # Shared by the O_BLK circuits (O_BLK > 1 only without preacts).
                enc = _encodings(x, params[0][1], params[0][2], REPS, PREACTS)
                # This thread's grad_x contribution, summed over its circuits.
                g_x = zero
                for k in range_constexpr(O_BLK):
                    trig = params[k][0]
                    ws = params[k][1]
                    psi = _simulate(enc, trig, REPS, ANSATZ)
                    go = _load_f32(GO, out_base + ois[k], GO_BF16)
                    adj = _measure_adjoint(psi, go, ANSATZ, FAST)

                    # Reverse sweep: at each gate take the parameter gradient
                    # from the post-gate state, then un-apply the gate to state
                    # and adjoint.
                    if const_expr(ANSATZ == "real"):
                        p0, p1 = psi
                        a0, a1 = adj
                        for l in range_constexpr(REPS - 1, -1, -1):
                            g_enc = _ry_real_grad(a0, a1, p0, p1)
                            p0, p1 = _ry_real_inv(p0, p1, *enc[l])
                            a0, a1 = _ry_real_inv(a0, a1, *enc[l])
                            if const_expr(PREACTS):
                                g_w[k][l] = g_w[k][l] + g_enc * x
                                g_b[k][l] = g_b[k][l] + g_enc
                                g_x = g_x + g_enc * ws[l]
                            else:
                                g_x = g_x + g_enc
                            p1 = 0.0 - p1  # Z
                            a1 = 0.0 - a1
                            g_th[k][l][0] = g_th[k][l][0] + _ry_real_grad(
                                a0, a1, p0, p1
                            )
                            p0, p1 = _ry_real_inv(p0, p1, *trig[l][0])
                            a0, a1 = _ry_real_inv(a0, a1, *trig[l][0])
                            p0, p1 = p1, p0  # X
                            a0, a1 = a1, a0
                    else:
                        # The final layer is indexed as [-1]: a name inside a
                        # subscript-assignment target (g_th[k][REPS]) would be
                        # treated by the AST rewriter as loop state and lose its
                        # compile-time value.
                        if const_expr(ANSATZ == "pz"):
                            g_th[k][-1][1] = g_th[k][-1][1] + _ry_grad(adj, psi)
                            psi = _ry_inv(*psi, *trig[-1][1])
                            adj = _ry_inv(*adj, *trig[-1][1])
                            g_th[k][-1][0] = g_th[k][-1][0] + _rz_grad(adj, psi)
                            psi = _rz_inv(*psi, *trig[-1][0])
                            adj = _rz_inv(*adj, *trig[-1][0])
                        else:  # rpz
                            g_th[k][-1][0] = g_th[k][-1][0] + _ry_grad(adj, psi)
                            psi = _ry_inv(*psi, *trig[-1][0])
                            adj = _ry_inv(*adj, *trig[-1][0])

                        for l in range_constexpr(REPS - 1, -1, -1):
                            g_enc = _rz_grad(adj, psi)
                            psi = _rz_inv(*psi, *enc[l])
                            adj = _rz_inv(*adj, *enc[l])
                            if const_expr(PREACTS):
                                g_w[k][l] = g_w[k][l] + g_enc * x
                                g_b[k][l] = g_b[k][l] + g_enc
                                g_x = g_x + g_enc * ws[l]
                            else:
                                g_x = g_x + g_enc

                            if const_expr(ANSATZ == "pz"):
                                g_th[k][l][1] = g_th[k][l][1] + _ry_grad(adj, psi)
                                psi = _ry_inv(*psi, *trig[l][1])
                                adj = _ry_inv(*adj, *trig[l][1])
                                g_th[k][l][0] = g_th[k][l][0] + _rz_grad(adj, psi)
                                psi = _rz_inv(*psi, *trig[l][0])
                                adj = _rz_inv(*adj, *trig[l][0])
                            else:  # rpz
                                g_th[k][l][0] = g_th[k][l][0] + _ry_grad(adj, psi)
                                psi = _ry_inv(*psi, *trig[l][0])
                                adj = _ry_inv(*adj, *trig[l][0])

                GXP[fx.Int64(b) * n_threads + t] = g_x

            # Each batch chunk writes its partial sums to its own row; the host
            # reduces the rows. No atomics: the per-parameter atomic adds of
            # thousands of chunks serialized on the same addresses. Theta
            # entries no gate reads get an explicit zero, so the buffer needs
            # no zero-fill.
            row_base = fx.Int64(fx.block_idx.y) * n_oi
            for k in range_constexpr(O_BLK):
                th_base = (row_base + ois[k]) * N_TH
                for l in range_constexpr(len(g_th[k])):
                    for p in range_constexpr(len(g_th[k][l])):
                        GT[th_base + (l * TH_P + p)] = g_th[k][l][p]
                    for p in range_constexpr(len(g_th[k][l]), TH_P):
                        GT[th_base + (l * TH_P + p)] = zero
                if const_expr(PREACTS):
                    pw_base = (row_base + ois[k]) * REPS
                    for l in range_constexpr(REPS):
                        GW[pw_base + l] = g_w[k][l]
                        GB[pw_base + l] = g_b[k][l]

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
        n_threads: fx.Int32,
        b_chunk: fx.Int32,
        n_b_chunks: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
        X_BF16: fx.Constexpr[bool],
        P_BF16: fx.Constexpr[bool],
        OUT_BF16: fx.Constexpr[bool],
        O_BLK: fx.Constexpr[int],
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
            n_threads,
            b_chunk,
            REPS,
            N_TH,
            TH_P,
            ANSATZ,
            PREACTS,
            FAST,
            X_BF16,
            P_BF16,
            OUT_BF16,
            O_BLK,
        ).launch(
            grid=((n_threads + _BLOCK - 1) // _BLOCK, n_b_chunks, 1),
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
        n_threads: fx.Int32,
        b_chunk: fx.Int32,
        n_b_chunks: fx.Int32,
        REPS: fx.Constexpr[int],
        N_TH: fx.Constexpr[int],
        TH_P: fx.Constexpr[int],
        ANSATZ: fx.Constexpr[str],
        PREACTS: fx.Constexpr[bool],
        FAST: fx.Constexpr[bool],
        X_BF16: fx.Constexpr[bool],
        P_BF16: fx.Constexpr[bool],
        GO_BF16: fx.Constexpr[bool],
        O_BLK: fx.Constexpr[int],
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
            n_threads,
            b_chunk,
            REPS,
            N_TH,
            TH_P,
            ANSATZ,
            PREACTS,
            FAST,
            X_BF16,
            P_BF16,
            GO_BF16,
            O_BLK,
        ).launch(
            grid=((n_threads + _BLOCK - 1) // _BLOCK, n_b_chunks, 1),
            block=(_BLOCK, 1, 1),
            stream=stream,
        )


# ---------------------------------------------------------------------------
# Host-side wrappers
# ---------------------------------------------------------------------------


def _use_bf16(c_dtype: torch.dtype) -> bool:
    """bf16 output for bf16 / fp8 compute dtypes (as in the CuTe backend)."""
    return c_dtype in (torch.bfloat16, torch.float8_e4m3fn)


@functools.lru_cache(maxsize=None)
def _compute_units(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _o_block(out_dim: int, in_dim: int, preacts: bool, max_blk: int) -> int:
    """Circuits per thread: the largest power of two up to ``max_blk`` that
    divides ``out_dim`` and still leaves a full wavefront of threads per batch
    chunk; 1 when the encoding differs per (o, i)."""
    if preacts:
        return 1
    o_blk = 1 << (max(1, max_blk).bit_length() - 1)
    while o_blk > 1 and (out_dim % o_blk or out_dim // o_blk * in_dim < _BLOCK):
        o_blk //= 2
    return o_blk


def _batch_chunks(
    n_threads: int, batch: int, device: torch.device, threads_per_cu: int
) -> tuple[int, int]:
    """Split the batch so roughly ``threads_per_cu`` threads run per CU.

    Fewer, longer chunks reuse the per-thread trig registers across more
    samples and write fewer gradient partial rows; more chunks fill the GPU when
    the ``n_threads`` threads of one chunk alone cannot.
    """
    n_cu = _compute_units(device)
    want = max(1, math.ceil(n_cu * threads_per_cu / max(n_threads, 1)))
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


def _flat(t: torch.Tensor, bf16: bool) -> torch.Tensor:
    """Flat contiguous bf16 or float32 view (the two dtypes the kernels read)."""
    return t.to(torch.bfloat16 if bf16 else torch.float32).contiguous().view(-1)


# Compiled launchers keyed by launcher and every compile-time argument. The
# kernels take flat tensors whose sizes are runtime values, so one compiled
# launcher serves every shape. Calling it directly skips the per-call
# signature binding and cache-key building of ``@flyc.jit`` dispatch, which
# costs more host time than the kernel takes on the GPU at QKAN layer sizes.
_COMPILED: dict = {}


def _launch(launcher, constexprs, *args):
    key = (launcher, constexprs)
    compiled = _COMPILED.get(key)
    if compiled is not None:
        compiled(*args)
        return
    # flyc.compile also runs this first launch.
    compiled = flyc.compile(launcher, *args)
    if compiled is not None:
        _COMPILED[key] = compiled


def _prepare(x, theta, preacts_w, preacts_b, reps, ansatz, preacts_trainable):
    """Validate and flatten the circuit parameters for a kernel launch.

    The parameters stay bf16 when all of them are bf16 (the kernels widen on
    load) and are cast to float32 otherwise.
    """
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

    p_bf16 = theta.dtype == torch.bfloat16 and (
        not preacts
        or (preacts_w.dtype == torch.bfloat16 and preacts_b.dtype == torch.bfloat16)
    )
    th = _flat(theta, p_bf16)
    if preacts:
        pw, pb = _flat(preacts_w, p_bf16), _flat(preacts_b, p_bf16)
    else:
        # Unused by the kernel, but every tensor argument needs real storage.
        pw = pb = th
    n_th = theta.shape[2] * theta.shape[3]
    return kind, preacts, th, pw, pb, n_th, theta.shape[3], p_bf16


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
    kind, preacts, th, pw, pb, n_th, th_p, p_bf16 = _prepare(
        x, theta, preacts_w, preacts_b, reps, ansatz, preacts_trainable
    )
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    n_oi = out_dim * in_dim
    out_bf16 = _use_bf16(c_dtype)
    x_bf16 = x.dtype == torch.bfloat16

    out = torch.empty(
        batch,
        out_dim,
        in_dim,
        device=x.device,
        dtype=torch.bfloat16 if out_bf16 else torch.float32,
    )
    if out.numel() == 0:
        return out
    o_blk = _o_block(out_dim, in_dim, preacts, _MAX_O_BLOCK_FWD)
    n_threads = out_dim // o_blk * in_dim
    b_chunk, n_b_chunks = _batch_chunks(n_threads, batch, x.device, _THREADS_PER_CU_FWD)
    constexprs = (
        reps,
        n_th,
        th_p,
        kind,
        preacts,
        fast_measure,
        x_bf16,
        p_bf16,
        out_bf16,
        o_blk,
    )
    _launch(
        _launch_forward,
        constexprs,
        _flat(x, x_bf16),
        th,
        pw,
        pb,
        out.view(-1),
        batch,
        in_dim,
        n_oi,
        n_threads,
        b_chunk,
        n_b_chunks,
        *constexprs,
        torch.cuda.current_stream(x.device),
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
    kind, preacts, th, pw, pb, n_th, th_p, p_bf16 = _prepare(
        x, theta, preacts_w, preacts_b, reps, ansatz, preacts_trainable
    )
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    n_oi = out_dim * in_dim
    device = x.device

    if batch == 0 or n_oi == 0:
        grad_x = torch.zeros(batch, in_dim, device=device, dtype=torch.float32)
        grad_theta = torch.zeros(theta.shape, device=device, dtype=torch.float32)
        if not preacts:
            return grad_x, grad_theta, None, None
        zeros_pw = torch.zeros(preacts_w.shape, device=device, dtype=torch.float32)
        return grad_x, grad_theta, zeros_pw, torch.zeros_like(zeros_pw)

    x_bf16 = x.dtype == torch.bfloat16
    go_bf16 = grad_output.dtype == torch.bfloat16
    max_blk = min(_MAX_O_BLOCK_BWD, max(1, _O_BLOCK_BWD_THETAS // n_th))
    o_blk = _o_block(out_dim, in_dim, preacts, max_blk)
    n_threads = out_dim // o_blk * in_dim
    # grad_x contributions per (b, thread): each thread has already summed
    # over its o_blk outputs; the rest of the sum over o is done below.
    grad_x_part = torch.empty(
        batch, out_dim // o_blk, in_dim, device=device, dtype=torch.float32
    )
    b_chunk, n_b_chunks = _batch_chunks(n_threads, batch, device, _THREADS_PER_CU_BWD)
    # One row of parameter-gradient partial sums per batch chunk, summed over
    # chunks below.
    theta_part = torch.empty(
        (n_b_chunks, *theta.shape), device=device, dtype=torch.float32
    )
    pw_part: torch.Tensor | None = None
    pb_part: torch.Tensor | None = None
    if preacts:
        pw_part = torch.empty(
            (n_b_chunks, *preacts_w.shape), device=device, dtype=torch.float32
        )
        pb_part = torch.empty_like(pw_part)
    constexprs = (
        reps,
        n_th,
        th_p,
        kind,
        preacts,
        fast_measure,
        x_bf16,
        p_bf16,
        go_bf16,
        o_blk,
    )
    _launch(
        _launch_backward,
        constexprs,
        _flat(x, x_bf16),
        th,
        pw,
        pb,
        _flat(grad_output, go_bf16),
        theta_part.view(-1),
        pw_part.view(-1) if pw_part is not None else theta_part.view(-1),
        pb_part.view(-1) if pb_part is not None else theta_part.view(-1),
        grad_x_part.view(-1),
        batch,
        in_dim,
        n_oi,
        n_threads,
        b_chunk,
        n_b_chunks,
        *constexprs,
        torch.cuda.current_stream(device),
    )
    grad_pw = pw_part.sum(dim=0) if pw_part is not None else None
    grad_pb = pb_part.sum(dim=0) if pb_part is not None else None
    return grad_x_part.sum(dim=1), theta_part.sum(dim=0), grad_pw, grad_pb
