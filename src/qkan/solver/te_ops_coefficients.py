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
Triton kernels for the Fourier coefficients of every edge (``solver="te"``).

With ``fast_measure=False`` and the preact weights fixed at 1, the circuit on an
edge is a trigonometric polynomial of degree ``reps`` in its input, so its
values at ``2*reps+1`` or more equispaced points determine it. The forward
kernel runs every edge's circuit at those points and takes the DFT of the
samples; the backward kernel differentiates both with the adjoint method. A
program owns a block of edges and all of their samples, as a (samples, edges)
tile.

theta and preacts_bias are float32 or bfloat16. All math runs in float32.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from .flydsl.flydsl_ops import _first_order_only

_INV_SQRT2 = tl.constexpr(0.7071067811865476)
_TWO_PI = tl.constexpr(6.283185307179586)

# reps 16 would need 64 samples an edge, more than the one thread that holds
# them keeps in registers: measured 60x slower than reps 15 or worse, slower
# than sampling pz and rpz through flash, and a minute of JIT.
# From reps 8 an edge takes 32 samples, about 1.6x the cost per sample, and
# the first call compiles for 10 s (reps 8) to 40 s (reps 15), once per
# Triton cache.
MAX_REPS = 15

_DTYPES = {torch.float32, torch.bfloat16}

# ansatz -> (kernel's name for it, theta's T - reps, theta's P)
_ANSATZES = {
    "pz_encoding": ("pz", 1, 2),
    "pz": ("pz", 1, 2),
    "rpz_encoding": ("rpz", 1, 1),
    "rpz": ("rpz", 1, 1),
    "real": ("real", 0, 1),
}

# One edge per thread: a thread then holds every sample of its edge, so the
# sums over the samples stay in its registers. Splitting an edge over threads,
# or an (edges, samples) tile, measured 2x slower or worse.
_NUM_WARPS = 4
_BLOCK = 32 * _NUM_WARPS


# ── Gates on psi = (r0 + i*i0, r1 + i*i1), as in flash/fused_ops.py ────────


@triton.jit
def _rz(r0, i0, r1, i1, c, s):
    """Rz(t) with c, s = cos, sin of t/2; pass -s for its inverse."""
    return r0 * c + i0 * s, i0 * c - r0 * s, r1 * c - i1 * s, i1 * c + r1 * s


@triton.jit
def _ry(r0, i0, r1, i1, c, s):
    """Ry(t) with c, s = cos, sin of t/2; pass -s for its inverse."""
    return c * r0 - s * r1, c * i0 - s * i1, s * r0 + c * r1, s * i0 + c * i1


@triton.jit
def _trig(ptr, offs, mask):
    """cos, sin of half the angle each edge has at ptr[offs], as (1, BLOCK) rows."""
    a = 0.5 * tl.load(ptr + offs, mask=mask, other=0.0).to(tl.float32)
    return tl.cos(a)[None, :], tl.sin(a)[None, :]


@triton.jit
def _final_state(
    theta_ptr,
    bias_ptr,
    t_offs,  # [BLOCK] offset of each edge's theta
    b_offs,  # [BLOCK] offset of each edge's bias
    mask,
    cx,  # [SAMPLES, 1] cos, sin of half the sample angles
    sx,
    ANSATZ: tl.constexpr,
    REPS: tl.constexpr,
    SAMPLES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The state each circuit of the block ends in at each sample: 4 x [SAMPLES, BLOCK]."""
    r0 = tl.full([SAMPLES, BLOCK], _INV_SQRT2, tl.float32)
    i0 = tl.zeros([SAMPLES, BLOCK], tl.float32)
    r1 = tl.full([SAMPLES, BLOCK], _INV_SQRT2, tl.float32)
    i1 = tl.zeros([SAMPLES, BLOCK], tl.float32)
    # pz and rpz end on a layer of theta that has no data gate after it.
    for l in tl.static_range(REPS + 1):
        if ANSATZ == "pz":
            c, s = _trig(theta_ptr, t_offs + 2 * l, mask)
            r0, i0, r1, i1 = _rz(r0, i0, r1, i1, c, s)
            c, s = _trig(theta_ptr, t_offs + 2 * l + 1, mask)
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, c, s)
            if l < REPS:
                r0, i0, r1, i1 = _rz(r0, i0, r1, i1, cx, sx)
        elif ANSATZ == "rpz":
            c, s = _trig(theta_ptr, t_offs + l, mask)
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, c, s)
            if l < REPS:
                # Rz(x + bias) = Rz(x) Rz(bias): no trig per (sample, edge).
                c, s = _trig(bias_ptr, b_offs + l, mask)
                r0, i0, r1, i1 = _rz(r0, i0, r1, i1, c, s)
                r0, i0, r1, i1 = _rz(r0, i0, r1, i1, cx, sx)
        elif l < REPS:
            r0, i0, r1, i1 = r1, i1, r0, i0  # X
            c, s = _trig(theta_ptr, t_offs + l, mask)
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, c, s)
            r1, i1 = -r1, -i1  # Z
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, cx, sx)
    return r0, i0, r1, i1


# Adjoint method. For a gate U = exp(-i t G / 2), with psi the state right
# after U and lambda the adjoint state there (Z psi_final pulled back through
# the later gates), d<Z>/dt = Im <lambda|G|psi>.


@triton.jit
def _grad_rz(r0, i0, r1, i1, lr0, li0, lr1, li1):
    """Im <lambda|Z|psi>, summed over the samples."""
    return tl.sum((lr0 * i0 - li0 * r0) - (lr1 * i1 - li1 * r1), axis=0)


@triton.jit
def _grad_ry(r0, i0, r1, i1, lr0, li0, lr1, li1):
    """Im <lambda|Y|psi>, summed over the samples."""
    return tl.sum((lr1 * r0 + li1 * i0) - (lr0 * r1 + li0 * i1), axis=0)


# ── Kernels ────────────────────────────────────────────────────────────────


# Sizes are not specialized: each class of in_dim or edge count (1, a multiple
# of 16, other) would compile these kernels again, for no measured gain.
@triton.jit(do_not_specialize=["n_edges", "in_dim", "stride_t"])
def _coefficients_fwd_kernel(
    theta_ptr,  # [O, I, T, P]
    bias_ptr,  # [O, I, R] for rpz, else None
    c_ptr,  # [O, 2R+1, I] float32
    n_edges,  # O * I
    in_dim,
    stride_t,  # T * P
    ANSATZ: tl.constexpr,
    REPS: tl.constexpr,
    SAMPLES: tl.constexpr,  # a power of two >= 2R+1
    BLOCK: tl.constexpr,
):
    """c[o] = [a0, a_1..a_R, b_1..b_R] of f_oi(x) = <Z> after the circuit of edge (o, i).

    The samples are x_j = 2*pi*j/SAMPLES. A tile dim has to be a power of two
    and any count >= 2R+1 gives the same coefficients, so no row is padding.
    Grid: (cdiv(O * I, BLOCK),).
    """
    pid = tl.program_id(0)
    # int64: every offset is a product with the edge index, and theta and c
    # pass 2^31 elements before the edges do (#31).
    edge = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = edge < n_edges
    x = tl.arange(0, SAMPLES).to(tl.float32) * (_TWO_PI / SAMPLES)

    r0, i0, r1, i1 = _final_state(
        theta_ptr,
        bias_ptr,
        edge * stride_t,
        edge * REPS,
        mask,
        tl.cos(0.5 * x)[:, None],
        tl.sin(0.5 * x)[:, None],
        ANSATZ,
        REPS,
        SAMPLES,
        BLOCK,
    )
    f = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    # DFT over the samples.
    row = (edge // in_dim) * (2 * REPS + 1)
    col = edge % in_dim
    a0 = tl.sum(f, axis=0) * (1.0 / SAMPLES)
    tl.store(c_ptr + row * in_dim + col, a0, mask=mask)
    for k in tl.static_range(1, REPS + 1):
        a = tl.sum(tl.cos(k * x)[:, None] * f, axis=0) * (2.0 / SAMPLES)
        b = tl.sum(tl.sin(k * x)[:, None] * f, axis=0) * (2.0 / SAMPLES)
        tl.store(c_ptr + (row + k) * in_dim + col, a, mask=mask)
        tl.store(c_ptr + (row + REPS + k) * in_dim + col, b, mask=mask)


@triton.jit(do_not_specialize=["n_edges", "in_dim", "stride_t"])
def _coefficients_bwd_kernel(
    theta_ptr,  # [O, I, T, P]
    bias_ptr,  # [O, I, R] for rpz, else None
    grad_c_ptr,  # [O, 2R+1, I]
    grad_theta_ptr,  # [O, I, T, P]
    grad_bias_ptr,  # [O, I, R] for rpz, else None
    n_edges,
    in_dim,
    stride_t,
    ANSATZ: tl.constexpr,
    REPS: tl.constexpr,
    SAMPLES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Adjoint of ``_coefficients_fwd_kernel``, same grid.

    Every gradient of an edge is written by the one program that owns it, so
    there are no atomics and no saved states: the circuit is re-run to its
    final state and then undone gate by gate.
    """
    pid = tl.program_id(0)
    edge = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = edge < n_edges
    x = tl.arange(0, SAMPLES).to(tl.float32) * (_TWO_PI / SAMPLES)
    cx = tl.cos(0.5 * x)[:, None]
    sx = tl.sin(0.5 * x)[:, None]
    t_offs = edge * stride_t
    b_offs = edge * REPS

    r0, i0, r1, i1 = _final_state(
        theta_ptr, bias_ptr, t_offs, b_offs, mask, cx, sx, ANSATZ, REPS, SAMPLES, BLOCK
    )

    # Upstream of every sample: the transpose of the forward DFT.
    row = (edge // in_dim) * (2 * REPS + 1)
    col = edge % in_dim
    g0 = tl.load(grad_c_ptr + row * in_dim + col, mask=mask, other=0.0)
    g = 0.5 * g0.to(tl.float32)[None, :]
    for k in tl.static_range(1, REPS + 1):
        ga = tl.load(grad_c_ptr + (row + k) * in_dim + col, mask=mask, other=0.0)
        gb = tl.load(grad_c_ptr + (row + REPS + k) * in_dim + col, mask=mask, other=0.0)
        g += tl.cos(k * x)[:, None] * ga.to(tl.float32)[None, :]
        g += tl.sin(k * x)[:, None] * gb.to(tl.float32)[None, :]
    g *= 2.0 / SAMPLES

    # lambda = upstream * Z psi
    lr0, li0, lr1, li1 = g * r0, g * i0, -g * r1, -g * i1

    for l in tl.static_range(REPS, -1, -1):
        if ANSATZ == "pz":
            if l < REPS:
                r0, i0, r1, i1 = _rz(r0, i0, r1, i1, cx, -sx)
                lr0, li0, lr1, li1 = _rz(lr0, li0, lr1, li1, cx, -sx)

            offs = t_offs + 2 * l + 1
            grad = _grad_ry(r0, i0, r1, i1, lr0, li0, lr1, li1)
            tl.store(grad_theta_ptr + offs, grad, mask=mask)
            c, s = _trig(theta_ptr, offs, mask)
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, c, -s)
            lr0, li0, lr1, li1 = _ry(lr0, li0, lr1, li1, c, -s)

            offs = t_offs + 2 * l
            grad = _grad_rz(r0, i0, r1, i1, lr0, li0, lr1, li1)
            tl.store(grad_theta_ptr + offs, grad, mask=mask)
            c, s = _trig(theta_ptr, offs, mask)
            r0, i0, r1, i1 = _rz(r0, i0, r1, i1, c, -s)
            lr0, li0, lr1, li1 = _rz(lr0, li0, lr1, li1, c, -s)
        elif ANSATZ == "rpz":
            if l < REPS:
                r0, i0, r1, i1 = _rz(r0, i0, r1, i1, cx, -sx)
                lr0, li0, lr1, li1 = _rz(lr0, li0, lr1, li1, cx, -sx)

                # The bias is the angle of an Rz of its own (see _final_state).
                offs = b_offs + l
                grad = _grad_rz(r0, i0, r1, i1, lr0, li0, lr1, li1)
                tl.store(grad_bias_ptr + offs, grad, mask=mask)
                c, s = _trig(bias_ptr, offs, mask)
                r0, i0, r1, i1 = _rz(r0, i0, r1, i1, c, -s)
                lr0, li0, lr1, li1 = _rz(lr0, li0, lr1, li1, c, -s)

            offs = t_offs + l
            grad = _grad_ry(r0, i0, r1, i1, lr0, li0, lr1, li1)
            tl.store(grad_theta_ptr + offs, grad, mask=mask)
            c, s = _trig(theta_ptr, offs, mask)
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, c, -s)
            lr0, li0, lr1, li1 = _ry(lr0, li0, lr1, li1, c, -s)
        elif l < REPS:
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, cx, -sx)
            lr0, li0, lr1, li1 = _ry(lr0, li0, lr1, li1, cx, -sx)
            r1, i1, lr1, li1 = -r1, -i1, -lr1, -li1  # Z

            offs = t_offs + l
            grad = _grad_ry(r0, i0, r1, i1, lr0, li0, lr1, li1)
            tl.store(grad_theta_ptr + offs, grad, mask=mask)
            c, s = _trig(theta_ptr, offs, mask)
            r0, i0, r1, i1 = _ry(r0, i0, r1, i1, c, -s)
            lr0, li0, lr1, li1 = _ry(lr0, li0, lr1, li1, c, -s)

            r0, i0, r1, i1 = r1, i1, r0, i0  # X
            lr0, li0, lr1, li1 = lr1, li1, lr0, li0


# ── Launchers ──────────────────────────────────────────────────────────────


def _coefficients_forward(theta, preacts_bias, reps: int, ansatz: str) -> torch.Tensor:
    """Launch the forward kernel: (O, 2R+1, I) float32. Only rpz touches preacts_bias."""
    out_dim, in_dim = theta.shape[:2]
    n = 2 * reps + 1
    c = torch.empty(out_dim, n, in_dim, device=theta.device, dtype=torch.float32)
    n_edges = out_dim * in_dim
    if n_edges == 0:
        return c
    _coefficients_fwd_kernel[(triton.cdiv(n_edges, _BLOCK),)](
        theta.contiguous(),
        preacts_bias.contiguous() if ansatz == "rpz" else None,
        c,
        n_edges,
        in_dim,
        # Not theta.stride(1): a size-1 dim keeps the stride of whatever view
        # made it and still counts as contiguous.
        theta.shape[2] * theta.shape[3],
        ANSATZ=ansatz,
        REPS=reps,
        SAMPLES=triton.next_power_of_2(n),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return c


def _coefficients_backward(grad_c, theta, preacts_bias, reps: int, ansatz: str):
    """Launch the backward kernel: (grad_theta, grad_bias), grad_bias None unless rpz."""
    theta = theta.contiguous()
    grad_theta = torch.empty_like(theta)
    if ansatz == "rpz":
        preacts_bias = preacts_bias.contiguous()
        grad_bias = torch.empty_like(preacts_bias)
    else:
        preacts_bias = grad_bias = None
    out_dim, in_dim = theta.shape[:2]
    n_edges = out_dim * in_dim
    if n_edges == 0:
        return grad_theta, grad_bias
    _coefficients_bwd_kernel[(triton.cdiv(n_edges, _BLOCK),)](
        theta,
        preacts_bias,
        grad_c.contiguous(),
        grad_theta,
        grad_bias,
        n_edges,
        in_dim,
        theta.shape[2] * theta.shape[3],
        ANSATZ=ansatz,
        REPS=reps,
        SAMPLES=triton.next_power_of_2(2 * reps + 1),
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return grad_theta, grad_bias


class _FourierCoefficients(torch.autograd.Function):
    """``fourier_coefficients`` with the adjoint-method backward."""

    @staticmethod
    def forward(ctx, theta, preacts_bias, reps, ansatz):
        ctx.save_for_backward(theta, preacts_bias)
        ctx.reps, ctx.ansatz = reps, ansatz
        return _coefficients_forward(theta, preacts_bias, reps, ansatz)

    @staticmethod
    def backward(ctx, grad_c):
        theta, preacts_bias = ctx.saved_tensors
        grad_theta, grad_bias = _first_order_only(
            _coefficients_backward(grad_c, theta, preacts_bias, ctx.reps, ctx.ansatz),
            (theta, preacts_bias, grad_c),
            "fourier_coefficients has no second derivatives, so its gradients "
            "cannot be differentiated",
        )
        return grad_theta, grad_bias, None, None


def fourier_coefficients(theta, preacts_bias, reps, ansatz) -> torch.Tensor:
    """Fourier series of every edge's circuit expectation value.

    ``f_oi(x) = a0 + sum_{k=1..R} a_k cos(kx) + b_k sin(kx)`` is <Z> after the
    ``ansatz`` circuit of edge (o, i) on input x, with ``fast_measure=False``
    and preact weights of 1.

    Args:
        theta: (O, I, T, P) on CUDA, float32 or bfloat16, any contiguity.
            (T, P) is (R+1, 2) for pz, (R+1, 1) for rpz and (R, 1) for real.
        preacts_bias: (O, I, R) on CUDA, float32 or bfloat16, for rpz. pz and
            real have no bias: whatever is passed is ignored and gets no
            gradient.
        reps: R, the number of data re-uploads, at most ``MAX_REPS``.
        ansatz: "pz_encoding" / "pz", "rpz_encoding" / "rpz" or "real".

    Returns:
        (O, 2R+1, I) float32 with ``c[o, 0, i] = a0``, ``c[o, k, i] = a_k`` and
        ``c[o, R+k, i] = b_k``. Differentiable once in theta and, for rpz, in
        preacts_bias; gradients come back in the dtype of their tensor.

    Raises:
        ValueError: for another ansatz, reps past ``MAX_REPS``, shapes that do
            not match (ansatz, reps), or another dtype.
    """
    if ansatz not in _ANSATZES:
        raise ValueError(
            f"Unsupported ansatz: {ansatz!r}. Supported: {sorted(_ANSATZES)}"
        )
    reps = int(reps)  # a numpy integer is not a Triton constexpr
    if reps > MAX_REPS:
        raise ValueError(
            f"reps={reps} is past MAX_REPS={MAX_REPS}, where these kernels are "
            "slower than sampling the circuit"
        )
    kind, extra, p = _ANSATZES[ansatz]
    if theta.shape[2:] != (reps + extra, p):
        raise ValueError(
            f"theta for ansatz={ansatz!r}, reps={reps} must be "
            f"(O, I, {reps + extra}, {p}), got {tuple(theta.shape)}"
        )
    if kind != "rpz":
        # pz and real have no bias. Dropped here, it stays out of the graph
        # too, where it could only buy a backward launch for a None gradient.
        preacts_bias = None
    elif preacts_bias.shape != (*theta.shape[:2], reps):
        raise ValueError(
            f"preacts_bias must be {(*theta.shape[:2], reps)}, "
            f"got {tuple(preacts_bias.shape)}"
        )
    dtypes = [t.dtype for t in (theta, preacts_bias) if t is not None]
    if not _DTYPES.issuperset(dtypes):
        # All math is float32: float64 would lose its precision silently.
        raise ValueError(
            f"theta and preacts_bias must be float32 or bfloat16, got {dtypes}"
        )
    return _FourierCoefficients.apply(theta, preacts_bias, reps, kind)
