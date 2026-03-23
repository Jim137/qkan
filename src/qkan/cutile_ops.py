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
cuTile-fused kernels for QKAN quantum circuit simulation.

Implements pz_encoding, rpz_encoding, and real ansatz forward and backward
passes as fused cuTile kernels, avoiding materialization of intermediate
complex state vectors. This is a direct port of the Triton kernels in
fused_ops.py to NVIDIA's cuTile programming model.
"""

import math

import cuda.tile as ct  # type: ignore
import torch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# ── pz_encoding forward kernel ────────────────────────────────────────────


@ct.kernel
def _ct_pz_encoding_kernel(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps+1, 2]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    out,  # [Batch, Out_Dim, In_Dim]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """
    Fused QKAN pz_encoding forward kernel.

    Grid: (out_dim * in_dim, cdiv(batch, BLOCK_B), 1).
    Circuit: H|0> -> [Rz(t0) Ry(t1) Rz(enc)]×reps -> Rz(t0) Ry(t1) -> measure Z
    """
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size

    # Load x[b_offs, idx_i]
    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps):
        # Rz(t0)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Ry(t1)
        t1 = ct.gather(theta, (idx_o, idx_i, layer, 1))
        a = t1 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(enc)
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Rz(t0), Ry(t1)
    t0 = ct.gather(theta, (idx_o, idx_i, reps, 0))
    t1 = ct.gather(theta, (idx_o, idx_i, reps, 1))

    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    a = t1 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if FAST_MEASURE:
        result = ct.sqrt(r0 * r0 + i0 * i0) - ct.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    ct.scatter(out, (b_offs, idx_o, idx_i), result, mask=b_mask)


def cutile_pz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
) -> torch.Tensor:
    """
    Launch the cuTile pz_encoding forward kernel.

    Args:
        x: (batch, in_dim) float32 on CUDA
        theta: (out_dim, in_dim, reps+1, 2) float32 on CUDA
        preacts_w: (out_dim, in_dim, reps) float32
        preacts_b: (out_dim, in_dim, reps) float32
        preacts_trainable: whether preacts are used
        fast_measure: True for |alpha|-|beta|, False for |alpha|^2-|beta|^2

    Returns:
        (batch, out_dim, in_dim) float32
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    x = x.contiguous()
    theta = theta.contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=x.dtype)

    BLOCK_B = 32
    grid = (out_dim * in_dim, math.ceil(batch / BLOCK_B), 1)

    if preacts_trainable:
        preacts_w = preacts_w.contiguous()
        preacts_b = preacts_b.contiguous()

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _ct_pz_encoding_kernel,
        (
            x,
            theta,
            preacts_w,
            preacts_b,
            output,
            batch,
            in_dim,
            out_dim,
            reps,
            preacts_trainable,
            fast_measure,
            BLOCK_B,
        ),
    )

    return output


# ── rpz_encoding forward kernel ───────────────────────────────────────────


@ct.kernel
def _ct_rpz_encoding_kernel(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps+1, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    out,  # [Batch, Out_Dim, In_Dim]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """
    Fused rpz_encoding forward kernel.

    Grid: (out_dim * in_dim, cdiv(batch, BLOCK_B), 1).
    Circuit: H|0> -> [Ry(t0) Rz(w*x+b)]×reps -> Ry(t0) -> measure Z
    """
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size

    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps):
        # Ry(theta)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(w*x+b)
        w = ct.gather(pw, (idx_o, idx_i, layer))
        b = ct.gather(pb, (idx_o, idx_i, layer))
        enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Ry(theta[reps, 0])
    t0 = ct.gather(theta, (idx_o, idx_i, reps, 0))
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if FAST_MEASURE:
        result = ct.sqrt(r0 * r0 + i0 * i0) - ct.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    ct.scatter(out, (b_offs, idx_o, idx_i), result, mask=b_mask)


def cutile_rpz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    fast_measure: bool = True,
) -> torch.Tensor:
    """
    Launch the cuTile rpz_encoding forward kernel.

    Args:
        x: (batch, in_dim) float32
        theta: (out_dim, in_dim, reps+1, 1) float32
        preacts_w: (out_dim, in_dim, reps) float32
        preacts_b: (out_dim, in_dim, reps) float32
        fast_measure: measurement mode

    Returns:
        (batch, out_dim, in_dim) float32
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    x = x.contiguous()
    theta = theta.contiguous()
    preacts_w = preacts_w.contiguous()
    preacts_b = preacts_b.contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=x.dtype)

    BLOCK_B = 32
    grid = (out_dim * in_dim, math.ceil(batch / BLOCK_B), 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _ct_rpz_encoding_kernel,
        (
            x,
            theta,
            preacts_w,
            preacts_b,
            output,
            batch,
            in_dim,
            out_dim,
            reps,
            fast_measure,
            BLOCK_B,
        ),
    )

    return output


# ── real ansatz forward kernel ─────────────────────────────────────────────


@ct.kernel
def _ct_real_encoding_kernel_bf16(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    out,  # [Batch, Out_Dim, In_Dim]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """
    Fused real ansatz forward kernel — real-only (bf16) fast path.

    Circuit: H|0> -> [X, Ry(theta), Z, Ry(enc)]×reps -> measure Z
    No imaginary components needed.
    """
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size

    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)

    for layer in range(reps):
        r0, r1 = r1, r0  # X gate

        # Ry(theta)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = ct.astype(t0, ct.float32) * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        nr1 = s * r0 + c * r1
        r0, r1 = nr0, nr1

        r1 = -r1  # Z gate

        # Ry(enc)
        enc = ct.astype(x_vals, ct.float32)
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = ct.astype(w, ct.float32) * ct.astype(x_vals, ct.float32) + ct.astype(
                b, ct.float32
            )

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        nr1 = s * r0 + c * r1
        r0, r1 = nr0, nr1

    if FAST_MEASURE:
        result = ct.abs(r0) - ct.abs(r1)
    else:
        result = r0 * r0 - r1 * r1

    ct.scatter(out, (b_offs, idx_o, idx_i), result, mask=b_mask)


@ct.kernel
def _ct_real_encoding_kernel_f32(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    out,  # [Batch, Out_Dim, In_Dim]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """
    Fused real ansatz forward kernel — full complex path.

    Circuit: H|0> -> [X, Ry(theta), Z, Ry(enc)]×reps -> measure Z
    """
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size

    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps):
        r0, i0, r1, i1 = r1, i1, r0, i0  # X gate

        # Ry(theta)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        r1 = -r1  # Z gate
        i1 = -i1

        # Ry(enc)
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if FAST_MEASURE:
        result = ct.sqrt(r0 * r0 + i0 * i0) - ct.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    ct.scatter(out, (b_offs, idx_o, idx_i), result, mask=b_mask)


def cutile_real_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Launch the cuTile real ansatz forward kernel.

    Args:
        x: (batch, in_dim)
        theta: (out_dim, in_dim, reps, 1) — no +1 final layer
        preacts_w: (out_dim, in_dim, reps)
        preacts_b: (out_dim, in_dim, reps)
        preacts_trainable: whether preacts are used
        fast_measure: measurement mode
        c_dtype: compute dtype (torch.bfloat16 or torch.float32)

    Returns:
        (batch, out_dim, in_dim) in c_dtype
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2]

    x = x.to(c_dtype).contiguous()
    theta = theta.to(c_dtype).contiguous()
    preacts_w = preacts_w.to(c_dtype).contiguous()
    preacts_b = preacts_b.to(c_dtype).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=c_dtype)

    compute_bf16 = c_dtype == torch.bfloat16
    BLOCK_B = 32 if compute_bf16 else 1
    grid = (out_dim * in_dim, math.ceil(batch / BLOCK_B), 1)

    kernel = (
        _ct_real_encoding_kernel_bf16 if compute_bf16 else _ct_real_encoding_kernel_f32
    )

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            x,
            theta,
            preacts_w,
            preacts_b,
            output,
            batch,
            in_dim,
            out_dim,
            reps,
            preacts_trainable,
            fast_measure,
            BLOCK_B,
        ),
    )

    return output


# ── pz_encoding backward kernel ───────────────────────────────────────────


@ct.kernel
def _ct_pz_encoding_backward_kernel(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps+1, 2]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, BLOCK_B, 4]
    grad_theta,  # [Out_Dim, In_Dim, Reps+1, 2]
    grad_x,  # [Batch, In_Dim]
    grad_pw,  # [Out_Dim, In_Dim, Reps]
    grad_pb,  # [Out_Dim, In_Dim, Reps]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    n_b_blocks: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """Backward kernel for pz_encoding ansatz."""
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size
    b_range = ct.arange(BLOCK_B, dtype=ct.int32)

    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    program_idx = pid_oi * n_b_blocks + pid_b

    # ── Phase 1: Forward recompute, saving states ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    # Save H state
    ct.scatter(states, (program_idx, 0, b_range, 0), r0, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 1), i0, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 2), r1, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 3), i1, mask=b_mask)

    state_idx = 1
    for layer in range(reps):
        # Rz(t0)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

        # Ry(t1)
        t1 = ct.gather(theta, (idx_o, idx_i, layer, 1))
        a = t1 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

        # Rz(enc)
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

    # Final Rz(t0)
    t0 = ct.gather(theta, (idx_o, idx_i, reps, 0))
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
    ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
    ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
    ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
    state_idx = state_idx + 1

    # Final Ry(t1)
    t1 = ct.gather(theta, (idx_o, idx_i, reps, 1))
    a = t1 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = ct.gather(grad_out, (b_offs, idx_o, idx_i), mask=b_mask, padding_value=0.0)

    if FAST_MEASURE:
        alpha_norm = ct.sqrt(r0 * r0 + i0 * i0)
        beta_norm = ct.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = ct.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = ct.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
        ar0 = go * r0 * inv_alpha
        ai0 = go * i0 * inv_alpha
        ar1 = -go * r1 * inv_beta
        ai1 = -go * i1 * inv_beta
    else:
        ar0 = 2.0 * go * r0
        ai0 = 2.0 * go * i0
        ar1 = -2.0 * go * r1
        ai1 = -2.0 * go * i1

    # ── Phase 3: Backward sweep ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    # Backward through final Ry(t1_final)
    state_idx = state_idx - 1
    sr0 = ct.gather(
        states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
    )
    si0 = ct.gather(
        states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
    )
    sr1 = ct.gather(
        states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
    )
    si1 = ct.gather(
        states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
    )

    t1 = ct.gather(theta, (idx_o, idx_i, reps, 1))
    a = t1 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)

    grad_t1_vec = 0.5 * (
        ar0 * (-s * sr0 - c * sr1)
        + ai0 * (-s * si0 - c * si1)
        + ar1 * (c * sr0 - s * sr1)
        + ai1 * (c * si0 - s * si1)
    )
    ct.atomic_add(
        grad_theta,
        (idx_o, idx_i, reps, 1),
        ct.sum(ct.where(b_mask, grad_t1_vec, 0.0)),
    )

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # Backward through final Rz(t0_final)
    state_idx = state_idx - 1
    sr0 = ct.gather(
        states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
    )
    si0 = ct.gather(
        states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
    )
    sr1 = ct.gather(
        states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
    )
    si1 = ct.gather(
        states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
    )

    t0 = ct.gather(theta, (idx_o, idx_i, reps, 0))
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)

    grad_t0_vec = 0.5 * (
        -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
        + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
    )
    ct.atomic_add(
        grad_theta,
        (idx_o, idx_i, reps, 0),
        ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
    )

    nar0 = c * ar0 - s * ai0
    nai0 = s * ar0 + c * ai0
    nar1 = c * ar1 + s * ai1
    nai1 = -s * ar1 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    for layer in range(reps - 1, -1, -1):
        # Backward through Rz(enc)
        state_idx = state_idx - 1
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        si0 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
        )
        si1 = ct.gather(
            states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
        )

        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_enc = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        if PREACTS_TRAINABLE:
            ct.atomic_add(
                grad_pw,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc * x_vals, 0.0)),
            )
            ct.atomic_add(
                grad_pb,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc, 0.0)),
            )
            grad_x_local = grad_x_local + grad_enc * w
        else:
            grad_x_local = grad_x_local + grad_enc

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(t1)
        state_idx = state_idx - 1
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        si0 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
        )
        si1 = ct.gather(
            states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
        )

        t1 = ct.gather(theta, (idx_o, idx_i, layer, 1))
        a = t1 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_t1_vec = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 1),
            ct.sum(ct.where(b_mask, grad_t1_vec, 0.0)),
        )

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Rz(t0)
        state_idx = state_idx - 1
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        si0 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
        )
        si1 = ct.gather(
            states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
        )

        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_t0_vec = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
        )

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    ct.atomic_add(grad_x, (b_offs, idx_i), grad_x_local, mask=b_mask)


def cutile_pz_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    pw: torch.Tensor,
    pb: torch.Tensor,
    grad_output: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool,
) -> tuple:
    """Launch pz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    x = x.contiguous()
    theta = theta.contiguous()
    grad_output = grad_output.contiguous()

    BLOCK_B = 32
    n_states = 3 * reps + 3
    n_b_blocks = math.ceil(batch / BLOCK_B)
    n_programs = out_dim * in_dim * n_b_blocks
    states = torch.empty(
        n_programs, n_states, BLOCK_B, 4, device=x.device, dtype=x.dtype
    )

    grad_theta = torch.zeros_like(theta)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=x.dtype)

    if preacts_trainable:
        pw = pw.contiguous()
        pb = pb.contiguous()
        grad_pw = torch.zeros_like(pw)
        grad_pb = torch.zeros_like(pb)
    else:
        grad_pw = torch.zeros(1, device=x.device)
        grad_pb = torch.zeros(1, device=x.device)

    grid = (out_dim * in_dim, n_b_blocks, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _ct_pz_encoding_backward_kernel,
        (
            x,
            theta,
            pw,
            pb,
            grad_output,
            states,
            grad_theta,
            grad_x,
            grad_pw,
            grad_pb,
            batch,
            in_dim,
            out_dim,
            reps,
            n_b_blocks,
            preacts_trainable,
            fast_measure,
            BLOCK_B,
        ),
    )

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )


# ── rpz_encoding backward kernel ──────────────────────────────────────────


@ct.kernel
def _ct_rpz_encoding_backward_kernel(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps+1, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, BLOCK_B, 4]
    grad_theta,  # [Out_Dim, In_Dim, Reps+1, 1]
    grad_x,  # [Batch, In_Dim]
    grad_pw,  # [Out_Dim, In_Dim, Reps]
    grad_pb,  # [Out_Dim, In_Dim, Reps]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    n_b_blocks: ConstInt,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """Backward kernel for rpz_encoding ansatz."""
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size
    b_range = ct.arange(BLOCK_B, dtype=ct.int32)

    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    program_idx = pid_oi * n_b_blocks + pid_b

    # ── Phase 1: Forward recompute, saving states ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    ct.scatter(states, (program_idx, 0, b_range, 0), r0, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 1), i0, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 2), r1, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 3), i1, mask=b_mask)

    state_idx = 1
    for layer in range(reps):
        # Ry(theta)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

        # Rz(w*x+b)
        w = ct.gather(pw, (idx_o, idx_i, layer))
        b = ct.gather(pb, (idx_o, idx_i, layer))
        enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

    # Final Ry(theta[reps,0])
    t0 = ct.gather(theta, (idx_o, idx_i, reps, 0))
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = ct.gather(grad_out, (b_offs, idx_o, idx_i), mask=b_mask, padding_value=0.0)

    if FAST_MEASURE:
        alpha_norm = ct.sqrt(r0 * r0 + i0 * i0)
        beta_norm = ct.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = ct.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = ct.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
        ar0 = go * r0 * inv_alpha
        ai0 = go * i0 * inv_alpha
        ar1 = -go * r1 * inv_beta
        ai1 = -go * i1 * inv_beta
    else:
        ar0 = 2.0 * go * r0
        ai0 = 2.0 * go * i0
        ar1 = -2.0 * go * r1
        ai1 = -2.0 * go * i1

    # ── Phase 3: Backward sweep ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    # Backward through final Ry(theta[reps,0])
    state_idx = state_idx - 1
    sr0 = ct.gather(
        states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
    )
    si0 = ct.gather(
        states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
    )
    sr1 = ct.gather(
        states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
    )
    si1 = ct.gather(
        states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
    )

    t0 = ct.gather(theta, (idx_o, idx_i, reps, 0))
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)

    grad_t0_vec = 0.5 * (
        ar0 * (-s * sr0 - c * sr1)
        + ai0 * (-s * si0 - c * si1)
        + ar1 * (c * sr0 - s * sr1)
        + ai1 * (c * si0 - s * si1)
    )
    ct.atomic_add(
        grad_theta,
        (idx_o, idx_i, reps, 0),
        ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
    )

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    for layer in range(reps - 1, -1, -1):
        # Backward through Rz(enc)
        state_idx = state_idx - 1
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        si0 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
        )
        si1 = ct.gather(
            states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
        )

        w = ct.gather(pw, (idx_o, idx_i, layer))
        b = ct.gather(pb, (idx_o, idx_i, layer))
        enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_enc = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        ct.atomic_add(
            grad_pw,
            (idx_o, idx_i, layer),
            ct.sum(ct.where(b_mask, grad_enc * x_vals, 0.0)),
        )
        ct.atomic_add(
            grad_pb,
            (idx_o, idx_i, layer),
            ct.sum(ct.where(b_mask, grad_enc, 0.0)),
        )
        grad_x_local = grad_x_local + grad_enc * w

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(theta[l,0])
        state_idx = state_idx - 1
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        si0 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
        )
        si1 = ct.gather(
            states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
        )

        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_t0_vec = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
        )

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    ct.atomic_add(grad_x, (b_offs, idx_i), grad_x_local, mask=b_mask)


def cutile_rpz_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    pw: torch.Tensor,
    pb: torch.Tensor,
    grad_output: torch.Tensor,
    fast_measure: bool,
) -> tuple:
    """Launch rpz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    x = x.contiguous()
    theta = theta.contiguous()
    pw = pw.contiguous()
    pb = pb.contiguous()
    grad_output = grad_output.contiguous()

    BLOCK_B = 32
    n_states = 2 * reps + 2
    n_b_blocks = math.ceil(batch / BLOCK_B)
    n_programs = out_dim * in_dim * n_b_blocks
    states = torch.empty(
        n_programs, n_states, BLOCK_B, 4, device=x.device, dtype=x.dtype
    )

    grad_theta = torch.zeros_like(theta)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=x.dtype)
    grad_pw = torch.zeros_like(pw)
    grad_pb = torch.zeros_like(pb)

    grid = (out_dim * in_dim, n_b_blocks, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _ct_rpz_encoding_backward_kernel,
        (
            x,
            theta,
            pw,
            pb,
            grad_output,
            states,
            grad_theta,
            grad_x,
            grad_pw,
            grad_pb,
            batch,
            in_dim,
            out_dim,
            reps,
            n_b_blocks,
            fast_measure,
            BLOCK_B,
        ),
    )

    return grad_x, grad_theta, grad_pw, grad_pb


# ── real ansatz backward kernel ────────────────────────────────────────────


@ct.kernel
def _ct_real_encoding_backward_kernel_bf16(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, BLOCK_B, 2]
    grad_theta,  # [Out_Dim, In_Dim, Reps, 1] (float32)
    grad_x,  # [Batch, In_Dim] (float32)
    grad_pw,  # [Out_Dim, In_Dim, Reps] (float32)
    grad_pb,  # [Out_Dim, In_Dim, Reps] (float32)
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    n_b_blocks: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """Backward kernel for real ansatz — real-only (bf16) path."""
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size
    b_range = ct.arange(BLOCK_B, dtype=ct.int32)

    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    program_idx = pid_oi * n_b_blocks + pid_b

    # ── Phase 1: Forward recompute ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)

    ct.scatter(states, (program_idx, 0, b_range, 0), r0, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 1), r1, mask=b_mask)

    state_idx = 1
    for layer in range(reps):
        r0, r1 = r1, r0  # X gate

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), r1, mask=b_mask)
        state_idx = state_idx + 1

        # Ry(theta)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = ct.astype(t0, ct.float32) * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        nr1 = s * r0 + c * r1
        r0, r1 = nr0, nr1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), r1, mask=b_mask)
        state_idx = state_idx + 1

        r1 = -r1  # Z gate

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), r1, mask=b_mask)
        state_idx = state_idx + 1

        # Ry(enc)
        enc = ct.astype(x_vals, ct.float32)
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = ct.astype(w, ct.float32) * ct.astype(x_vals, ct.float32) + ct.astype(
                b, ct.float32
            )

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        nr1 = s * r0 + c * r1
        r0, r1 = nr0, nr1

    # ── Phase 2: Measurement gradient ──
    go = ct.gather(grad_out, (b_offs, idx_o, idx_i), mask=b_mask, padding_value=0.0)

    if FAST_MEASURE:
        ar0 = ct.where(ct.abs(r0) > 1e-30, go * r0 / ct.abs(r0), 0.0)
        ar1 = ct.where(ct.abs(r1) > 1e-30, -go * r1 / ct.abs(r1), 0.0)
    else:
        ar0 = 2.0 * go * r0
        ar1 = -2.0 * go * r1

    # ── Phase 3: Backward sweep ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps - 1, -1, -1):
        # Backward through Ry(enc)
        state_idx = state_idx - 1
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )

        enc = ct.astype(x_vals, ct.float32)
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = ct.astype(w, ct.float32) * ct.astype(x_vals, ct.float32) + ct.astype(
                b, ct.float32
            )

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_enc = 0.5 * (ar0 * (-s * sr0 - c * sr1) + ar1 * (c * sr0 - s * sr1))

        if PREACTS_TRAINABLE:
            ct.atomic_add(
                grad_pw,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc * ct.astype(x_vals, ct.float32), 0.0)),
            )
            ct.atomic_add(
                grad_pb,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc, 0.0)),
            )
            grad_x_local = grad_x_local + grad_enc * ct.astype(w, ct.float32)
        else:
            grad_x_local = grad_x_local + grad_enc

        nar0 = c * ar0 + s * ar1
        nar1 = -s * ar0 + c * ar1
        ar0, ar1 = nar0, nar1

        # Backward through Z gate
        ar1 = -ar1

        # Backward through Ry(theta)
        state_idx = state_idx - 2
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )

        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = ct.astype(t0, ct.float32) * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_t0_vec = 0.5 * (ar0 * (-s * sr0 - c * sr1) + ar1 * (c * sr0 - s * sr1))
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
        )

        nar0 = c * ar0 + s * ar1
        nar1 = -s * ar0 + c * ar1
        ar0, ar1 = nar0, nar1

        # Backward through X gate
        ar0, ar1 = ar1, ar0

    ct.atomic_add(grad_x, (b_offs, idx_i), grad_x_local, mask=b_mask)


@ct.kernel
def _ct_real_encoding_backward_kernel_f32(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, BLOCK_B, 4]
    grad_theta,  # [Out_Dim, In_Dim, Reps, 1] (float32)
    grad_x,  # [Batch, In_Dim] (float32)
    grad_pw,  # [Out_Dim, In_Dim, Reps] (float32)
    grad_pb,  # [Out_Dim, In_Dim, Reps] (float32)
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    n_b_blocks: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """Backward kernel for real ansatz — full complex path."""
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size
    b_range = ct.arange(BLOCK_B, dtype=ct.int32)

    x_vals = ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0)

    program_idx = pid_oi * n_b_blocks + pid_b

    # ── Phase 1: Forward recompute ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    ct.scatter(states, (program_idx, 0, b_range, 0), r0, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 1), i0, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 2), r1, mask=b_mask)
    ct.scatter(states, (program_idx, 0, b_range, 3), i1, mask=b_mask)

    state_idx = 1
    for layer in range(reps):
        r0, i0, r1, i1 = r1, i1, r0, i0  # X gate

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

        # Ry(theta)
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

        r1 = -r1  # Z gate
        i1 = -i1

        ct.scatter(states, (program_idx, state_idx, b_range, 0), r0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 1), i0, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 2), r1, mask=b_mask)
        ct.scatter(states, (program_idx, state_idx, b_range, 3), i1, mask=b_mask)
        state_idx = state_idx + 1

        # Ry(enc)
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = ct.gather(grad_out, (b_offs, idx_o, idx_i), mask=b_mask, padding_value=0.0)

    if FAST_MEASURE:
        alpha_norm = ct.sqrt(r0 * r0 + i0 * i0)
        beta_norm = ct.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = ct.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = ct.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
        ar0 = go * r0 * inv_alpha
        ai0 = go * i0 * inv_alpha
        ar1 = -go * r1 * inv_beta
        ai1 = -go * i1 * inv_beta
    else:
        ar0 = 2.0 * go * r0
        ai0 = 2.0 * go * i0
        ar1 = -2.0 * go * r1
        ai1 = -2.0 * go * i1

    # ── Phase 3: Backward sweep ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps - 1, -1, -1):
        # Backward through Ry(enc)
        state_idx = state_idx - 1
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        si0 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
        )
        si1 = ct.gather(
            states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
        )

        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_enc = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )

        if PREACTS_TRAINABLE:
            ct.atomic_add(
                grad_pw,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc * x_vals, 0.0)),
            )
            ct.atomic_add(
                grad_pb,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc, 0.0)),
            )
            grad_x_local = grad_x_local + grad_enc * w
        else:
            grad_x_local = grad_x_local + grad_enc

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Z gate
        ar1 = -ar1
        ai1 = -ai1

        # Backward through Ry(theta)
        state_idx = state_idx - 2
        sr0 = ct.gather(
            states, (program_idx, state_idx, b_range, 0), mask=b_mask, padding_value=0.0
        )
        si0 = ct.gather(
            states, (program_idx, state_idx, b_range, 1), mask=b_mask, padding_value=0.0
        )
        sr1 = ct.gather(
            states, (program_idx, state_idx, b_range, 2), mask=b_mask, padding_value=0.0
        )
        si1 = ct.gather(
            states, (program_idx, state_idx, b_range, 3), mask=b_mask, padding_value=0.0
        )

        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)

        grad_t0_vec = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
        )

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through X gate
        ar0, ai0, ar1, ai1 = ar1, ai1, ar0, ai0

    ct.atomic_add(grad_x, (b_offs, idx_i), grad_x_local, mask=b_mask)


def cutile_real_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    pw: torch.Tensor,
    pb: torch.Tensor,
    grad_output: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool,
    c_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Launch real ansatz backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2]

    x = x.to(c_dtype).contiguous()
    theta = theta.to(c_dtype).contiguous()
    pw = pw.to(c_dtype).contiguous()
    pb = pb.to(c_dtype).contiguous()
    grad_output = grad_output.contiguous()

    compute_bf16 = c_dtype == torch.bfloat16
    BLOCK_B = 32 if compute_bf16 else 1

    n_states = 3 * reps + 1
    n_b_blocks = math.ceil(batch / BLOCK_B)
    n_programs = out_dim * in_dim * n_b_blocks
    n_components = 2 if compute_bf16 else 4
    states = torch.empty(
        n_programs,
        n_states,
        BLOCK_B,
        n_components,
        device=x.device,
        dtype=torch.float32,
    )

    grad_theta = torch.zeros(theta.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)

    if preacts_trainable:
        grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)
    else:
        grad_pw = torch.zeros(1, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(1, device=x.device, dtype=torch.float32)

    grid = (out_dim * in_dim, n_b_blocks, 1)
    kernel = (
        _ct_real_encoding_backward_kernel_bf16
        if compute_bf16
        else _ct_real_encoding_backward_kernel_f32
    )

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            x,
            theta,
            pw,
            pb,
            grad_output,
            states,
            grad_theta,
            grad_x,
            grad_pw,
            grad_pb,
            batch,
            in_dim,
            out_dim,
            reps,
            n_b_blocks,
            preacts_trainable,
            fast_measure,
            BLOCK_B,
        ),
    )

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )
