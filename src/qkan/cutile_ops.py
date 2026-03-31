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

from qkan._kernel_utils import _select_block_b

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
    COMPUTE_BF16: ConstBool,
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

    x_vals = ct.astype(
        ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0), ct.float32
    )

    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps):
        # Rz(t0)
        t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, layer, 0)), ct.float32)
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Ry(t1)
        t1 = ct.astype(ct.gather(theta, (idx_o, idx_i, layer, 1)), ct.float32)
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
            w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
            b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
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
    t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, reps, 0)), ct.float32)
    t1 = ct.astype(ct.gather(theta, (idx_o, idx_i, reps, 1)), ct.float32)

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

    if COMPUTE_BF16:
        ct.scatter(
            out, (b_offs, idx_o, idx_i), ct.astype(result, ct.bfloat16), mask=b_mask
        )
    else:
        ct.scatter(out, (b_offs, idx_o, idx_i), result, mask=b_mask)


def cutile_pz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Launch the cuTile pz_encoding forward kernel.

    Args:
        x: (batch, in_dim) on CUDA
        theta: (out_dim, in_dim, reps+1, 2) on CUDA
        preacts_w: (out_dim, in_dim, reps)
        preacts_b: (out_dim, in_dim, reps)
        preacts_trainable: whether preacts are used
        fast_measure: True for |alpha|-|beta|, False for |alpha|^2-|beta|^2
        c_dtype: compute/storage dtype (torch.float32 or torch.bfloat16)

    Returns:
        (batch, out_dim, in_dim) in c_dtype
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=io_dtype)

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    grid = (n_oi, math.ceil(batch / BLOCK_B), 1)

    if preacts_trainable:
        preacts_w = preacts_w.to(io_dtype).contiguous()
        preacts_b = preacts_b.to(io_dtype).contiguous()

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
            compute_bf16,
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
    COMPUTE_BF16: ConstBool,
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

    x_vals = ct.astype(
        ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0), ct.float32
    )

    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps):
        # Ry(theta)
        t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, layer, 0)), ct.float32)
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(w*x+b)
        w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
        b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
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
    t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, reps, 0)), ct.float32)
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

    if COMPUTE_BF16:
        ct.scatter(
            out, (b_offs, idx_o, idx_i), ct.astype(result, ct.bfloat16), mask=b_mask
        )
    else:
        ct.scatter(out, (b_offs, idx_o, idx_i), result, mask=b_mask)


def cutile_rpz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Launch the cuTile rpz_encoding forward kernel.

    Args:
        x: (batch, in_dim) on CUDA
        theta: (out_dim, in_dim, reps+1, 1) on CUDA
        preacts_w: (out_dim, in_dim, reps)
        preacts_b: (out_dim, in_dim, reps)
        fast_measure: measurement mode
        c_dtype: compute/storage dtype (torch.float32 or torch.bfloat16)

    Returns:
        (batch, out_dim, in_dim) in c_dtype
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    preacts_w = preacts_w.to(io_dtype).contiguous()
    preacts_b = preacts_b.to(io_dtype).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=io_dtype)

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    grid = (n_oi, math.ceil(batch / BLOCK_B), 1)

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
            compute_bf16,
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

    ct.scatter(out, (b_offs, idx_o, idx_i), ct.astype(result, ct.bfloat16), mask=b_mask)


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

    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    preacts_w = preacts_w.to(io_dtype).contiguous()
    preacts_b = preacts_b.to(io_dtype).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=io_dtype)

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch, 32 if compute_bf16 else 1)
    grid = (n_oi, math.ceil(batch / BLOCK_B), 1)

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


@ct.kernel(occupancy=4)
def _ct_pz_encoding_backward_kernel(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps+1, 2]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, 4, BLOCK_B]
    trig_cache,  # [N_Programs, 2*Reps+2, 2] — cached cos/sin for theta gates
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
    COMPUTE_BF16: ConstBool,
    COMPUTE_FP8: ConstBool,
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

    x_vals = ct.astype(
        ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0), ct.float32
    )

    pi = pid_oi * n_b_blocks + pid_b

    def _save4(sidx, v0, v1, v2, v3):
        if COMPUTE_FP8:
            FP8_S = 224.0
            ct.scatter(
                states,
                (pi, sidx, 0, b_range),
                ct.astype(v0 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 1, b_range),
                ct.astype(v1 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 2, b_range),
                ct.astype(v2 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 3, b_range),
                ct.astype(v3 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
        elif COMPUTE_BF16:
            ct.scatter(
                states, (pi, sidx, 0, b_range), ct.astype(v0, ct.bfloat16), mask=b_mask
            )
            ct.scatter(
                states, (pi, sidx, 1, b_range), ct.astype(v1, ct.bfloat16), mask=b_mask
            )
            ct.scatter(
                states, (pi, sidx, 2, b_range), ct.astype(v2, ct.bfloat16), mask=b_mask
            )
            ct.scatter(
                states, (pi, sidx, 3, b_range), ct.astype(v3, ct.bfloat16), mask=b_mask
            )
        else:
            ct.scatter(states, (pi, sidx, 0, b_range), v0, mask=b_mask)
            ct.scatter(states, (pi, sidx, 1, b_range), v1, mask=b_mask)
            ct.scatter(states, (pi, sidx, 2, b_range), v2, mask=b_mask)
            ct.scatter(states, (pi, sidx, 3, b_range), v3, mask=b_mask)

    def _load4(sidx):
        if COMPUTE_FP8:
            INV_S = 0.00446428571428  # 1/224
            return (
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
            )
        elif COMPUTE_BF16:
            return (
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
            )
        else:
            return (
                ct.gather(
                    states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                ),
            )

    # ── Phase 1: Forward recompute, saving states + trig cache ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    _save4(0, r0, i0, r1, i1)

    state_idx = 1
    trig_idx = 0
    for layer in range(reps):
        # Rz(t0) — cache trig
        t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, layer, 0)), ct.float32)
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        ct.scatter(trig_cache, (pi, trig_idx, 0), c)
        ct.scatter(trig_cache, (pi, trig_idx, 1), s)
        trig_idx = trig_idx + 1
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        # Ry(t1) — cache trig
        t1 = ct.astype(ct.gather(theta, (idx_o, idx_i, layer, 1)), ct.float32)
        a = t1 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        ct.scatter(trig_cache, (pi, trig_idx, 0), c)
        ct.scatter(trig_cache, (pi, trig_idx, 1), s)
        trig_idx = trig_idx + 1
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        # Rz(enc) — NOT cached (depends on x_vals per batch element)
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
            b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

    # Final Rz(t0) — cache trig
    t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, reps, 0)), ct.float32)
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    ct.scatter(trig_cache, (pi, trig_idx, 0), c)
    ct.scatter(trig_cache, (pi, trig_idx, 1), s)
    trig_idx = trig_idx + 1
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    _save4(state_idx, r0, i0, r1, i1)
    state_idx = state_idx + 1

    # Final Ry(t1) — cache trig
    t1 = ct.astype(ct.gather(theta, (idx_o, idx_i, reps, 1)), ct.float32)
    a = t1 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    ct.scatter(trig_cache, (pi, trig_idx, 0), c)
    ct.scatter(trig_cache, (pi, trig_idx, 1), s)
    trig_idx = trig_idx + 1
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = ct.astype(
        ct.gather(grad_out, (b_offs, idx_o, idx_i), mask=b_mask, padding_value=0.0),
        ct.float32,
    )

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

    # ── Phase 3: Backward sweep (using cached trig for theta gates) ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    # Backward through final Ry(t1_final) — load cached trig
    trig_idx = trig_idx - 1
    state_idx = state_idx - 1
    sr0, si0, sr1, si1 = _load4(state_idx)

    c = ct.gather(trig_cache, (pi, trig_idx, 0))
    s = ct.gather(trig_cache, (pi, trig_idx, 1))

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
        memory_order=ct.MemoryOrder.RELAXED,
    )

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # Backward through final Rz(t0_final) — load cached trig
    trig_idx = trig_idx - 1
    state_idx = state_idx - 1
    sr0, si0, sr1, si1 = _load4(state_idx)

    c = ct.gather(trig_cache, (pi, trig_idx, 0))
    s = ct.gather(trig_cache, (pi, trig_idx, 1))

    grad_t0_vec = 0.5 * (
        -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
        + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
    )
    ct.atomic_add(
        grad_theta,
        (idx_o, idx_i, reps, 0),
        ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
        memory_order=ct.MemoryOrder.RELAXED,
    )

    nar0 = c * ar0 - s * ai0
    nai0 = s * ar0 + c * ai0
    nar1 = c * ar1 + s * ai1
    nai1 = -s * ar1 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    for _ri in range(reps):
        layer = reps - 1 - _ri
        # Backward through Rz(enc) — must recompute (depends on x_vals)
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)

        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
            b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
            enc = w * x_vals + b

        ae = enc * 0.5
        ce = ct.cos(ae)
        se = ct.sin(ae)

        grad_enc = 0.5 * (
            -se * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + ce * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        if PREACTS_TRAINABLE:
            ct.atomic_add(
                grad_pw,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc * x_vals, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            ct.atomic_add(
                grad_pb,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            grad_x_local = grad_x_local + grad_enc * w
        else:
            grad_x_local = grad_x_local + grad_enc

        nar0 = ce * ar0 - se * ai0
        nai0 = se * ar0 + ce * ai0
        nar1 = ce * ar1 + se * ai1
        nai1 = -se * ar1 + ce * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(t1) — load cached trig
        trig_idx = trig_idx - 1
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)

        ct1 = ct.gather(trig_cache, (pi, trig_idx, 0))
        st1 = ct.gather(trig_cache, (pi, trig_idx, 1))

        grad_t1_vec = 0.5 * (
            ar0 * (-st1 * sr0 - ct1 * sr1)
            + ai0 * (-st1 * si0 - ct1 * si1)
            + ar1 * (ct1 * sr0 - st1 * sr1)
            + ai1 * (ct1 * si0 - st1 * si1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 1),
            ct.sum(ct.where(b_mask, grad_t1_vec, 0.0)),
            memory_order=ct.MemoryOrder.RELAXED,
        )

        nar0 = ct1 * ar0 + st1 * ar1
        nai0 = ct1 * ai0 + st1 * ai1
        nar1 = -st1 * ar0 + ct1 * ar1
        nai1 = -st1 * ai0 + ct1 * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Rz(t0) — load cached trig
        trig_idx = trig_idx - 1
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)

        ct0 = ct.gather(trig_cache, (pi, trig_idx, 0))
        st0 = ct.gather(trig_cache, (pi, trig_idx, 1))

        grad_t0_vec = 0.5 * (
            -st0 * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + ct0 * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
            memory_order=ct.MemoryOrder.RELAXED,
        )

        nar0 = ct0 * ar0 - st0 * ai0
        nai0 = st0 * ar0 + ct0 * ai0
        nar1 = ct0 * ar1 + st0 * ai1
        nai1 = -st0 * ar1 + ct0 * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    ct.atomic_add(
        grad_x,
        (b_offs, idx_i),
        ct.where(b_mask, grad_x_local, 0.0),
        memory_order=ct.MemoryOrder.RELAXED,
    )


def cutile_pz_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    pw: torch.Tensor,
    pb: torch.Tensor,
    grad_output: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool,
    c_dtype: torch.dtype = torch.float32,
) -> tuple:
    """Launch pz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    grad_output = grad_output.contiguous()

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    n_states = 3 * reps + 3
    n_b_blocks = math.ceil(batch / BLOCK_B)
    n_programs = n_oi * n_b_blocks
    # BF16: store states in bf16 to halve memory traffic (dominant cost)
    compute_fp8 = c_dtype == torch.float8_e4m3fn
    if compute_fp8:
        states_dtype = torch.float8_e4m3fn
    elif compute_bf16:
        states_dtype = torch.bfloat16
    else:
        states_dtype = torch.float32
    states = torch.empty(
        n_programs, n_states, 4, BLOCK_B, device=x.device, dtype=states_dtype
    )
    # Trig cache always f32 (small, accuracy-sensitive)
    n_trig = 2 * reps + 2
    trig_cache = torch.empty(
        n_programs, n_trig, 2, device=x.device, dtype=torch.float32
    )

    grad_theta = torch.zeros(theta.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)

    if preacts_trainable:
        pw = pw.to(io_dtype).contiguous()
        pb = pb.to(io_dtype).contiguous()
        grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)
    else:
        grad_pw = torch.zeros(1, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(1, device=x.device, dtype=torch.float32)

    grid = (n_oi, n_b_blocks, 1)
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
            trig_cache,
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
            compute_bf16,
            compute_fp8,
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


@ct.kernel(occupancy=4)
def _ct_rpz_encoding_backward_kernel(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps+1, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, 4, BLOCK_B]
    trig_cache,  # [N_Programs, Reps+1, 2] — cached cos/sin for Ry(theta) gates
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
    COMPUTE_BF16: ConstBool,
    COMPUTE_FP8: ConstBool,
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

    x_vals = ct.astype(
        ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0), ct.float32
    )

    pi = pid_oi * n_b_blocks + pid_b

    def _save4(sidx, v0, v1, v2, v3):
        if COMPUTE_FP8:
            FP8_S = 224.0
            ct.scatter(
                states,
                (pi, sidx, 0, b_range),
                ct.astype(v0 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 1, b_range),
                ct.astype(v1 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 2, b_range),
                ct.astype(v2 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 3, b_range),
                ct.astype(v3 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
        elif COMPUTE_BF16:
            ct.scatter(
                states, (pi, sidx, 0, b_range), ct.astype(v0, ct.bfloat16), mask=b_mask
            )
            ct.scatter(
                states, (pi, sidx, 1, b_range), ct.astype(v1, ct.bfloat16), mask=b_mask
            )
            ct.scatter(
                states, (pi, sidx, 2, b_range), ct.astype(v2, ct.bfloat16), mask=b_mask
            )
            ct.scatter(
                states, (pi, sidx, 3, b_range), ct.astype(v3, ct.bfloat16), mask=b_mask
            )
        else:
            ct.scatter(states, (pi, sidx, 0, b_range), v0, mask=b_mask)
            ct.scatter(states, (pi, sidx, 1, b_range), v1, mask=b_mask)
            ct.scatter(states, (pi, sidx, 2, b_range), v2, mask=b_mask)
            ct.scatter(states, (pi, sidx, 3, b_range), v3, mask=b_mask)

    def _load4(sidx):
        if COMPUTE_FP8:
            INV_S = 0.00446428571428  # 1/224
            return (
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
            )
        elif COMPUTE_BF16:
            return (
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                ),
            )
        else:
            return (
                ct.gather(
                    states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                ),
            )

    # ── Phase 1: Forward recompute, saving states + trig cache ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    _save4(0, r0, i0, r1, i1)

    state_idx = 1
    trig_idx = 0
    for layer in range(reps):
        # Ry(theta) — cache trig
        t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, layer, 0)), ct.float32)
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        ct.scatter(trig_cache, (pi, trig_idx, 0), c)
        ct.scatter(trig_cache, (pi, trig_idx, 1), s)
        trig_idx = trig_idx + 1
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        # Rz(w*x+b) — NOT cached (depends on x_vals per batch element)
        w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
        b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
        enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

    # Final Ry(theta[reps,0]) — cache trig
    t0 = ct.astype(ct.gather(theta, (idx_o, idx_i, reps, 0)), ct.float32)
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    ct.scatter(trig_cache, (pi, trig_idx, 0), c)
    ct.scatter(trig_cache, (pi, trig_idx, 1), s)
    trig_idx = trig_idx + 1
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = ct.astype(
        ct.gather(grad_out, (b_offs, idx_o, idx_i), mask=b_mask, padding_value=0.0),
        ct.float32,
    )

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

    # ── Phase 3: Backward sweep (using cached trig for theta gates) ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    # Backward through final Ry(theta[reps,0]) — load cached trig
    trig_idx = trig_idx - 1
    state_idx = state_idx - 1
    sr0, si0, sr1, si1 = _load4(state_idx)

    c = ct.gather(trig_cache, (pi, trig_idx, 0))
    s = ct.gather(trig_cache, (pi, trig_idx, 1))

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
        memory_order=ct.MemoryOrder.RELAXED,
    )

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    for _ri in range(reps):
        layer = reps - 1 - _ri
        # Backward through Rz(enc) — must recompute (depends on x_vals)
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)

        w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
        b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
        enc = w * x_vals + b

        ae = enc * 0.5
        ce = ct.cos(ae)
        se = ct.sin(ae)

        grad_enc = 0.5 * (
            -se * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + ce * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        ct.atomic_add(
            grad_pw,
            (idx_o, idx_i, layer),
            ct.sum(ct.where(b_mask, grad_enc * x_vals, 0.0)),
            memory_order=ct.MemoryOrder.RELAXED,
        )
        ct.atomic_add(
            grad_pb,
            (idx_o, idx_i, layer),
            ct.sum(ct.where(b_mask, grad_enc, 0.0)),
            memory_order=ct.MemoryOrder.RELAXED,
        )
        grad_x_local = grad_x_local + grad_enc * w

        nar0 = ce * ar0 - se * ai0
        nai0 = se * ar0 + ce * ai0
        nar1 = ce * ar1 + se * ai1
        nai1 = -se * ar1 + ce * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(theta[l,0]) — load cached trig
        trig_idx = trig_idx - 1
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)

        ct0 = ct.gather(trig_cache, (pi, trig_idx, 0))
        st0 = ct.gather(trig_cache, (pi, trig_idx, 1))

        grad_t0_vec = 0.5 * (
            ar0 * (-st0 * sr0 - ct0 * sr1)
            + ai0 * (-st0 * si0 - ct0 * si1)
            + ar1 * (ct0 * sr0 - st0 * sr1)
            + ai1 * (ct0 * si0 - st0 * si1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
            memory_order=ct.MemoryOrder.RELAXED,
        )

        nar0 = ct0 * ar0 + st0 * ar1
        nai0 = ct0 * ai0 + st0 * ai1
        nar1 = -st0 * ar0 + ct0 * ar1
        nai1 = -st0 * ai0 + ct0 * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    ct.atomic_add(
        grad_x,
        (b_offs, idx_i),
        ct.where(b_mask, grad_x_local, 0.0),
        memory_order=ct.MemoryOrder.RELAXED,
    )


def cutile_rpz_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    pw: torch.Tensor,
    pb: torch.Tensor,
    grad_output: torch.Tensor,
    fast_measure: bool,
    c_dtype: torch.dtype = torch.float32,
) -> tuple:
    """Launch rpz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    pw = pw.to(io_dtype).contiguous()
    pb = pb.to(io_dtype).contiguous()
    grad_output = grad_output.contiguous()

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    n_states = 2 * reps + 2
    n_b_blocks = math.ceil(batch / BLOCK_B)
    n_programs = n_oi * n_b_blocks
    compute_fp8 = c_dtype == torch.float8_e4m3fn
    if compute_fp8:
        states_dtype = torch.float8_e4m3fn
    elif compute_bf16:
        states_dtype = torch.bfloat16
    else:
        states_dtype = torch.float32
    states = torch.empty(
        n_programs, n_states, 4, BLOCK_B, device=x.device, dtype=states_dtype
    )
    # Trig cache always f32 (small, accuracy-sensitive)
    n_trig = reps + 1
    trig_cache = torch.empty(
        n_programs, n_trig, 2, device=x.device, dtype=torch.float32
    )

    grad_theta = torch.zeros(theta.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)
    grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
    grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)

    grid = (n_oi, n_b_blocks, 1)
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
            trig_cache,
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
            compute_bf16,
            compute_fp8,
            BLOCK_B,
        ),
    )

    return grad_x, grad_theta, grad_pw, grad_pb


# ── real ansatz backward kernel ────────────────────────────────────────────


@ct.kernel(occupancy=4)
def _ct_real_encoding_backward_kernel_bf16(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, 2, BLOCK_B]
    trig_cache,  # [N_Programs, Reps, 2] — cached cos/sin for Ry(theta) gates
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
    COMPUTE_FP8: ConstBool,
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
    x_vals_f32 = ct.astype(x_vals, ct.float32)

    pi = pid_oi * n_b_blocks + pid_b

    def _save2(sidx, v0, v1):
        if COMPUTE_FP8:
            FP8_S = 224.0
            ct.scatter(
                states,
                (pi, sidx, 0, b_range),
                ct.astype(v0 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 1, b_range),
                ct.astype(v1 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
        else:
            ct.scatter(states, (pi, sidx, 0, b_range), v0, mask=b_mask)
            ct.scatter(states, (pi, sidx, 1, b_range), v1, mask=b_mask)

    def _load2(sidx):
        if COMPUTE_FP8:
            INV_S = 0.00446428571428  # 1/224
            return (
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
            )
        else:
            return (
                ct.gather(
                    states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                ),
            )

    # ── Phase 1: Forward recompute + trig cache ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)

    _save2(0, r0, r1)

    state_idx = 1
    trig_idx = 0
    for layer in range(reps):
        r0, r1 = r1, r0  # X gate

        _save2(state_idx, r0, r1)
        state_idx = state_idx + 1

        # Ry(theta) — cache trig
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = ct.astype(t0, ct.float32) * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        ct.scatter(trig_cache, (pi, trig_idx, 0), c)
        ct.scatter(trig_cache, (pi, trig_idx, 1), s)
        trig_idx = trig_idx + 1
        nr0 = c * r0 - s * r1
        nr1 = s * r0 + c * r1
        r0, r1 = nr0, nr1

        _save2(state_idx, r0, r1)
        state_idx = state_idx + 1

        r1 = -r1  # Z gate

        _save2(state_idx, r0, r1)
        state_idx = state_idx + 1

        # Ry(enc) — NOT cached (depends on x_vals per batch element)
        enc = x_vals_f32
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = ct.astype(w, ct.float32) * x_vals_f32 + ct.astype(b, ct.float32)

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

    # ── Phase 3: Backward sweep (using cached trig for theta gates) ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for _ri in range(reps):
        layer = reps - 1 - _ri
        # Backward through Ry(enc) — must recompute (depends on x_vals)
        state_idx = state_idx - 1
        sr0, sr1 = _load2(state_idx)

        enc = x_vals_f32
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            w_f32 = ct.astype(w, ct.float32)
            enc = w_f32 * x_vals_f32 + ct.astype(b, ct.float32)

        ae = enc * 0.5
        ce = ct.cos(ae)
        se = ct.sin(ae)

        grad_enc = 0.5 * (ar0 * (-se * sr0 - ce * sr1) + ar1 * (ce * sr0 - se * sr1))

        if PREACTS_TRAINABLE:
            ct.atomic_add(
                grad_pw,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc * x_vals_f32, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            ct.atomic_add(
                grad_pb,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            grad_x_local = grad_x_local + grad_enc * w_f32
        else:
            grad_x_local = grad_x_local + grad_enc

        nar0 = ce * ar0 + se * ar1
        nar1 = -se * ar0 + ce * ar1
        ar0, ar1 = nar0, nar1

        # Backward through Z gate
        ar1 = -ar1

        # Backward through Ry(theta) — load cached trig
        trig_idx = trig_idx - 1
        state_idx = state_idx - 2
        sr0, sr1 = _load2(state_idx)

        ct0 = ct.gather(trig_cache, (pi, trig_idx, 0))
        st0 = ct.gather(trig_cache, (pi, trig_idx, 1))

        grad_t0_vec = 0.5 * (
            ar0 * (-st0 * sr0 - ct0 * sr1) + ar1 * (ct0 * sr0 - st0 * sr1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
            memory_order=ct.MemoryOrder.RELAXED,
        )

        nar0 = ct0 * ar0 + st0 * ar1
        nar1 = -st0 * ar0 + ct0 * ar1
        ar0, ar1 = nar0, nar1

        # Backward through X gate
        ar0, ar1 = ar1, ar0

    ct.atomic_add(
        grad_x,
        (b_offs, idx_i),
        ct.where(b_mask, grad_x_local, 0.0),
        memory_order=ct.MemoryOrder.RELAXED,
    )


@ct.kernel(occupancy=4)
def _ct_real_encoding_backward_kernel_f32(
    x,  # [Batch, In_Dim]
    theta,  # [Out_Dim, In_Dim, Reps, 1]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, 4, BLOCK_B]
    trig_cache,  # [N_Programs, Reps, 2] — cached cos/sin for Ry(theta) gates
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
    COMPUTE_FP8: ConstBool,
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

    pi = pid_oi * n_b_blocks + pid_b

    def _save4(sidx, v0, v1, v2, v3):
        if COMPUTE_FP8:
            FP8_S = 224.0
            ct.scatter(
                states,
                (pi, sidx, 0, b_range),
                ct.astype(v0 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 1, b_range),
                ct.astype(v1 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 2, b_range),
                ct.astype(v2 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
            ct.scatter(
                states,
                (pi, sidx, 3, b_range),
                ct.astype(v3 * FP8_S, ct.float8_e4m3fn),
                mask=b_mask,
            )
        else:
            ct.scatter(states, (pi, sidx, 0, b_range), v0, mask=b_mask)
            ct.scatter(states, (pi, sidx, 1, b_range), v1, mask=b_mask)
            ct.scatter(states, (pi, sidx, 2, b_range), v2, mask=b_mask)
            ct.scatter(states, (pi, sidx, 3, b_range), v3, mask=b_mask)

    def _load4(sidx):
        if COMPUTE_FP8:
            INV_S = 0.00446428571428  # 1/224
            return (
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
                ct.astype(
                    ct.gather(
                        states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                    ),
                    ct.float32,
                )
                * INV_S,
            )
        else:
            return (
                ct.gather(
                    states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0
                ),
                ct.gather(
                    states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0
                ),
            )

    # ── Phase 1: Forward recompute + trig cache ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    _save4(0, r0, i0, r1, i1)

    state_idx = 1
    trig_idx = 0
    for layer in range(reps):
        r0, i0, r1, i1 = r1, i1, r0, i0  # X gate

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        # Ry(theta) — cache trig
        t0 = ct.gather(theta, (idx_o, idx_i, layer, 0))
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        ct.scatter(trig_cache, (pi, trig_idx, 0), c)
        ct.scatter(trig_cache, (pi, trig_idx, 1), s)
        trig_idx = trig_idx + 1
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        r1 = -r1  # Z gate
        i1 = -i1

        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        # Ry(enc) — NOT cached (depends on x_vals per batch element)
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

    # ── Phase 3: Backward sweep (using cached trig for theta gates) ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for _ri in range(reps):
        layer = reps - 1 - _ri
        # Backward through Ry(enc) — must recompute (depends on x_vals)
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)

        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.gather(pw, (idx_o, idx_i, layer))
            b = ct.gather(pb, (idx_o, idx_i, layer))
            enc = w * x_vals + b

        ae = enc * 0.5
        ce = ct.cos(ae)
        se = ct.sin(ae)

        grad_enc = 0.5 * (
            ar0 * (-se * sr0 - ce * sr1)
            + ai0 * (-se * si0 - ce * si1)
            + ar1 * (ce * sr0 - se * sr1)
            + ai1 * (ce * si0 - se * si1)
        )

        if PREACTS_TRAINABLE:
            ct.atomic_add(
                grad_pw,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc * x_vals, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            ct.atomic_add(
                grad_pb,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            grad_x_local = grad_x_local + grad_enc * w
        else:
            grad_x_local = grad_x_local + grad_enc

        nar0 = ce * ar0 + se * ar1
        nai0 = ce * ai0 + se * ai1
        nar1 = -se * ar0 + ce * ar1
        nai1 = -se * ai0 + ce * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Z gate
        ar1 = -ar1
        ai1 = -ai1

        # Backward through Ry(theta) — load cached trig
        trig_idx = trig_idx - 1
        state_idx = state_idx - 2
        sr0, si0, sr1, si1 = _load4(state_idx)

        ct0 = ct.gather(trig_cache, (pi, trig_idx, 0))
        st0 = ct.gather(trig_cache, (pi, trig_idx, 1))

        grad_t0_vec = 0.5 * (
            ar0 * (-st0 * sr0 - ct0 * sr1)
            + ai0 * (-st0 * si0 - ct0 * si1)
            + ar1 * (ct0 * sr0 - st0 * sr1)
            + ai1 * (ct0 * si0 - st0 * si1)
        )
        ct.atomic_add(
            grad_theta,
            (idx_o, idx_i, layer, 0),
            ct.sum(ct.where(b_mask, grad_t0_vec, 0.0)),
            memory_order=ct.MemoryOrder.RELAXED,
        )

        nar0 = ct0 * ar0 + st0 * ar1
        nai0 = ct0 * ai0 + st0 * ai1
        nar1 = -st0 * ar0 + ct0 * ar1
        nai1 = -st0 * ai0 + ct0 * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through X gate
        ar0, ai0, ar1, ai1 = ar1, ai1, ar0, ai0

    ct.atomic_add(
        grad_x,
        (b_offs, idx_i),
        ct.where(b_mask, grad_x_local, 0.0),
        memory_order=ct.MemoryOrder.RELAXED,
    )


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

    compute_fp8 = c_dtype == torch.float8_e4m3fn
    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    io_dtype = torch.bfloat16 if compute_fp8 else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    pw = pw.to(io_dtype).contiguous()
    pb = pb.to(io_dtype).contiguous()
    grad_output = grad_output.contiguous()

    n_oi = out_dim * in_dim
    # cuTile real: always use BLOCK_B>=32 (coalesced layout makes larger blocks efficient)
    BLOCK_B = _select_block_b(n_oi, batch)

    n_states = 3 * reps + 1
    n_b_blocks = math.ceil(batch / BLOCK_B)
    n_programs = n_oi * n_b_blocks
    n_components = 2 if compute_bf16 else 4
    # cuTile: keep states in f32 for bf16 mode (ct.astype overhead > bandwidth savings
    # for 2-component states). Only use fp8 for fp8 mode (prescale offsets the cost).
    states_dtype = torch.float8_e4m3fn if compute_fp8 else torch.float32
    states = torch.empty(
        n_programs,
        n_states,
        n_components,
        BLOCK_B,
        device=x.device,
        dtype=states_dtype,
    )
    # Trig cache: reps Ry(theta) gates, each storing (cos, sin)
    trig_cache = torch.empty(n_programs, reps, 2, device=x.device, dtype=torch.float32)

    grad_theta = torch.zeros(theta.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)

    if preacts_trainable:
        grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)
    else:
        grad_pw = torch.zeros(1, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(1, device=x.device, dtype=torch.float32)

    grid = (n_oi, n_b_blocks, 1)
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
            trig_cache,
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
            compute_fp8,
            BLOCK_B,
        ),
    )

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )
