"""
Triton-fused kernels for QKAN quantum circuit simulation.

Implements pz_encoding, rpz_encoding, and real ansatz forward passes as fused
Triton kernels, avoiding materialization of intermediate complex state vectors.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _pz_encoding_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 2]
    pw_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts weights)
    pb_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts bias)
    out_ptr,  # [Batch, Out_Dim, In_Dim]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for output
    stride_o_b,
    stride_o_o,
    stride_o_i,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
):
    """
    Fused QKAN pz_encoding forward kernel.

    Each program instance computes one element of the output[batch, out, in].
    The quantum circuit:
        H |0> -> for l in reps: Rz(theta[l,0]) Ry(theta[l,1]) Rz(enc_x) -> Rz(theta[L,0]) Ry(theta[L,1]) -> measure Z
    """
    pid = tl.program_id(0)

    # Decompose linear PID -> (batch, out, in)
    idx_i = pid % in_dim
    tmp = pid // in_dim
    idx_o = tmp % out_dim
    idx_b = tmp // out_dim

    if idx_b >= batch_size:
        return

    # Load input x[b, i]
    x_val = tl.load(x_ptr + idx_b * stride_x_b + idx_i * stride_x_i)

    # Initialize state: H|0> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
    # State is (r0 + j*i0)|0> + (r1 + j*i1)|1>
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = INV_SQRT2
    i0 = 0.0
    r1 = INV_SQRT2
    i1 = 0.0

    # Base pointer for theta[o, i, :, :]
    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i

    # Main circuit loop
    for layer in range(reps):
        # Load theta parameters for this layer
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        t1 = tl.load(theta_base + layer * stride_t_r + 1 * stride_t_p)

        # --- Apply Rz(t0) ---
        # Rz(t) = diag(e^{-it/2}, e^{it/2})
        # e^{-it/2} = cos(t/2) - j*sin(t/2)
        # |0> component: (r0+ji0) * (c-js) = (r0*c+i0*s) + j(i0*c-r0*s)
        # |1> component: (r1+ji1) * (c+js) = (r1*c-i1*s) + j(i1*c+r1*s)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # --- Apply Ry(t1) ---
        # Ry(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        a = t1 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # --- Apply Rz(encoded_x) (data re-uploading) ---
        enc = x_val
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            )
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            )
            enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final rotations: Rz(theta[reps, 0]), Ry(theta[reps, 1])
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p)
    t1 = tl.load(theta_base + reps * stride_t_r + 1 * stride_t_p)

    # Rz(t0)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Ry(t1)
    a = t1 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Measure Z expectation value
    if FAST_MEASURE:
        # Quantum-inspired: |alpha| - |beta|
        result = tl.sqrt(r0 * r0 + i0 * i0) - tl.sqrt(r1 * r1 + i1 * i1)
    else:
        # Standard quantum: |alpha|^2 - |beta|^2
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    # Store result
    tl.store(
        out_ptr + idx_b * stride_o_b + idx_o * stride_o_o + idx_i * stride_o_i,
        result,
    )


def triton_pz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
) -> torch.Tensor:
    """
    Launch the Triton pz_encoding kernel.

    Args:
        x: (batch, in_dim) float32 on CUDA
        theta: (out_dim, in_dim, reps+1, 2) float32 on CUDA, already expanded
        preacts_w: (out_dim, in_dim, reps) float32, or any tensor if not trainable
        preacts_b: (out_dim, in_dim, reps) float32, or any tensor if not trainable
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

    grid = (batch * out_dim * in_dim,)

    if preacts_trainable:
        preacts_w = preacts_w.contiguous()
        preacts_b = preacts_b.contiguous()
        pw_strides = (preacts_w.stride(0), preacts_w.stride(1), preacts_w.stride(2))
        pb_strides = (preacts_b.stride(0), preacts_b.stride(1), preacts_b.stride(2))
    else:
        # Kernel won't access these (dead code), pass dummy strides
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)

    _pz_encoding_kernel[grid](
        x,
        theta,
        preacts_w,
        preacts_b,
        output,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
    )

    return output


# ── rpz_encoding kernel ─────────────────────────────────────────────────────


@triton.jit
def _rpz_encoding_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 1]
    pw_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts weights, always loaded)
    pb_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts bias, always loaded)
    out_ptr,  # [Batch, Out_Dim, In_Dim]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for output
    stride_o_b,
    stride_o_o,
    stride_o_i,
    # Compile-time constants
    FAST_MEASURE: tl.constexpr,
):
    """
    Fused rpz_encoding forward kernel.

    Circuit: H|0> -> [Ry(theta[l,0]) Rz(w*x+b)]×reps -> Ry(theta[L,0]) -> measure Z
    rpz always uses encoded_x = w*x + b (even when preacts_trainable=False, w=1).
    """
    pid = tl.program_id(0)

    idx_i = pid % in_dim
    tmp = pid // in_dim
    idx_o = tmp % out_dim
    idx_b = tmp // out_dim

    if idx_b >= batch_size:
        return

    x_val = tl.load(x_ptr + idx_b * stride_x_b + idx_i * stride_x_i)

    # H|0> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = INV_SQRT2
    i0 = 0.0
    r1 = INV_SQRT2
    i1 = 0.0

    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    pw_base = pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i
    pb_base = pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i

    for layer in range(reps):
        # Ry(theta[l, 0])
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(w * x + b)  — always load preacts
        w = tl.load(pw_base + layer * stride_pw_r)
        b = tl.load(pb_base + layer * stride_pb_r)
        enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Ry(theta[reps, 0])
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Measure Z
    if FAST_MEASURE:
        result = tl.sqrt(r0 * r0 + i0 * i0) - tl.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    tl.store(
        out_ptr + idx_b * stride_o_b + idx_o * stride_o_o + idx_i * stride_o_i,
        result,
    )


def triton_rpz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    fast_measure: bool = True,
) -> torch.Tensor:
    """
    Launch the Triton rpz_encoding kernel.

    Args:
        x: (batch, in_dim) float32
        theta: (out_dim, in_dim, reps+1, 1) float32, already expanded
        preacts_w: (out_dim, in_dim, reps) float32, already expanded
        preacts_b: (out_dim, in_dim, reps) float32, already expanded
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
    grid = (batch * out_dim * in_dim,)

    _rpz_encoding_kernel[grid](
        x,
        theta,
        preacts_w,
        preacts_b,
        output,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        preacts_w.stride(0),
        preacts_w.stride(1),
        preacts_w.stride(2),
        preacts_b.stride(0),
        preacts_b.stride(1),
        preacts_b.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        FAST_MEASURE=fast_measure,
    )

    return output


@triton.jit
def _rpz_encoding_backward_kernel(
    # Pointers
    x_ptr,
    theta_ptr,
    pw_ptr,
    pb_ptr,
    grad_out_ptr,
    states_ptr,
    grad_theta_ptr,
    grad_x_ptr,
    grad_pw_ptr,
    grad_pb_ptr,
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for grad_output
    stride_go_b,
    stride_go_o,
    stride_go_i,
    # Strides for states buffer
    stride_s_n,
    stride_s_s,
    stride_s_c,
    # Strides for grad_theta
    stride_gt_o,
    stride_gt_i,
    stride_gt_r,
    stride_gt_p,
    # Strides for grad_x
    stride_gx_b,
    stride_gx_i,
    # Strides for grad_pw/pb
    stride_gpw_o,
    stride_gpw_i,
    stride_gpw_r,
    stride_gpb_o,
    stride_gpb_i,
    stride_gpb_r,
    # Compile-time constants
    FAST_MEASURE: tl.constexpr,
):
    """Backward kernel for rpz_encoding ansatz."""
    pid = tl.program_id(0)
    idx_i = pid % in_dim
    tmp = pid // in_dim
    idx_o = tmp % out_dim
    idx_b = tmp // out_dim

    if idx_b >= batch_size:
        return

    x_val = tl.load(x_ptr + idx_b * stride_x_b + idx_i * stride_x_i)
    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    pw_base = pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i
    pb_base = pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i
    states_base = states_ptr + pid * stride_s_n

    # ── Phase 1: Forward recompute, saving states ──
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = INV_SQRT2
    i0 = 0.0
    r1 = INV_SQRT2
    i1 = 0.0

    tl.store(states_base + 0 * stride_s_s + 0 * stride_s_c, r0)
    tl.store(states_base + 0 * stride_s_s + 1 * stride_s_c, i0)
    tl.store(states_base + 0 * stride_s_s + 2 * stride_s_c, r1)
    tl.store(states_base + 0 * stride_s_s + 3 * stride_s_c, i1)

    state_idx = 1
    for layer in range(reps):
        # Ry(theta[l,0])
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

        # Rz(w*x+b)
        w = tl.load(pw_base + layer * stride_pw_r)
        b = tl.load(pb_base + layer * stride_pb_r)
        enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

    # Final Ry(theta[reps,0])
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = tl.load(
        grad_out_ptr + idx_b * stride_go_b + idx_o * stride_go_o + idx_i * stride_go_i
    )

    if FAST_MEASURE:
        alpha_norm = tl.sqrt(r0 * r0 + i0 * i0)
        beta_norm = tl.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = tl.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = tl.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
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
    grad_x_local = 0.0
    gt_base = grad_theta_ptr + idx_o * stride_gt_o + idx_i * stride_gt_i

    # Backward through final Ry(theta[reps,0])
    state_idx -= 1
    sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
    si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
    sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
    si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)

    grad_t0 = 0.5 * (
        ar0 * (-s * sr0 - c * sr1)
        + ai0 * (-s * si0 - c * si1)
        + ar1 * (c * sr0 - s * sr1)
        + ai1 * (c * si0 - s * si1)
    )
    tl.atomic_add(gt_base + reps * stride_gt_r + 0 * stride_gt_p, grad_t0)

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # Backward through layers in reverse
    for layer in range(reps - 1, -1, -1):
        # Backward through Rz(enc)
        state_idx -= 1
        sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
        si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
        sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
        si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

        w = tl.load(pw_base + layer * stride_pw_r)
        b = tl.load(pb_base + layer * stride_pb_r)
        enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_enc = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        tl.atomic_add(
            grad_pw_ptr
            + idx_o * stride_gpw_o
            + idx_i * stride_gpw_i
            + layer * stride_gpw_r,
            grad_enc * x_val,
        )
        tl.atomic_add(
            grad_pb_ptr
            + idx_o * stride_gpb_o
            + idx_i * stride_gpb_i
            + layer * stride_gpb_r,
            grad_enc,
        )
        grad_x_local += grad_enc * w

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(theta[l,0])
        state_idx -= 1
        sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
        si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
        sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
        si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_t0 = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        tl.atomic_add(gt_base + layer * stride_gt_r + 0 * stride_gt_p, grad_t0)

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    tl.atomic_add(grad_x_ptr + idx_b * stride_gx_b + idx_i * stride_gx_i, grad_x_local)


def triton_rpz_backward(x, theta, pw, pb, grad_output, fast_measure):
    """Launch rpz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    x = x.contiguous()
    theta = theta.contiguous()
    pw = pw.contiguous()
    pb = pb.contiguous()
    grad_output = grad_output.contiguous()

    n_states = 2 * reps + 2  # H state + 2 per layer (after Ry, Rz)
    N = batch * out_dim * in_dim
    states = torch.empty(N, n_states, 4, device=x.device, dtype=x.dtype)

    grad_theta = torch.zeros_like(theta)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=x.dtype)
    grad_pw = torch.zeros_like(pw)
    grad_pb = torch.zeros_like(pb)

    grid = (N,)
    _rpz_encoding_backward_kernel[grid](
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
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        pw.stride(0),
        pw.stride(1),
        pw.stride(2),
        pb.stride(0),
        pb.stride(1),
        pb.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        grad_theta.stride(0),
        grad_theta.stride(1),
        grad_theta.stride(2),
        grad_theta.stride(3),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_pw.stride(0),
        grad_pw.stride(1),
        grad_pw.stride(2),
        grad_pb.stride(0),
        grad_pb.stride(1),
        grad_pb.stride(2),
        FAST_MEASURE=fast_measure,
    )

    return grad_x, grad_theta, grad_pw, grad_pb


# ── real ansatz kernel ───────────────────────────────────────────────────────


@triton.jit
def _real_encoding_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps, 1]  (no final layer)
    pw_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts weights)
    pb_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts bias)
    out_ptr,  # [Batch, Out_Dim, In_Dim]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for output
    stride_o_b,
    stride_o_o,
    stride_o_i,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
):
    """
    Fused real ansatz forward kernel.

    Circuit: H|0> -> [X, Ry(theta[l,0]), Z, Ry(enc_x)]×reps -> measure Z
    No final rotation. Data encoding uses Ry (not Rz).
    """
    pid = tl.program_id(0)

    idx_i = pid % in_dim
    tmp = pid // in_dim
    idx_o = tmp % out_dim
    idx_b = tmp // out_dim

    if idx_b >= batch_size:
        return

    x_val = tl.load(x_ptr + idx_b * stride_x_b + idx_i * stride_x_i)

    # H|0>
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = INV_SQRT2
    i0 = 0.0
    r1 = INV_SQRT2
    i1 = 0.0

    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i

    for layer in range(reps):
        # X gate: swap |0> <-> |1>
        r0, i0, r1, i1 = r1, i1, r0, i0

        # Ry(theta[l, 0])
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Z gate: negate |1> component
        r1 = -r1
        i1 = -i1

        # Ry(enc_x) — data encoding with Ry
        enc = x_val
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            )
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            )
            enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # No final rotation — go straight to measurement

    # Measure Z
    if FAST_MEASURE:
        result = tl.sqrt(r0 * r0 + i0 * i0) - tl.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    tl.store(
        out_ptr + idx_b * stride_o_b + idx_o * stride_o_o + idx_i * stride_o_i,
        result,
    )


# ── Backward kernels ───────────────────────────────────────────────────────


@triton.jit
def _pz_encoding_backward_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 2]
    pw_ptr,  # [Out_Dim, In_Dim, Reps]
    pb_ptr,  # [Out_Dim, In_Dim, Reps]
    grad_out_ptr,  # [Batch, Out_Dim, In_Dim]
    states_ptr,  # [batch*out*in, n_states, 4]
    grad_theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 2]
    grad_x_ptr,  # [Batch, In_Dim]
    grad_pw_ptr,  # [Out_Dim, In_Dim, Reps]
    grad_pb_ptr,  # [Out_Dim, In_Dim, Reps]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for grad_output
    stride_go_b,
    stride_go_o,
    stride_go_i,
    # Strides for states buffer [N, n_states, 4]
    stride_s_n,
    stride_s_s,
    stride_s_c,
    # Strides for grad_theta
    stride_gt_o,
    stride_gt_i,
    stride_gt_r,
    stride_gt_p,
    # Strides for grad_x
    stride_gx_b,
    stride_gx_i,
    # Strides for grad_pw/pb
    stride_gpw_o,
    stride_gpw_i,
    stride_gpw_r,
    stride_gpb_o,
    stride_gpb_i,
    stride_gpb_r,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
):
    """Backward kernel for pz_encoding ansatz."""
    pid = tl.program_id(0)
    idx_i = pid % in_dim
    tmp = pid // in_dim
    idx_o = tmp % out_dim
    idx_b = tmp // out_dim

    if idx_b >= batch_size:
        return

    x_val = tl.load(x_ptr + idx_b * stride_x_b + idx_i * stride_x_i)
    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    states_base = states_ptr + pid * stride_s_n

    # ── Phase 1: Forward recompute, saving states ──
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = INV_SQRT2
    i0 = 0.0
    r1 = INV_SQRT2
    i1 = 0.0

    # Save initial state (state index 0)
    tl.store(states_base + 0 * stride_s_s + 0 * stride_s_c, r0)
    tl.store(states_base + 0 * stride_s_s + 1 * stride_s_c, i0)
    tl.store(states_base + 0 * stride_s_s + 2 * stride_s_c, r1)
    tl.store(states_base + 0 * stride_s_s + 3 * stride_s_c, i1)

    state_idx = 1
    for layer in range(reps):
        # Rz(t0)
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

        # Ry(t1)
        t1 = tl.load(theta_base + layer * stride_t_r + 1 * stride_t_p)
        a = t1 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

        # Rz(enc)
        enc = x_val
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            )
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            )
            enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

    # Final Rz(t0)
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
    tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
    tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
    tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
    state_idx += 1

    # Final Ry(t1)
    t1 = tl.load(theta_base + reps * stride_t_r + 1 * stride_t_p)
    a = t1 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1
    # final state — no need to save

    # ── Phase 2: Measurement gradient ──
    go = tl.load(
        grad_out_ptr + idx_b * stride_go_b + idx_o * stride_go_o + idx_i * stride_go_i
    )

    if FAST_MEASURE:
        alpha_norm = tl.sqrt(r0 * r0 + i0 * i0)
        beta_norm = tl.sqrt(r1 * r1 + i1 * i1)
        # Avoid division by zero
        inv_alpha = tl.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = tl.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
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
    grad_x_local = 0.0
    # Total states saved = 3*reps + 2 (indices 0..state_idx-1 before final Ry)
    # We walk backward: final Ry, final Rz, then loop (Rz_enc, Ry, Rz) per layer reversed

    # -- Backward through final Ry(t1_final) --
    state_idx -= 1  # pre-Ry state
    sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
    si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
    sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
    si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

    t1 = tl.load(theta_base + reps * stride_t_r + 1 * stride_t_p)
    a = t1 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)

    # Param grad for Ry: v = 0.5*(-s*r0-c*r1, -s*i0-c*i1, c*r0-s*r1, c*i0-s*i1)
    grad_t1 = 0.5 * (
        ar0 * (-s * sr0 - c * sr1)
        + ai0 * (-s * si0 - c * si1)
        + ar1 * (c * sr0 - s * sr1)
        + ai1 * (c * si0 - s * si1)
    )

    gt_base = grad_theta_ptr + idx_o * stride_gt_o + idx_i * stride_gt_i
    tl.atomic_add(gt_base + reps * stride_gt_r + 1 * stride_gt_p, grad_t1)

    # Adjoint prop Ry^T: new_ar0=c*ar0+s*ar1, etc.
    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # -- Backward through final Rz(t0_final) --
    state_idx -= 1
    sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
    si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
    sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
    si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)

    # Param grad for Rz
    grad_t0 = 0.5 * (
        -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
        + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
    )
    tl.atomic_add(gt_base + reps * stride_gt_r + 0 * stride_gt_p, grad_t0)

    # Adjoint prop Rz^T
    nar0 = c * ar0 - s * ai0
    nai0 = s * ar0 + c * ai0
    nar1 = c * ar1 + s * ai1
    nai1 = -s * ar1 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # -- Backward through layers in reverse --
    for layer in range(reps - 1, -1, -1):
        # Backward through Rz(enc)
        state_idx -= 1
        sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
        si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
        sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
        si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

        enc = x_val
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            )
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            )
            enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_enc = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        if PREACTS_TRAINABLE:
            tl.atomic_add(
                grad_pw_ptr
                + idx_o * stride_gpw_o
                + idx_i * stride_gpw_i
                + layer * stride_gpw_r,
                grad_enc * x_val,
            )
            tl.atomic_add(
                grad_pb_ptr
                + idx_o * stride_gpb_o
                + idx_i * stride_gpb_i
                + layer * stride_gpb_r,
                grad_enc,
            )
            grad_x_local += grad_enc * w
        else:
            grad_x_local += grad_enc

        # Adjoint prop Rz^T
        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(t1)
        state_idx -= 1
        sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
        si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
        sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
        si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

        t1 = tl.load(theta_base + layer * stride_t_r + 1 * stride_t_p)
        a = t1 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_t1 = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        tl.atomic_add(gt_base + layer * stride_gt_r + 1 * stride_gt_p, grad_t1)

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Rz(t0)
        state_idx -= 1
        sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
        si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
        sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
        si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_t0 = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )
        tl.atomic_add(gt_base + layer * stride_gt_r + 0 * stride_gt_p, grad_t0)

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # Accumulate grad_x across out_dim
    tl.atomic_add(grad_x_ptr + idx_b * stride_gx_b + idx_i * stride_gx_i, grad_x_local)


def triton_pz_backward(x, theta, pw, pb, grad_output, preacts_trainable, fast_measure):
    """Launch pz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    x = x.contiguous()
    theta = theta.contiguous()
    grad_output = grad_output.contiguous()

    n_states = (
        3 * reps + 3
    )  # H state + 3 per layer (after Rz, Ry, Rz_enc) + after final Rz
    N = batch * out_dim * in_dim
    states = torch.empty(N, n_states, 4, device=x.device, dtype=x.dtype)

    grad_theta = torch.zeros_like(theta)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=x.dtype)

    if preacts_trainable:
        pw = pw.contiguous()
        pb = pb.contiguous()
        grad_pw = torch.zeros_like(pw)
        grad_pb = torch.zeros_like(pb)
        pw_strides = (pw.stride(0), pw.stride(1), pw.stride(2))
        pb_strides = (pb.stride(0), pb.stride(1), pb.stride(2))
        gpw_strides = (grad_pw.stride(0), grad_pw.stride(1), grad_pw.stride(2))
        gpb_strides = (grad_pb.stride(0), grad_pb.stride(1), grad_pb.stride(2))
    else:
        grad_pw = torch.zeros(1, device=x.device)
        grad_pb = torch.zeros(1, device=x.device)
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)
        gpw_strides = (0, 0, 0)
        gpb_strides = (0, 0, 0)

    grid = (N,)
    _pz_encoding_backward_kernel[grid](
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
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        grad_theta.stride(0),
        grad_theta.stride(1),
        grad_theta.stride(2),
        grad_theta.stride(3),
        grad_x.stride(0),
        grad_x.stride(1),
        *gpw_strides,
        *gpb_strides,
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
    )

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )


def triton_real_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
) -> torch.Tensor:
    """
    Launch the Triton real ansatz kernel.

    Args:
        x: (batch, in_dim) float32
        theta: (out_dim, in_dim, reps, 1) float32, already expanded (no +1 layer)
        preacts_w: (out_dim, in_dim, reps) float32, or any tensor if not trainable
        preacts_b: (out_dim, in_dim, reps) float32, or any tensor if not trainable
        preacts_trainable: whether preacts are used
        fast_measure: measurement mode

    Returns:
        (batch, out_dim, in_dim) float32
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2]  # No +1 for real ansatz

    x = x.contiguous()
    theta = theta.contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=x.dtype)
    grid = (batch * out_dim * in_dim,)

    if preacts_trainable:
        preacts_w = preacts_w.contiguous()
        preacts_b = preacts_b.contiguous()
        pw_strides = (preacts_w.stride(0), preacts_w.stride(1), preacts_w.stride(2))
        pb_strides = (preacts_b.stride(0), preacts_b.stride(1), preacts_b.stride(2))
    else:
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)

    _real_encoding_kernel[grid](
        x,
        theta,
        preacts_w,
        preacts_b,
        output,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
    )

    return output


@triton.jit
def _real_encoding_backward_kernel(
    # Pointers
    x_ptr,
    theta_ptr,
    pw_ptr,
    pb_ptr,
    grad_out_ptr,
    states_ptr,
    grad_theta_ptr,
    grad_x_ptr,
    grad_pw_ptr,
    grad_pb_ptr,
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for grad_output
    stride_go_b,
    stride_go_o,
    stride_go_i,
    # Strides for states buffer
    stride_s_n,
    stride_s_s,
    stride_s_c,
    # Strides for grad_theta
    stride_gt_o,
    stride_gt_i,
    stride_gt_r,
    stride_gt_p,
    # Strides for grad_x
    stride_gx_b,
    stride_gx_i,
    # Strides for grad_pw/pb
    stride_gpw_o,
    stride_gpw_i,
    stride_gpw_r,
    stride_gpb_o,
    stride_gpb_i,
    stride_gpb_r,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
):
    """Backward kernel for real ansatz. Circuit: H|0> -> [X, Ry(theta), Z, Ry(enc)]×reps -> measure Z"""
    pid = tl.program_id(0)
    idx_i = pid % in_dim
    tmp = pid // in_dim
    idx_o = tmp % out_dim
    idx_b = tmp // out_dim

    if idx_b >= batch_size:
        return

    x_val = tl.load(x_ptr + idx_b * stride_x_b + idx_i * stride_x_i)
    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    states_base = states_ptr + pid * stride_s_n

    # ── Phase 1: Forward recompute, saving states ──
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = INV_SQRT2
    i0 = 0.0
    r1 = INV_SQRT2
    i1 = 0.0

    tl.store(states_base + 0 * stride_s_s + 0 * stride_s_c, r0)
    tl.store(states_base + 0 * stride_s_s + 1 * stride_s_c, i0)
    tl.store(states_base + 0 * stride_s_s + 2 * stride_s_c, r1)
    tl.store(states_base + 0 * stride_s_s + 3 * stride_s_c, i1)

    state_idx = 1
    for layer in range(reps):
        # X gate: swap |0> <-> |1>
        r0, i0, r1, i1 = r1, i1, r0, i0

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

        # Ry(theta[l,0])
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

        # Z gate: negate |1>
        r1 = -r1
        i1 = -i1

        tl.store(states_base + state_idx * stride_s_s + 0 * stride_s_c, r0)
        tl.store(states_base + state_idx * stride_s_s + 1 * stride_s_c, i0)
        tl.store(states_base + state_idx * stride_s_s + 2 * stride_s_c, r1)
        tl.store(states_base + state_idx * stride_s_s + 3 * stride_s_c, i1)
        state_idx += 1

        # Ry(enc)
        enc = x_val
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            )
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            )
            enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Don't save after Ry(enc): its output is the next iteration's
        # input (or the final state for measurement). This keeps state_idx
        # aligned so backward state_idx-1 points to the pre-gate state.

    # ── Phase 2: Measurement gradient ──
    go = tl.load(
        grad_out_ptr + idx_b * stride_go_b + idx_o * stride_go_o + idx_i * stride_go_i
    )

    if FAST_MEASURE:
        alpha_norm = tl.sqrt(r0 * r0 + i0 * i0)
        beta_norm = tl.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = tl.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = tl.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
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
    grad_x_local = 0.0
    gt_base = grad_theta_ptr + idx_o * stride_gt_o + idx_i * stride_gt_i

    for layer in range(reps - 1, -1, -1):
        # Backward through Ry(enc) — pre-gate state = after Z = state[3l+3]
        state_idx -= 1
        sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
        si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
        sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
        si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

        enc = x_val
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            )
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            )
            enc = w * x_val + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_enc = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )

        if PREACTS_TRAINABLE:
            tl.atomic_add(
                grad_pw_ptr
                + idx_o * stride_gpw_o
                + idx_i * stride_gpw_i
                + layer * stride_gpw_r,
                grad_enc * x_val,
            )
            tl.atomic_add(
                grad_pb_ptr
                + idx_o * stride_gpb_o
                + idx_i * stride_gpb_i
                + layer * stride_gpb_r,
                grad_enc,
            )
            grad_x_local += grad_enc * w
        else:
            grad_x_local += grad_enc

        # Adjoint prop Ry^T
        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Z gate: negate ar1, ai1 (no state needed)
        ar1 = -ar1
        ai1 = -ai1

        # Backward through Ry(theta[l,0]) — pre-gate state = after X = state[3l+1]
        state_idx -= 2  # skip over after-Ry(theta) state to reach after-X state
        sr0 = tl.load(states_base + state_idx * stride_s_s + 0 * stride_s_c)
        si0 = tl.load(states_base + state_idx * stride_s_s + 1 * stride_s_c)
        sr1 = tl.load(states_base + state_idx * stride_s_s + 2 * stride_s_c)
        si1 = tl.load(states_base + state_idx * stride_s_s + 3 * stride_s_c)

        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_t0 = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        tl.atomic_add(gt_base + layer * stride_gt_r + 0 * stride_gt_p, grad_t0)

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through X gate: swap adjoint components (no state needed)
        ar0, ai0, ar1, ai1 = ar1, ai1, ar0, ai0

    tl.atomic_add(grad_x_ptr + idx_b * stride_gx_b + idx_i * stride_gx_i, grad_x_local)


def triton_real_backward(
    x, theta, pw, pb, grad_output, preacts_trainable, fast_measure
):
    """Launch real ansatz backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2]  # No +1 for real ansatz

    x = x.contiguous()
    theta = theta.contiguous()
    grad_output = grad_output.contiguous()

    n_states = 3 * reps + 1  # H state + 3 per layer (after X, Ry_theta, Z)
    N = batch * out_dim * in_dim
    states = torch.empty(N, n_states, 4, device=x.device, dtype=x.dtype)

    grad_theta = torch.zeros_like(theta)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=x.dtype)

    if preacts_trainable:
        pw = pw.contiguous()
        pb = pb.contiguous()
        grad_pw = torch.zeros_like(pw)
        grad_pb = torch.zeros_like(pb)
        pw_strides = (pw.stride(0), pw.stride(1), pw.stride(2))
        pb_strides = (pb.stride(0), pb.stride(1), pb.stride(2))
        gpw_strides = (grad_pw.stride(0), grad_pw.stride(1), grad_pw.stride(2))
        gpb_strides = (grad_pb.stride(0), grad_pb.stride(1), grad_pb.stride(2))
    else:
        grad_pw = torch.zeros(1, device=x.device)
        grad_pb = torch.zeros(1, device=x.device)
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)
        gpw_strides = (0, 0, 0)
        gpb_strides = (0, 0, 0)

    grid = (N,)
    _real_encoding_backward_kernel[grid](
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
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        grad_theta.stride(0),
        grad_theta.stride(1),
        grad_theta.stride(2),
        grad_theta.stride(3),
        grad_x.stride(0),
        grad_x.stride(1),
        *gpw_strides,
        *gpb_strides,
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
    )

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )
