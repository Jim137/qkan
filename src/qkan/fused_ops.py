"""
Triton-fused kernels for QKAN quantum circuit simulation.

Implements the pz_encoding ansatz forward pass as a single fused Triton kernel,
avoiding materialization of intermediate complex state vectors.
"""

import torch
import triton
import triton.language as tl


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
