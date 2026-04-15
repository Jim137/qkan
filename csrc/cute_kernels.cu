// Copyright (c) 2026, Jiun-Cheng Jiang. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CuTe DSL CUDA kernels for QKAN quantum circuit simulation.
//
// Uses CUTLASS CuTe for tensor layout abstractions (make_tensor, make_gmem_ptr,
// make_layout), __sincosf intrinsics for simultaneous sin/cos, shared memory
// trig caching (eliminates redundant trig across batch tiles), and warp shuffle
// reductions for efficient gradient accumulation.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

using namespace cute;

// Precision types for I/O and backward state checkpoints
using bf16_t = cutlass::bfloat16_t;     // 2 bytes, native bf16 I/O & state
using fp8_t  = cutlass::float_e4m3_t;   // 1 byte,  prescale 224.0
using fp4_t  = cutlass::float_e2m1_t;   // 0.5 byte, prescale 6.0 (experimental)

// ====================================================================
// Constants & Utilities
// ====================================================================

static constexpr float INV_SQRT2 = 0.7071067811865476f;

// Row-major layout macros (CuTe defaults to column-major / LayoutLeft).
// PyTorch tensors are contiguous in the LAST dimension, so we must
// provide explicit strides matching C order.
#define ROWMAJOR2(d0, d1) \
    make_layout(make_shape((d0), (d1)), make_stride((d1), 1))
#define ROWMAJOR3(d0, d1, d2) \
    make_layout(make_shape((d0), (d1), (d2)), make_stride((d1)*(d2), (d2), 1))

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/// Block-wide sum → single atomicAdd.  Specialized fast path for single-warp
/// blocks (BLOCK_B ≤ 32); multi-warp path uses smem for cross-warp reduction.
template <int BLOCK_B>
__device__ __forceinline__ void block_reduce_atomic_add(
    float val, bool valid, float* reduce_smem, float* target
) {
    constexpr int NUM_WARPS = (BLOCK_B + 31) / 32;
    float masked = valid ? val : 0.0f;

    if constexpr (NUM_WARPS == 1) {
        // Fast path: single warp — no smem, no syncthreads
        float sum = warp_reduce_sum(masked);
        if ((threadIdx.x & 31) == 0 && sum != 0.0f) atomicAdd(target, sum);
    } else {
        int tid     = threadIdx.x;
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        float warp_sum = warp_reduce_sum(masked);
        if (lane_id == 0) reduce_smem[warp_id] = warp_sum;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < NUM_WARPS) ? reduce_smem[tid] : 0.0f;
            v = warp_reduce_sum(v);
            if (tid == 0 && v != 0.0f) atomicAdd(target, v);
        }
        __syncthreads();
    }
}

// ====================================================================
// PZ-encoding forward kernel
// ====================================================================
// Circuit: H|0> → [Rz(θ₀) Ry(θ₁) Rz(enc)]×reps → Rz(θ₀_final) Ry(θ₁_final) → ⟨Z⟩
//
// Shared memory caches cos/sin of all theta half-angles so trig is computed
// once per block rather than once per thread.

template <typename IOT, int BLOCK_B>
__global__ void cute_pz_fwd_kernel(
    const IOT* __restrict__ x_ptr,
    const IOT* __restrict__ theta_ptr,
    const IOT* __restrict__ pw_ptr,
    const IOT* __restrict__ pb_ptr,
    IOT* __restrict__ out_ptr,
    int batch_size, int in_dim, int out_dim, int reps,
    bool preacts_trainable, bool fast_measure)
{
    extern __shared__ float smem[];

    int pid_oi = blockIdx.x;
    int pid_b  = blockIdx.y;
    int tid    = threadIdx.x;

    int idx_i = pid_oi % in_dim;
    int idx_o = pid_oi / in_dim;
    if (idx_o >= out_dim) return;

    // ── CuTe tensor views (global memory) ──
    auto gTheta = make_tensor(make_gmem_ptr(theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps + 1, 2),
                    make_stride(in_dim * (reps + 1) * 2, (reps + 1) * 2, 2, 1)));

    // ── Shared memory trig cache ──
    int n_trig = (reps + 1) * 2;
    float* s_cos = smem;
    float* s_sin = smem + n_trig;

    if (tid == 0) {
        for (int l = 0; l <= reps; l++) {
            __sincosf(float(gTheta(idx_o, idx_i, l, 0)) * 0.5f,
                      &s_sin[l * 2 + 0], &s_cos[l * 2 + 0]);
            __sincosf(float(gTheta(idx_o, idx_i, l, 1)) * 0.5f,
                      &s_sin[l * 2 + 1], &s_cos[l * 2 + 1]);
        }
    }
    __syncthreads();

    int b = pid_b * BLOCK_B + tid;
    if (b >= batch_size) return;

    // CuTe views for x & output (IOT element type handles bf16/f32 I/O)
    auto gX   = make_tensor(make_gmem_ptr(x_ptr),
                  ROWMAJOR2(batch_size, in_dim));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr),
                  ROWMAJOR3(batch_size, out_dim, in_dim));

    float x_val = float(gX(b, idx_i));

    // H|0⟩ = 1/√2 (|0⟩ + |1⟩)
    float r0 = INV_SQRT2, i0 = 0.0f;
    float r1 = INV_SQRT2, i1 = 0.0f;

    // CuTe view for preacts
    auto gPW = make_tensor(make_gmem_ptr(pw_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));
    auto gPB = make_tensor(make_gmem_ptr(pb_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));

    #pragma unroll
    for (int l = 0; l < reps; l++) {
        // Rz(θ[l,0])
        float cz = s_cos[l * 2], sz = s_sin[l * 2];
        float nr0 = r0 * cz + i0 * sz;
        float ni0 = i0 * cz - r0 * sz;
        float nr1 = r1 * cz - i1 * sz;
        float ni1 = i1 * cz + r1 * sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;

        // Ry(θ[l,1])
        float cy = s_cos[l * 2 + 1], sy = s_sin[l * 2 + 1];
        nr0 = cy * r0 - sy * r1;
        ni0 = cy * i0 - sy * i1;
        nr1 = sy * r0 + cy * r1;
        ni1 = sy * i0 + cy * i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;

        // Rz(enc)
        float enc = x_val;
        if (preacts_trainable) enc = float(gPW(idx_o, idx_i, l)) * x_val + float(gPB(idx_o, idx_i, l));
        float ce, se;
        __sincosf(enc * 0.5f, &se, &ce);
        nr0 = r0 * ce + i0 * se;
        ni0 = i0 * ce - r0 * se;
        nr1 = r1 * ce - i1 * se;
        ni1 = i1 * ce + r1 * se;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    // Final Rz, Ry
    {
        float cz = s_cos[reps * 2], sz = s_sin[reps * 2];
        float nr0 = r0 * cz + i0 * sz;
        float ni0 = i0 * cz - r0 * sz;
        float nr1 = r1 * cz - i1 * sz;
        float ni1 = i1 * cz + r1 * sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }
    {
        float cy = s_cos[reps * 2 + 1], sy = s_sin[reps * 2 + 1];
        float nr0 = cy * r0 - sy * r1;
        float ni0 = cy * i0 - sy * i1;
        float nr1 = sy * r0 + cy * r1;
        float ni1 = sy * i0 + cy * i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    float result;
    if (fast_measure) result = sqrtf(r0*r0 + i0*i0) - sqrtf(r1*r1 + i1*i1);
    else              result = (r0*r0 + i0*i0) - (r1*r1 + i1*i1);

    gOut(b, idx_o, idx_i) = IOT(result);
}

// ====================================================================
// PZ-encoding backward kernel
// ====================================================================

template <typename StateT, int BLOCK_B>
__global__ void cute_pz_bwd_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ theta_ptr,
    const float* __restrict__ pw_ptr,
    const float* __restrict__ pb_ptr,
    const float* __restrict__ grad_out_ptr,
    StateT* __restrict__ states_ptr,
    float* __restrict__ grad_theta_ptr,
    float* __restrict__ grad_x_ptr,
    float* __restrict__ grad_pw_ptr,
    float* __restrict__ grad_pb_ptr,
    int batch_size, int in_dim, int out_dim, int reps,
    bool preacts_trainable, bool fast_measure, float fp8_prescale)
{
    extern __shared__ float smem[];
    constexpr int NUM_WARPS = (BLOCK_B + 31) / 32;
    const float fp8_inv = 1.0f / fp8_prescale;

    int pid_oi = blockIdx.x;
    int pid_b  = blockIdx.y;
    int tid    = threadIdx.x;

    int idx_i = pid_oi % in_dim;
    int idx_o = pid_oi / in_dim;
    if (idx_o >= out_dim) return;

    // ── CuTe global tensor views ──
    auto gTheta = make_tensor(make_gmem_ptr(theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps + 1, 2),
                    make_stride(in_dim * (reps + 1) * 2, (reps + 1) * 2, 2, 1)));
    auto gX = make_tensor(make_gmem_ptr(x_ptr),
                ROWMAJOR2(batch_size, in_dim));
    auto gGradOut = make_tensor(make_gmem_ptr(grad_out_ptr),
                      ROWMAJOR3(batch_size, out_dim, in_dim));
    auto gPW = make_tensor(make_gmem_ptr(pw_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));
    auto gPB = make_tensor(make_gmem_ptr(pb_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));

    // ── Shared memory layout ──
    int n_trig = (reps + 1) * 2;
    float* s_cos = smem;
    float* s_sin = smem + n_trig;
    float* s_reduce = smem + 2 * n_trig;  // NUM_WARPS floats

    // Thread 0: precompute all theta trig (reused in both forward & backward)
    if (tid == 0) {
        for (int l = 0; l <= reps; l++) {
            __sincosf(gTheta(idx_o, idx_i, l, 0) * 0.5f,
                      &s_sin[l * 2 + 0], &s_cos[l * 2 + 0]);
            __sincosf(gTheta(idx_o, idx_i, l, 1) * 0.5f,
                      &s_sin[l * 2 + 1], &s_cos[l * 2 + 1]);
        }
    }
    __syncthreads();

    int b = pid_b * BLOCK_B + tid;
    bool valid = b < batch_size;

    float x_val = valid ? gX(b, idx_i) : 0.0f;

    // ── State buffer: (n_programs, n_states, 4, BLOCK_B) with StateT elements ──
    int n_states = 3 * reps + 3;
    int n_b_blocks = (batch_size + BLOCK_B - 1) / BLOCK_B;
    int program_idx = pid_oi * n_b_blocks + pid_b;
    int state_stride_n = n_states * 4 * BLOCK_B;
    int state_stride_s = 4 * BLOCK_B;
    int state_stride_c = BLOCK_B;
    StateT* my_states = states_ptr + program_idx * state_stride_n;

    // State save: f32 → prescale → StateT.  State load: StateT → f32 → unprescale.
    #define STATE_ADDR(sidx, comp) (my_states + (sidx) * state_stride_s + (comp) * state_stride_c + tid)
    #define SAVE_STATE(sidx, v0, v1, v2, v3) do { \
        if (valid) { \
            *STATE_ADDR(sidx, 0) = StateT((v0) * fp8_prescale); \
            *STATE_ADDR(sidx, 1) = StateT((v1) * fp8_prescale); \
            *STATE_ADDR(sidx, 2) = StateT((v2) * fp8_prescale); \
            *STATE_ADDR(sidx, 3) = StateT((v3) * fp8_prescale); \
        } \
    } while(0)
    #define LOAD_STATE4(sidx, v0, v1, v2, v3) do { \
        v0 = valid ? float(*STATE_ADDR(sidx, 0)) * fp8_inv : 0.0f; \
        v1 = valid ? float(*STATE_ADDR(sidx, 1)) * fp8_inv : 0.0f; \
        v2 = valid ? float(*STATE_ADDR(sidx, 2)) * fp8_inv : 0.0f; \
        v3 = valid ? float(*STATE_ADDR(sidx, 3)) * fp8_inv : 0.0f; \
    } while(0)

    // ── Phase 1: Forward recompute with state checkpointing ──
    float r0 = INV_SQRT2, i0 = 0.0f, r1 = INV_SQRT2, i1 = 0.0f;
    SAVE_STATE(0, r0, i0, r1, i1);

    int state_idx = 1;
    #pragma unroll
    for (int l = 0; l < reps; l++) {
        // Rz(θ[l,0])
        float cz = s_cos[l * 2], sz = s_sin[l * 2];
        float nr0 = r0 * cz + i0 * sz;
        float ni0 = i0 * cz - r0 * sz;
        float nr1 = r1 * cz - i1 * sz;
        float ni1 = i1 * cz + r1 * sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
        SAVE_STATE(state_idx, r0, i0, r1, i1);
        state_idx++;

        // Ry(θ[l,1])
        float cy = s_cos[l * 2 + 1], sy = s_sin[l * 2 + 1];
        nr0 = cy * r0 - sy * r1;
        ni0 = cy * i0 - sy * i1;
        nr1 = sy * r0 + cy * r1;
        ni1 = sy * i0 + cy * i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
        SAVE_STATE(state_idx, r0, i0, r1, i1);
        state_idx++;

        // Rz(enc)
        float enc = x_val;
        if (preacts_trainable) enc = gPW(idx_o, idx_i, l) * x_val + gPB(idx_o, idx_i, l);
        float ce, se;
        __sincosf(enc * 0.5f, &se, &ce);
        nr0 = r0 * ce + i0 * se;
        ni0 = i0 * ce - r0 * se;
        nr1 = r1 * ce - i1 * se;
        ni1 = i1 * ce + r1 * se;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
        SAVE_STATE(state_idx, r0, i0, r1, i1);
        state_idx++;
    }

    // Final Rz(θ[reps,0])
    {
        float cz = s_cos[reps * 2], sz = s_sin[reps * 2];
        float nr0 = r0 * cz + i0 * sz;
        float ni0 = i0 * cz - r0 * sz;
        float nr1 = r1 * cz - i1 * sz;
        float ni1 = i1 * cz + r1 * sz;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }
    SAVE_STATE(state_idx, r0, i0, r1, i1);
    state_idx++;

    // Final Ry(θ[reps,1])
    {
        float cy = s_cos[reps * 2 + 1], sy = s_sin[reps * 2 + 1];
        float nr0 = cy * r0 - sy * r1;
        float ni0 = cy * i0 - sy * i1;
        float nr1 = sy * r0 + cy * r1;
        float ni1 = sy * i0 + cy * i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    // ── Phase 2: Measurement gradient ──
    float go = valid ? gGradOut(b, idx_o, idx_i) : 0.0f;
    float ar0, ai0, ar1, ai1;
    if (fast_measure) {
        float alpha_norm = sqrtf(r0*r0 + i0*i0);
        float beta_norm  = sqrtf(r1*r1 + i1*i1);
        float inv_alpha  = (alpha_norm > 1e-30f) ? 1.0f / alpha_norm : 0.0f;
        float inv_beta   = (beta_norm  > 1e-30f) ? 1.0f / beta_norm  : 0.0f;
        ar0 =  go * r0 * inv_alpha;
        ai0 =  go * i0 * inv_alpha;
        ar1 = -go * r1 * inv_beta;
        ai1 = -go * i1 * inv_beta;
    } else {
        ar0 =  2.0f * go * r0;
        ai0 =  2.0f * go * i0;
        ar1 = -2.0f * go * r1;
        ai1 = -2.0f * go * i1;
    }

    // ── Phase 3: Backward sweep ──
    auto gGradTheta = make_tensor(make_gmem_ptr(grad_theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps + 1, 2),
                    make_stride(in_dim * (reps + 1) * 2, (reps + 1) * 2, 2, 1)));
    float grad_x_local = 0.0f;

    // Backward through final Ry(θ[reps,1])
    float sr0, si0, sr1, si1;
    state_idx--;
    LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
    {
        float cy = s_cos[reps * 2 + 1], sy = s_sin[reps * 2 + 1];
        float gv = 0.5f * (ar0 * (-sy * sr0 - cy * sr1) +
                            ai0 * (-sy * si0 - cy * si1) +
                            ar1 * ( cy * sr0 - sy * sr1) +
                            ai1 * ( cy * si0 - sy * si1));
        block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
            &gGradTheta(idx_o, idx_i, reps, 1));

        float nar0 =  cy * ar0 + sy * ar1;
        float nai0 =  cy * ai0 + sy * ai1;
        float nar1 = -sy * ar0 + cy * ar1;
        float nai1 = -sy * ai0 + cy * ai1;
        ar0 = nar0; ai0 = nai0; ar1 = nar1; ai1 = nai1;
    }

    // Backward through final Rz(θ[reps,0])
    state_idx--;
    LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
    {
        float cz = s_cos[reps * 2], sz = s_sin[reps * 2];
        float gv = 0.5f * (-sz * (ar0*sr0 + ai0*si0 + ar1*sr1 + ai1*si1) +
                             cz * (ar0*si0 - ai0*sr0 - ar1*si1 + ai1*sr1));
        block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
            &gGradTheta(idx_o, idx_i, reps, 0));

        float nar0 =  cz * ar0 - sz * ai0;
        float nai0 =  sz * ar0 + cz * ai0;
        float nar1 =  cz * ar1 + sz * ai1;
        float nai1 = -sz * ar1 + cz * ai1;
        ar0 = nar0; ai0 = nai0; ar1 = nar1; ai1 = nai1;
    }

    // Loop backward through layers
    #pragma unroll
    for (int l = reps - 1; l >= 0; l--) {
        // Backward Rz(enc)
        state_idx--;
        LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
        {
            float enc = x_val;
            if (preacts_trainable)
                enc = gPW(idx_o, idx_i, l) * x_val + gPB(idx_o, idx_i, l);
            float ce, se;
            __sincosf(enc * 0.5f, &se, &ce);

            float grad_enc = 0.5f * (
                -se * (ar0*sr0 + ai0*si0 + ar1*sr1 + ai1*si1) +
                 ce * (ar0*si0 - ai0*sr0 - ar1*si1 + ai1*sr1));

            if (preacts_trainable) {
                float w = gPW(idx_o, idx_i, l);
                block_reduce_atomic_add<BLOCK_B>(grad_enc * x_val, valid, s_reduce,
                    &grad_pw_ptr[(idx_o * in_dim + idx_i) * reps + l]);
                block_reduce_atomic_add<BLOCK_B>(grad_enc, valid, s_reduce,
                    &grad_pb_ptr[(idx_o * in_dim + idx_i) * reps + l]);
                grad_x_local += grad_enc * w;
            } else {
                grad_x_local += grad_enc;
            }

            float nar0 =  ce * ar0 - se * ai0;
            float nai0 =  se * ar0 + ce * ai0;
            float nar1 =  ce * ar1 + se * ai1;
            float nai1 = -se * ar1 + ce * ai1;
            ar0 = nar0; ai0 = nai0; ar1 = nar1; ai1 = nai1;
        }

        // Backward Ry(θ[l,1])
        state_idx--;
        LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
        {
            float cy = s_cos[l * 2 + 1], sy = s_sin[l * 2 + 1];
            float gv = 0.5f * (ar0 * (-sy * sr0 - cy * sr1) +
                                ai0 * (-sy * si0 - cy * si1) +
                                ar1 * ( cy * sr0 - sy * sr1) +
                                ai1 * ( cy * si0 - sy * si1));
            block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
                &gGradTheta(idx_o, idx_i, l, 1));

            float nar0 =  cy * ar0 + sy * ar1;
            float nai0 =  cy * ai0 + sy * ai1;
            float nar1 = -sy * ar0 + cy * ar1;
            float nai1 = -sy * ai0 + cy * ai1;
            ar0 = nar0; ai0 = nai0; ar1 = nar1; ai1 = nai1;
        }

        // Backward Rz(θ[l,0])
        state_idx--;
        LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
        {
            float cz = s_cos[l * 2], sz = s_sin[l * 2];
            float gv = 0.5f * (-sz * (ar0*sr0 + ai0*si0 + ar1*sr1 + ai1*si1) +
                                 cz * (ar0*si0 - ai0*sr0 - ar1*si1 + ai1*sr1));
            block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
                &gGradTheta(idx_o, idx_i, l, 0));

            float nar0 =  cz * ar0 - sz * ai0;
            float nai0 =  sz * ar0 + cz * ai0;
            float nar1 =  cz * ar1 + sz * ai1;
            float nai1 = -sz * ar1 + cz * ai1;
            ar0 = nar0; ai0 = nai0; ar1 = nar1; ai1 = nai1;
        }
    }

    // grad_x: atomic because multiple (o,i) programs accumulate to the same x element
    if (valid && grad_x_local != 0.0f)
        atomicAdd(&grad_x_ptr[b * in_dim + idx_i], grad_x_local);

    #undef STATE_ADDR
    #undef SAVE_STATE
    #undef LOAD_STATE4
}

// ====================================================================
// RPZ-encoding forward kernel
// ====================================================================
// Circuit: H|0> → [Ry(θ) Rz(w·x+b)]×reps → Ry(θ_final) → ⟨Z⟩

template <typename IOT, int BLOCK_B>
__global__ void cute_rpz_fwd_kernel(
    const IOT* __restrict__ x_ptr,
    const IOT* __restrict__ theta_ptr,
    const IOT* __restrict__ pw_ptr,
    const IOT* __restrict__ pb_ptr,
    IOT* __restrict__ out_ptr,
    int batch_size, int in_dim, int out_dim, int reps,
    bool fast_measure)
{
    extern __shared__ float smem[];

    int pid_oi = blockIdx.x, pid_b = blockIdx.y, tid = threadIdx.x;
    int idx_i = pid_oi % in_dim, idx_o = pid_oi / in_dim;
    if (idx_o >= out_dim) return;

    // CuTe views
    auto gTheta = make_tensor(make_gmem_ptr(theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps + 1, 1),
                    make_stride(in_dim * (reps + 1), (reps + 1), 1, 1)));
    auto gPW = make_tensor(make_gmem_ptr(pw_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));
    auto gPB = make_tensor(make_gmem_ptr(pb_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));

    // Shared trig cache (reps+1 Ry parameters)
    float* s_cos = smem;
    float* s_sin = smem + (reps + 1);

    if (tid == 0) {
        for (int l = 0; l <= reps; l++)
            __sincosf(float(gTheta(idx_o, idx_i, l, 0)) * 0.5f,
                      &s_sin[l], &s_cos[l]);
    }
    __syncthreads();

    int b = pid_b * BLOCK_B + tid;
    if (b >= batch_size) return;

    auto gX   = make_tensor(make_gmem_ptr(x_ptr),
                  ROWMAJOR2(batch_size, in_dim));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr),
                  ROWMAJOR3(batch_size, out_dim, in_dim));
    float x_val = float(gX(b, idx_i));

    float r0 = INV_SQRT2, i0 = 0.0f, r1 = INV_SQRT2, i1 = 0.0f;

    #pragma unroll
    for (int l = 0; l < reps; l++) {
        // Ry(θ[l])
        float cy = s_cos[l], sy = s_sin[l];
        float nr0 = cy * r0 - sy * r1;
        float ni0 = cy * i0 - sy * i1;
        float nr1 = sy * r0 + cy * r1;
        float ni1 = sy * i0 + cy * i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;

        // Rz(w·x+b)
        float enc = float(gPW(idx_o, idx_i, l)) * x_val + float(gPB(idx_o, idx_i, l));
        float ce, se;
        __sincosf(enc * 0.5f, &se, &ce);
        nr0 = r0 * ce + i0 * se;
        ni0 = i0 * ce - r0 * se;
        nr1 = r1 * ce - i1 * se;
        ni1 = i1 * ce + r1 * se;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    // Final Ry(θ[reps])
    {
        float cy = s_cos[reps], sy = s_sin[reps];
        float nr0 = cy * r0 - sy * r1;
        float ni0 = cy * i0 - sy * i1;
        float nr1 = sy * r0 + cy * r1;
        float ni1 = sy * i0 + cy * i1;
        r0 = nr0; i0 = ni0; r1 = nr1; i1 = ni1;
    }

    float result;
    if (fast_measure) result = sqrtf(r0*r0 + i0*i0) - sqrtf(r1*r1 + i1*i1);
    else              result = (r0*r0 + i0*i0) - (r1*r1 + i1*i1);

    gOut(b, idx_o, idx_i) = IOT(result);
}

// ====================================================================
// RPZ-encoding backward kernel
// ====================================================================

template <typename StateT, int BLOCK_B>
__global__ void cute_rpz_bwd_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ theta_ptr,
    const float* __restrict__ pw_ptr,
    const float* __restrict__ pb_ptr,
    const float* __restrict__ grad_out_ptr,
    StateT* __restrict__ states_ptr,
    float* __restrict__ grad_theta_ptr,
    float* __restrict__ grad_x_ptr,
    float* __restrict__ grad_pw_ptr,
    float* __restrict__ grad_pb_ptr,
    int batch_size, int in_dim, int out_dim, int reps,
    bool fast_measure, float fp8_prescale)
{
    extern __shared__ float smem[];
    constexpr int NUM_WARPS = (BLOCK_B + 31) / 32;
    const float fp8_inv = 1.0f / fp8_prescale;

    int pid_oi = blockIdx.x, pid_b = blockIdx.y, tid = threadIdx.x;
    int idx_i = pid_oi % in_dim, idx_o = pid_oi / in_dim;
    if (idx_o >= out_dim) return;

    auto gTheta = make_tensor(make_gmem_ptr(theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps + 1, 1),
                    make_stride(in_dim * (reps + 1), (reps + 1), 1, 1)));
    auto gX = make_tensor(make_gmem_ptr(x_ptr),
                ROWMAJOR2(batch_size, in_dim));
    auto gGradOut = make_tensor(make_gmem_ptr(grad_out_ptr),
                      ROWMAJOR3(batch_size, out_dim, in_dim));
    auto gPW = make_tensor(make_gmem_ptr(pw_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));
    auto gPB = make_tensor(make_gmem_ptr(pb_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));

    float* s_cos = smem;
    float* s_sin = smem + (reps + 1);
    float* s_reduce = smem + 2 * (reps + 1);

    if (tid == 0) {
        for (int l = 0; l <= reps; l++)
            __sincosf(gTheta(idx_o, idx_i, l, 0) * 0.5f,
                      &s_sin[l], &s_cos[l]);
    }
    __syncthreads();

    int b = pid_b * BLOCK_B + tid;
    bool valid = b < batch_size;
    float x_val = valid ? gX(b, idx_i) : 0.0f;

    int n_states = 2 * reps + 2;
    int n_b_blocks = (batch_size + BLOCK_B - 1) / BLOCK_B;
    int program_idx = pid_oi * n_b_blocks + pid_b;
    StateT* my_states = states_ptr + program_idx * n_states * 4 * BLOCK_B;
    int state_stride_s = 4 * BLOCK_B, state_stride_c = BLOCK_B;

    #define STATE_ADDR(sidx, comp) (my_states + (sidx) * state_stride_s + (comp) * state_stride_c + tid)
    #define SAVE_STATE(sidx, v0, v1, v2, v3) do { \
        if (valid) { *STATE_ADDR(sidx,0)=StateT((v0)*fp8_prescale); *STATE_ADDR(sidx,1)=StateT((v1)*fp8_prescale); \
                     *STATE_ADDR(sidx,2)=StateT((v2)*fp8_prescale); *STATE_ADDR(sidx,3)=StateT((v3)*fp8_prescale); } } while(0)
    #define LOAD_STATE4(sidx, v0, v1, v2, v3) do { \
        v0=valid?float(*STATE_ADDR(sidx,0))*fp8_inv:0.f; v1=valid?float(*STATE_ADDR(sidx,1))*fp8_inv:0.f; \
        v2=valid?float(*STATE_ADDR(sidx,2))*fp8_inv:0.f; v3=valid?float(*STATE_ADDR(sidx,3))*fp8_inv:0.f; } while(0)

    // Phase 1: Forward recompute
    float r0 = INV_SQRT2, i0 = 0.0f, r1 = INV_SQRT2, i1 = 0.0f;
    SAVE_STATE(0, r0, i0, r1, i1);
    int state_idx = 1;

    #pragma unroll
    for (int l = 0; l < reps; l++) {
        float cy = s_cos[l], sy = s_sin[l];
        float nr0 = cy*r0 - sy*r1, ni0 = cy*i0 - sy*i1;
        float nr1 = sy*r0 + cy*r1, ni1 = sy*i0 + cy*i1;
        r0=nr0; i0=ni0; r1=nr1; i1=ni1;
        SAVE_STATE(state_idx, r0, i0, r1, i1); state_idx++;

        float enc = gPW(idx_o, idx_i, l) * x_val + gPB(idx_o, idx_i, l);
        float ce, se; __sincosf(enc * 0.5f, &se, &ce);
        nr0 = r0*ce + i0*se; ni0 = i0*ce - r0*se;
        nr1 = r1*ce - i1*se; ni1 = i1*ce + r1*se;
        r0=nr0; i0=ni0; r1=nr1; i1=ni1;
        SAVE_STATE(state_idx, r0, i0, r1, i1); state_idx++;
    }

    // Final Ry
    {
        float cy = s_cos[reps], sy = s_sin[reps];
        float nr0 = cy*r0 - sy*r1, ni0 = cy*i0 - sy*i1;
        float nr1 = sy*r0 + cy*r1, ni1 = sy*i0 + cy*i1;
        r0=nr0; i0=ni0; r1=nr1; i1=ni1;
    }

    // Phase 2: Measurement gradient
    float go = valid ? gGradOut(b, idx_o, idx_i) : 0.0f;
    float ar0, ai0, ar1, ai1;
    if (fast_measure) {
        float an = sqrtf(r0*r0+i0*i0), bn = sqrtf(r1*r1+i1*i1);
        float ia = (an>1e-30f)?1.f/an:0.f, ib = (bn>1e-30f)?1.f/bn:0.f;
        ar0=go*r0*ia; ai0=go*i0*ia; ar1=-go*r1*ib; ai1=-go*i1*ib;
    } else {
        ar0=2.f*go*r0; ai0=2.f*go*i0; ar1=-2.f*go*r1; ai1=-2.f*go*i1;
    }

    auto gGradTheta = make_tensor(make_gmem_ptr(grad_theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps + 1, 1),
                    make_stride(in_dim * (reps + 1), (reps + 1), 1, 1)));
    float grad_x_local = 0.0f;

    // Phase 3: Backward sweep
    float sr0, si0, sr1, si1;

    // Backward final Ry(θ[reps])
    state_idx--;
    LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
    {
        float cy = s_cos[reps], sy = s_sin[reps];
        float gv = 0.5f * (ar0*(-sy*sr0 - cy*sr1) + ai0*(-sy*si0 - cy*si1) +
                            ar1*( cy*sr0 - sy*sr1) + ai1*( cy*si0 - sy*si1));
        block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
            &gGradTheta(idx_o, idx_i, reps, 0));
        float nar0= cy*ar0+sy*ar1, nai0= cy*ai0+sy*ai1;
        float nar1=-sy*ar0+cy*ar1, nai1=-sy*ai0+cy*ai1;
        ar0=nar0; ai0=nai0; ar1=nar1; ai1=nai1;
    }

    #pragma unroll
    for (int l = reps - 1; l >= 0; l--) {
        // Backward Rz(enc)
        state_idx--;
        LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
        {
            float w = gPW(idx_o, idx_i, l), bp = gPB(idx_o, idx_i, l);
            float enc = w * x_val + bp;
            float ce, se; __sincosf(enc * 0.5f, &se, &ce);
            float grad_enc = 0.5f * (-se*(ar0*sr0+ai0*si0+ar1*sr1+ai1*si1) +
                                       ce*(ar0*si0-ai0*sr0-ar1*si1+ai1*sr1));
            block_reduce_atomic_add<BLOCK_B>(grad_enc * x_val, valid, s_reduce,
                &grad_pw_ptr[(idx_o * in_dim + idx_i) * reps + l]);
            block_reduce_atomic_add<BLOCK_B>(grad_enc, valid, s_reduce,
                &grad_pb_ptr[(idx_o * in_dim + idx_i) * reps + l]);
            grad_x_local += grad_enc * w;

            float nar0= ce*ar0-se*ai0, nai0= se*ar0+ce*ai0;
            float nar1= ce*ar1+se*ai1, nai1=-se*ar1+ce*ai1;
            ar0=nar0; ai0=nai0; ar1=nar1; ai1=nai1;
        }

        // Backward Ry(θ[l])
        state_idx--;
        LOAD_STATE4(state_idx, sr0, si0, sr1, si1);
        {
            float cy = s_cos[l], sy = s_sin[l];
            float gv = 0.5f * (ar0*(-sy*sr0 - cy*sr1) + ai0*(-sy*si0 - cy*si1) +
                                ar1*( cy*sr0 - sy*sr1) + ai1*( cy*si0 - sy*si1));
            block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
                &gGradTheta(idx_o, idx_i, l, 0));
            float nar0= cy*ar0+sy*ar1, nai0= cy*ai0+sy*ai1;
            float nar1=-sy*ar0+cy*ar1, nai1=-sy*ai0+cy*ai1;
            ar0=nar0; ai0=nai0; ar1=nar1; ai1=nai1;
        }
    }

    if (valid && grad_x_local != 0.0f)
        atomicAdd(&grad_x_ptr[b * in_dim + idx_i], grad_x_local);

    #undef STATE_ADDR
    #undef SAVE_STATE
    #undef LOAD_STATE4
}

// ====================================================================
// Real-encoding forward kernel
// ====================================================================
// Circuit: H|0> → [X, Ry(θ), Z, Ry(enc)]×reps → ⟨Z⟩

template <typename IOT, int BLOCK_B>
__global__ void cute_real_fwd_kernel(
    const IOT* __restrict__ x_ptr,
    const IOT* __restrict__ theta_ptr,
    const IOT* __restrict__ pw_ptr,
    const IOT* __restrict__ pb_ptr,
    IOT* __restrict__ out_ptr,
    int batch_size, int in_dim, int out_dim, int reps,
    bool preacts_trainable, bool fast_measure, bool compute_bf16)
{
    extern __shared__ float smem[];

    int pid_oi = blockIdx.x, pid_b = blockIdx.y, tid = threadIdx.x;
    int idx_i = pid_oi % in_dim, idx_o = pid_oi / in_dim;
    if (idx_o >= out_dim) return;

    // CuTe views — real ansatz theta: [out_dim, in_dim, reps, 1]
    auto gTheta = make_tensor(make_gmem_ptr(theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps, 1),
                    make_stride(in_dim * reps, reps, 1, 1)));
    auto gPW = make_tensor(make_gmem_ptr(pw_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));
    auto gPB = make_tensor(make_gmem_ptr(pb_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));

    // Shared trig cache (reps Ry parameters — no final layer)
    float* s_cos = smem;
    float* s_sin = smem + reps;
    if (tid == 0) {
        for (int l = 0; l < reps; l++)
            __sincosf(float(gTheta(idx_o, idx_i, l, 0)) * 0.5f,
                      &s_sin[l], &s_cos[l]);
    }
    __syncthreads();

    int b = pid_b * BLOCK_B + tid;
    if (b >= batch_size) return;

    auto gX   = make_tensor(make_gmem_ptr(x_ptr),
                  ROWMAJOR2(batch_size, in_dim));
    auto gOut = make_tensor(make_gmem_ptr(out_ptr),
                  ROWMAJOR3(batch_size, out_dim, in_dim));
    float x_val = float(gX(b, idx_i));

    if (compute_bf16) {
        float r0 = INV_SQRT2, r1 = INV_SQRT2;
        #pragma unroll
        for (int l = 0; l < reps; l++) {
            float tmp = r0; r0 = r1; r1 = tmp;
            float cy = s_cos[l], sy = s_sin[l];
            float nr0 = cy * r0 - sy * r1;
            float nr1 = sy * r0 + cy * r1;
            r0 = nr0; r1 = nr1;
            r1 = -r1;
            float enc = x_val;
            if (preacts_trainable) enc = float(gPW(idx_o, idx_i, l)) * x_val + float(gPB(idx_o, idx_i, l));
            float ce, se;
            __sincosf(enc * 0.5f, &se, &ce);
            nr0 = ce * r0 - se * r1;
            nr1 = se * r0 + ce * r1;
            r0 = nr0; r1 = nr1;
        }
        float result;
        if (fast_measure) result = fabsf(r0) - fabsf(r1);
        else              result = r0*r0 - r1*r1;
        gOut(b, idx_o, idx_i) = IOT(result);
    } else {
        float r0 = INV_SQRT2, i0 = 0.0f, r1 = INV_SQRT2, i1 = 0.0f;
        #pragma unroll
        for (int l = 0; l < reps; l++) {
            float tr = r0, ti = i0; r0 = r1; i0 = i1; r1 = tr; i1 = ti;
            float cy = s_cos[l], sy = s_sin[l];
            float nr0 = cy*r0 - sy*r1, ni0 = cy*i0 - sy*i1;
            float nr1 = sy*r0 + cy*r1, ni1 = sy*i0 + cy*i1;
            r0=nr0; i0=ni0; r1=nr1; i1=ni1;
            r1 = -r1; i1 = -i1;
            float enc = x_val;
            if (preacts_trainable) enc = float(gPW(idx_o, idx_i, l)) * x_val + float(gPB(idx_o, idx_i, l));
            float ce, se;
            __sincosf(enc * 0.5f, &se, &ce);
            nr0 = ce*r0 - se*r1; ni0 = ce*i0 - se*i1;
            nr1 = se*r0 + ce*r1; ni1 = se*i0 + ce*i1;
            r0=nr0; i0=ni0; r1=nr1; i1=ni1;
        }
        float result;
        if (fast_measure) result = sqrtf(r0*r0+i0*i0) - sqrtf(r1*r1+i1*i1);
        else              result = (r0*r0+i0*i0) - (r1*r1+i1*i1);
        gOut(b, idx_o, idx_i) = IOT(result);
    }
}

// ====================================================================
// Real-encoding backward kernel
// ====================================================================

template <typename StateT, int BLOCK_B>
__global__ void cute_real_bwd_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ theta_ptr,
    const float* __restrict__ pw_ptr,
    const float* __restrict__ pb_ptr,
    const float* __restrict__ grad_out_ptr,
    StateT* __restrict__ states_ptr,
    float* __restrict__ grad_theta_ptr,
    float* __restrict__ grad_x_ptr,
    float* __restrict__ grad_pw_ptr,
    float* __restrict__ grad_pb_ptr,
    int batch_size, int in_dim, int out_dim, int reps,
    bool preacts_trainable, bool fast_measure, bool compute_bf16,
    int n_components, float fp8_prescale)
{
    extern __shared__ float smem[];
    constexpr int NUM_WARPS = (BLOCK_B + 31) / 32;
    const float fp8_inv = 1.0f / fp8_prescale;

    int pid_oi = blockIdx.x, pid_b = blockIdx.y, tid = threadIdx.x;
    int idx_i = pid_oi % in_dim, idx_o = pid_oi / in_dim;
    if (idx_o >= out_dim) return;

    auto gTheta = make_tensor(make_gmem_ptr(theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps, 1),
                    make_stride(in_dim * reps, reps, 1, 1)));
    auto gX = make_tensor(make_gmem_ptr(x_ptr),
                ROWMAJOR2(batch_size, in_dim));
    auto gGradOut = make_tensor(make_gmem_ptr(grad_out_ptr),
                      ROWMAJOR3(batch_size, out_dim, in_dim));
    auto gPW = make_tensor(make_gmem_ptr(pw_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));
    auto gPB = make_tensor(make_gmem_ptr(pb_ptr),
                 ROWMAJOR3(out_dim, in_dim, reps));

    float* s_cos = smem;
    float* s_sin = smem + reps;
    float* s_reduce = smem + 2 * reps;

    if (tid == 0) {
        for (int l = 0; l < reps; l++)
            __sincosf(gTheta(idx_o, idx_i, l, 0) * 0.5f,
                      &s_sin[l], &s_cos[l]);
    }
    __syncthreads();

    int b = pid_b * BLOCK_B + tid;
    bool valid = b < batch_size;
    float x_val = valid ? gX(b, idx_i) : 0.0f;

    int n_states = 3 * reps + 1;
    int n_b_blocks = (batch_size + BLOCK_B - 1) / BLOCK_B;
    int program_idx = pid_oi * n_b_blocks + pid_b;
    // n_components = 2 for bf16 path, 4 for full complex
    StateT* my_states = states_ptr + program_idx * n_states * n_components * BLOCK_B;
    int state_stride_s = n_components * BLOCK_B;
    int state_stride_c = BLOCK_B;

    #define STATE_ADDR(sidx, comp) (my_states + (sidx) * state_stride_s + (comp) * state_stride_c + tid)

    auto gGradTheta = make_tensor(make_gmem_ptr(grad_theta_ptr),
        make_layout(make_shape(out_dim, in_dim, reps, 1),
                    make_stride(in_dim * reps, reps, 1, 1)));
    float grad_x_local = 0.0f;

    if (compute_bf16) {
        // ── Real-only backward (2 components) ──
        #define SAVE2(sidx, v0, v1) do { if(valid){ *STATE_ADDR(sidx,0)=StateT((v0)*fp8_prescale); *STATE_ADDR(sidx,1)=StateT((v1)*fp8_prescale); } } while(0)
        #define LOAD2(sidx, v0, v1) do { v0=valid?float(*STATE_ADDR(sidx,0))*fp8_inv:0.f; v1=valid?float(*STATE_ADDR(sidx,1))*fp8_inv:0.f; } while(0)

        float r0 = INV_SQRT2, r1 = INV_SQRT2;
        SAVE2(0, r0, r1);
        int sidx = 1;

        #pragma unroll
        for (int l = 0; l < reps; l++) {
            // X gate
            float tmp = r0; r0 = r1; r1 = tmp;
            SAVE2(sidx, r0, r1); sidx++;

            // Ry(θ)
            float cy = s_cos[l], sy = s_sin[l];
            float nr0 = cy*r0 - sy*r1, nr1 = sy*r0 + cy*r1;
            r0 = nr0; r1 = nr1;
            SAVE2(sidx, r0, r1); sidx++;

            // Z gate + Ry(enc)
            r1 = -r1;
            float enc = x_val;
            if (preacts_trainable) enc = gPW(idx_o, idx_i, l) * x_val + gPB(idx_o, idx_i, l);
            float ce, se; __sincosf(enc * 0.5f, &se, &ce);
            nr0 = ce*r0 - se*r1; nr1 = se*r0 + ce*r1;
            r0 = nr0; r1 = nr1;
            SAVE2(sidx, r0, r1); sidx++;
        }

        // Measurement gradient
        float go = valid ? gGradOut(b, idx_o, idx_i) : 0.0f;
        float ar0, ar1;
        if (fast_measure) {
            ar0 = go * ((r0 >= 0.f) ? 1.f : -1.f);
            ar1 = -go * ((r1 >= 0.f) ? 1.f : -1.f);
        } else {
            ar0 = 2.f * go * r0;
            ar1 = -2.f * go * r1;
        }

        // Backward sweep — use direct index: 3l+2 = after Ry_θ (input to Z+Ry_enc),
        //                                      3l+1 = after X   (input to Ry_θ)
        float sr0, sr1;
        #pragma unroll
        for (int l = reps - 1; l >= 0; l--) {
            // Backward Ry(enc) + Z gate — input state is after Ry_θ[l]
            LOAD2(3*l + 2, sr0, sr1);
            float z_sr1 = -sr1;  // state after Z gate (sr0, -sr1)
            float enc = x_val;
            if (preacts_trainable) enc = gPW(idx_o, idx_i, l) * x_val + gPB(idx_o, idx_i, l);
            float ce, se; __sincosf(enc * 0.5f, &se, &ce);
            float grad_enc = 0.5f * (-se * (ar0 * sr0 + ar1 * z_sr1) +
                                       ce * (-ar0 * z_sr1 + ar1 * sr0));

            if (preacts_trainable) {
                float w = gPW(idx_o, idx_i, l);
                block_reduce_atomic_add<BLOCK_B>(grad_enc * x_val, valid, s_reduce,
                    &grad_pw_ptr[(idx_o * in_dim + idx_i) * reps + l]);
                block_reduce_atomic_add<BLOCK_B>(grad_enc, valid, s_reduce,
                    &grad_pb_ptr[(idx_o * in_dim + idx_i) * reps + l]);
                grad_x_local += grad_enc * w;
            } else {
                grad_x_local += grad_enc;
            }

            // Adjoint through Ry_enc
            float nar0 =  ce * ar0 + se * ar1;
            float nar1 = -se * ar0 + ce * ar1;
            ar0 = nar0; ar1 = nar1;

            // Adjoint through Z gate: ar1 = -ar1
            ar1 = -ar1;

            // Backward Ry(θ) — input state is after X[l]
            LOAD2(3*l + 1, sr0, sr1);
            {
                float cy = s_cos[l], sy = s_sin[l];
                float gv = 0.5f * (ar0 * (-sy * sr0 - cy * sr1) +
                                    ar1 * ( cy * sr0 - sy * sr1));
                block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
                    &gGradTheta(idx_o, idx_i, l, 0));
                nar0 =  cy * ar0 + sy * ar1;
                nar1 = -sy * ar0 + cy * ar1;
                ar0 = nar0; ar1 = nar1;
            }

            // Backward X gate: swap adjoint (no state needed)
            {
                float t = ar0; ar0 = ar1; ar1 = t;
            }
        }

        #undef SAVE2
        #undef LOAD2
    } else {
        // ── Full complex backward (4 components) ──
        #define SAVE4(sidx, v0,v1,v2,v3) do { if(valid){ \
            *STATE_ADDR(sidx,0)=StateT((v0)*fp8_prescale); *STATE_ADDR(sidx,1)=StateT((v1)*fp8_prescale); \
            *STATE_ADDR(sidx,2)=StateT((v2)*fp8_prescale); *STATE_ADDR(sidx,3)=StateT((v3)*fp8_prescale); } } while(0)
        #define LOAD4(sidx, v0,v1,v2,v3) do { \
            v0=valid?float(*STATE_ADDR(sidx,0))*fp8_inv:0.f; v1=valid?float(*STATE_ADDR(sidx,1))*fp8_inv:0.f; \
            v2=valid?float(*STATE_ADDR(sidx,2))*fp8_inv:0.f; v3=valid?float(*STATE_ADDR(sidx,3))*fp8_inv:0.f; } while(0)

        float r0 = INV_SQRT2, i0 = 0.f, r1 = INV_SQRT2, i1 = 0.f;
        SAVE4(0, r0, i0, r1, i1);
        int sidx = 1;

        #pragma unroll
        for (int l = 0; l < reps; l++) {
            // X gate
            float tr=r0, ti=i0; r0=r1; i0=i1; r1=tr; i1=ti;
            SAVE4(sidx, r0, i0, r1, i1); sidx++;

            // Ry(θ)
            float cy = s_cos[l], sy = s_sin[l];
            float nr0=cy*r0-sy*r1, ni0=cy*i0-sy*i1;
            float nr1=sy*r0+cy*r1, ni1=sy*i0+cy*i1;
            r0=nr0; i0=ni0; r1=nr1; i1=ni1;
            SAVE4(sidx, r0, i0, r1, i1); sidx++;

            // Z gate
            r1=-r1; i1=-i1;

            // Ry(enc)
            float enc = x_val;
            if (preacts_trainable) enc = gPW(idx_o, idx_i, l) * x_val + gPB(idx_o, idx_i, l);
            float ce, se; __sincosf(enc * 0.5f, &se, &ce);
            nr0=ce*r0-se*r1; ni0=ce*i0-se*i1;
            nr1=se*r0+ce*r1; ni1=se*i0+ce*i1;
            r0=nr0; i0=ni0; r1=nr1; i1=ni1;
            SAVE4(sidx, r0, i0, r1, i1); sidx++;
        }

        // Measurement gradient
        float go = valid ? gGradOut(b, idx_o, idx_i) : 0.0f;
        float ar0, ai0, ar1, ai1;
        if (fast_measure) {
            float an = sqrtf(r0*r0+i0*i0), bn = sqrtf(r1*r1+i1*i1);
            float ia = (an>1e-30f)?1.f/an:0.f, ib = (bn>1e-30f)?1.f/bn:0.f;
            ar0=go*r0*ia; ai0=go*i0*ia; ar1=-go*r1*ib; ai1=-go*i1*ib;
        } else {
            ar0=2.f*go*r0; ai0=2.f*go*i0; ar1=-2.f*go*r1; ai1=-2.f*go*i1;
        }

        float sr0, si0, sr1, si1;
        // Backward sweep — direct index: 3l+2 = after Ry_θ, 3l+1 = after X
        #pragma unroll
        for (int l = reps - 1; l >= 0; l--) {
            // Backward Ry(enc) + Z gate — input is after Ry_θ[l]
            LOAD4(3*l + 2, sr0, si0, sr1, si1);
            float z_sr1 = -sr1, z_si1 = -si1;  // state after Z gate
            float enc = x_val;
            if (preacts_trainable) enc = gPW(idx_o, idx_i, l) * x_val + gPB(idx_o, idx_i, l);
            float ce, se; __sincosf(enc * 0.5f, &se, &ce);

            float grad_enc = 0.5f * (
                ar0*(-se*sr0 - ce*z_sr1) + ai0*(-se*si0 - ce*z_si1) +
                ar1*(ce*sr0 - se*z_sr1) + ai1*(ce*si0 - se*z_si1));

            if (preacts_trainable) {
                float w = gPW(idx_o, idx_i, l);
                block_reduce_atomic_add<BLOCK_B>(grad_enc * x_val, valid, s_reduce,
                    &grad_pw_ptr[(idx_o * in_dim + idx_i) * reps + l]);
                block_reduce_atomic_add<BLOCK_B>(grad_enc, valid, s_reduce,
                    &grad_pb_ptr[(idx_o * in_dim + idx_i) * reps + l]);
                grad_x_local += grad_enc * w;
            } else {
                grad_x_local += grad_enc;
            }

            // Adjoint through Ry_enc
            float nar0= ce*ar0+se*ar1, nai0= ce*ai0+se*ai1;
            float nar1=-se*ar0+ce*ar1, nai1=-se*ai0+ce*ai1;
            ar0=nar0; ai0=nai0; ar1=nar1; ai1=nai1;

            // Adjoint through Z gate
            ar1 = -ar1; ai1 = -ai1;

            // Backward Ry(θ) — input is after X[l]
            LOAD4(3*l + 1, sr0, si0, sr1, si1);
            {
                float cy = s_cos[l], sy = s_sin[l];
                float gv = 0.5f * (ar0*(-sy*sr0 - cy*sr1) + ai0*(-sy*si0 - cy*si1) +
                                    ar1*( cy*sr0 - sy*sr1) + ai1*( cy*si0 - sy*si1));
                block_reduce_atomic_add<BLOCK_B>(gv, valid, s_reduce,
                    &gGradTheta(idx_o, idx_i, l, 0));
                nar0= cy*ar0+sy*ar1; nai0= cy*ai0+sy*ai1;
                nar1=-sy*ar0+cy*ar1; nai1=-sy*ai0+cy*ai1;
                ar0=nar0; ai0=nai0; ar1=nar1; ai1=nai1;
            }

            // Backward X gate: swap adjoint (no state needed)
            {
                float t;
                t=ar0; ar0=ar1; ar1=t;
                t=ai0; ai0=ai1; ai1=t;
            }
        }

        #undef SAVE4
        #undef LOAD4
    }

    if (valid && grad_x_local != 0.0f)
        atomicAdd(&grad_x_ptr[b * in_dim + idx_i], grad_x_local);

    #undef STATE_ADDR
}

// ====================================================================
// Host launcher helpers
// ====================================================================

static int select_block_b(int n_oi, int batch, int base = 32) {
    if (base != 32) return base;
    // CuTe benefits from large BLOCK_B: better smem trig amortization + fewer
    // atomicAdd operations in the backward (fewer batch blocks per grad element).
    // Unlike Triton (which auto-tunes), we pick aggressively based on batch size.
    if (batch >= 512)  return 256;
    if (batch >= 128)  return 128;
    if (batch >= 64)   return 64;
    return 32;
}

// Forward kernels: template<typename IOT, int BLOCK_B>
#define DISPATCH_FWD(block_b, IOT_TYPE, KERNEL, grid, smem, ...) \
    do { \
        switch (block_b) { \
            case 32:  KERNEL<IOT_TYPE, 32> <<<grid, 32,  smem>>>(__VA_ARGS__); break; \
            case 64:  KERNEL<IOT_TYPE, 64> <<<grid, 64,  smem>>>(__VA_ARGS__); break; \
            case 128: KERNEL<IOT_TYPE, 128><<<grid, 128, smem>>>(__VA_ARGS__); break; \
            case 256: KERNEL<IOT_TYPE, 256><<<grid, 256, smem>>>(__VA_ARGS__); break; \
            default:  KERNEL<IOT_TYPE, 32> <<<grid, 32,  smem>>>(__VA_ARGS__); break; \
        } \
    } while(0)

// Backward kernels: template<typename StateT, int BLOCK_B>
#define DISPATCH_BWD(block_b, STATE_T, KERNEL, grid, smem, ...) \
    do { \
        switch (block_b) { \
            case 32:  KERNEL<STATE_T, 32> <<<grid, 32,  smem>>>(__VA_ARGS__); break; \
            case 64:  KERNEL<STATE_T, 64> <<<grid, 64,  smem>>>(__VA_ARGS__); break; \
            case 128: KERNEL<STATE_T, 128><<<grid, 128, smem>>>(__VA_ARGS__); break; \
            case 256: KERNEL<STATE_T, 256><<<grid, 256, smem>>>(__VA_ARGS__); break; \
            default:  KERNEL<STATE_T, 32> <<<grid, 32,  smem>>>(__VA_ARGS__); break; \
        } \
    } while(0)

// ====================================================================
// Host helpers
// ====================================================================

/// Fast tensor preparation: ensure contiguity first, then dtype conversion.
static inline torch::Tensor prep(torch::Tensor t, torch::ScalarType dtype) {
    if (!t.is_contiguous()) t = t.contiguous();
    if (t.scalar_type() != dtype) t = t.to(dtype);
    return t;
}

// ====================================================================
// Python-facing launchers
// ====================================================================

torch::Tensor cute_pz_forward(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    bool preacts_trainable, bool fast_measure, bool use_bf16)
{
    int batch   = x.size(0);
    int in_dim  = x.size(1);
    int out_dim = theta.size(0);
    int reps    = theta.size(2) - 1;

    auto io_dtype = use_bf16 ? torch::kBFloat16 : torch::kFloat32;
    x     = prep(x, io_dtype);
    theta = prep(theta, io_dtype);
    pw = prep(pw, io_dtype);
    pb = prep(pb, io_dtype);

    auto output = torch::empty({batch, out_dim, in_dim},
        torch::TensorOptions().device(x.device()).dtype(io_dtype));
    int n_oi = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch);
    dim3 grid(n_oi, (batch + block_b - 1) / block_b);
    int smem = (reps + 1) * 2 * 2 * sizeof(float);

    if (use_bf16) {
        DISPATCH_FWD(block_b, bf16_t, cute_pz_fwd_kernel, grid, smem,
            reinterpret_cast<const bf16_t*>(x.data_ptr()),
            reinterpret_cast<const bf16_t*>(theta.data_ptr()),
            reinterpret_cast<const bf16_t*>(pw.data_ptr()),
            reinterpret_cast<const bf16_t*>(pb.data_ptr()),
            reinterpret_cast<bf16_t*>(output.data_ptr()),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure);
    } else {
        DISPATCH_FWD(block_b, float, cute_pz_fwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            output.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure);
    }

    return output;
}

std::vector<torch::Tensor> cute_pz_backward(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    torch::Tensor grad_output,
    bool preacts_trainable, bool fast_measure, int state_bits)
{
    // state_bits: 32 = f32 states, 8 = fp8 prescaled, 4 = fp4 prescaled (experimental)
    int batch   = x.size(0);
    int in_dim  = x.size(1);
    int out_dim = theta.size(0);
    int reps    = theta.size(2) - 1;

    x     = prep(x, torch::kFloat32);
    theta = prep(theta, torch::kFloat32);
    grad_output = prep(grad_output, torch::kFloat32);
    pw = prep(pw, torch::kFloat32);
    pb = prep(pb, torch::kFloat32);

    int n_oi    = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch);
    int n_b_blk = (batch + block_b - 1) / block_b;
    int n_prog  = n_oi * n_b_blk;
    int n_states = 3 * reps + 3;

    // State dtype & prescale selection
    auto dev_opts = torch::TensorOptions().device(x.device());
    torch::Tensor states;
    float prescale;
    if (state_bits == 8) {
        states = torch::empty({n_prog, n_states, 4, block_b}, dev_opts.dtype(torch::kFloat8_e4m3fn));
        prescale = 224.0f;
    } else if (state_bits == 16) {
        states = torch::empty({n_prog, n_states, 4, block_b}, dev_opts.dtype(torch::kBFloat16));
        prescale = 1.0f;
    } else {
        states = torch::empty({n_prog, n_states, 4, block_b}, dev_opts.dtype(torch::kFloat32));
        prescale = 1.0f;
    }

    auto grad_theta = torch::zeros_like(theta, torch::kFloat32);
    auto grad_x     = torch::zeros({batch, in_dim}, dev_opts.dtype(torch::kFloat32));
    auto grad_pw    = preacts_trainable
        ? torch::zeros(pw.sizes(), dev_opts.dtype(torch::kFloat32))
        : torch::zeros({1}, dev_opts.dtype(torch::kFloat32));
    auto grad_pb    = preacts_trainable
        ? torch::zeros(pb.sizes(), dev_opts.dtype(torch::kFloat32))
        : torch::zeros({1}, dev_opts.dtype(torch::kFloat32));

    dim3 grid(n_oi, n_b_blk);
    int n_trig = (reps + 1) * 2;
    int num_warps = (block_b + 31) / 32;
    int smem = (2 * n_trig + num_warps) * sizeof(float);

    if (state_bits == 8) {
        DISPATCH_BWD(block_b, fp8_t, cute_pz_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            reinterpret_cast<fp8_t*>(states.data_ptr()),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, prescale);
    } else if (state_bits == 16) {
        DISPATCH_BWD(block_b, bf16_t, cute_pz_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            reinterpret_cast<bf16_t*>(states.data_ptr()),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, prescale);
    } else {
        DISPATCH_BWD(block_b, float, cute_pz_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            states.data_ptr<float>(),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, prescale);
    }

    return {grad_x, grad_theta,
            preacts_trainable ? grad_pw : torch::Tensor(),
            preacts_trainable ? grad_pb : torch::Tensor()};
}

torch::Tensor cute_rpz_forward(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    bool fast_measure, bool use_bf16)
{
    int batch = x.size(0), in_dim = x.size(1);
    int out_dim = theta.size(0), reps = theta.size(2) - 1;

    auto io_dtype = use_bf16 ? torch::kBFloat16 : torch::kFloat32;
    x = prep(x, io_dtype);
    theta = prep(theta, io_dtype);
    pw = prep(pw, io_dtype);
    pb = prep(pb, io_dtype);

    auto output = torch::empty({batch, out_dim, in_dim},
        torch::TensorOptions().device(x.device()).dtype(io_dtype));
    int n_oi = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch);
    dim3 grid(n_oi, (batch + block_b - 1) / block_b);
    int smem = (reps + 1) * 2 * sizeof(float);

    if (use_bf16) {
        DISPATCH_FWD(block_b, bf16_t, cute_rpz_fwd_kernel, grid, smem,
            reinterpret_cast<const bf16_t*>(x.data_ptr()),
            reinterpret_cast<const bf16_t*>(theta.data_ptr()),
            reinterpret_cast<const bf16_t*>(pw.data_ptr()),
            reinterpret_cast<const bf16_t*>(pb.data_ptr()),
            reinterpret_cast<bf16_t*>(output.data_ptr()),
            batch, in_dim, out_dim, reps, fast_measure);
    } else {
        DISPATCH_FWD(block_b, float, cute_rpz_fwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            output.data_ptr<float>(),
            batch, in_dim, out_dim, reps, fast_measure);
    }

    return output;
}

std::vector<torch::Tensor> cute_rpz_backward(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    torch::Tensor grad_output,
    bool fast_measure, int state_bits)
{
    int batch = x.size(0), in_dim = x.size(1);
    int out_dim = theta.size(0), reps = theta.size(2) - 1;

    x = prep(x, torch::kFloat32);
    theta = prep(theta, torch::kFloat32);
    pw = prep(pw, torch::kFloat32);
    pb = prep(pb, torch::kFloat32);
    grad_output = prep(grad_output, torch::kFloat32);

    int n_oi = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch);
    int n_b_blk = (batch + block_b - 1) / block_b;
    int n_prog = n_oi * n_b_blk;
    int n_states = 2 * reps + 2;

    auto dev_opts = torch::TensorOptions().device(x.device());
    torch::Tensor states;
    float prescale;
    if (state_bits == 8) {
        states = torch::empty({n_prog, n_states, 4, block_b}, dev_opts.dtype(torch::kFloat8_e4m3fn));
        prescale = 224.0f;
    } else if (state_bits == 16) {
        states = torch::empty({n_prog, n_states, 4, block_b}, dev_opts.dtype(torch::kBFloat16));
        prescale = 1.0f;
    } else {
        states = torch::empty({n_prog, n_states, 4, block_b}, dev_opts.dtype(torch::kFloat32));
        prescale = 1.0f;
    }

    auto grad_theta = torch::zeros_like(theta, torch::kFloat32);
    auto grad_x  = torch::zeros({batch, in_dim}, dev_opts.dtype(torch::kFloat32));
    auto grad_pw = torch::zeros(pw.sizes(), dev_opts.dtype(torch::kFloat32));
    auto grad_pb = torch::zeros(pb.sizes(), dev_opts.dtype(torch::kFloat32));

    dim3 grid(n_oi, n_b_blk);
    int num_warps = (block_b + 31) / 32;
    int smem = (2 * (reps + 1) + num_warps) * sizeof(float);

    if (state_bits == 8) {
        DISPATCH_BWD(block_b, fp8_t, cute_rpz_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            reinterpret_cast<fp8_t*>(states.data_ptr()),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps, fast_measure, prescale);
    } else if (state_bits == 16) {
        DISPATCH_BWD(block_b, bf16_t, cute_rpz_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            reinterpret_cast<bf16_t*>(states.data_ptr()),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps, fast_measure, prescale);
    } else {
        DISPATCH_BWD(block_b, float, cute_rpz_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            states.data_ptr<float>(),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps, fast_measure, prescale);
    }

    return {grad_x, grad_theta, grad_pw, grad_pb};
}

torch::Tensor cute_real_forward(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    bool preacts_trainable, bool fast_measure, bool compute_bf16,
    bool use_bf16)
{
    int batch = x.size(0), in_dim = x.size(1);
    int out_dim = theta.size(0), reps = theta.size(2);

    auto io_dtype = use_bf16 ? torch::kBFloat16 : torch::kFloat32;
    x = prep(x, io_dtype);
    theta = prep(theta, io_dtype);
    pw = prep(pw, io_dtype);
    pb = prep(pb, io_dtype);

    auto output = torch::empty({batch, out_dim, in_dim},
        torch::TensorOptions().device(x.device()).dtype(io_dtype));
    int n_oi = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch, compute_bf16 ? 32 : 32);
    dim3 grid(n_oi, (batch + block_b - 1) / block_b);
    int smem = reps * 2 * sizeof(float);

    if (use_bf16) {
        DISPATCH_FWD(block_b, bf16_t, cute_real_fwd_kernel, grid, smem,
            reinterpret_cast<const bf16_t*>(x.data_ptr()),
            reinterpret_cast<const bf16_t*>(theta.data_ptr()),
            reinterpret_cast<const bf16_t*>(pw.data_ptr()),
            reinterpret_cast<const bf16_t*>(pb.data_ptr()),
            reinterpret_cast<bf16_t*>(output.data_ptr()),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, compute_bf16);
    } else {
        DISPATCH_FWD(block_b, float, cute_real_fwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            output.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, compute_bf16);
    }

    return output;
}

std::vector<torch::Tensor> cute_real_backward(
    torch::Tensor x, torch::Tensor theta,
    torch::Tensor pw, torch::Tensor pb,
    torch::Tensor grad_output,
    bool preacts_trainable, bool fast_measure, bool compute_bf16,
    int state_bits)
{
    int batch = x.size(0), in_dim = x.size(1);
    int out_dim = theta.size(0), reps = theta.size(2);

    x = prep(x, torch::kFloat32);
    theta = prep(theta, torch::kFloat32);
    pw = prep(pw, torch::kFloat32);
    pb = prep(pb, torch::kFloat32);
    grad_output = prep(grad_output, torch::kFloat32);

    int n_oi = out_dim * in_dim;
    int block_b = select_block_b(n_oi, batch, compute_bf16 ? 32 : 32);
    int n_b_blk = (batch + block_b - 1) / block_b;
    int n_prog = n_oi * n_b_blk;
    int n_states = 3 * reps + 1;
    int n_components = compute_bf16 ? 2 : 4;

    auto dev_opts = torch::TensorOptions().device(x.device());
    torch::Tensor states;
    float prescale;
    if (state_bits == 8) {
        states = torch::empty({n_prog, n_states, n_components, block_b}, dev_opts.dtype(torch::kFloat8_e4m3fn));
        prescale = 224.0f;
    } else {
        states = torch::empty({n_prog, n_states, n_components, block_b}, dev_opts.dtype(torch::kFloat32));
        prescale = 1.0f;
    }

    auto grad_theta = torch::zeros_like(theta, torch::kFloat32);
    auto grad_x = torch::zeros({batch, in_dim}, dev_opts.dtype(torch::kFloat32));
    auto grad_pw = preacts_trainable
        ? torch::zeros(pw.sizes(), dev_opts.dtype(torch::kFloat32))
        : torch::zeros({1}, dev_opts.dtype(torch::kFloat32));
    auto grad_pb = preacts_trainable
        ? torch::zeros(pb.sizes(), dev_opts.dtype(torch::kFloat32))
        : torch::zeros({1}, dev_opts.dtype(torch::kFloat32));

    dim3 grid(n_oi, n_b_blk);
    int num_warps = (block_b + 31) / 32;
    int smem = (2 * reps + num_warps) * sizeof(float);

    if (state_bits == 8) {
        DISPATCH_BWD(block_b, fp8_t, cute_real_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            reinterpret_cast<fp8_t*>(states.data_ptr()),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, compute_bf16, n_components, prescale);
    } else if (state_bits == 16) {
        DISPATCH_BWD(block_b, bf16_t, cute_real_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            reinterpret_cast<bf16_t*>(states.data_ptr()),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, compute_bf16, n_components, prescale);
    } else {
        DISPATCH_BWD(block_b, float, cute_real_bwd_kernel, grid, smem,
            x.data_ptr<float>(), theta.data_ptr<float>(),
            pw.data_ptr<float>(), pb.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            states.data_ptr<float>(),
            grad_theta.data_ptr<float>(), grad_x.data_ptr<float>(),
            grad_pw.data_ptr<float>(), grad_pb.data_ptr<float>(),
            batch, in_dim, out_dim, reps,
            preacts_trainable, fast_measure, compute_bf16, n_components, prescale);
    }

    return {grad_x, grad_theta,
            preacts_trainable ? grad_pw : torch::Tensor(),
            preacts_trainable ? grad_pb : torch::Tensor()};
}

#undef DISPATCH_BLOCK_B

// ====================================================================
// pybind11 module
// ====================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pz_forward",    &cute_pz_forward,    "CuTe PZ forward");
    m.def("pz_backward",   &cute_pz_backward,   "CuTe PZ backward");
    m.def("rpz_forward",   &cute_rpz_forward,   "CuTe RPZ forward");
    m.def("rpz_backward",  &cute_rpz_backward,  "CuTe RPZ backward");
    m.def("real_forward",  &cute_real_forward,   "CuTe Real forward");
    m.def("real_backward", &cute_real_backward,  "CuTe Real backward");
}
