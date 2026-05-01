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

// CuTe DSL CUDA pointwise activation kernels for the QKAN base path.
//
// QKANLayer applies `base_activation(x) * base_weight` alongside the quantum
// solver. When the solver runs on cuTe kernels, we want the base activation
// to stay on the same accelerator path — these kernels provide efficient
// f32/bf16 forward + backward for the standard activation set used by callers
// (silu / gelu_exact / gelu_tanh / relu / tanh / sigmoid).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>

using namespace cute;

using bf16_t = cutlass::bfloat16_t;

// ====================================================================
// Activation kind enum (host-facing dispatch tag)
// ====================================================================

enum class ActKind : int {
    Silu      = 0,
    GeluExact = 1,
    GeluTanh  = 2,
    Relu      = 3,
    Tanh      = 4,
    Sigmoid   = 5,
};

// ====================================================================
// Constants
// ====================================================================

namespace {
constexpr int BLOCK_B = 256;
// gelu_tanh constants: y = 0.5 x (1 + tanh(SQRT_2_OVER_PI (x + GELU_C * x^3)))
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float GELU_C         = 0.044715f;
// gelu_exact derivative uses the standard normal pdf φ(x) = (1/√(2π)) exp(-x²/2)
constexpr float INV_SQRT_2     = 0.7071067811865476f;
constexpr float INV_SQRT_2PI   = 0.3989422804014327f;
}  // namespace

// ====================================================================
// Pointwise math (compute always in f32)
// ====================================================================

__device__ __forceinline__ float silu_fwd(float x) {
    // s = sigmoid(x) = 1 / (1 + exp(-x));  y = x * s
    float s = 1.0f / (1.0f + __expf(-x));
    return x * s;
}

__device__ __forceinline__ float silu_bwd(float x, float gy) {
    // dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float s = 1.0f / (1.0f + __expf(-x));
    return gy * (s * (1.0f + x * (1.0f - s)));
}

__device__ __forceinline__ float gelu_exact_fwd(float x) {
    // y = 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * INV_SQRT_2));
}

__device__ __forceinline__ float gelu_exact_bwd(float x, float gy) {
    // dy/dx = 0.5 * (1 + erf(x/√2)) + x * φ(x), where φ(x) = (1/√2π) exp(-x²/2)
    float erf_term = erff(x * INV_SQRT_2);
    float pdf      = INV_SQRT_2PI * __expf(-0.5f * x * x);
    return gy * (0.5f * (1.0f + erf_term) + x * pdf);
}

__device__ __forceinline__ float gelu_tanh_fwd(float x) {
    // y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    float x3 = x * x * x;
    float u  = SQRT_2_OVER_PI * (x + GELU_C * x3);
    return 0.5f * x * (1.0f + tanhf(u));
}

__device__ __forceinline__ float gelu_tanh_bwd(float x, float gy) {
    // Let u = √(2/π) (x + 0.044715 x³), t = tanh(u).
    // dy/dx = 0.5*(1 + t) + 0.5*x*(1 - t²) * du/dx
    //       = 0.5*(1 + t) + 0.5*x*(1 - t²) * √(2/π) * (1 + 3*0.044715*x²)
    float x2 = x * x;
    float x3 = x2 * x;
    float u  = SQRT_2_OVER_PI * (x + GELU_C * x3);
    float t  = tanhf(u);
    float dudx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_C * x2);
    float dy = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dudx;
    return gy * dy;
}

__device__ __forceinline__ float relu_fwd(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ __forceinline__ float relu_bwd(float x, float gy) {
    return x > 0.0f ? gy : 0.0f;
}

__device__ __forceinline__ float tanh_fwd(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float tanh_bwd(float x, float gy) {
    // dy/dx = 1 - tanh²(x)  — recompute tanh, cheap on f32 fast-math
    float t = tanhf(x);
    return gy * (1.0f - t * t);
}

__device__ __forceinline__ float sigmoid_fwd(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float sigmoid_bwd(float x, float gy) {
    float s = 1.0f / (1.0f + __expf(-x));
    return gy * (s * (1.0f - s));
}

// ====================================================================
// Templated 1D forward / backward kernels
// ====================================================================
//
// One kernel per (kind × direction). IOT is the I/O dtype (float or bf16_t);
// compute is always f32. CuTe DSL is used here purely for tensor view helpers
// (make_tensor + make_layout for the 1D contiguous range) — the math itself
// is bog-standard pointwise.

template <typename IOT, ActKind KIND>
__global__ void cute_act_fwd_kernel(
    const IOT* __restrict__ x_ptr,
    IOT* __restrict__ y_ptr,
    int n)
{
    int idx = blockIdx.x * BLOCK_B + threadIdx.x;
    if (idx >= n) return;

    // CuTe 1D contiguous views; explicit unit stride for clarity / parity
    // with cute_kernels.cu's row-major layout convention.
    auto gX = make_tensor(make_gmem_ptr(x_ptr),
                          make_layout(make_shape(n), make_stride(_1{})));
    auto gY = make_tensor(make_gmem_ptr(y_ptr),
                          make_layout(make_shape(n), make_stride(_1{})));

    float xv = float(gX(idx));
    float yv;
    if constexpr (KIND == ActKind::Silu)        yv = silu_fwd(xv);
    else if constexpr (KIND == ActKind::GeluExact) yv = gelu_exact_fwd(xv);
    else if constexpr (KIND == ActKind::GeluTanh)  yv = gelu_tanh_fwd(xv);
    else if constexpr (KIND == ActKind::Relu)      yv = relu_fwd(xv);
    else if constexpr (KIND == ActKind::Tanh)      yv = tanh_fwd(xv);
    else /* Sigmoid */                              yv = sigmoid_fwd(xv);
    gY(idx) = IOT(yv);
}

template <typename IOT, ActKind KIND>
__global__ void cute_act_bwd_kernel(
    const IOT* __restrict__ x_ptr,
    const IOT* __restrict__ grad_y_ptr,
    IOT* __restrict__ grad_x_ptr,
    int n)
{
    int idx = blockIdx.x * BLOCK_B + threadIdx.x;
    if (idx >= n) return;

    auto gX  = make_tensor(make_gmem_ptr(x_ptr),
                           make_layout(make_shape(n), make_stride(_1{})));
    auto gGy = make_tensor(make_gmem_ptr(grad_y_ptr),
                           make_layout(make_shape(n), make_stride(_1{})));
    auto gGx = make_tensor(make_gmem_ptr(grad_x_ptr),
                           make_layout(make_shape(n), make_stride(_1{})));

    float xv  = float(gX(idx));
    float gyv = float(gGy(idx));
    float gxv;
    if constexpr (KIND == ActKind::Silu)        gxv = silu_bwd(xv, gyv);
    else if constexpr (KIND == ActKind::GeluExact) gxv = gelu_exact_bwd(xv, gyv);
    else if constexpr (KIND == ActKind::GeluTanh)  gxv = gelu_tanh_bwd(xv, gyv);
    else if constexpr (KIND == ActKind::Relu)      gxv = relu_bwd(xv, gyv);
    else if constexpr (KIND == ActKind::Tanh)      gxv = tanh_bwd(xv, gyv);
    else /* Sigmoid */                              gxv = sigmoid_bwd(xv, gyv);
    gGx(idx) = IOT(gxv);
}

// ====================================================================
// Dispatch helpers
// ====================================================================

// Switch on runtime kind → call a templated kernel for each compile-time KIND.
// `if constexpr` cannot operate on a runtime value, so we expand a switch.
#define DISPATCH_KIND(kind, IOT, KERNEL, grid, n, ...) \
    do { \
        auto stream = at::cuda::getCurrentCUDAStream(); \
        switch (kind) { \
            case ActKind::Silu: \
                KERNEL<IOT, ActKind::Silu>     <<<grid, BLOCK_B, 0, stream>>>(__VA_ARGS__); break; \
            case ActKind::GeluExact: \
                KERNEL<IOT, ActKind::GeluExact><<<grid, BLOCK_B, 0, stream>>>(__VA_ARGS__); break; \
            case ActKind::GeluTanh: \
                KERNEL<IOT, ActKind::GeluTanh> <<<grid, BLOCK_B, 0, stream>>>(__VA_ARGS__); break; \
            case ActKind::Relu: \
                KERNEL<IOT, ActKind::Relu>     <<<grid, BLOCK_B, 0, stream>>>(__VA_ARGS__); break; \
            case ActKind::Tanh: \
                KERNEL<IOT, ActKind::Tanh>     <<<grid, BLOCK_B, 0, stream>>>(__VA_ARGS__); break; \
            case ActKind::Sigmoid: \
                KERNEL<IOT, ActKind::Sigmoid>  <<<grid, BLOCK_B, 0, stream>>>(__VA_ARGS__); break; \
        } \
    } while (0)

static inline ActKind to_act_kind(int kind_int) {
    switch (kind_int) {
        case 0: return ActKind::Silu;
        case 1: return ActKind::GeluExact;
        case 2: return ActKind::GeluTanh;
        case 3: return ActKind::Relu;
        case 4: return ActKind::Tanh;
        case 5: return ActKind::Sigmoid;
        default: TORCH_CHECK(false, "Unknown CuTe activation kind: ", kind_int);
    }
    return ActKind::Silu;  // unreachable
}

// ====================================================================
// Python-facing launchers (bound from cute_kernels.cu's PYBIND11_MODULE)
// ====================================================================

torch::Tensor cute_activation_forward(torch::Tensor x, int kind_int) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    auto x_c = x.is_contiguous() ? x : x.contiguous();
    int n = static_cast<int>(x_c.numel());
    auto y = torch::empty_like(x_c);
    if (n == 0) return y;

    ActKind kind = to_act_kind(kind_int);
    dim3 grid((n + BLOCK_B - 1) / BLOCK_B);

    auto dt = x_c.scalar_type();
    if (dt == torch::kBFloat16) {
        DISPATCH_KIND(kind, bf16_t, cute_act_fwd_kernel, grid, n,
            reinterpret_cast<const bf16_t*>(x_c.data_ptr()),
            reinterpret_cast<bf16_t*>(y.data_ptr()),
            n);
    } else if (dt == torch::kFloat32) {
        DISPATCH_KIND(kind, float, cute_act_fwd_kernel, grid, n,
            x_c.data_ptr<float>(), y.data_ptr<float>(), n);
    } else {
        TORCH_CHECK(false, "CuTe activation forward: unsupported dtype ", dt);
    }
    return y;
}

torch::Tensor cute_activation_backward(torch::Tensor grad_y, torch::Tensor x, int kind_int) {
    TORCH_CHECK(x.is_cuda() && grad_y.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == grad_y.scalar_type(),
                "x and grad_y must have the same dtype");
    auto x_c  = x.is_contiguous() ? x : x.contiguous();
    auto gy_c = grad_y.is_contiguous() ? grad_y : grad_y.contiguous();
    int n = static_cast<int>(x_c.numel());
    auto gx = torch::empty_like(x_c);
    if (n == 0) return gx;

    ActKind kind = to_act_kind(kind_int);
    dim3 grid((n + BLOCK_B - 1) / BLOCK_B);

    auto dt = x_c.scalar_type();
    if (dt == torch::kBFloat16) {
        DISPATCH_KIND(kind, bf16_t, cute_act_bwd_kernel, grid, n,
            reinterpret_cast<const bf16_t*>(x_c.data_ptr()),
            reinterpret_cast<const bf16_t*>(gy_c.data_ptr()),
            reinterpret_cast<bf16_t*>(gx.data_ptr()),
            n);
    } else if (dt == torch::kFloat32) {
        DISPATCH_KIND(kind, float, cute_act_bwd_kernel, grid, n,
            x_c.data_ptr<float>(), gy_c.data_ptr<float>(),
            gx.data_ptr<float>(), n);
    } else {
        TORCH_CHECK(false, "CuTe activation backward: unsupported dtype ", dt);
    }
    return gx;
}

#undef DISPATCH_KIND
