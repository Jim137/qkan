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

// CuTe DSL CUDA kernel for nn.Linear (y = x @ W^T + b).
//
// Drop-in replacement for torch.nn.Linear forward.  Backward delegates to
// cuBLAS via torch::matmul to keep this file small — the goal is a stream-
// correct cuTe forward path that captures cleanly under torch.cuda.CUDAGraph.
//
// Layout:
//   x: (M, K)  row-major
//   W: (N, K)  row-major
//   y: (M, N)  row-major
//   b: (N,)    optional
//
// Compute: f32 accumulation, IOT load/store (f32 or bf16).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>

using namespace cute;

using bf16_t = cutlass::bfloat16_t;

// Row-major layout helpers (CuTe defaults to column-major).
#ifndef ROWMAJOR2
#define ROWMAJOR2(d0, d1) \
    make_layout(make_shape((d0), (d1)), make_stride((d1), 1))
#endif

// ====================================================================
// Forward kernel: y = x @ W^T + b
// ====================================================================
// Tile shape: BM x BN, with each thread computing one output element by
// looping over K.  This is a naive blocked GEMM — no smem staging, no
// double-buffering — but it is correct, stream-aware, and serves as a
// drop-in cuTe path for CUDA graphs.
template <typename IOT, int BM, int BN>
__global__ void cute_linear_fwd_kernel(
    const IOT* __restrict__ x_ptr,
    const IOT* __restrict__ w_ptr,
    const IOT* __restrict__ b_ptr,  // may be nullptr
    IOT* __restrict__ y_ptr,
    int M, int N, int K)
{
    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;
    int tx = threadIdx.x;  // [0, BN)
    int ty = threadIdx.y;  // [0, BM)

    int m = block_m + ty;
    int n = block_n + tx;
    if (m >= M || n >= N) return;

    auto gX = make_tensor(make_gmem_ptr(x_ptr), ROWMAJOR2(M, K));
    auto gW = make_tensor(make_gmem_ptr(w_ptr), ROWMAJOR2(N, K));
    auto gY = make_tensor(make_gmem_ptr(y_ptr), ROWMAJOR2(M, N));

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < K; k++) {
        acc += float(gX(m, k)) * float(gW(n, k));
    }
    if (b_ptr != nullptr) acc += float(b_ptr[n]);
    gY(m, n) = IOT(acc);
}

// ====================================================================
// Host launcher
// ====================================================================

static inline torch::Tensor prep_linear(torch::Tensor t, torch::ScalarType dtype) {
    if (!t.is_contiguous()) t = t.contiguous();
    if (t.scalar_type() != dtype) t = t.to(dtype);
    return t;
}

// Forward: y = x @ W^T + b
//   x: (M, K), W: (N, K), b: optional (N,), returns y: (M, N)
torch::Tensor cute_linear_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt)
{
    TORCH_CHECK(x.is_cuda(), "cute_linear_forward: x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "cute_linear_forward: weight must be CUDA");
    TORCH_CHECK(x.dim() == 2, "cute_linear_forward expects x of shape (M, K)");
    TORCH_CHECK(weight.dim() == 2, "cute_linear_forward expects weight of shape (N, K)");

    int M = x.size(0);
    int K = x.size(1);
    int N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K,
        "weight last dim must match x last dim (got K=", K, ", weight K=", weight.size(1), ")");

    auto io_dtype = x.scalar_type();
    TORCH_CHECK(io_dtype == torch::kFloat32 || io_dtype == torch::kBFloat16,
        "cute_linear_forward: only f32 / bf16 supported (got ", io_dtype, ")");

    x = prep_linear(x, io_dtype);
    weight = prep_linear(weight, io_dtype);
    bool has_bias = bias_opt.has_value() && bias_opt->defined();
    torch::Tensor bias;
    if (has_bias) {
        bias = prep_linear(bias_opt.value(), io_dtype);
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N,
            "bias must be a 1-D tensor of size N");
    }

    auto y = torch::empty({M, N},
        torch::TensorOptions().device(x.device()).dtype(io_dtype));

    constexpr int BM = 16;
    constexpr int BN = 16;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN, BM);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (io_dtype == torch::kBFloat16) {
        cute_linear_fwd_kernel<bf16_t, BM, BN><<<grid, block, 0, stream>>>(
            reinterpret_cast<const bf16_t*>(x.data_ptr()),
            reinterpret_cast<const bf16_t*>(weight.data_ptr()),
            has_bias ? reinterpret_cast<const bf16_t*>(bias.data_ptr()) : nullptr,
            reinterpret_cast<bf16_t*>(y.data_ptr()),
            M, N, K);
    } else {
        cute_linear_fwd_kernel<float, BM, BN><<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            y.data_ptr<float>(),
            M, N, K);
    }
    return y;
}

// Backward: delegate to torch::matmul (cuBLAS) for simplicity.
//   grad_x = grad_y @ W
//   grad_W = grad_y^T @ x
//   grad_b = grad_y.sum(0)  (only if has_bias)
//
// Using torch APIs keeps the math correct and identical to nn.Linear's
// backward.  These calls also launch on the current CUDA stream.
std::vector<torch::Tensor> cute_linear_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor weight,
    bool has_bias)
{
    TORCH_CHECK(grad_y.is_cuda() && x.is_cuda() && weight.is_cuda(),
        "cute_linear_backward: all tensors must be CUDA");

    grad_y = grad_y.contiguous();
    x = x.contiguous();
    weight = weight.contiguous();

    auto grad_x = torch::matmul(grad_y, weight);          // (M, K)
    auto grad_w = torch::matmul(grad_y.transpose(0, 1), x); // (N, K)
    torch::Tensor grad_b;
    if (has_bias) {
        grad_b = grad_y.sum(0);
    }
    return {grad_x, grad_w, grad_b};
}
