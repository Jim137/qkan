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

// nn.Linear forward/backward (y = x @ W^T + b) delegated to cuBLAS.
//
// History: the original implementation here was a hand-rolled CuTe DSL kernel
// where each thread computed one output element by looping over K (no shared
// memory, no Tensor Core MMA).  On a 4096x4096 bf16 GEMM it ran ~100x slower
// than ``torch.nn.Linear`` because it did not use the Tensor Cores at all.
//
// Rather than re-implement a tuned bf16/f32 GEMM with MMA atoms — which would
// duplicate cuBLAS and need per-arch tuning — we delegate to ``at::addmm``
// (cuBLAS).  cuBLAS GEMM is fully stream-aware and CUDA-graph capturable, so
// this still satisfies the original "stream-correct forward path that captures
// cleanly under ``torch.cuda.CUDAGraph``" goal of this file.  Backward already
// delegated to ``torch::matmul`` for the same reason.
//
// Layout:
//   x: (M, K)  row-major
//   W: (N, K)  row-major
//   y: (M, N)  row-major
//   b: (N,)    optional

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

    int64_t K = x.size(1);
    int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K,
        "weight last dim must match x last dim (got K=", K, ", weight K=", weight.size(1), ")");

    auto io_dtype = x.scalar_type();
    TORCH_CHECK(io_dtype == torch::kFloat32 || io_dtype == torch::kBFloat16,
        "cute_linear_forward: only f32 / bf16 supported (got ", io_dtype, ")");

    if (!x.is_contiguous()) x = x.contiguous();
    if (!weight.is_contiguous()) weight = weight.contiguous();
    if (weight.scalar_type() != io_dtype) weight = weight.to(io_dtype);

    bool has_bias = bias_opt.has_value() && bias_opt->defined();
    if (has_bias) {
        TORCH_CHECK(bias_opt->dim() == 1 && bias_opt->size(0) == N,
            "bias must be a 1-D tensor of size N");
    }

    // Delegate to cuBLAS.  ``at::addmm(bias, x, W^T)`` matches
    // ``torch.nn.functional.linear`` bit-exactly and runs on the current CUDA
    // stream (so it captures into CUDA graphs just like nn.Linear).
    auto wT = weight.transpose(0, 1);
    if (has_bias) {
        auto bias = bias_opt.value();
        if (bias.scalar_type() != io_dtype) bias = bias.to(io_dtype);
        return at::addmm(bias, x, wT);
    }
    return at::matmul(x, wT);
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
