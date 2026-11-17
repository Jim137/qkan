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

"""solver="flash" refuses what it has no Triton kernel for.

An ansatz outside pz / rpz / real used to run torch_exact_solver, and a
double-precision c_dtype the float32 kernels behind a float64 tensor.
TritonActivation used to run the torch built-in on CPU tensors and without
Triton, and its kernels computed a float64 input in float32. The checks are
host-side Python; the tests that launch kernels need Triton and CUDA.
"""

import re

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qkan import QKANLayer  # noqa: E402
from qkan.solver import (  # noqa: E402
    _FLASH_AVAILABLE,
    flash_exact_solver,
    get_solver,
)
from qkan.solver.flash import triton_activation  # noqa: E402
from qkan.solver.flash.triton_activation import TritonActivation  # noqa: E402

needs_triton = pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton")
needs_gpu = pytest.mark.skipif(
    not (_FLASH_AVAILABLE and torch.cuda.is_available()),
    reason="needs triton and CUDA",
)

ANSATZES = ["pz_encoding", "pz", "rpz_encoding", "rpz", "real"]
OTHER_ANSATZES = [
    "px_encoding",
    "px",
    "mix",
    pytest.param(lambda *args, **kwargs: None, id="callable"),
]
C_DTYPES = [torch.complex64, torch.float32, torch.bfloat16, torch.float8_e4m3fn]
DOUBLE_C_DTYPES = [torch.complex128, torch.float64]

BATCH, OUT_DIM, IN_DIM, REPS = 8, 2, 3, 2

NO_KERNEL = "solver='flash' has no kernel for ansatz"


def no_support_for(c_dtype):
    return re.escape(
        f"solver='flash' computes in float32 and does not support c_dtype={c_dtype}:"
    )


def solve(ansatz, fn=flash_exact_solver, device="cuda", grad=False, **kwargs):
    """Call a solver function on inputs shaped as the layer hands them over.

    Returns the output, and with ``grad`` also the gradient of theta.
    """
    torch.manual_seed(0)
    last = 2 if ansatz in ("pz_encoding", "pz", "mix") else 1
    axis = REPS if ansatz == "real" else REPS + 1
    x = torch.rand(BATCH, IN_DIM, device=device) * 2 - 1
    theta = torch.rand(OUT_DIM, IN_DIM, axis, last, device=device)
    pw = torch.ones(OUT_DIM, IN_DIM, REPS, device=device)
    pb = torch.zeros(OUT_DIM, IN_DIM, REPS, device=device)
    theta.requires_grad_(grad)
    out = fn(x, theta, pw, pb, REPS, ansatz=ansatz, out_dim=OUT_DIM, **kwargs)
    if not grad:
        return out
    out.float().sum().backward()
    return out.detach(), theta.grad


# ---------------------------------------------------------------------------
# QKANSolver.check(): what a layer asks when it is built
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ansatz", OTHER_ANSATZES)
def test_check_refuses_other_ansatzes(ansatz):
    with pytest.raises(ValueError, match=NO_KERNEL):
        get_solver("flash").check(ansatz, torch.complex64, REPS)


@pytest.mark.parametrize("c_dtype", DOUBLE_C_DTYPES + [torch.int32])
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_check_refuses_other_c_dtypes(ansatz, c_dtype):
    with pytest.raises(ValueError, match=no_support_for(c_dtype)):
        get_solver("flash").check(ansatz, c_dtype, REPS)


def test_check_accepts_what_the_kernels_cover():
    for ansatz in ANSATZES:
        for c_dtype in C_DTYPES:
            get_solver("flash").check(ansatz, c_dtype, REPS)
    # float32 compute behind float16 buffers: pz and rpz only.
    for ansatz in ("pz_encoding", "pz", "rpz_encoding", "rpz"):
        get_solver("flash").check(ansatz, torch.float16, REPS)


def test_check_refuses_float16_for_the_real_ansatz():
    with pytest.raises(ValueError, match=no_support_for(torch.float16)):
        get_solver("flash").check("real", torch.float16, REPS)


# ---------------------------------------------------------------------------
# flash_exact_solver: direct callers get the same errors
# ---------------------------------------------------------------------------


@needs_triton
@pytest.mark.parametrize("ansatz", ["px_encoding", "px", "mix"])
def test_solver_function_refuses_other_ansatzes(ansatz):
    # CPU tensors: the fallback never touched Triton, so it ran there too.
    with pytest.raises(ValueError, match=NO_KERNEL):
        solve(ansatz, device="cpu", preacts_trainable=ansatz.startswith("px"))


@needs_gpu
@pytest.mark.parametrize("c_dtype", DOUBLE_C_DTYPES)
@pytest.mark.parametrize("ansatz", ["pz", "rpz", "real"])
def test_solver_function_refuses_double_precision(ansatz, c_dtype):
    with pytest.raises(ValueError, match=no_support_for(c_dtype)):
        solve(ansatz, dtype=c_dtype)


@needs_gpu
def test_solver_function_refuses_float16_for_the_real_ansatz():
    with pytest.raises(ValueError, match=no_support_for(torch.float16)):
        solve("real", dtype=torch.float16)


@needs_gpu
@pytest.mark.parametrize("ansatz", ["pz", "rpz", "real"])
def test_complex64_runs_the_float32_kernels(ansatz):
    out32, grad32 = solve(ansatz, grad=True, dtype=torch.float32)
    # complex64 is the function's own default and the layer's.
    for kwargs in ({}, {"dtype": torch.complex64}):
        out, grad = solve(ansatz, grad=True, **kwargs)
        assert out.dtype == torch.float32
        assert torch.equal(out, out32)
        assert torch.equal(grad, grad32)
        assert torch.equal(solve(ansatz, **kwargs), out32)


@needs_gpu
@pytest.mark.parametrize(
    "ansatz, c_dtype, out_dtype",
    [
        (ansatz, c_dtype, out_dtype)
        for ansatz in ("pz", "rpz", "real")
        for c_dtype, out_dtype in (
            (torch.bfloat16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.bfloat16),
            (torch.float16, torch.float16),
        )
        if not (ansatz == "real" and c_dtype == torch.float16)
    ],
)
def test_reduced_precision_buffers_still_run(ansatz, c_dtype, out_dtype):
    out32, grad32 = solve(ansatz, grad=True, dtype=torch.float32)
    out, grad = solve(ansatz, grad=True, dtype=c_dtype)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out.float(), out32, atol=3e-2, rtol=0)
    cosine = F.cosine_similarity(grad.float().flatten(), grad32.flatten(), dim=0)
    assert cosine > 0.99, cosine.item()


# ---------------------------------------------------------------------------
# CPU tensors
# ---------------------------------------------------------------------------


@needs_triton
def test_registered_solver_refuses_cpu_tensors():
    with pytest.raises(RuntimeError, match="solver='flash'.*CUDA"):
        solve("pz", fn=get_solver("flash"), device="cpu", dtype=torch.float32)


@needs_triton
@pytest.mark.parametrize(
    "base_activation", ["silu", torch.nn.SiLU()], ids=["triton-silu", "nn-silu"]
)
def test_cpu_layer_raises_an_error_that_names_cuda(base_activation):
    layer = QKANLayer(
        IN_DIM, OUT_DIM, reps=REPS, solver="flash", base_activation=base_activation
    )
    with pytest.raises(RuntimeError, match="solver='flash'.*CUDA"):
        layer(torch.rand(BATCH, IN_DIM))


# ---------------------------------------------------------------------------
# TritonActivation and its kernels
# ---------------------------------------------------------------------------


def test_triton_activation_needs_triton(monkeypatch):
    monkeypatch.setattr(triton_activation, "_TRITON_ACT_AVAILABLE", False)
    with pytest.raises(ImportError, match="Install triton"):
        TritonActivation("silu")


@needs_triton
def test_triton_activation_refuses_cpu_tensors():
    with pytest.raises(RuntimeError, match="CUDA"):
        TritonActivation("silu")(torch.randn(16, 4))


@needs_gpu
@pytest.mark.parametrize("kind", ["silu", "gelu_exact"])
def test_activation_kernels_refuse_float64(kind):
    from qkan.solver.flash.triton_activations import (
        triton_activation_backward,
        triton_activation_forward,
    )

    x = torch.randn(4096, device="cuda", dtype=torch.float64)
    with pytest.raises(ValueError, match="float64"):
        triton_activation_forward(x, kind)
    with pytest.raises(ValueError, match="float64"):
        triton_activation_backward(torch.ones_like(x), x, kind)
    # The module refuses whether or not the input reaches the kernels.
    for requires_grad in (False, True):
        with pytest.raises(ValueError, match="float64"):
            TritonActivation(kind)(x.clone().requires_grad_(requires_grad))


@needs_gpu
@pytest.mark.parametrize(
    "dtype, tol",
    [(torch.float32, 1e-5), (torch.bfloat16, 2e-2), (torch.float16, 2e-3)],
    ids=["float32", "bfloat16", "float16"],
)
def test_activation_kernels_keep_their_other_dtypes(dtype, tol):
    from qkan.solver.flash.triton_activations import (
        triton_activation_backward,
        triton_activation_forward,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4096, device="cuda", generator=g).to(dtype)
    grad_y = torch.randn(4096, device="cuda", generator=g).to(dtype)
    ref_x = x.float().requires_grad_()
    ref = F.silu(ref_x)
    ref.backward(grad_y.float())

    y = triton_activation_forward(x, "silu")
    grad_x = triton_activation_backward(grad_y, x, "silu")
    assert y.dtype == dtype and grad_x.dtype == dtype
    torch.testing.assert_close(y.float(), ref.detach(), atol=tol, rtol=tol)
    torch.testing.assert_close(grad_x.float(), ref_x.grad, atol=tol, rtol=tol)
