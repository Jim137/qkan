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

"""Fused feature map of solver="te" against the eager one evaluated in float64."""

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qkan.solver.flash.triton_activation import (  # noqa: E402
    _TORCH_FUNCTIONAL as ACTS,
)
from qkan.solver.flash.triton_activations import SUPPORTED_KINDS  # noqa: E402
from qkan.solver.te_ops_features import (  # noqa: E402
    _features_bwd_kernel,
    _features_fwd_kernel,
    _tiling,
    fourier_features,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP32, BF16, FP16 = torch.float32, torch.bfloat16, torch.float16


def reference(x, reps, act):
    """The eager feature map of qkan.solver.te."""
    k = torch.arange(1, reps + 1, device=x.device, dtype=x.dtype)
    kx = k[:, None] * x[:, None, :]
    return torch.cat([kx.cos(), kx.sin(), act(x)[:, None]], dim=1).flatten(1)


def randn(*shape, dtype=FP32, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=gen) * 3).to(dtype)


def ulps(a, b):
    """Largest distance between two fp16/bf16 tensors, in representable values."""

    def lattice(t):
        bits = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (lattice(a) - lattice(b)).abs().max().item()


def assert_grad_close(grad, ref):
    """ref is float64; grad carries the rounding of its own dtype."""
    scale = ref.abs().max().item()
    if grad.dtype == FP32:
        assert (grad - ref).abs().max().item() <= 1e-5 * scale
    else:  # fp32 arithmetic, then one rounding to 8 (bf16) or 11 (fp16) bits
        torch.testing.assert_close(grad.double(), ref, rtol=2**-7, atol=1e-5 * scale)


def check(x, reps, kind, dtype=FP32, align=1):
    """Values, padding and input gradient of one call."""
    batch, in_dim = x.shape
    k = (2 * reps + 1) * in_dim
    shape = (batch + -batch % align, k + -k % align)
    x = x.detach().requires_grad_()
    x64 = x.detach().double().requires_grad_()
    ref = reference(x64, reps, ACTS[kind])

    # torch.empty hands the block just freed back, so untouched padding
    # would show up as NaN instead of passing as zero by luck.
    poison = torch.full(shape, float("nan"), device="cuda", dtype=dtype)
    del poison
    out = fourier_features(x, reps, kind, dtype, align)
    assert out.dtype == dtype and out.shape == shape
    assert not out[batch:].any() and not out[:, k:].any()

    out32 = out if dtype == FP32 else fourier_features(x, reps, kind, FP32, align)
    assert (out32[:batch, :k] - ref).abs().max().item() <= 1e-5
    if dtype != FP32:
        assert ulps(out, out32.to(dtype)) <= 1

    # Non-zero on the padding too: it must not leak into the gradient.
    upstream = randn(*shape, dtype=dtype, seed=1)
    out.backward(upstream)
    ref.backward(upstream[:batch, :k].double())
    assert x.grad.dtype == x.dtype and x.grad.shape == x.shape
    assert_grad_close(x.grad, x64.grad)


@pytest.mark.parametrize("kind", SUPPORTED_KINDS)
def test_activation_kinds(kind):
    check(randn(37, 13), 2, kind)


@pytest.mark.parametrize("reps", [1, 3, 5, 8])
def test_harmonics(reps):
    check(randn(37, 13), reps, "silu")


@pytest.mark.parametrize("x_dtype", [FP32, BF16, FP16])
@pytest.mark.parametrize("dtype", [FP32, BF16, FP16])
def test_dtypes(x_dtype, dtype):
    check(randn(37, 13, dtype=x_dtype), 3, "gelu_tanh", dtype)


@pytest.mark.parametrize("dtype", [FP32, BF16])
def test_alignment_padding(dtype):
    # 37 rows and 5 * 13 = 65 columns: neither is a multiple of 32.
    check(randn(37, 13), 2, "silu", dtype, 32)
    check(randn(64, 32, dtype=BF16), 1, "silu", dtype, 32)  # nothing to pad


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 300), (300, 1), (5, 1030), (1100, 3), (67, 129)]
)
def test_edge_shapes(shape):
    check(randn(*shape), 2, "tanh", align=32)


@pytest.mark.parametrize("kind", SUPPORTED_KINDS)
def test_zeros_tails_and_large_arguments(kind):
    # What randn * 3 never draws: relu's kink, the activation tails, and
    # arguments many periods out, where reducing x in float32 costs the 1e-5.
    special = [0.0, -0.0, 1e-4, -1e-4, 15.0, -15.0, 40.0, -40.0, 1e3, -1e3, 12345.678]
    x = randn(37, 13)
    x[0, : len(special)] = torch.tensor(special, device="cuda")
    check(x, 3, kind)


def test_noncontiguous_input():
    base = randn(26, 37)
    for view in (base[:, ::2], base.t(), base[::3]):
        assert not view.is_contiguous()
        check(view, 2, "silu")


def test_numpy_reps(monkeypatch, tmp_path):
    # reps out of np.arange in a sweep.  A numpy integer is no Triton constexpr
    # but hashes like the int, so a cached int variant would hide the failure:
    # compile from scratch (no other test uses reps=4).
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    check(randn(37, 13), np.int64(4), "silu")


@pytest.mark.parametrize("shape", [(37, 13), (5, 1030), (1100, 3)])
def test_kernels_stay_inside_their_buffers(shape):
    # grad_x has no padding that would show a stray store, so both kernels
    # write into buffers that sit between sentinels.  Not NaN: a stray lane
    # computes from whatever it loaded, which may be NaN itself.
    guard, sentinel = 4096, -7.0
    batch, in_dim = shape
    k = 5 * in_dim

    def guarded(cols):
        store = torch.full((batch * cols + 2 * guard,), sentinel, device="cuda")
        return store, store[guard:-guard].view(batch, cols)

    x = randn(*shape)
    out_store, out = guarded(k)
    gx_store, gx = guarded(in_dim)
    grid, block_b, block_i = _tiling(batch, in_dim)
    consts = dict(REPS=2, KIND="silu", BLOCK_B=block_b, BLOCK_I=block_i)
    _features_fwd_kernel[grid](x, out, batch, in_dim, k, **consts)
    _features_bwd_kernel[grid](x, out, gx, batch, in_dim, k, **consts)

    for store, body in ((out_store, out), (gx_store, gx)):
        assert (store[:guard] == sentinel).all() and (store[-guard:] == sentinel).all()
        assert (body != sentinel).all()


def test_no_output_sized_temporaries():
    # The output is written once, in its dtype, and only x is kept for
    # backward.  A float32 buffer cast afterwards would peak at 3x.
    x = randn(2048, 1024, dtype=BF16).requires_grad_()
    upstream = randn(2048, 3 * 1024, dtype=BF16, seed=1)

    def peak_bytes(fn):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        result = fn()
        return result, torch.cuda.max_memory_allocated() - before

    out, forward_peak = peak_bytes(lambda: fourier_features(x, 1, "silu", BF16))
    (saved,) = out.grad_fn.saved_tensors
    assert saved is x
    (grad,), backward_peak = peak_bytes(lambda: torch.autograd.grad(out, x, upstream))
    assert forward_peak < 1.5 * out.numel() * out.element_size()
    assert backward_peak < 1.5 * grad.numel() * grad.element_size()


@pytest.mark.parametrize("dtype", [FP32, BF16])
@pytest.mark.parametrize("kind", SUPPORTED_KINDS)
def test_second_derivatives(kind, dtype):
    # A PINN loss is built on du/dx.  Under create_graph=True that gradient has
    # to stay differentiable in x and in the upstream gradient, which is where
    # the GEMM weights enter; as a constant it would drop the term silently.
    reps, batch, in_dim = 2, 37, 13
    k = (2 * reps + 1) * in_dim
    x = randn(batch, in_dim).requires_grad_()
    upstream = randn(64, 96, dtype=dtype, seed=1).requires_grad_()  # padded shape
    x64 = x.detach().double().requires_grad_()
    upstream64 = upstream.detach().double().requires_grad_()

    def gradients(features, x, upstream, weights):
        (dx,) = torch.autograd.grad(features, x, weights, create_graph=True)
        return dx, *torch.autograd.grad(dx.square().sum(), (x, upstream))

    out = fourier_features(x, reps, kind, dtype, 32)
    ref = reference(x64, reps, ACTS[kind])
    got = gradients(out, x, upstream, upstream)
    want = gradients(ref, x64, upstream64, upstream64[:batch, :k])
    for grad, grad64 in zip(got, want):
        assert_grad_close(grad, grad64)


def test_expanded_upstream_gradient():
    # sum() sends a stride-0 gradient into the backward launcher.
    x = randn(37, 13).requires_grad_()
    x64 = x.detach().double().requires_grad_()
    fourier_features(x, 2, "silu", FP32, 32).sum().backward()
    reference(x64, 2, F.silu).sum().backward()
    assert_grad_close(x.grad, x64.grad)


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_empty(shape):
    x = torch.empty(shape, device="cuda", requires_grad=True)
    out = fourier_features(x, 2, "silu", BF16, 32)
    # 5 rows pad to 32; 5 * 5 = 25 columns pad to 32.
    assert out.shape == (32 * (shape[0] > 0), 32 * (shape[1] > 0))
    assert not out.any()
    out.sum().backward()
    assert x.grad.shape == shape


def test_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported activation kind"):
        fourier_features(randn(2, 2), 1, "swish", FP32)


@pytest.mark.skipif(
    os.environ.get("QKAN_LARGE_TESTS") != "1",
    reason="allocates ~10 GB; set QKAN_LARGE_TESTS=1",
)
def test_output_past_int32(run_snippet):
    # 704 columns a row, so the output passes 2^31 elements at row 3050402.
    # Rows are compared with the float64 reference, never with a round trip
    # through the same offsets (see test_large_offsets.py); with the row
    # offset left in int32 this fails.
    result = run_snippet(
        """
        import torch
        import torch.nn.functional as F
        from qkan.solver.te_ops_features import fourier_features

        torch.manual_seed(0)
        batch, in_dim, reps = 3_060_000, 64, 5
        k = (2 * reps + 1) * in_dim
        assert batch * k > 2**31
        edge = 2**31 // k
        bf16 = torch.bfloat16

        x = torch.randn(batch, in_dim, device="cuda", dtype=bf16).requires_grad_()
        out = fourier_features(x, reps, "silu", bf16)
        upstream = torch.randn(batch, k, device="cuda", dtype=bf16)
        out.backward(upstream)

        def reference(x):
            f = torch.arange(1, reps + 1, device="cuda", dtype=x.dtype)
            fx = f[:, None] * x[:, None, :]
            return torch.cat([fx.cos(), fx.sin(), F.silu(x)[:, None]], dim=1).flatten(1)

        # First rows, both sides of the 2^31 boundary, last rows.
        for rows in (slice(0, 64), slice(edge - 64, edge + 64), slice(batch - 64, batch)):
            x64 = x[rows].detach().double().requires_grad_()
            ref = reference(x64)
            ref.backward(upstream[rows].double())
            torch.testing.assert_close(
                out[rows].double(), ref.detach(), rtol=2**-7, atol=1e-5
            )
            torch.testing.assert_close(
                x.grad[rows].double(), x64.grad, rtol=2**-7, atol=1e-3
            )
        print("OK")
        """,
        preamble="",
        timeout=1800,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
