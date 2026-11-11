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

"""Fused Fourier-coefficient kernels against the float64 exact-solver reference."""

import math
import os

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from qkan.solver import te  # noqa: E402
from qkan.solver.te_ops_coefficients import (  # noqa: E402
    MAX_REPS,
    fourier_coefficients,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ANSATZES = ["pz_encoding", "rpz_encoding", "real"]


def params(ansatz, reps, out_dim=4, in_dim=5, dtype=torch.float32):
    """Random theta (O, I, T, P) and preacts_bias (O, I, R) on CUDA."""
    g = torch.Generator().manual_seed(0)
    t = reps if ansatz == "real" else reps + 1
    p = 2 if ansatz == "pz_encoding" else 1
    theta = torch.randn(out_dim, in_dim, t, p, generator=g) * 2
    bias = torch.randn(out_dim, in_dim, reps, generator=g)
    return theta.to("cuda", dtype), bias.to("cuda", dtype)


def reference(theta, bias, reps, ansatz):
    """te.fourier_coefficients on float64 CPU inputs: the exact solver, good to 1e-12."""
    out_dim, in_dim = theta.shape[:2]
    return te.fourier_coefficients(
        theta,
        torch.ones_like(bias),
        bias,
        reps,
        in_dim,
        ansatz=ansatz,
        fast_measure=False,
        out_dim=out_dim,
    )


def run(fn, theta, bias, upstream):
    """Coefficients and the gradients of the linear functional <upstream, c>."""
    # detach(), not clone(): the leaves keep the strides the test gave them.
    theta = theta.detach().requires_grad_()
    bias = bias.detach().requires_grad_()
    c = fn(theta, bias)
    (c * upstream.to(c)).sum().backward()
    return c.detach(), theta.grad, bias.grad


def assert_grad_close(got, ref):
    """Within 1e-4 of the largest gradient; a bf16 gradient is the fp32 one
    rounded, which adds half an ulp (2^-8 relative) per element."""
    ref = ref.to(got.device)
    rounding = 2.0**-8 if got.dtype == torch.bfloat16 else 0.0
    err = (got.double() - ref).abs()
    tol = 1e-4 * ref.abs().max() + rounding * ref.abs()
    assert (err <= tol).all(), (err.max().item(), ref.abs().max().item())


def check(theta, bias, reps, ansatz):
    """Values and gradients of the kernel at (theta, bias) against the reference."""
    n = 2 * reps + 1
    out_dim, in_dim = theta.shape[:2]
    upstream = torch.randn(
        out_dim, n, in_dim, generator=torch.Generator().manual_seed(1)
    )
    c, grad_theta, grad_bias = run(
        lambda t, b: fourier_coefficients(t, b, reps, ansatz), theta, bias, upstream
    )
    # .double() of a bf16 tensor is its rounded value, where the kernel ran.
    ref_c, ref_theta, ref_bias = run(
        lambda t, b: reference(t, b, reps, ansatz),
        theta.double().cpu(),
        bias.double().cpu(),
        upstream,
    )
    assert c.shape == (out_dim, n, in_dim) and c.dtype == torch.float32
    torch.testing.assert_close(c.double().cpu(), ref_c, rtol=0, atol=2e-6)

    assert grad_theta.dtype == theta.dtype and grad_theta.shape == theta.shape
    assert_grad_close(grad_theta, ref_theta)
    if ansatz.startswith("rpz"):
        assert grad_bias.dtype == bias.dtype and grad_bias.shape == bias.shape
        assert_grad_close(grad_bias, ref_bias)
    else:
        assert grad_bias is None and ref_bias is None


@pytest.mark.parametrize("reps", [1, 2, 3, 5])
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_matches_reference(ansatz, reps):
    check(*params(ansatz, reps), reps, ansatz)


@pytest.mark.parametrize("ansatz", ANSATZES)
def test_bf16(ansatz):
    check(*params(ansatz, 2, dtype=torch.bfloat16), 2, ansatz)


@pytest.mark.parametrize("ansatz", ANSATZES)
def test_largest_tile(ansatz):
    """reps 8 to MAX_REPS run on 32 samples an edge, a tile no smaller reps
    compiles."""
    check(*params(ansatz, 8), 8, ansatz)


@pytest.mark.parametrize(
    "out_dim, in_dim",
    # One edge, one row, one column, rows longer than any block (one of them
    # past 1024 columns), blocks that span many rows, and whole blocks only.
    [(1, 1), (1, 7), (7, 1), (3, 701), (2, 1031), (301, 7), (4, 64)],
)
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_shapes(ansatz, out_dim, in_dim):
    check(*params(ansatz, 2, out_dim, in_dim), 2, ansatz)


@pytest.mark.parametrize(
    "out_dim, in_dim",
    # Edge counts at, one below and one above a block, and in between.
    [(1, 1), (1, 127), (1, 128), (1, 129), (3, 43), (5, 257)],
)
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_writes_stay_inside_the_outputs(ansatz, out_dim, in_dim, monkeypatch):
    """A mask that lets one more edge through passes every value check, so the
    outputs are allocated inside poisoned buffers: a launch has to write all of
    its output and nothing around it."""
    guard, poison = 4096, 12345.0
    buffers = []

    def empty(*shape, **kwargs):
        buffer = torch.full((math.prod(shape) + 2 * guard,), poison, **kwargs)
        buffers.append(buffer)
        return buffer[guard:-guard].view(shape)

    def empty_like(t):
        return empty(*t.shape, device=t.device, dtype=t.dtype)

    theta, bias = params(ansatz, 2, out_dim, in_dim)
    theta.requires_grad_()
    bias.requires_grad_()
    upstream = torch.randn(out_dim, 5, in_dim, device="cuda")
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    fourier_coefficients(theta, bias, 2, ansatz).backward(upstream)
    monkeypatch.undo()

    # c, grad_theta and, for rpz, grad_bias.
    assert len(buffers) == 2 + ansatz.startswith("rpz")
    for buffer in buffers:
        assert (buffer[:guard] == poison).all() and (buffer[-guard:] == poison).all()
        assert not (buffer[guard:-guard] == poison).any()


@pytest.mark.parametrize("ansatz", ANSATZES)
def test_non_contiguous(ansatz):
    theta, bias = params(ansatz, 2, out_dim=6, in_dim=5)
    theta, bias = theta.transpose(0, 1), bias.transpose(0, 1)[:, ::2]
    theta = theta[:, ::2]
    assert not theta.is_contiguous() and not bias.is_contiguous()
    check(theta, bias, 2, ansatz)


@pytest.mark.parametrize("ansatz", ANSATZES)
def test_non_contiguous_upstream(ansatz):
    theta, bias = params(ansatz, 1)
    upstream = torch.randn(4, 3, 10, device="cuda")[:, :, ::2]
    _, expected, _ = run(
        lambda t, b: fourier_coefficients(t, b, 1, ansatz), theta, bias, upstream
    )
    theta.requires_grad_()
    fourier_coefficients(theta, bias, 1, ansatz).backward(upstream)
    torch.testing.assert_close(theta.grad, expected, rtol=0, atol=0)


def test_size_one_dim_with_a_foreign_stride():
    """A size-1 dim keeps the stride of the view that made it and still counts
    as contiguous, so theta's strides say nothing about its layout."""
    stored, bias = params("pz_encoding", 2, out_dim=1, in_dim=6)
    theta = stored.transpose(0, 1)  # (6, 1, T, P) with stride(1) = 6 * T * P
    assert theta.is_contiguous() and theta.stride(1) != theta[0, 0].numel()
    check(theta, bias.transpose(0, 1), 2, "pz_encoding")


@pytest.mark.parametrize("ansatz", ["pz_encoding", "real"])
def test_pz_and_real_ignore_bias(ansatz):
    theta, bias = params(ansatz, 3)
    theta.requires_grad_()
    bias.requires_grad_()
    c = fourier_coefficients(theta, bias, 3, ansatz)
    # Whatever it is: other values, another device and dtype, or nothing.
    for other in (torch.randn_like(bias) * 3, bias.double().cpu(), None):
        assert torch.equal(c, fourier_coefficients(theta, other, 3, ansatz))
    c.sum().backward()
    assert bias.grad is None and theta.grad is not None


@pytest.mark.parametrize("ansatz", ["pz_encoding", "real"])
def test_pz_and_real_bias_builds_no_graph(ansatz):
    """A bias that requires grad must not buy a backward launch that returns None."""
    theta, bias = params(ansatz, 2)
    c = fourier_coefficients(theta, bias.requires_grad_(), 2, ansatz)
    assert not c.requires_grad and c.grad_fn is None


def test_rpz_bias_gradient_with_frozen_theta():
    theta, bias = params("rpz_encoding", 2)
    upstream = torch.randn(4, 5, 5, device="cuda")
    _, _, expected = run(
        lambda t, b: fourier_coefficients(t, b, 2, "rpz_encoding"),
        theta,
        bias,
        upstream,
    )
    bias.requires_grad_()
    c = fourier_coefficients(theta, bias, 2, "rpz_encoding")
    (c * upstream).sum().backward()
    assert theta.grad is None
    torch.testing.assert_close(bias.grad, expected, rtol=0, atol=0)


def test_second_derivatives_raise():
    """The backward is a kernel launch, a constant to autograd: differentiating
    it has to raise, not drop the second-order term."""
    theta, bias = params("rpz_encoding", 2)
    theta.requires_grad_()
    c = fourier_coefficients(theta, bias, 2, "rpz_encoding")
    (expected,) = torch.autograd.grad(c.sum(), theta, retain_graph=True)
    (grad,) = torch.autograd.grad(c.sum(), theta, create_graph=True)
    assert torch.equal(grad, expected)  # first order still works
    with pytest.raises(RuntimeError, match="second derivatives"):
        grad.square().sum().backward()


def test_numpy_reps(monkeypatch, tmp_path):
    """reps out of np.arange in a sweep. A numpy integer is no Triton constexpr
    but hashes like the int, so a cached int variant would hide the failure:
    compile from scratch (no other test uses reps=4)."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    check(*params("real", 4), np.int64(4), "real")


@pytest.mark.parametrize("name", ["pz_encoding", "rpz_encoding"])
def test_short_ansatz_names(name):
    theta, bias = params(name, 2)
    short = name.removesuffix("_encoding")
    assert torch.equal(
        fourier_coefficients(theta, bias, 2, short),
        fourier_coefficients(theta, bias, 2, name),
    )


@pytest.mark.parametrize("out_dim, in_dim", [(0, 5), (4, 0)])
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_empty(ansatz, out_dim, in_dim):
    theta, bias = params(ansatz, 2, out_dim, in_dim)
    theta.requires_grad_()
    bias.requires_grad_()
    c = fourier_coefficients(theta, bias, 2, ansatz)
    assert c.shape == (out_dim, 5, in_dim) and c.dtype == torch.float32
    c.sum().backward()
    assert theta.grad.shape == theta.shape


def test_rejects_other_ansatzes_and_layouts():
    theta, bias = params("pz_encoding", 2)
    with pytest.raises(ValueError, match="ansatz"):
        fourier_coefficients(theta, bias, 2, "mix")
    # reps=3 would read a fourth layer this theta does not have.
    with pytest.raises(ValueError, match="theta"):
        fourier_coefficients(theta, bias, 3, "pz_encoding")
    theta, bias = params("rpz_encoding", 2)
    with pytest.raises(ValueError, match="preacts_bias"):
        fourier_coefficients(theta, bias[:, :, :1], 2, "rpz_encoding")


def test_rejects_reps_past_max_reps():
    """Past MAX_REPS the kernels are slower than the sampling they replace."""
    theta, bias = params("real", MAX_REPS + 1)
    with pytest.raises(ValueError, match="MAX_REPS"):
        fourier_coefficients(theta, bias, MAX_REPS + 1, "real")


def test_rejects_other_dtypes():
    """All math is float32: a float64 input would lose its precision silently."""
    theta, bias = params("rpz_encoding", 2)
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        fourier_coefficients(theta.double(), bias, 2, "rpz_encoding")
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        fourier_coefficients(theta, bias.double(), 2, "rpz_encoding")


@pytest.mark.skipif(
    os.environ.get("QKAN_LARGE_TESTS") != "1",
    reason="allocates ~19 GB; set QKAN_LARGE_TESTS=1",
)
def test_offsets_past_int32(run_snippet):
    # pz at reps=1 holds 4 theta elements and 3 coefficients an edge, so both
    # tensors pass 2^31 elements here: theta in row 525828, c in row 701104.
    # Rows on both sides of both boundaries are compared with the float64
    # reference, never with another launch through the same offsets (see
    # test_large_offsets.py); with the edge index, the theta offsets or the c
    # offsets left in int32 this fails.
    result = run_snippet(
        """
        import torch
        from qkan.solver import te
        from qkan.solver.te_ops_coefficients import fourier_coefficients

        out_dim, in_dim = 705_000, 1021
        gen = torch.Generator(device="cuda").manual_seed(0)
        theta = torch.empty(out_dim, in_dim, 2, 2, device="cuda", dtype=torch.bfloat16)
        for rows in theta.split(15_000):  # no float32 copy of the whole theta
            rows.copy_(torch.randn(rows.shape, device="cuda", generator=gen) * 2)
        theta.requires_grad_()
        assert theta.numel() > 2**31 and out_dim * 3 * in_dim > 2**31

        c = fourier_coefficients(theta, None, 1, "pz_encoding")
        c.backward(c.detach())  # c is its own upstream: no second 8.6 GB

        def reference(theta):
            ones = torch.ones(len(theta), in_dim, 1, device="cuda", dtype=theta.dtype)
            return te.fourier_coefficients(
                theta, ones, ones, 1, in_dim,
                ansatz="pz_encoding", fast_measure=False, out_dim=len(theta),
            )

        theta_edge = 2**31 // (in_dim * 4)
        c_edge = 2**31 // (in_dim * 3)
        for start in (0, theta_edge - 8, c_edge - 8, out_dim - 16):
            rows = slice(start, start + 16)
            theta64 = theta[rows].detach().double().requires_grad_()
            ref = reference(theta64)
            ref.backward(c[rows].detach().double())
            torch.testing.assert_close(
                c[rows].detach().double(), ref.detach(), rtol=0, atol=2e-6
            )
            # theta.grad is bf16: half an ulp of rounding on top.
            torch.testing.assert_close(
                theta.grad[rows].double(), theta64.grad, rtol=2**-8, atol=1e-4
            )
        print("OK")
        """,
        preamble="",
        timeout=1800,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
