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

"""Parity tests for the FlyDSL (ROCm) solver against the torch exact solver.

The FlyDSL backward pass recovers intermediate states by un-applying gates
instead of checkpointing them, so the gradients are checked against autograd
through the exact solver for every ansatz / preacts / measurement mode.
"""

import itertools

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan.solver import _FLYDSL_AVAILABLE, torch_exact_solver  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (_FLYDSL_AVAILABLE and torch.cuda.is_available()),
    reason="FlyDSL kernels require flydsl and a ROCm GPU",
)

TOL = 2e-5


def _inputs(ansatz, batch=37, in_dim=5, out_dim=3, reps=3, seed=1):
    gen = torch.Generator().manual_seed(seed)
    last = 2 if ansatz == "pz" else 1
    axis = reps if ansatz == "real" else reps + 1

    def param(*shape):
        return torch.randn(*shape, generator=gen).cuda().requires_grad_()

    return (
        param(batch, in_dim),
        param(out_dim, in_dim, axis, last),
        param(out_dim, in_dim, reps),
        param(out_dim, in_dim, reps),
    )


def _run(solver, ansatz, preacts, fast, dtype, reps=3, out_dim=3, in_dim=5):
    x, theta, pw, pb = _inputs(ansatz, reps=reps, out_dim=out_dim, in_dim=in_dim)
    out = solver(
        x,
        theta,
        pw,
        pb,
        reps,
        ansatz=ansatz,
        preacts_trainable=preacts,
        fast_measure=fast,
        out_dim=theta.shape[0],
        dtype=dtype,
    )
    grad_out = torch.randn(out.shape, generator=torch.Generator().manual_seed(2))
    (out.float() * grad_out.cuda()).sum().backward()
    grads = [
        t.grad if t.grad is not None else torch.zeros_like(t)
        for t in (x, theta, pw, pb)
    ]
    return [out.detach().float(), *grads]


def _assert_close(expected, actual, tol):
    names = ("out", "grad_x", "grad_theta", "grad_pw", "grad_pb")
    for name, e, a in zip(names, expected, actual):
        scale = e.abs().max().item() + 1e-6
        err = (e - a).abs().max().item() / scale
        assert err < tol, f"{name}: relative error {err:.2e}"


@pytest.mark.parametrize(
    "ansatz,preacts,fast",
    list(itertools.product(["pz", "rpz", "real"], [False, True], [True, False])),
)
def test_matches_exact_solver(ansatz, preacts, fast):
    from qkan.solver import flydsl_exact_solver

    expected = _run(torch_exact_solver, ansatz, preacts, fast, torch.complex64)
    actual = _run(flydsl_exact_solver, ansatz, preacts, fast, torch.float32)
    _assert_close(expected, actual, TOL)


@pytest.mark.parametrize("reps", [1, 8])
def test_reps_extremes(reps):
    from qkan.solver import flydsl_exact_solver

    expected = _run(torch_exact_solver, "pz", True, True, torch.complex64, reps)
    actual = _run(flydsl_exact_solver, "pz", True, True, torch.float32, reps)
    _assert_close(expected, actual, TOL)


@pytest.mark.parametrize(
    "ansatz,preacts,reps,out_dim",
    list(itertools.product(["pz", "rpz", "real"], [False, True], [1, 3], [32, 48])),
)
def test_output_blocking(ansatz, preacts, reps, out_dim):
    """Threads running several outputs: with in_dim 16, out_dim 32 / 48 give
    4 circuits per thread forward and 1-8 backward, depending on the thetas
    per circuit."""
    from qkan.solver import flydsl_exact_solver

    shape = dict(reps=reps, out_dim=out_dim, in_dim=16)
    expected = _run(torch_exact_solver, ansatz, preacts, True, torch.complex64, **shape)
    actual = _run(flydsl_exact_solver, ansatz, preacts, True, torch.float32, **shape)
    _assert_close(expected, actual, TOL)


@pytest.mark.parametrize(
    "ansatz,preacts", [("pz", False), ("rpz", True), ("real", True)]
)
def test_several_samples_per_thread(monkeypatch, ansatz, preacts):
    """With one CU, each thread runs a chunk of 19 samples and sums its
    gradients over them, and the second of the two chunks is a sample short."""
    from qkan.solver import flydsl_exact_solver
    from qkan.solver.flydsl import flydsl_ops as ops

    monkeypatch.setattr(ops, "_compute_units", lambda device: 1)
    monkeypatch.setattr(ops, "_THREADS_PER_CU_FWD", 16)
    monkeypatch.setattr(ops, "_THREADS_PER_CU_BWD", 16)
    expected = _run(torch_exact_solver, ansatz, preacts, True, torch.complex64)
    actual = _run(flydsl_exact_solver, ansatz, preacts, True, torch.float32)
    _assert_close(expected, actual, TOL)


@pytest.mark.parametrize("ansatz", ["pz", "rpz", "real"])
def test_bf16_output_is_the_rounded_f32_output(ansatz):
    """QKANLayer runs every real-ansatz layer with bf16 output."""
    from qkan.solver.flydsl import flydsl_ops as ops

    x, theta, pw, pb = (t.detach() for t in _inputs(ansatz))
    out = ops.flydsl_forward(x, theta, pw, pb, 3, ansatz, True)
    out_bf16 = ops.flydsl_forward(
        x, theta, pw, pb, 3, ansatz, True, c_dtype=torch.bfloat16
    )
    assert out_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(out_bf16.float(), out, rtol=2**-8, atol=0.0)


def test_bf16_output_and_grad_dtypes():
    from qkan.solver import flydsl_exact_solver

    x, theta, pw, pb = (t.detach().bfloat16().requires_grad_() for t in _inputs("pz"))
    out = flydsl_exact_solver(
        x, theta, pw, pb, 3, ansatz="pz", out_dim=3, dtype=torch.bfloat16
    )
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert x.grad.dtype == torch.bfloat16
    assert theta.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("ansatz", ["pz", "rpz", "real"])
def test_bf16_inputs_read_natively(ansatz):
    """bf16 x / params / grad_output give the same result as their f32 values."""
    from qkan.solver.flydsl import flydsl_ops as ops

    bf16 = [t.detach().bfloat16() for t in _inputs(ansatz)]
    f32 = [t.float() for t in bf16]
    go = torch.randn(37, 3, 5, device="cuda").bfloat16()
    for fn, extra in (
        (ops.flydsl_forward, ()),
        (ops.flydsl_backward, (go,)),
    ):
        args_bf16 = (*bf16, *extra, 3, ansatz, True)
        args_f32 = (*f32, *(t.float() for t in extra), 3, ansatz, True)
        results = fn(*args_bf16), fn(*args_f32)
        if fn is ops.flydsl_forward:
            results = ([results[0]], [results[1]])
        for a, b in zip(*results):
            assert torch.equal(a, b)


@pytest.mark.parametrize("ansatz", ["pz", "real"])
def test_unused_theta_slots_get_zero_grad(ansatz):
    """Theta entries no gate reads get exactly zero gradient."""
    from qkan.solver.flydsl import flydsl_ops as ops

    x, theta, pw, pb = (t.detach() for t in _inputs(ansatz))
    used = theta.shape[3]
    padded = torch.cat([theta, torch.randn_like(theta)], dim=3)
    go = torch.randn(37, 3, 5, device="cuda")
    for _ in range(3):  # reused allocations must not leak stale values
        _, grad_theta, _, _ = ops.flydsl_backward(
            x, padded, pw, pb, go, 3, ansatz, False
        )
        _, expected, _, _ = ops.flydsl_backward(x, theta, pw, pb, go, 3, ansatz, False)
        assert torch.equal(grad_theta[..., used:], torch.zeros_like(expected))
        assert torch.equal(grad_theta[..., :used], expected)


def test_empty_batch():
    from qkan.solver import flydsl_exact_solver

    x, theta, pw, pb = _inputs("pz", batch=0)
    with torch.no_grad():
        out = flydsl_exact_solver(x, theta, pw, pb, 3, ansatz="pz", out_dim=3)
    assert out.shape == (0, 3, 5)


# mix has no FlyDSL kernel and falls back to the exact solver, which needs
# the layer to keep its complex c_dtype.
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "mix"])
def test_layer_matches_exact(ansatz):
    from qkan import QKAN

    x = torch.randn(64, 6, device="cuda")
    ref = QKAN([6, 5, 3], reps=3, device="cuda", solver="exact", seed=7, ansatz=ansatz)
    model = QKAN(
        [6, 5, 3], reps=3, device="cuda", solver="flydsl", seed=7, ansatz=ansatz
    )
    model.load_state_dict(ref.state_dict())
    ref(x).pow(2).sum().backward()
    model(x).pow(2).sum().backward()
    for (name, p_ref), p in zip(ref.named_parameters(), model.parameters()):
        if p_ref.grad is None:
            continue
        err = (p_ref.grad - p.grad).abs().max().item()
        assert err < 1e-4 * (p_ref.grad.abs().max().item() + 1), name


@pytest.mark.parametrize(
    "kind,ref",
    [
        ("silu", torch.nn.SiLU()),
        ("gelu_exact", torch.nn.GELU()),
        ("gelu_tanh", torch.nn.GELU(approximate="tanh")),
        ("relu", torch.nn.ReLU()),
        ("tanh", torch.nn.Tanh()),
        ("sigmoid", torch.nn.Sigmoid()),
    ],
)
def test_activation_matches_torch(kind, ref):
    from qkan.solver.flydsl.flydsl_activation import FlyDSLActivation

    x = (torch.randn(1000, 33, device="cuda") * 3).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    y, y_ref = FlyDSLActivation(kind)(x), ref(x_ref)
    grad = torch.randn_like(y)
    (y * grad).sum().backward()
    (y_ref * grad).sum().backward()
    assert (y - y_ref).abs().max().item() < 1e-5
    assert (x.grad - x_ref.grad).abs().max().item() < 1e-4
