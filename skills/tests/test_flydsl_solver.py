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


def _run(solver, ansatz, preacts, fast, dtype, reps=3):
    x, theta, pw, pb = _inputs(ansatz, reps=reps)
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


def test_empty_batch():
    from qkan.solver import flydsl_exact_solver

    x, theta, pw, pb = _inputs("pz", batch=0)
    with torch.no_grad():
        out = flydsl_exact_solver(x, theta, pw, pb, 3, ansatz="pz", out_dim=3)
    assert out.shape == (0, 3, 5)


@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding"])
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
