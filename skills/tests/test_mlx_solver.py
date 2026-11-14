# Copyright (c) 2026, Jiun-Cheng Jiang and the QKAN contributors. All rights reserved.
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

"""Parity tests for the MLX (Apple silicon) solver against the torch exact solver.

The MLX backward pass rebuilds the intermediate states instead of
checkpointing them (kept in registers for shallow circuits, recovered by
un-applying gates for deep ones), so the gradients are checked against
autograd through the exact solver for every ansatz / preacts / measurement
mode. The reference always runs on the CPU; the MLX solver runs with its
tensors on the CPU and, where available, on MPS, where it shares Metal buffers
with PyTorch.

Both routes are covered: the solver that returns ``(batch, out, in)`` postacts,
and the contracted solver QKANLayer uses by default, which sums over the
inputs inside the kernel.
"""

import itertools

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan import QKAN  # noqa: E402
from qkan.solver import (  # noqa: E402
    _MLX_AVAILABLE,
    mlx_contracted_solver,
    mlx_exact_solver,
    torch_exact_solver,
)
from qkan.solver.mlx import mlx_ops  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (_MLX_AVAILABLE and mlx_ops._kernels_available()),
    reason="MLX kernels require mlx and an Apple silicon GPU",
)

TOL = 2e-5
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
ANSATZES = ("pz", "rpz", "real", "mix")
CASES = list(itertools.product(ANSATZES, (False, True), (False, True)))


def _inputs(ansatz, device, batch=37, in_dim=5, out_dim=3, reps=3, seed=1):
    gen = torch.Generator().manual_seed(seed)
    last = 2 if ansatz in ("pz", "mix") else 1
    axis = reps if ansatz == "real" else reps + 1

    def param(*shape):
        return torch.randn(*shape, generator=gen).to(device).requires_grad_()

    return (
        param(batch, in_dim),
        param(out_dim, in_dim, axis, last),
        param(out_dim, in_dim, reps),
        param(out_dim, in_dim, reps),
    )


def _run(solver, ansatz, preacts, fast, device, **shape):
    x, theta, pw, pb = _inputs(ansatz, device, **shape)
    out = solver(
        x,
        theta,
        pw,
        pb,
        shape.get("reps", 3),
        ansatz=ansatz,
        preacts_trainable=preacts,
        fast_measure=fast,
        out_dim=theta.shape[0],
        dtype=torch.complex64,
    )
    grad_out = torch.randn(out.shape, generator=torch.Generator().manual_seed(2))
    (out * grad_out.to(device)).sum().backward()
    grads = [
        t.grad if t.grad is not None else torch.zeros_like(t)
        for t in (x, theta, pw, pb)
    ]
    return [t.detach().float().cpu() for t in (out, *grads)]


def _assert_close(expected, actual, tol=TOL):
    names = ("out", "grad_x", "grad_theta", "grad_pw", "grad_pb")
    for name, e, a in zip(names, expected, actual):
        assert e.shape == a.shape, name
        err = (e - a).abs().max().item() / max(1.0, e.abs().max().item())
        assert err < tol, f"{name}: rel err {err:.2e}"


@pytest.fixture
def many_rows_per_thread(monkeypatch):
    """Keep the batch chunk long on tiny shapes.

    The launcher shortens the chunk until the grid fills the GPU, which on a
    test-sized layer leaves one batch row per thread and never runs the
    in-thread batch loop or a ragged final chunk.
    """
    monkeypatch.setattr(mlx_ops, "_MIN_THREADS", 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ansatz,preacts,fast", CASES)
def test_matches_exact_solver(device, ansatz, preacts, fast):
    expected = _run(torch_exact_solver, ansatz, preacts, fast, "cpu")
    actual = _run(mlx_exact_solver, ansatz, preacts, fast, device)
    _assert_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ansatz,preacts,fast", CASES)
def test_batch_chunks(device, ansatz, preacts, fast, many_rows_per_thread):
    """Several rows per thread, with a ragged last chunk (150 = 4 * 32 + 22)."""
    shape = dict(batch=150, in_dim=3, out_dim=2)
    expected = _run(torch_exact_solver, ansatz, preacts, fast, "cpu", **shape)
    actual = _run(mlx_exact_solver, ansatz, preacts, fast, device, **shape)
    _assert_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shape",
    [
        dict(reps=1),
        dict(reps=6, batch=9),
        dict(batch=1, in_dim=9, out_dim=1),
        dict(batch=130, in_dim=1, out_dim=7),
    ],
)
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_shapes(device, ansatz, shape):
    expected = _run(torch_exact_solver, ansatz, True, True, "cpu", **shape)
    actual = _run(mlx_exact_solver, ansatz, True, True, device, **shape)
    _assert_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
def test_output_stays_on_the_input_device(device):
    x, theta, pw, pb = _inputs("pz", device)
    out = mlx_exact_solver(x, theta, pw, pb, 3, ansatz="pz", out_dim=3)
    assert out.device.type == device
    assert out.dtype == torch.float32
    assert out.shape == (37, 3, 5)


@pytest.mark.parametrize("device", DEVICES)
def test_first_layer_input_needs_no_gradient(device):
    """Without x.requires_grad the backward kernel skips grad_x altogether."""
    x, theta, pw, pb = _inputs("pz", device)
    xr, thr, pwr, pbr = _inputs("pz", "cpu")
    kw = dict(ansatz="pz", preacts_trainable=True, out_dim=3)
    mlx_exact_solver(x.detach(), theta, pw, pb, 3, **kw).sum().backward()
    torch_exact_solver(xr.detach(), thr, pwr, pbr, 3, **kw).sum().backward()
    _assert_close(
        [thr.grad, pwr.grad, pbr.grad],
        [theta.grad.cpu(), pw.grad.cpu(), pb.grad.cpu()],
    )


@pytest.mark.parametrize("device", DEVICES)
def test_no_grad_skips_autograd(device):
    x, theta, pw, pb = _inputs("pz", device)
    with torch.no_grad():
        out = mlx_exact_solver(x, theta, pw, pb, 3, ansatz="pz", out_dim=3)
    assert out.grad_fn is None and not out.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_empty_batch(device):
    x, theta, pw, pb = _inputs("pz", device, batch=0)
    out = mlx_exact_solver(x, theta, pw, pb, 3, ansatz="pz", out_dim=3)
    assert out.shape == (0, 3, 5)
    out.sum().backward()
    assert theta.grad.abs().sum().item() == 0.0


def test_unused_thetas_get_zero_gradient():
    """A theta wider than the ansatz reads keeps its shape and a zero gradient."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(11, 4, generator=gen)
    theta = torch.randn(2, 4, 5, 3, generator=gen, requires_grad=True)
    ref = theta.detach().clone().requires_grad_()
    kw = dict(ansatz="pz", out_dim=2)
    mlx_exact_solver(x, theta, x, x, 3, **kw).sum().backward()
    torch_exact_solver(x, ref, x, x, 3, **kw).sum().backward()
    _assert_close([ref.grad], [theta.grad])
    assert theta.grad[:, :, 4:].abs().sum().item() == 0.0
    assert theta.grad[:, :, :, 2:].abs().sum().item() == 0.0


def test_second_derivatives_raise():
    x, theta, pw, pb = _inputs("pz", "cpu")
    out = mlx_exact_solver(x, theta, pw, pb, 3, ansatz="pz", out_dim=3)
    (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    with pytest.raises(RuntimeError, match="no second derivatives"):
        grad.sum().backward()


def test_unsupported_configurations_raise():
    x, theta, pw, pb = _inputs("rpz", "cpu")
    # px is not fused; double precision is beyond Metal.
    for kw, match in (
        (dict(ansatz="px", preacts_trainable=True), "no kernel for ansatz 'px'"),
        (dict(ansatz="rpz", dtype=torch.complex128), "Metal has no float64"),
        (dict(ansatz="rpz", dtype=torch.float64), "Metal has no float64"),
    ):
        with pytest.raises(ValueError, match=match):
            mlx_exact_solver(x, theta, pw, pb, 3, out_dim=3, **kw)


@pytest.mark.skipif("mps" not in DEVICES, reason="needs the MPS device")
def test_reads_pending_mps_writes():
    """PyTorch work still queued on its own command queue must land first."""
    gen = torch.Generator().manual_seed(0)
    theta = torch.randn(4, 64, 4, 2, generator=gen)
    base = torch.randn(256, 64, generator=gen)
    kw = dict(ansatz="pz", out_dim=4)
    # Moved up front: a host-to-MPS copy flushes the queue by itself.
    theta_mps, base_mps = theta.to("mps"), base.to("mps")
    for trial in range(20):
        expected = torch_exact_solver(
            torch.sin(base @ base.T @ base * 1e-3) + trial, theta, base, base, 3, **kw
        )
        x = torch.sin(base_mps @ base_mps.T @ base_mps * 1e-3)
        x.add_(float(trial))  # queued in place, right before the launch
        actual = mlx_exact_solver(x, theta_mps, base_mps, base_mps, 3, **kw)
        assert (expected - actual.cpu()).abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# Contracted solver: sum_i postacts * weight inside the kernel
# ---------------------------------------------------------------------------


def _run_contracted(fused, ansatz, preacts, fast, device, x_grad=True, **shape):
    weight_grad = shape.pop("weight_grad", True)
    x, theta, pw, pb = _inputs(ansatz, device, **shape)
    gen = torch.Generator().manual_seed(5)
    weight = torch.randn(theta.shape[0], x.shape[1], generator=gen).to(device)
    weight.requires_grad_(weight_grad)
    if not x_grad:
        x = x.detach()
    kw = dict(
        ansatz=ansatz,
        preacts_trainable=preacts,
        fast_measure=fast,
        out_dim=theta.shape[0],
        dtype=torch.complex64,
    )
    reps = shape.get("reps", 3)
    if fused:
        out = mlx_contracted_solver(x, theta, pw, pb, weight, reps, **kw)
        assert out is not None
    else:
        out = (torch_exact_solver(x, theta, pw, pb, reps, **kw) * weight).sum(dim=2)
    grad_out = torch.randn(out.shape, generator=torch.Generator().manual_seed(2))
    (out * grad_out.to(device)).sum().backward()
    grads = [
        t.grad if t.grad is not None else torch.zeros_like(t)
        for t in (x, theta, pw, pb, weight)
    ]
    return [t.detach().float().cpu() for t in (out, *grads)]


def _assert_contracted_close(expected, actual, tol=TOL):
    names = ("out", "grad_x", "grad_theta", "grad_pw", "grad_pb", "grad_weight")
    for name, e, a in zip(names, expected, actual):
        assert e.shape == a.shape, name
        err = (e - a).abs().max().item() / max(1.0, e.abs().max().item())
        assert err < tol, f"{name}: rel err {err:.2e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ansatz,preacts,fast", CASES)
def test_contracted_matches_exact(device, ansatz, preacts, fast):
    expected = _run_contracted(False, ansatz, preacts, fast, "cpu")
    actual = _run_contracted(True, ansatz, preacts, fast, device)
    assert actual[0].shape == (37, 3)
    _assert_contracted_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "min_threads",
    [1, 10**9],
    ids=["one-block-long-chunks", "one-circuit-one-row"],
)
@pytest.mark.parametrize("ansatz,preacts,fast", CASES)
def test_contracted_launch_shapes(
    device, ansatz, preacts, fast, min_threads, monkeypatch
):
    """Both ends of the launch: a thread that loops over every input (or
    output) for a ragged chunk of batch rows (150 = 4 * 32 + 22), and the
    grid split all the way down to one circuit and one row per thread, where
    the host adds one partial sum per input (or output)."""
    monkeypatch.setattr(mlx_ops, "_MIN_THREADS_CONTRACT", min_threads)
    shape = dict(batch=150, in_dim=7, out_dim=4)
    expected = _run_contracted(False, ansatz, preacts, fast, "cpu", **shape)
    actual = _run_contracted(True, ansatz, preacts, fast, device, **shape)
    _assert_contracted_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shape",
    [
        dict(reps=1),
        dict(reps=6, batch=9),
        dict(batch=1, in_dim=9, out_dim=1),
        dict(batch=130, in_dim=1, out_dim=7),
    ],
)
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_contracted_shapes(device, ansatz, shape):
    """reps=6 is past the register budget: states are recovered by un-applying."""
    expected = _run_contracted(False, ansatz, True, True, "cpu", **shape)
    actual = _run_contracted(True, ansatz, True, True, device, **shape)
    _assert_contracted_close(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("x_grad,weight_grad", [(False, True), (True, False)])
def test_contracted_skips_unneeded_gradients(device, x_grad, weight_grad):
    """A first layer (no grad_x) and frozen postact weights (no grad_weight)
    compile kernels without those sums; the remaining gradients still match."""
    kw = dict(x_grad=x_grad, weight_grad=weight_grad)
    expected = _run_contracted(False, "pz", True, True, "cpu", **kw)
    actual = _run_contracted(True, "pz", True, True, device, **kw)
    _assert_contracted_close(expected, actual)


def test_contracted_declines_what_it_does_not_fuse():
    x, theta, pw, pb = _inputs("pz", "cpu")
    weight = torch.randn(3, 5)
    kw = dict(out_dim=3)
    # px is not fused: mlx_exact_solver, which the layer calls next, refuses it.
    assert mlx_contracted_solver(x, theta, pw, pb, weight, 3, ansatz="px", **kw) is None
    # Metal has no double precision.
    with pytest.raises(ValueError, match="Metal has no float64"):
        mlx_contracted_solver(
            x, theta, pw, pb, weight, 3, ansatz="pz", dtype=torch.complex128, **kw
        )
    # One row of thetas shared by every output: the unfused route evaluates
    # those circuits once instead of once per output.
    shared = theta[:1]
    assert (
        mlx_contracted_solver(x, shared, pw, pb, weight, 3, ansatz="pz", **kw) is None
    )


def test_contracted_second_derivatives_raise():
    x, theta, pw, pb = _inputs("pz", "cpu")
    weight = torch.randn(3, 5)
    out = mlx_contracted_solver(x, theta, pw, pb, weight, 3, ansatz="pz", out_dim=3)
    (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    with pytest.raises(RuntimeError, match="no second derivatives"):
        grad.sum().backward()


@pytest.mark.parametrize("device", DEVICES)
def test_contracted_empty_batch(device):
    x, theta, pw, pb = _inputs("pz", device, batch=0)
    weight = torch.randn(3, 5, device=device)
    out = mlx_contracted_solver(x, theta, pw, pb, weight, 3, ansatz="pz", out_dim=3)
    assert out.shape == (0, 3)
    out.sum().backward()
    assert theta.grad.abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# Through QKAN
# ---------------------------------------------------------------------------

LAYER_CASES = [
    dict(),
    dict(ansatz="rpz_encoding"),
    dict(ansatz="mix", preact_trainable=True),
    dict(fast_measure=False),
    dict(p_dim=2),
    dict(reps=1),
    # A group that does not divide in_dim: theta is tiled and sliced, which
    # leaves it non-contiguous and has to be copied before the launch.
    dict(group=2),
    dict(group=5),
    dict(ansatz="rpz", group=3),
    dict(postact_weight_trainable=True, postact_bias_trainable=True),
]


def _set_fused(model, fused):
    for layer in model.layers:
        layer.set_fused_epilogue(fused)


def _prune_some_edges(model):
    """Zero a few mask entries and set a nonzero frozen postact bias, the way
    prune_edge and a loaded checkpoint leave a layer."""
    gen = torch.Generator().manual_seed(7)
    for layer in model.layers:
        keep = (torch.rand(layer.mask.shape, generator=gen) > 0.3).float()
        layer.mask.data.copy_(keep)
        layer._mask_is_identity = False
        layer.postact_bias.data.copy_(
            torch.randn(layer.postact_bias.shape, generator=gen)
        )
        layer._pb_is_nonzero = True


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("pruned", [False, True], ids=["dense", "pruned"])
@pytest.mark.parametrize("kwargs", LAYER_CASES, ids=str)
def test_qkan_matches_exact(device, kwargs, fused, pruned):
    ref = QKAN([5, 7, 3], solver="exact", device="cpu", seed=0, **kwargs)
    if pruned:
        _prune_some_edges(ref)
    model = QKAN([5, 7, 3], solver="mlx", device=device, seed=0, **kwargs)
    model.load_state_dict(ref.state_dict())
    model.to(device)
    _set_fused(model, fused)
    x = torch.randn(33, 5, generator=torch.Generator().manual_seed(3))

    expected = ref(x)
    actual = model(x.to(device))
    assert actual.device.type == device
    assert actual.dtype == expected.dtype
    _assert_close([expected.detach()], [actual.detach().cpu()])

    expected.pow(2).sum().backward()
    actual.pow(2).sum().backward()
    for (name, p), q in zip(model.named_parameters(), ref.parameters()):
        if q.grad is None:
            assert p.grad is None, name
        else:
            _assert_close([q.grad], [p.grad.cpu()])

    ref.eval()
    model.eval()
    with torch.no_grad():
        _assert_close([ref(x)], [model(x.to(device)).cpu()])


def test_qkan_fuses_the_contraction_by_default(monkeypatch):
    """solver="mlx" contracts inside the kernel unless told otherwise; the
    other solvers keep the fused epilogue opt-in."""
    monkeypatch.delenv("QKAN_FUSED_EPILOGUE", raising=False)
    assert QKAN([3, 2], solver="mlx").layers[0]._use_fused_epilogue
    assert not QKAN([3, 2], solver="exact").layers[0]._use_fused_epilogue
    monkeypatch.setenv("QKAN_FUSED_EPILOGUE", "0")
    assert not QKAN([3, 2], solver="mlx").layers[0]._use_fused_epilogue
    monkeypatch.setenv("QKAN_FUSED_EPILOGUE", "1")
    assert QKAN([3, 2], solver="exact").layers[0]._use_fused_epilogue


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_qkan_routes_through_the_selected_kernels(fused, monkeypatch):
    """The fused route never builds postacts, and switching it off never
    touches the contraction kernels."""
    from qkan.solver.mlx import mlx as mlx_solver

    def unexpected(*args, **kwargs):
        raise AssertionError("wrong MLX route")

    model = QKAN([5, 7, 3], solver="mlx", seed=0)
    _set_fused(model, fused)
    blocked = "mlx_forward" if fused else "mlx_contract_forward"
    monkeypatch.setattr(mlx_solver, blocked, unexpected)
    x = torch.randn(9, 5)
    model(x).sum().backward()
    model.eval()
    with torch.no_grad():
        model(x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qkan_output_dtype(device, fused, dtype):
    """As for the other solvers: the output keeps the parameter dtype, and
    follows autocast under it. The kernels themselves compute in float32."""
    from qkan import QKANLayer

    torch.manual_seed(0)
    layer = QKANLayer(12, 10, reps=2, solver="mlx", device=device)
    layer = layer.to(dtype)
    layer.set_fused_epilogue(fused)
    x = torch.randn(48, 12, device=device, dtype=dtype, requires_grad=True)
    autocast = dict(device_type=device, dtype=torch.bfloat16)

    assert layer(x).dtype == dtype
    with torch.autocast(**autocast):
        y = layer(x)
    assert y.dtype == torch.bfloat16
    y.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == dtype
    assert layer.theta.grad.dtype == dtype

    layer.eval()
    with torch.no_grad():
        assert layer(x).dtype == dtype
        with torch.autocast(**autocast):
            assert layer(x).dtype == torch.bfloat16


@pytest.mark.parametrize("device", DEVICES)
def test_qkan_trains(device):
    torch.manual_seed(0)
    model = QKAN([2, 4, 1], solver="mlx", device=device, seed=0)
    x = torch.rand(256, 2, device=device) * 2 - 1
    y = torch.sin(torch.pi * x[:, :1]) * x[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
