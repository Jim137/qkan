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

"""Host-side tests for the FlyDSL solver.

FlyDSL is replaced by a stand-in for ``flyc.compile`` that checks every launch
the way FlyDSL's argument conversion does, keeps each compile's constants and
device as FlyDSL does, and runs small torch "kernels" with the real launchers'
signatures and output layout. The tests therefore run on any CUDA or ROCm
machine without flydsl, and cover what the host code owns: input validation,
the tensors and launch configuration handed to FlyDSL, and how the results
are assembled. Kernel numerics are covered on ROCm by test_flydsl_solver.py.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan import QKAN, QKANLayer  # noqa: E402
from qkan.solver.flydsl import flydsl as solver  # noqa: E402
from qkan.solver.flydsl import flydsl_activation as act  # noqa: E402
from qkan.solver.flydsl import flydsl_ops as ops  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src"

# FlyDSL packs every tensor size into an int32, and HIP caps the launch grid's
# y dimension, which carries the batch chunks.
INT32_MAX = 2**31 - 1
GRID_Y_MAX = 65535
ARCH_VARS = ("ARCH", "FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION")


class FakeFlyDSL:
    """Stand-in for ``flydsl.compiler``.

    Like FlyDSL, ``compile`` runs the first launch and returns a function that
    keeps that launch's compile-time constants and runs on the device that was
    current then, so the host has to look up the right one for both.
    """

    def __init__(self):
        self.max_numel = INT32_MAX
        self.compile_only = False  # FlyDSL's COMPILE_ONLY environment variable
        self.compiles = []  # (current device, target arch, constants) per compile
        self.launches = 0

    def compile(self, launcher, *args):
        device = torch.cuda.current_device()
        n = launcher.n_runtime_args
        constants = args[n:-1]
        self.compiles.append((device, _flydsl_arch(), constants))
        if self.compile_only:
            return None

        def compiled(*call_args):
            # FlyDSL ignores the constants of later calls.
            assert call_args[n:-1] == constants, (call_args[n:-1], constants)
            assert torch.cuda.current_device() == device
            self._launch(launcher, call_args)

        self._launch(launcher, args)
        return compiled

    def _launch(self, launcher, args):
        *values, stream = args
        assert isinstance(stream, torch.cuda.Stream)
        device = torch.device("cuda", torch.cuda.current_device())
        for v in values:
            if isinstance(v, torch.Tensor):
                # Flat, unit stride, int32-sized, on the launching device.
                assert v.dim() == 1 and v.stride() == (1,), (v.shape, v.stride())
                assert v.numel() <= self.max_numel, v.numel()
                assert v.device == device, (v.device, device)
            else:
                # Scalars and compile-time constants must be builtin types.
                assert type(v) in (int, bool, float, str), type(v)
        self.launches += 1
        launcher(*args)


def _flydsl_arch():
    """The arch FlyDSL targets: its environment overrides, in FlyDSL's order
    of precedence, else what it detects on the machine."""
    for var in ARCH_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return "detected"


def fake_forward(
    X, TH, PW, PB, OUT, batch, in_dim, n_oi, n_threads, b_chunk, n_b, *rest
):
    """Stand-in for ``_launch_forward``: out[b, o, i] = x[b, i] + o, written
    for the batch rows the launch grid covers."""
    assert n_b <= GRID_Y_MAX
    out_dim = n_oi // in_dim
    rows = min(batch, b_chunk * n_b)
    out = OUT.view(batch, out_dim, in_dim)
    out[:rows] = _expected_out(X.view(batch, in_dim)[:rows], out_dim).to(OUT.dtype)


def fake_backward(
    X,
    TH,
    PW,
    PB,
    GO,
    GT,
    GW,
    GB,
    GXP,
    batch,
    in_dim,
    n_oi,
    n_threads,
    b_chunk,
    n_b,
    *rest,
):
    """Stand-in for ``_launch_backward``, writing the real kernel's layout: the
    grad_x partial sum of every (sample, thread) and one row of parameter sums
    per batch chunk. Every derivative of the stand-in circuit is 1, so
    grad_x[b, i] = sum_o go[b, o, i] and each theta or preacts entry of
    circuit (o, i) gets sum_b go[b, o, i]."""
    reps, n_th, _, _, preacts, _, _, _, _, o_blk = rest[:10]
    assert n_b <= GRID_Y_MAX
    out_dim = n_oi // in_dim
    rows = min(batch, b_chunk * n_b)
    go = GO.view(batch, out_dim, in_dim).float()
    gxp = GXP.view(batch, out_dim // o_blk, in_dim)
    gxp[:rows] = go[:rows].view(rows, out_dim // o_blk, o_blk, in_dim).sum(2)
    for y in range(n_b):
        chunk = go[y * b_chunk : (y + 1) * b_chunk].sum(0).view(n_oi, 1)
        GT.view(n_b, n_oi, n_th)[y] = chunk
        if preacts:
            GW.view(n_b, n_oi, reps)[y] = chunk
            GB.view(n_b, n_oi, reps)[y] = chunk


def fake_act_forward(X, Y, n, n_blocks, *rest):
    """Stand-in for ``_launch_act_forward``: y = 2x."""
    Y.copy_(X * 2)


def fake_act_backward(GY, X, GX, n, n_blocks, *rest):
    """Stand-in for ``_launch_act_backward``: dy/dx = 2."""
    GX.copy_(GY * 2)


# Arguments before the compile-time constants, as in the real launchers.
fake_forward.n_runtime_args = 11
fake_backward.n_runtime_args = 15
fake_act_forward.n_runtime_args = 4
fake_act_backward.n_runtime_args = 5


@pytest.fixture
def fake_flydsl(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA or ROCm GPU")
    fake = FakeFlyDSL()
    monkeypatch.setattr(ops, "flyc", fake, raising=False)
    monkeypatch.setattr(ops, "_launch_forward", fake_forward, raising=False)
    monkeypatch.setattr(ops, "_launch_backward", fake_backward, raising=False)
    monkeypatch.setattr(act, "_launch_act_forward", fake_act_forward, raising=False)
    monkeypatch.setattr(act, "_launch_act_backward", fake_act_backward, raising=False)
    monkeypatch.setattr(ops, "_FLYDSL_KERNELS_AVAILABLE", True)
    monkeypatch.setattr(act, "_FLYDSL_KERNELS_AVAILABLE", True)
    monkeypatch.setattr(solver, "_FLYDSL_AVAILABLE", True)
    monkeypatch.setattr(ops, "_COMPILED", {})
    monkeypatch.setattr(ops, "_compute_units", lambda device: 304)  # MI300X
    for var in ARCH_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ops, "_device_arch", lambda device: "gfx90a", raising=False)
    return fake


def _params(ansatz, out_dim=3, in_dim=5, reps=2, batch=7, device="cuda"):
    """x, theta, preacts weight and preacts bias, in the solver's layouts."""
    gen = torch.Generator().manual_seed(0)
    last = 2 if ansatz == "pz" else 1
    layers = reps if ansatz == "real" else reps + 1
    shapes = [
        (batch, in_dim),
        (out_dim, in_dim, layers, last),
        (out_dim, in_dim, reps),
        (out_dim, in_dim, reps),
    ]
    return [torch.randn(*s, generator=gen).to(device) for s in shapes]


def _grad_out(*shape, device="cuda"):
    """Integer-valued grad_output, so the stand-in's sums are exact in f32."""
    gen = torch.Generator().manual_seed(1)
    return torch.randint(-3, 4, shape, generator=gen).float().to(device)


def _expected_out(x, out_dim):
    """The stand-in forward: out[b, o, i] = x[b, i] + o."""
    o = torch.arange(out_dim, device=x.device, dtype=torch.float32)
    return x.float()[:, None, :] + o[:, None]


def _solver_pz(x, theta, pw, pb):
    """The pz solver on 3 outputs, through autograd when an input needs it."""
    return solver.flydsl_exact_solver(
        x, theta, pw, pb, 2, ansatz="pz", out_dim=3, dtype=torch.float32
    )


def test_zero_reps_real_ansatz_backward(fake_flydsl):
    # theta of the real ansatz with reps=0 has no layers: (out, in, 0, 1).
    x, theta, pw, pb = _params("real", reps=0)
    go = _grad_out(7, 3, 5)
    grad_x, grad_theta, grad_pw, grad_pb = ops.flydsl_backward(
        x, theta, pw, pb, go, 0, "real", False
    )
    assert torch.equal(grad_x, go.sum(1))
    assert grad_theta.shape == (3, 5, 0, 1)
    assert grad_pw is None and grad_pb is None


def test_numpy_hyperparameters_reach_flydsl_as_builtins(fake_flydsl):
    # e.g. reps from an np.arange sweep or a pandas config row
    x, theta, pw, pb = _params("pz")
    go = _grad_out(7, 3, 5)
    reps, preacts, fast = np.int64(2), np.bool_(True), np.bool_(False)
    out = ops.flydsl_forward(x, theta, pw, pb, reps, "pz", preacts, fast)
    grad_x, *_ = ops.flydsl_backward(x, theta, pw, pb, go, reps, "pz", preacts, fast)
    assert torch.equal(out, _expected_out(x, 3))
    assert torch.equal(grad_x, go.sum(1))


def test_batch_chunks_fit_the_grid_y_limit(fake_flydsl):
    # One circuit and a long batch: the occupancy heuristic alone asks for
    # one batch chunk per sample.
    x, theta, pw, pb = _params("pz", out_dim=1, in_dim=1, batch=100_000)
    go = _grad_out(100_000, 1, 1)
    out = ops.flydsl_forward(x, theta, pw, pb, 2, "pz", False)
    grad_x, grad_theta, _, _ = ops.flydsl_backward(x, theta, pw, pb, go, 2, "pz", False)
    assert torch.equal(out, _expected_out(x, 1))
    assert torch.equal(grad_x, go.sum(1))
    assert torch.equal(grad_theta, go.sum(0)[..., None, None].expand_as(theta))


@pytest.mark.parametrize("n_cu", [1, 38, 304])
def test_batch_chunk_ends_fit_int32(monkeypatch, n_cu):
    # The kernels compute the end of a batch chunk, b_start + b_chunk, in
    # int32. A single circuit makes the longest launches.
    monkeypatch.setattr(ops, "_compute_units", lambda device: n_cu)
    device = torch.device("cuda")
    for batch in (INT32_MAX, 2**31 + 12345, 3 * 2**30 + 1):
        covered = 0
        for b0, rows in ops._batch_slices(batch, 1):
            assert b0 == covered
            covered += rows
            for threads_per_cu in (ops._THREADS_PER_CU_FWD, ops._THREADS_PER_CU_BWD):
                b_chunk, n_b = ops._batch_chunks(1, rows, device, threads_per_cu)
                assert n_b <= GRID_Y_MAX
                assert b_chunk * n_b <= INT32_MAX, (batch, rows, b_chunk, n_b)
        assert covered == batch


@pytest.mark.parametrize(
    "ansatz,preacts,out_dim,in_dim,reps,limit",
    [
        # 3x5 circuits with preacts: one circuit per thread.
        ("rpz", True, 3, 5, 2, 64),
        # 8x16 pz circuits without preacts: two circuits per backward thread,
        # so grad_x has fewer partial sums per sample than out has outputs.
        ("pz", False, 8, 16, 1, 600),
    ],
)
def test_large_tensors_are_split_across_launches(
    fake_flydsl, monkeypatch, ansatz, preacts, out_dim, in_dim, reps, limit
):
    # Shrink FlyDSL's int32 size limit so that 37 samples need several
    # launches while theta still fits in one.
    monkeypatch.setattr(ops, "_MAX_KERNEL_NUMEL", limit, raising=False)
    fake_flydsl.max_numel = limit
    x, theta, pw, pb = _params(ansatz, out_dim, in_dim, reps, batch=37)
    go = _grad_out(37, out_dim, in_dim)
    out = ops.flydsl_forward(x, theta, pw, pb, reps, ansatz, preacts)
    grad_x, grad_theta, grad_pw, grad_pb = ops.flydsl_backward(
        x, theta, pw, pb, go, reps, ansatz, preacts
    )
    assert torch.equal(out, _expected_out(x, out_dim))
    assert torch.equal(grad_x, go.sum(1))
    per_circuit = go.sum(0)
    assert torch.equal(grad_theta, per_circuit[..., None, None].expand_as(theta))
    if preacts:
        assert torch.equal(grad_pw, per_circuit[..., None].expand_as(pw))
        assert torch.equal(grad_pb, per_circuit[..., None].expand_as(pb))


def test_activation_splits_long_tensors_across_launches(fake_flydsl, monkeypatch):
    # Shrink FlyDSL's int32 size limit so that 100 elements need 3 launches.
    monkeypatch.setattr(ops, "_MAX_KERNEL_NUMEL", 40, raising=False)
    fake_flydsl.max_numel = 40
    x = torch.randn(10, 10, device="cuda", requires_grad=True)
    y = act.FlyDSLActivation("silu")(x)
    y.sum().backward()
    assert torch.equal(y, 2 * x)
    assert torch.equal(x.grad, torch.full_like(x, 2.0))
    assert fake_flydsl.launches == 6


def test_parameters_beyond_int32_sizes_are_rejected(fake_flydsl, monkeypatch):
    # theta cannot be split like the batch: 3x5 circuits x 3 layers = 45 > 40.
    monkeypatch.setattr(ops, "_MAX_KERNEL_NUMEL", 40, raising=False)
    fake_flydsl.max_numel = 40
    x, theta, pw, pb = _params("rpz")
    with pytest.raises(ValueError, match="elements"):
        ops.flydsl_forward(x, theta, pw, pb, 2, "rpz", True)
    assert fake_flydsl.launches == 0


def test_tensors_on_another_device_are_rejected(fake_flydsl):
    x, theta, pw, pb = _params("pz")
    go = _grad_out(7, 3, 5)
    calls = [
        lambda: ops.flydsl_forward(x, theta.cpu(), pw, pb, 2, "pz", False),
        # preacts are read only when trainable
        lambda: ops.flydsl_forward(x, theta, pw.cpu(), pb, 2, "pz", True),
        lambda: ops.flydsl_backward(x, theta, pw, pb, go.cpu(), 2, "pz", False),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="device"):
            call()
    assert fake_flydsl.launches == 0


def test_one_element_grad_output_reaches_flydsl_with_unit_stride(fake_flydsl):
    # The gradient of out.sum() for a single output is an expanded scalar,
    # with stride 0.
    x, theta, pw, pb = _params("pz", out_dim=1, in_dim=1, batch=1)
    go = torch.ones((), device="cuda").expand(1, 1, 1)
    grad_x, *_ = ops.flydsl_backward(x, theta, pw, pb, go, 2, "pz", False)
    assert torch.equal(grad_x, torch.ones(1, 1, device="cuda"))


def test_activation_one_element_grad_reaches_flydsl_with_unit_stride(fake_flydsl):
    x = torch.randn(1, device="cuda", requires_grad=True)
    act.FlyDSLActivation("silu")(x).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def test_gradient_penalty_through_the_solver_is_refused(fake_flydsl):
    # A gradient penalty needs second derivatives, which the kernels do not
    # provide. Rather than silently contribute nothing, it fails once
    # differentiated, including by autograd.grad for the parameters alone.
    x, theta, pw, pb = (t.requires_grad_() for t in _params("pz"))
    out = _solver_pz(x, theta, pw, pb)
    (grad_x,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    loss = out.pow(2).sum() + grad_x.pow(2).sum()
    with pytest.raises(RuntimeError, match="second derivatives"):
        torch.autograd.grad(loss, theta)


def test_gradient_penalty_through_the_activation_is_refused(fake_flydsl):
    x = torch.randn(4, device="cuda", requires_grad=True)
    y = act.FlyDSLActivation("silu")(x)
    (grad_x,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    with pytest.raises(RuntimeError, match="second derivatives"):
        grad_x.pow(2).sum().backward()


def test_create_graph_keeps_the_first_order_gradients(fake_flydsl):
    # e.g. an optimizer that differentiates its update step: create_graph=True
    # alone must neither refuse nor change the first-order gradients.
    x, theta, pw, pb = (t.requires_grad_() for t in _params("pz"))
    go = _grad_out(7, 3, 5)
    expected = torch.autograd.grad(_solver_pz(x, theta, pw, pb), (x, theta), go)
    actual = torch.autograd.grad(
        _solver_pz(x, theta, pw, pb), (x, theta), go, create_graph=True
    )
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    y = act.FlyDSLActivation("silu")(x)
    expected = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True)
    actual = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)
    assert torch.equal(actual[0], expected[0])


def test_gradients_downstream_of_the_solver_can_be_differentiated(fake_flydsl):
    # A penalty on another layer's gradient needs only the solver's first
    # derivatives, applied to the penalty's gradient as well as the loss's.
    x, theta, pw, pb = _params("pz")
    theta.requires_grad_()
    lin = torch.nn.Linear(5, 2, device="cuda")

    def loss_of(post):
        return lin(post).pow(2).sum()

    loss = loss_of(_solver_pz(x, theta, pw, pb))
    _, grad_w = torch.autograd.grad(loss, (theta, lin.weight), create_graph=True)
    (loss + grad_w.pow(2).sum()).backward()
    # The stand-in's theta gradient is the post gradient summed over the batch.
    post = _solver_pz(x, theta, pw, pb).detach().requires_grad_()
    loss = loss_of(post)
    (grad_w,) = torch.autograd.grad(loss, lin.weight, create_graph=True)
    (grad_post,) = torch.autograd.grad(loss + grad_w.pow(2).sum(), post)
    expected = grad_post.sum(0)[..., None, None].expand_as(theta)
    torch.testing.assert_close(theta.grad, expected)


@pytest.mark.parametrize("grad", [True, False])
def test_half_precision_model_runs(fake_flydsl, grad):
    # The kernels write float32 postacts, which a float16 layer has to cast:
    # the next layer's float16 weights take no float32 input.
    model = QKAN(
        [3, 4, 2],
        reps=2,
        device="cuda",
        solver="flydsl",
        c_dtype=torch.float16,
        p_dtype=torch.float16,
    )
    with torch.set_grad_enabled(grad):
        y = model(torch.randn(5, 3, device="cuda", dtype=torch.float16))
    assert y.dtype == torch.float16
    if grad:
        y.float().sum().backward()


def test_half_compute_keeps_float32_postacts(fake_flydsl):
    # c_dtype=float16 with float32 parameters: the float32 postacts the kernels
    # write must not take a round trip through float16, which would flush
    # these small gradients to zero on the way back.
    layer = QKANLayer(
        5,
        3,
        reps=2,
        device="cuda",
        solver="flydsl",
        c_dtype=torch.float16,
        p_dtype=torch.float32,
    )
    y = layer(torch.randn(7, 5, device="cuda"))
    (y * 1e-9).sum().backward()
    assert layer.theta.grad.abs().min() > 0


def test_compiles_for_the_device_arch(fake_flydsl):
    # The fixture reports the device as gfx90a, e.g. an MI210 on a host where
    # FlyDSL would otherwise fall back to gfx942.
    x, theta, pw, pb = _params("pz")
    ops.flydsl_forward(x, theta, pw, pb, 2, "pz", False)
    [(_, arch, constants)] = fake_flydsl.compiles
    assert arch == "gfx90a"
    assert "FLYDSL_GPU_ARCH" not in os.environ
    # FlyDSL files compiled code under the arch of the launcher's first
    # compile in the process, so the DEVICE and ARCH constants must tell
    # compiles for other devices and arches apart.
    assert constants[-2:] == (x.device.index, "gfx90a")


@pytest.mark.parametrize(
    "var,value", [("FLYDSL_GPU_ARCH", "gfx942"), ("HSA_OVERRIDE_GFX_VERSION", "9.4.2")]
)
def test_user_arch_override_is_kept(fake_flydsl, monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    x, theta, pw, pb = _params("pz")
    go = _grad_out(7, 3, 5)
    ops.flydsl_forward(x, theta, pw, pb, 2, "pz", False)
    ops.flydsl_backward(x, theta, pw, pb, go, 2, "pz", False)
    targets = [(arch, constants[-1]) for _, arch, constants in fake_flydsl.compiles]
    assert targets == [(value, value)] * 2
    assert os.environ[var] == value


def test_compile_without_launch_is_an_error(fake_flydsl):
    # FlyDSL's COMPILE_ONLY environment variable makes flyc.compile return None
    # without launching, which would leave the output uninitialized.
    fake_flydsl.compile_only = True
    x, theta, pw, pb = _params("pz")
    with pytest.raises(RuntimeError, match="COMPILE_ONLY"):
        ops.flydsl_forward(x, theta, pw, pb, 2, "pz", False)


def test_each_compile_time_configuration_compiles_once(fake_flydsl):
    # The stand-in, like FlyDSL, ignores the constants of later calls, so a
    # configuration that reused another's compile would fail its checks.
    x, theta, pw, pb = _params("pz")
    go = _grad_out(7, 3, 5)
    for _ in range(2):
        for preacts in (False, True):
            out = ops.flydsl_forward(x, theta, pw, pb, 2, "pz", preacts)
            ops.flydsl_backward(x, theta, pw, pb, go, 2, "pz", preacts)
            assert torch.equal(out, _expected_out(x, 3))
    assert len(fake_flydsl.compiles) == 4


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_each_device_compiles_and_launches_its_own_kernels(fake_flydsl):
    # Each device twice, so the later launches reuse a compiled launcher while
    # another device is current.
    for index in (0, 1, 0, 1):
        x, theta, pw, pb = _params("pz", device=f"cuda:{index}")
        out = ops.flydsl_forward(x, theta, pw, pb, 2, "pz", False)
        assert torch.equal(out, _expected_out(x, 3))
    assert [device for device, _, _ in fake_flydsl.compiles] == [0, 1]


@pytest.mark.skipif(torch.version.hip is not None, reason="ROCm builds use FlyDSL")
def test_non_rocm_torch_never_imports_flydsl():
    # FlyDSL loads large MLIR/LLVM bindings and installs a sys.excepthook.
    code = """
import importlib.abc
import sys


class NoFlyDSL(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "flydsl":
            raise AssertionError("imported " + name)


sys.meta_path.insert(0, NoFlyDSL())
import qkan

print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(SRC)),
        timeout=300,
    )
    assert "OK" in result.stdout, result.stderr
