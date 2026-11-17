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

"""solver="cute" and solver="cutile" refuse what they have no kernel for.

An ansatz outside pz / rpz / real used to run torch_exact_solver, and a
c_dtype outside float32 (complex64) / bfloat16 / float8_e4m3fn the float32
kernels. The checks are host-side Python, so the CuTe ones also run against
a recording stand-in for qkan._C on machines without the extension; the
tests that launch kernels are skipped where the backend is missing.
"""

import re

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan.solver import (  # noqa: E402
    _CUTE_AVAILABLE,
    _CUTILE_AVAILABLE,
    cute_exact_solver,
    cutile_flash_exact_solver,
    get_solver,
    torch_exact_solver,
)
from qkan.solver.cute import cute as cute_mod  # noqa: E402
from qkan.solver.cute import cute_ops  # noqa: E402

SOLVERS = ["cute", "cutile"]
SOLVER_FNS = {"cute": cute_exact_solver, "cutile": cutile_flash_exact_solver}
# Launching kernels needs the backend; the checks alone do not.
BACKENDS = [
    pytest.param(
        "cute",
        marks=pytest.mark.skipif(not _CUTE_AVAILABLE, reason="no CuTe extension"),
    ),
    pytest.param(
        "cutile",
        marks=pytest.mark.skipif(not _CUTILE_AVAILABLE, reason="no cuda-tile"),
    ),
]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_cutile = pytest.mark.skipif(
    not (_CUTILE_AVAILABLE and torch.cuda.is_available()),
    reason="needs cuda-tile and CUDA",
)

ANSATZES = ["pz_encoding", "pz", "rpz_encoding", "rpz", "real"]
OTHER_ANSATZES = [
    "px_encoding",
    "px",
    "mix",
    pytest.param(lambda *args, **kwargs: None, id="callable"),
]
C_DTYPES = [torch.complex64, torch.float32, torch.bfloat16, torch.float8_e4m3fn]
OTHER_C_DTYPES = [torch.complex128, torch.float64, torch.float16, torch.int32]

BATCH, OUT_DIM, IN_DIM, REPS = 8, 2, 3, 2


def no_kernel_for(solver):
    return f"solver='{solver}' has no kernel for ansatz"


def no_support_for(solver, c_dtype):
    return re.escape(
        f"solver='{solver}' computes in float32 and does not support c_dtype={c_dtype}:"
    )


def solver_inputs(ansatz="pz_encoding", device="cuda"):
    """x, theta, preacts_weight, preacts_bias as the layer hands them over."""
    torch.manual_seed(0)
    last = 2 if ansatz in ("pz_encoding", "pz") else 1
    axis = REPS if ansatz == "real" else REPS + 1
    x = torch.rand(BATCH, IN_DIM, device=device)
    theta = torch.rand(OUT_DIM, IN_DIM, axis, last, device=device)
    pw = torch.rand(OUT_DIM, IN_DIM, REPS, device=device) + 0.5
    pb = torch.rand(OUT_DIM, IN_DIM, REPS, device=device)
    return x, theta, pw, pb


def run_solver(fn, ansatz, **kwargs):
    """Forward and backward of a solver function; returns the output and
    the gradients of x and theta."""
    x, theta, pw, pb = solver_inputs(ansatz)
    x.requires_grad_()
    theta.requires_grad_()
    out = fn(
        x,
        theta,
        pw,
        pb,
        REPS,
        ansatz=ansatz,
        out_dim=OUT_DIM,
        preacts_trainable=True,
        **kwargs,
    )
    out.float().sum().backward()
    return out.detach(), x.grad, theta.grad


# ---------------------------------------------------------------------------
# QKANSolver.check(): what a layer asks when it is built
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ansatz", OTHER_ANSATZES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_check_refuses_other_ansatzes(solver, ansatz):
    with pytest.raises(ValueError, match=no_kernel_for(solver)):
        get_solver(solver).check(ansatz, torch.complex64, REPS)


@pytest.mark.parametrize("c_dtype", OTHER_C_DTYPES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_check_refuses_other_c_dtypes(solver, c_dtype):
    with pytest.raises(ValueError, match=no_support_for(solver, c_dtype)):
        get_solver(solver).check("pz_encoding", c_dtype, REPS)


@pytest.mark.parametrize("solver", SOLVERS)
def test_check_accepts_what_the_kernels_cover(solver):
    for ansatz in ANSATZES:
        for c_dtype in C_DTYPES:
            get_solver(solver).check(ansatz, c_dtype, REPS)


# ---------------------------------------------------------------------------
# CuTe host side, against a stand-in for the extension
# ---------------------------------------------------------------------------


class RecordingExt:
    """Stand-in for ``qkan._C``: records what each launcher is handed and
    returns tensors of the dtype the real launcher allocates."""

    def __init__(self):
        self.launches = []

    def _forward(self, name, x, theta, use_bf16):
        self.launches.append((name, x.dtype, use_bf16))
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        return torch.zeros(x.shape[0], theta.shape[0], x.shape[1], dtype=dtype)

    def _backward(self, name, x, theta, pw, pb, state_bits):
        self.launches.append((name, x.dtype, state_bits))
        return [torch.zeros(t.shape) for t in (x, theta, pw, pb)]

    def pz_forward(self, x, theta, pw, pb, trainable, fast, use_bf16):
        return self._forward("pz_forward", x, theta, use_bf16)

    def rpz_forward(self, x, theta, pw, pb, fast, use_bf16):
        return self._forward("rpz_forward", x, theta, use_bf16)

    def real_forward(self, x, theta, pw, pb, trainable, fast, compute_bf16, use_bf16):
        return self._forward("real_forward", x, theta, use_bf16)

    def pz_backward(self, x, theta, pw, pb, grad, trainable, fast, state_bits):
        return self._backward("pz_backward", x, theta, pw, pb, state_bits)

    def rpz_backward(self, x, theta, pw, pb, grad, fast, state_bits):
        return self._backward("rpz_backward", x, theta, pw, pb, state_bits)

    def real_backward(
        self, x, theta, pw, pb, grad, trainable, fast, compute_bf16, state_bits
    ):
        return self._backward("real_backward", x, theta, pw, pb, state_bits)


@pytest.fixture
def cute_ext(monkeypatch):
    ext = RecordingExt()
    monkeypatch.setattr(cute_mod, "_CUTE_AVAILABLE", True)
    monkeypatch.setattr(cute_ops, "_get_ext", lambda: ext)
    return ext


@pytest.mark.parametrize("ansatz", OTHER_ANSATZES)
def test_cute_solver_refuses_other_ansatzes(cute_ext, ansatz):
    inputs = solver_inputs(device="cpu")
    with pytest.raises(ValueError, match=no_kernel_for("cute")):
        cute_exact_solver(*inputs, REPS, ansatz=ansatz, out_dim=OUT_DIM)
    assert cute_ext.launches == []


@pytest.mark.parametrize("needs_grad", [False, True])
@pytest.mark.parametrize("c_dtype", OTHER_C_DTYPES)
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "real"])
def test_cute_solver_refuses_other_c_dtypes(cute_ext, ansatz, c_dtype, needs_grad):
    x, theta, pw, pb = solver_inputs(ansatz, device="cpu")
    theta.requires_grad_(needs_grad)
    with pytest.raises(ValueError, match=no_support_for("cute", c_dtype)):
        cute_exact_solver(
            x, theta, pw, pb, REPS, ansatz=ansatz, out_dim=OUT_DIM, dtype=c_dtype
        )
    assert cute_ext.launches == []


CUTE_WRAPPER_ARGS = {
    # the flags each kernel wrapper takes after the tensors
    "cute_pz_forward": (True, True),
    "cute_rpz_forward": (True,),
    "cute_real_forward": (True, True),
    "cute_pz_backward": (True, True),
    "cute_rpz_backward": (True,),
    "cute_real_backward": (True, True),
}


@pytest.mark.parametrize("c_dtype", OTHER_C_DTYPES)
@pytest.mark.parametrize("wrapper", list(CUTE_WRAPPER_ARGS))
def test_cute_kernel_wrappers_refuse_other_c_dtypes(cute_ext, wrapper, c_dtype):
    """Called directly, they took the float32 launch for any unknown dtype."""
    tensors = solver_inputs(device="cpu")
    if wrapper.endswith("backward"):
        tensors += (torch.zeros(BATCH, OUT_DIM, IN_DIM),)  # grad_output
    with pytest.raises(ValueError, match=no_support_for("cute", c_dtype)):
        getattr(cute_ops, wrapper)(
            *tensors, *CUTE_WRAPPER_ARGS[wrapper], c_dtype=c_dtype
        )
    assert cute_ext.launches == []


@pytest.mark.parametrize(
    "c_dtype, use_bf16, state_bits, io_dtype",
    [
        (torch.complex64, False, 32, torch.float32),
        (torch.float32, False, 32, torch.float32),
        (torch.bfloat16, True, 16, torch.bfloat16),
        (torch.float8_e4m3fn, True, 8, torch.bfloat16),
    ],
)
def test_cute_c_dtype_selects_the_kernel_path(
    cute_ext, c_dtype, use_bf16, state_bits, io_dtype
):
    """complex64 is float32: both take the float32 launch and checkpoints."""
    x, theta, pw, pb = solver_inputs(device="cpu")
    theta.requires_grad_()
    out = cute_exact_solver(x, theta, pw, pb, REPS, out_dim=OUT_DIM, dtype=c_dtype)
    out.float().sum().backward()
    assert cute_ext.launches == [
        ("pz_forward", torch.float32, use_bf16),
        ("pz_backward", io_dtype, state_bits),
    ]


# ---------------------------------------------------------------------------
# Solver functions on the real kernels
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("ansatz", OTHER_ANSATZES)
@pytest.mark.parametrize("solver", BACKENDS)
def test_solver_refuses_other_ansatzes(solver, ansatz):
    inputs = solver_inputs()
    with pytest.raises(ValueError, match=no_kernel_for(solver)):
        SOLVER_FNS[solver](*inputs, REPS, ansatz=ansatz, out_dim=OUT_DIM)


@needs_cuda
@pytest.mark.parametrize("c_dtype", OTHER_C_DTYPES)
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "real"])
@pytest.mark.parametrize("solver", BACKENDS)
def test_solver_refuses_other_c_dtypes(solver, ansatz, c_dtype):
    """float64 came back as the float32 result in a float64 tensor."""
    inputs = solver_inputs(ansatz)
    with pytest.raises(ValueError, match=no_support_for(solver, c_dtype)):
        SOLVER_FNS[solver](*inputs, REPS, ansatz=ansatz, out_dim=OUT_DIM, dtype=c_dtype)


@needs_cuda
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "real"])
@pytest.mark.parametrize("solver", BACKENDS)
def test_complex64_is_float32(solver, ansatz):
    """complex64 is the solver functions' own default dtype; cuTile used to
    hand it to the kernels, which have no complex arrays."""
    fn = SOLVER_FNS[solver]
    default = run_solver(fn, ansatz)
    for c_dtype in (torch.complex64, torch.float32):
        for got, want in zip(run_solver(fn, ansatz, dtype=c_dtype), default):
            assert got.dtype == torch.float32
            assert torch.equal(got, want)
    exact = run_solver(torch_exact_solver, ansatz, dtype=torch.complex64)
    torch.testing.assert_close(default, exact, rtol=1e-4, atol=1e-5)


@needs_cuda
@pytest.mark.parametrize("c_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "real"])
@pytest.mark.parametrize("solver", BACKENDS)
def test_reduced_precision_c_dtypes_still_run(solver, ansatz, c_dtype):
    fn = SOLVER_FNS[solver]
    out, grad_x, grad_theta = run_solver(fn, ansatz, dtype=c_dtype)
    ref, ref_grad_x, ref_grad_theta = run_solver(fn, ansatz, dtype=torch.float32)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=2e-2)
    for grad, grad_ref in ((grad_x, ref_grad_x), (grad_theta, ref_grad_theta)):
        assert (grad - grad_ref).abs().max() <= 0.1 * grad_ref.abs().max()


# ---------------------------------------------------------------------------
# CuTileActivation
# ---------------------------------------------------------------------------


@needs_cutile
@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_cutile_activation_refuses_other_dtypes(dtype):
    """They were converted to bfloat16 and came back as bfloat16."""
    from qkan.solver.cutile.cutile_activation import CuTileActivation
    from qkan.solver.cutile.cutile_activations import cutile_activation_backward

    x = torch.linspace(-3, 3, 9, device="cuda").to(dtype)
    match = re.escape(
        f"CuTileActivation supports float32, bfloat16 and float16, got {dtype}"
    )
    with pytest.raises(ValueError, match=match):
        CuTileActivation("silu")(x)
    if dtype.is_floating_point:
        with pytest.raises(ValueError, match=match):
            CuTileActivation("silu")(x.clone().requires_grad_())
    with pytest.raises(ValueError, match=match):
        cutile_activation_backward(x, x, "silu", dtype)


@needs_cutile
@pytest.mark.parametrize(
    "dtype, atol",
    [(torch.float32, 1e-6), (torch.bfloat16, 2e-2), (torch.float16, 2e-3)],
    ids=["float32", "bfloat16", "float16"],
)
def test_cutile_activation_keeps_its_dtypes(dtype, atol):
    """float16 runs the float32 kernel: it was rounded through bfloat16."""
    from qkan.solver.cutile.cutile_activation import CuTileActivation
    from qkan.solver.cutile.cutile_activations import cutile_activation_backward

    x = torch.linspace(-3, 3, 9, device="cuda", dtype=dtype, requires_grad=True)
    y = CuTileActivation("silu")(x)
    y.float().sum().backward()
    assert y.dtype == dtype and x.grad.dtype == dtype
    with torch.no_grad():
        assert CuTileActivation("silu")(x).dtype == dtype
        empty = CuTileActivation("silu")(x[:0])
        assert empty.dtype == dtype and empty.shape == (0,)
    grad = cutile_activation_backward(torch.ones_like(x), x.detach(), "silu", dtype)
    assert grad.dtype == dtype and torch.equal(grad, x.grad)

    x_ref = x.detach().double().requires_grad_()
    y_ref = torch.nn.functional.silu(x_ref)
    y_ref.sum().backward()
    torch.testing.assert_close(y.double(), y_ref, rtol=0, atol=atol)
    torch.testing.assert_close(x.grad.double(), x_ref.grad, rtol=0, atol=atol)


@needs_cutile
@pytest.mark.parametrize("ba_trainable", [False, True])
def test_cutile_model_under_default_autocast(ba_trainable):
    """torch.autocast("cuda") is float16: the hidden activations reach the
    base activation of the next layer as float16."""
    from qkan import QKAN

    torch.manual_seed(0)
    model = QKAN([4, 5, 3], solver="cutile", device="cuda", ba_trainable=ba_trainable)
    reference = QKAN(
        [4, 5, 3],
        solver="cutile",
        device="cuda",
        ba_trainable=ba_trainable,
        base_activation=torch.nn.SiLU(),
    )
    reference.load_state_dict(model.state_dict())
    x = torch.rand(BATCH, 4, device="cuda")
    with torch.autocast("cuda"):
        y, y_ref = model(x), reference(x)
    assert y.dtype == torch.float16
    # a bfloat16 round trip of the activation is 8e-3 away with ba_trainable
    torch.testing.assert_close(y.float(), y_ref.float(), rtol=0, atol=1e-3)
