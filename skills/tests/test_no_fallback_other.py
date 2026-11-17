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

"""solver="exact", "cutn", "flydsl" and "mlx" refuse what they cannot run.

The exact and cutn solvers used to simulate a complex-amplitude ansatz in a
real c_dtype by dropping the imaginary parts, and cutn, flydsl and mlx used to
hand an ansatz, depth or precision outside their kernels to torch_exact_solver
or to their float32 kernels. The checks are host-side Python, so everything
here runs on the CPU: where a test gets past them, FlyDSL's launches are
replaced by recorders, and MLX is only reported as available.
"""

import re

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan import QKANLayer  # noqa: E402
from qkan.solver import (  # noqa: E402
    cutn_solver,
    flydsl_exact_solver,
    get_solver,
    mlx_contracted_solver,
    mlx_exact_solver,
    torch_exact_solver,
)
from qkan.solver.flydsl import flydsl as flydsl_mod  # noqa: E402
from qkan.solver.flydsl import flydsl_activation as flydsl_act  # noqa: E402
from qkan.solver.flydsl import flydsl_ops  # noqa: E402
from qkan.solver.mlx import mlx as mlx_mod  # noqa: E402

SOLVER_FNS = {"exact": torch_exact_solver, "cutn": cutn_solver}
CALLABLE = pytest.param(lambda *args, **kwargs: None, id="callable")
PZ_RPZ = ["pz_encoding", "pz", "rpz_encoding", "rpz"]
# (solver, ansatz) pairs that simulate complex amplitudes.
COMPLEX_CASES = [("exact", a) for a in PZ_RPZ + ["px_encoding", "px", "mix"]]
COMPLEX_CASES += [("cutn", a) for a in PZ_RPZ]
REAL_TWINS = [(torch.float32, torch.complex64), (torch.float64, torch.complex128)]
MIXED_DTYPES = [torch.bfloat16, torch.float16, torch.float8_e4m3fn]
DOUBLE_DTYPES = [torch.complex128, torch.float64]
# What the float32 kernels of flydsl and mlx keep accepting.
SINGLE_DTYPES = [
    torch.complex64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
]

BATCH, OUT_DIM, IN_DIM = 5, 2, 3


def no_kernel_for(solver):
    return f"solver='{solver}' has no kernel for ansatz"


def no_support_for(solver, c_dtype):
    return re.escape(
        f"solver='{solver}' computes in float32 and does not support c_dtype={c_dtype}:"
    )


def solver_inputs(ansatz, reps=2):
    """x, theta, preacts_weight, preacts_bias as the layer hands them over."""
    gen = torch.Generator().manual_seed(0)
    last = 2 if ansatz in ("pz_encoding", "pz", "mix") else 1
    axis = reps if ansatz == "real" else reps + 1
    # Small values keep the acos of the px encoding inside its domain.
    x = torch.rand(BATCH, IN_DIM, generator=gen) * 0.5
    theta = torch.randn(OUT_DIM, IN_DIM, axis, last, generator=gen)
    pw = torch.rand(OUT_DIM, IN_DIM, reps, generator=gen) * 0.5
    pb = torch.rand(OUT_DIM, IN_DIM, reps, generator=gen) * 0.5
    return x, theta, pw, pb


def run(fn, ansatz, reps=2, **kwargs):
    return fn(
        *solver_inputs(ansatz, reps),
        reps,
        ansatz=ansatz,
        preacts_trainable=True,
        out_dim=OUT_DIM,
        **kwargs,
    )


def run_contracted(ansatz, **kwargs):
    """mlx_contracted_solver with all-ones postact weights."""
    x, theta, pw, pb = solver_inputs(ansatz)
    weight = torch.ones(OUT_DIM, IN_DIM)
    return mlx_contracted_solver(
        x, theta, pw, pb, weight, 2, ansatz=ansatz, out_dim=OUT_DIM, **kwargs
    )


# ---------------------------------------------------------------------------
# exact and cutn: a real c_dtype on complex amplitudes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver,ansatz", COMPLEX_CASES)
@pytest.mark.parametrize("real,cplx", REAL_TWINS)
def test_real_c_dtype_names_the_complex_one(solver, ansatz, real, cplx):
    """float32 / float64 used to cast the Rz / Rx gates to real, which drops
    their imaginary parts (pz, rpz) or fails inside torch (px, mix)."""
    fn = SOLVER_FNS[solver]
    get_solver(solver).check(ansatz, real, 2)
    assert torch.equal(run(fn, ansatz, dtype=real), run(fn, ansatz, dtype=cplx))


@pytest.mark.parametrize("solver,ansatz", COMPLEX_CASES)
@pytest.mark.parametrize("c_dtype", MIXED_DTYPES)
def test_mixed_precision_on_complex_amplitudes_is_refused(solver, ansatz, c_dtype):
    match = re.escape(
        f"solver='{solver}' simulates ansatz {ansatz!r} with complex amplitudes "
        f"and does not support c_dtype={c_dtype}:"
    )
    for call in (
        lambda: get_solver(solver).check(ansatz, c_dtype, 2),
        lambda: run(SOLVER_FNS[solver], ansatz, dtype=c_dtype),
    ):
        with pytest.raises(ValueError, match=match) as error:
            call()
        # Where mixed precision is implemented, and for which ansatzes.
        assert "solver='flash', 'cute' and 'cutile' for the pz, rpz" in str(error.value)


@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding"])
def test_python_types_as_c_dtype(ansatz):
    """torch takes float and complex for float64 and complex128."""
    reference = run(torch_exact_solver, ansatz, dtype=torch.complex128)
    for c_dtype in (float, complex):
        get_solver("exact").check(ansatz, c_dtype, 2)
        assert torch.equal(run(torch_exact_solver, ansatz, dtype=c_dtype), reference)
    for c_dtype in (int, bool, torch.int64):  # no complex amplitudes in those
        with pytest.raises(ValueError, match="with complex amplitudes"):
            get_solver("exact").check(ansatz, c_dtype, 2)


@pytest.mark.parametrize("ansatz", ["px_encoding", "px"])
def test_px_needs_trainable_preacts(ansatz):
    """The circuit read an encoding that frozen pre-activations never built:
    NameError."""
    with pytest.raises(ValueError, match="preact_trainable=True"):
        torch_exact_solver(*solver_inputs(ansatz), 2, ansatz=ansatz, out_dim=OUT_DIM)
    assert run(torch_exact_solver, ansatz).isfinite().all()


@pytest.mark.parametrize("solver", ["exact", "cutn"])
def test_real_ansatz_keeps_a_real_c_dtype(solver):
    """Its amplitudes are real, so bfloat16 (the layer's default for it) is
    what it is simulated in."""
    for c_dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn):
        get_solver(solver).check("real", c_dtype, 2)
    out = run(SOLVER_FNS[solver], "real", dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16


def test_exact_leaves_the_dtype_of_a_callable_ansatz_alone():
    for c_dtype in (torch.float32, torch.bfloat16):
        get_solver("exact").check(lambda theta: theta, c_dtype, 2)


@pytest.mark.parametrize("solver", ["exact", "cutn"])
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding"])
def test_layer_float32_c_dtype_matches_complex64(solver, ansatz):
    """QKANLayer documents c_dtype=torch.float32 as full-precision f32."""
    x = torch.rand(BATCH, IN_DIM)
    kwargs = dict(solver=solver, ansatz=ansatz)
    ref = QKANLayer(IN_DIM, OUT_DIM, c_dtype=torch.complex64, **kwargs)
    layer = QKANLayer(IN_DIM, OUT_DIM, c_dtype=torch.float32, **kwargs)
    layer.load_state_dict(ref.state_dict())
    assert torch.equal(layer(x), ref(x))


# ---------------------------------------------------------------------------
# cutn: ansatzes and depths outside its network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cutn", "tn"])
@pytest.mark.parametrize("ansatz", ["px_encoding", "px", "mix", CALLABLE])
def test_cutn_refuses_other_ansatzes(name, ansatz):
    for call in (
        lambda: get_solver(name).check(ansatz, torch.complex64, 2),
        lambda: run(get_solver(name), ansatz),
        lambda: run(cutn_solver, ansatz),
    ):
        with pytest.raises(ValueError, match=no_kernel_for("cutn")):
            call()


@pytest.mark.parametrize("ansatz,max_reps", [("pz", 11), ("rpz", 11), ("real", 12)])
def test_cutn_reps_limit(ansatz, max_reps):
    """The einsum has 26 chain letters: the deepest circuit they index still
    contracts, and one more rep is refused."""
    get_solver("cutn").check(ansatz, torch.complex64, max_reps)
    out = run(cutn_solver, ansatz, reps=max_reps)
    expected = run(torch_exact_solver, ansatz, reps=max_reps)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-5)

    match = re.escape(
        f"solver='cutn' supports reps <= {max_reps} for ansatz {ansatz!r}"
    )
    for call in (
        lambda: get_solver("cutn").check(ansatz, torch.complex64, max_reps + 1),
        lambda: run(cutn_solver, ansatz, reps=max_reps + 1),
    ):
        with pytest.raises(ValueError, match=match):
            call()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(solver="cutn", ansatz="mix"), no_kernel_for("cutn")),
        (dict(solver="tn", ansatz="px", preact_trainable=True), no_kernel_for("cutn")),
        (dict(solver="cutn", reps=12), "reps <= 11"),
        (dict(solver="exact", c_dtype=torch.bfloat16), "does not support c_dtype"),
    ],
    ids=["cutn-mix", "tn-px", "cutn-reps", "exact-bf16"],
)
def test_layer_refuses(kwargs, match):
    """Whether the layer is refused when built or on its first forward."""
    with pytest.raises(ValueError, match=match):
        QKANLayer(IN_DIM, OUT_DIM, **kwargs)(torch.rand(BATCH, IN_DIM) * 0.5)


# ---------------------------------------------------------------------------
# flydsl
# ---------------------------------------------------------------------------


@pytest.fixture
def flydsl_launches(monkeypatch):
    """Report FlyDSL as available and record the c_dtype of every forward
    launch instead of running it."""
    launches = []

    def forward(x, theta, pw, pb, reps, ansatz, preacts, fast, c_dtype):
        launches.append(c_dtype)
        return torch.zeros(x.shape[0], theta.shape[0], x.shape[1])

    monkeypatch.setattr(flydsl_mod, "_FLYDSL_AVAILABLE", True)
    monkeypatch.setattr(flydsl_mod, "flydsl_forward", forward, raising=False)
    return launches


@pytest.mark.parametrize("ansatz", ["px_encoding", "px", "mix", CALLABLE])
def test_flydsl_refuses_other_ansatzes(flydsl_launches, ansatz):
    for call in (
        lambda: get_solver("flydsl").check(ansatz, torch.complex64, 2),
        lambda: run(flydsl_exact_solver, ansatz),
    ):
        with pytest.raises(ValueError, match=no_kernel_for("flydsl")):
            call()


@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz", "real"])
@pytest.mark.parametrize("c_dtype", DOUBLE_DTYPES)
def test_flydsl_refuses_double_precision(flydsl_launches, ansatz, c_dtype):
    """It used to launch the float32 kernels."""
    for call in (
        lambda: get_solver("flydsl").check(ansatz, c_dtype, 2),
        lambda: run(flydsl_exact_solver, ansatz, dtype=c_dtype),
    ):
        with pytest.raises(ValueError, match=no_support_for("flydsl", c_dtype)):
            call()
    assert not flydsl_launches


@pytest.mark.parametrize("c_dtype", SINGLE_DTYPES)
def test_flydsl_keeps_its_other_c_dtypes(flydsl_launches, c_dtype):
    """complex64 and float16 select float32 postacts, fp8 the bf16 ones."""
    get_solver("flydsl").check("pz", c_dtype, 2)
    with torch.no_grad():
        run(flydsl_exact_solver, "pz", dtype=c_dtype)
    bf16_out = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert [flydsl_ops._use_bf16(seen) for seen in flydsl_launches] == [bf16_out]


def test_flydsl_activation_refuses_float64(monkeypatch):
    """It used to compute in float32 and label the result float64."""
    match = "FlyDSLActivation computes in float32 and does not support a torch.float64"
    x = torch.randn(4, dtype=torch.float64)
    with pytest.raises(ValueError, match=match):
        flydsl_act.flydsl_activation_forward(x, "silu")
    with pytest.raises(ValueError, match=match):
        flydsl_act.flydsl_activation_backward(torch.ones_like(x), x, "silu")

    monkeypatch.setattr(flydsl_act, "_FLYDSL_KERNELS_AVAILABLE", True)
    module = flydsl_act.FlyDSLActivation("silu")
    for requires_grad in (False, True):
        with pytest.raises(ValueError, match=match):
            module(x.clone().requires_grad_(requires_grad))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_flydsl_activation_keeps_its_other_dtypes(monkeypatch, dtype):
    """float16 is widened to the float32 kernel and cast back; bf16 is read
    as it is."""
    launches = []

    def launch(launcher, kind, bf16, device, x, y):
        launches.append((bf16, x.dtype))
        y.copy_(x * 2)

    monkeypatch.setattr(flydsl_act, "_launch_act_forward", None, raising=False)
    monkeypatch.setattr(flydsl_act, "_launch_pointwise", launch)
    x = torch.randn(4).to(dtype)
    y = flydsl_act.flydsl_activation_forward(x, "silu")
    assert y.dtype == dtype
    assert torch.equal(y, x * 2)
    is_bf16 = dtype == torch.bfloat16
    assert launches == [(is_bf16, torch.bfloat16 if is_bf16 else torch.float32)]


# ---------------------------------------------------------------------------
# mlx
# ---------------------------------------------------------------------------


@pytest.fixture
def mlx_kernels(monkeypatch):
    """Report the Metal kernels as available. Nothing here gets as far as
    launching one."""
    monkeypatch.setattr(mlx_mod, "_require_kernels", lambda: None)


@pytest.mark.parametrize("ansatz", ["px_encoding", "px", CALLABLE])
def test_mlx_refuses_other_ansatzes(mlx_kernels, ansatz):
    for call in (
        lambda: get_solver("mlx").check(ansatz, torch.complex64, 2),
        lambda: run(mlx_exact_solver, ansatz),
    ):
        with pytest.raises(ValueError, match=no_kernel_for("mlx")):
            call()
    # The contracted solver declines, and the layer then asks mlx_exact_solver.
    assert run_contracted(ansatz) is None


@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz", "real", "mix"])
@pytest.mark.parametrize("c_dtype", DOUBLE_DTYPES)
def test_mlx_refuses_double_precision(mlx_kernels, ansatz, c_dtype):
    """It used to run torch_exact_solver in double precision."""
    for call in (
        lambda: get_solver("mlx").check(ansatz, c_dtype, 2),
        lambda: run(mlx_exact_solver, ansatz, dtype=c_dtype),
        lambda: run_contracted(ansatz, dtype=c_dtype),
    ):
        with pytest.raises(ValueError, match=no_support_for("mlx", c_dtype)) as error:
            call()
        assert "Metal has no float64" in str(error.value)


@pytest.mark.parametrize(
    "ansatz", ["pz_encoding", "pz", "rpz_encoding", "rpz", "real", "mix"]
)
@pytest.mark.parametrize("c_dtype", SINGLE_DTYPES)
def test_mlx_check_accepts_what_the_kernels_run(ansatz, c_dtype):
    get_solver("mlx").check(ansatz, c_dtype, 2)


@pytest.mark.parametrize("c_dtype", [torch.int32, torch.bool, torch.complex32])
@pytest.mark.parametrize("solver", ["flydsl", "mlx"])
def test_dtypes_without_a_kernel_path_are_refused(solver, c_dtype):
    """Not only double precision: the check lists what the kernels take."""
    with pytest.raises(ValueError, match=no_support_for(solver, c_dtype)) as error:
        get_solver(solver).check("pz_encoding", c_dtype, 2)
    assert "double precision" not in str(error.value)
