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

"""QKANLayer and QKAN refuse, when they are built, what cannot run as asked.

A fused solver given an ansatz outside its kernels used to keep complex64 for
torch_exact_solver, and a double-precision c_dtype was rewritten to float64
and then computed in float32. solver="qml" ran its pz circuit whatever the
ansatz and the pre-activations, QKAN dropped unknown keyword arguments, a
backend activation that could not be built was replaced by the torch.nn one,
and the fused epilogue computed float64 parameters in float32. The checks at
construction are host-side Python and run on the CPU without any backend;
only the epilogue tests need Triton and CUDA.
"""

import importlib
import re

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from qkan import QKAN, QKANLayer  # noqa: E402
from qkan.qkan_epilogue import (  # noqa: E402
    _TRITON_AVAILABLE,
    QKANEpilogue,
    qkan_epilogue_forward,
)
from qkan.solver import (  # noqa: E402
    _CUTE_AVAILABLE,
    _CUTILE_AVAILABLE,
    _FLASH_AVAILABLE,
    _FLYDSL_AVAILABLE,
    get_registry,
    make_base_activation,
)

# Registered solvers with kernels for pz / rpz / real only, in float32.
FUSED = ["flash", "cute", "cutile", "flydsl"]
# Where each one's activation kernel lives, the flag that reports it as
# built, and the ImportError it raises when it is not.
BACKEND_ACTIVATIONS = {
    "flash": (
        "qkan.solver.flash.triton_activation",
        "_TRITON_ACT_AVAILABLE",
        "Triton is required for TritonActivation",
    ),
    "cute": (
        "qkan.solver.cute.cute_activation",
        "_CUTE_ACTIVATION_AVAILABLE",
        "CuTe activation kernels not available",
    ),
    "cutile": (
        "qkan.solver.cutile.cutile_activation",
        "_CUTILE_ACTIVATION_AVAILABLE",
        "cuda.tile is required for CuTileActivation",
    ),
    "flydsl": (
        "qkan.solver.flydsl.flydsl_activation",
        "_FLYDSL_KERNELS_AVAILABLE",
        "FlyDSL activation kernels need the flydsl package",
    ),
}
INSTALLED = [
    pytest.param(name, marks=pytest.mark.skipif(not available, reason=f"no {name}"))
    for name, available in (
        ("flash", _FLASH_AVAILABLE),
        ("cute", _CUTE_AVAILABLE),
        ("cutile", _CUTILE_AVAILABLE),
        ("flydsl", _FLYDSL_AVAILABLE),
    )
]
needs_gpu = pytest.mark.skipif(
    not (_TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="needs triton and CUDA",
)

IN_DIM, OUT_DIM, REPS = 3, 2, 2
CUSTOM_THETA = [OUT_DIM, IN_DIM, REPS + 1, 2]


def custom_ansatz(*args, **kwargs):
    return None


def build(**kwargs):
    """A layer with a torch.nn base activation: building it needs no backend,
    so only the configuration decides whether it is refused."""
    kwargs.setdefault("reps", REPS)
    kwargs.setdefault("base_activation", nn.SiLU())
    return QKANLayer(IN_DIM, OUT_DIM, **kwargs)


def no_kernel_for(solver, ansatz):
    return re.escape(f"solver='{solver}' has no kernel for ansatz {ansatz!r}:")


def no_support_for(solver, c_dtype):
    return re.escape(
        f"solver='{solver}' computes in float32 and does not support c_dtype={c_dtype}:"
    )


# ---------------------------------------------------------------------------
# Registered solvers: check() runs when the layer is built
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ansatz", ["mix", "px", "px_encoding"])
@pytest.mark.parametrize("solver", FUSED + ["cutn", "tn"])
def test_layer_refuses_an_ansatz_without_kernel(solver, ansatz):
    name = "cutn" if solver == "tn" else solver
    with pytest.raises(ValueError, match=no_kernel_for(name, ansatz)):
        build(solver=solver, ansatz=ansatz)


@pytest.mark.parametrize("ansatz", ["px", "px_encoding"])
def test_mlx_layer_refuses_an_ansatz_without_kernel(ansatz):
    with pytest.raises(ValueError, match=no_kernel_for("mlx", ansatz)):
        build(solver="mlx", ansatz=ansatz)


@pytest.mark.parametrize("solver", FUSED + ["mlx", "cutn"])
def test_layer_refuses_a_custom_ansatz_on_a_kernel_solver(solver):
    with pytest.raises(ValueError, match=f"solver='{solver}' has no kernel for ansatz"):
        build(solver=solver, ansatz=custom_ansatz, theta_size=CUSTOM_THETA)


@pytest.mark.parametrize("c_dtype", [torch.complex128, torch.float64])
@pytest.mark.parametrize("ansatz", ["pz", "rpz", "real"])
@pytest.mark.parametrize("solver", FUSED + ["mlx"])
def test_layer_refuses_double_precision_on_a_float32_solver(solver, ansatz, c_dtype):
    """The message names the c_dtype that was asked for: complex128 used to
    reach the solver as float64."""
    with pytest.raises(ValueError, match=no_support_for(solver, c_dtype)) as error:
        build(solver=solver, ansatz=ansatz, c_dtype=c_dtype)
    assert "Use solver='exact' for double precision." in str(error.value)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            dict(solver="flash", ansatz="real", c_dtype=torch.float16),
            no_support_for("flash", torch.float16),
        ),
        (
            dict(solver="cute", c_dtype=torch.float16),
            no_support_for("cute", torch.float16),
        ),
        (
            dict(solver="cutile", c_dtype=torch.float16),
            no_support_for("cutile", torch.float16),
        ),
        (
            dict(solver="exact", c_dtype=torch.bfloat16),
            "solver='exact' simulates ansatz 'pz_encoding' with complex amplitudes",
        ),
        (
            dict(solver="cutn", ansatz="rpz", c_dtype=torch.float8_e4m3fn),
            "solver='cutn' simulates ansatz 'rpz' with complex amplitudes",
        ),
        (dict(solver="cutn", reps=12), "solver='cutn' supports reps <= 11"),
        (
            dict(solver="tn", ansatz="real", reps=13),
            "solver='cutn' supports reps <= 12",
        ),
    ],
    ids=[
        "flash-real-float16",
        "cute-float16",
        "cutile-float16",
        "exact-bfloat16",
        "cutn-fp8",
        "cutn-reps",
        "tn-real-reps",
    ],
)
def test_layer_refuses_what_the_solver_check_refuses(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build(**kwargs)


@pytest.mark.parametrize("solver", FUSED)
def test_fused_layer_c_dtype(solver):
    """complex64, the layer default, is the real-valued kernels' float32; the
    dtypes they have a path for are kept as given."""
    assert build(solver=solver).c_dtype == torch.float32
    assert build(solver=solver, ansatz="rpz").c_dtype == torch.float32
    assert build(solver=solver, ansatz="real").c_dtype == torch.bfloat16
    for ansatz in ("pz", "rpz", "real"):
        layer = build(solver=solver, ansatz=ansatz, c_dtype=torch.complex64)
        assert layer.c_dtype == torch.float32
        for c_dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn):
            layer = build(solver=solver, ansatz=ansatz, c_dtype=c_dtype)
            assert layer.c_dtype == c_dtype


def test_other_solvers_keep_the_c_dtype():
    assert build(solver="exact").c_dtype == torch.complex64
    assert build(solver="exact", ansatz="mix").c_dtype == torch.complex64
    for c_dtype in (torch.complex128, torch.float64, torch.float32):
        assert build(solver="exact", c_dtype=c_dtype).c_dtype == c_dtype
    assert build(solver="mlx", ansatz="mix").c_dtype == torch.complex64
    assert build(solver="cutn", reps=11).c_dtype == torch.complex64
    assert build(solver="tn", ansatz="real", reps=12).c_dtype == torch.bfloat16


def test_callable_solver_is_not_checked():
    def solver(x, theta, *args, **kwargs):
        return x[:, None, :].expand(-1, OUT_DIM, -1)

    layer = build(solver=solver, ansatz="mix", c_dtype=torch.float64, theta_size=[1, 1])
    assert layer.c_dtype == torch.float64
    assert layer(torch.rand(4, IN_DIM)).shape == (4, OUT_DIM)


def test_refusals_come_before_anything_is_seeded_or_built(monkeypatch):
    """Neither a reseeded global RNG nor a missing-backend ImportError first."""
    import qkan.qkan as qkan_module

    def no_activation(*args, **kwargs):
        raise ImportError("backend not installed")

    monkeypatch.setattr(qkan_module, "make_base_activation", no_activation)
    torch.manual_seed(1)
    state = torch.get_rng_state()
    for build_model in (
        lambda: QKAN([IN_DIM, OUT_DIM], solver="flash", ansatz="mix", seed=0),
        lambda: QKANLayer(IN_DIM, OUT_DIM, solver="flash", ansatz="mix", seed=0),
    ):
        with pytest.raises(ValueError, match=no_kernel_for("flash", "mix")):
            build_model()
        assert torch.equal(torch.get_rng_state(), state)


class Unhashable:
    """A callable ansatz that cannot be looked up in a set."""

    __hash__ = None

    def __call__(self, theta):
        return theta


@pytest.mark.parametrize("solver", FUSED + ["cutn", "mlx"])
def test_unhashable_ansatz_object_is_refused_like_any_callable(solver):
    with pytest.raises(ValueError, match=f"solver='{solver}' has no kernel"):
        build(solver=solver, ansatz=Unhashable(), theta_size=CUSTOM_THETA)


@pytest.mark.parametrize("ansatz", ["px", "px_encoding"])
def test_px_needs_trainable_preacts_when_built(ansatz):
    """It used to build, and fail at the first forward with NameError."""
    for build_model in (
        lambda **kw: build(ansatz=ansatz, **kw),
        lambda **kw: QKAN([IN_DIM, OUT_DIM], ansatz=ansatz, **kw),
    ):
        with pytest.raises(ValueError, match="preact_trainable=True"):
            build_model()
        build_model(preact_trainable=True)


def test_unknown_solver_name_is_refused():
    """It used to build, and fail at the first forward with a bare
    NotImplementedError."""
    with pytest.raises(ValueError, match="Unknown solver 'flsh'.*'flash'"):
        build(solver="flsh")
    with pytest.raises(ValueError, match="Unknown solver 'flsh'"):
        QKAN([IN_DIM, OUT_DIM], solver="flsh")


def test_registry_entry_without_check_is_not_checked(monkeypatch):
    """A plain function put in the registry, not a QKANSolver."""

    def solver(x, theta, *args, **kwargs):
        return x[:, None, :].expand(-1, OUT_DIM, -1)

    monkeypatch.setitem(get_registry(), "plain", solver)
    layer = build(solver="plain")
    assert layer(torch.rand(4, IN_DIM)).shape == (4, OUT_DIM)


def test_qkan_refuses_when_built():
    with pytest.raises(ValueError, match=no_kernel_for("flash", "mix")):
        QKAN([3, 4, 2], solver="flash", ansatz="mix", base_activation=nn.SiLU())
    with pytest.raises(ValueError, match=no_support_for("cutile", torch.complex128)):
        QKAN(
            [3, 4, 2],
            solver="cutile",
            c_dtype=torch.complex128,
            base_activation=nn.SiLU(),
        )


@pytest.mark.parametrize("solver", INSTALLED)
def test_default_layer_of_an_installed_backend(solver):
    """The same with the default base activation, the backend's own kernel."""
    with pytest.raises(ValueError, match=no_kernel_for(solver, "mix")):
        QKANLayer(IN_DIM, OUT_DIM, solver=solver, ansatz="mix")
    with pytest.raises(ValueError, match=no_support_for(solver, torch.complex128)):
        QKANLayer(IN_DIM, OUT_DIM, solver=solver, c_dtype=torch.complex128)
    assert QKANLayer(IN_DIM, OUT_DIM, solver=solver).c_dtype == torch.float32


# ---------------------------------------------------------------------------
# solver="qml": one pz circuit on the raw input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ansatz",
    [
        "mix",
        "rpz_encoding",
        "rpz",
        "real",
        "px_encoding",
        "px",
        pytest.param(custom_ansatz, id="callable"),
    ],
)
def test_qml_refuses_other_ansatzes(ansatz):
    theta_size = CUSTOM_THETA if callable(ansatz) else None
    with pytest.raises(ValueError, match="solver='qml' has no circuit for ansatz"):
        build(solver="qml", ansatz=ansatz, theta_size=theta_size)


def test_qml_refuses_trainable_preacts():
    with pytest.raises(ValueError, match="solver='qml'.*preact_trainable=True"):
        build(solver="qml", preact_trainable=True)
    with pytest.raises(ValueError, match="solver='qml'.*preact_trainable=True"):
        QKAN([3, 2], solver="qml", preact_trainable=True)


@pytest.mark.parametrize("ansatz", ["pz_encoding", "pz"])
def test_qml_pz_layer_still_builds(ansatz):
    layer = build(solver="qml", ansatz=ansatz)
    assert layer.theta.shape == (OUT_DIM, IN_DIM, REPS + 1, 2)


# ---------------------------------------------------------------------------
# QKAN keyword arguments
# ---------------------------------------------------------------------------


def test_qkan_refuses_unknown_keywords():
    """A misspelt option used to be dropped, building the default model."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'solvr'"):
        QKAN([2, 3, 1], solvr="flash")
    with pytest.raises(TypeError, match="unexpected keyword argument 'c_dtyp'"):
        QKAN([2, 3, 1], solver="exact", c_dtyp=torch.bfloat16)


# ---------------------------------------------------------------------------
# String base activations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", FUSED)
def test_missing_backend_activation_raises(solver, monkeypatch):
    """base_activation="silu" names the solver's own kernel; without the
    backend the torch.nn module used to be returned in its place."""
    module, flag, message = BACKEND_ACTIVATIONS[solver]
    monkeypatch.setattr(importlib.import_module(module), flag, False)
    for call in (
        lambda: make_base_activation("silu", solver),
        lambda: QKANLayer(IN_DIM, OUT_DIM, solver=solver),
        lambda: QKAN([IN_DIM, OUT_DIM], solver=solver, base_activation="gelu"),
    ):
        with pytest.raises(ImportError, match=message):
            call()


@pytest.mark.parametrize("solver", FUSED)
def test_string_base_activation_is_the_backend_kernel(solver, monkeypatch):
    module, flag, _ = BACKEND_ACTIVATIONS[solver]
    monkeypatch.setattr(importlib.import_module(module), flag, True)
    assert type(make_base_activation("silu", solver)).__module__ == module


@pytest.mark.parametrize("solver", ["exact", "cutn", "mlx", "qml", custom_ansatz])
def test_other_solvers_get_the_torch_activation(solver):
    assert isinstance(make_base_activation("silu", solver), nn.SiLU)


# ---------------------------------------------------------------------------
# Fused epilogue with float64 parameters
# ---------------------------------------------------------------------------

F64 = dict(rtol=1e-12, atol=1e-12)


def _epilogue_operands(batch=37, out_dim=5, in_dim=21):
    """postacts, eff_pw, pb, base_input, base_w in float64, as leaves."""
    gen = torch.Generator().manual_seed(0)
    shapes = [
        (batch, out_dim, in_dim),
        (out_dim, in_dim),
        (out_dim, in_dim),
        (batch, in_dim),
        (out_dim, in_dim),
    ]
    return [
        torch.randn(shape, generator=gen, dtype=torch.float64)
        .to("cuda")
        .requires_grad_(True)
        for shape in shapes
    ]


@needs_gpu
def test_fused_epilogue_computes_float64_in_float64():
    """Both kernels accumulated in float32 and stored the result as float64,
    4e-7 away from the eager chain they replace."""
    operands = _epilogue_operands()
    postacts, eff_pw, pb, base_input, base_w = operands
    out = QKANEpilogue.apply(*operands)
    ref = ((postacts + pb) * eff_pw).sum(dim=2) + F.linear(base_input, base_w)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref, **F64)

    weight = torch.randn_like(ref)
    grads = torch.autograd.grad((out * weight).sum(), operands)
    ref_grads = torch.autograd.grad((ref * weight).sum(), operands)
    for grad, ref_grad in zip(grads, ref_grads):
        assert grad.dtype == torch.float64
        torch.testing.assert_close(grad, ref_grad, **F64)


@needs_gpu
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64]
)
def test_fused_epilogue_accumulates_in_float64_for_float64_only(dtype, monkeypatch):
    """The other dtypes keep the float32 launch they had."""
    import qkan.qkan_epilogue as epilogue

    launched = {}

    def spy(name):
        kernel = getattr(epilogue, name)

        class Launcher:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launched[name] = kwargs["ACC_F64"]
                    return kernel[grid](*args, **kwargs)

                return launch

        monkeypatch.setattr(epilogue, name, Launcher())

    spy("_epilogue_fwd_kernel")
    spy("_epilogue_bwd_kernel")
    operands = [t.detach().to(dtype).requires_grad_() for t in _epilogue_operands()]
    QKANEpilogue.apply(*operands).float().sum().backward()
    is_f64 = dtype == torch.float64
    assert launched == {"_epilogue_fwd_kernel": is_f64, "_epilogue_bwd_kernel": is_f64}


@needs_gpu
def test_fused_epilogue_eval_form_computes_float64_in_float64():
    """The eval route: no postact bias, a folded bias on the base term."""
    with torch.no_grad():
        postacts, eff_pw, _, base_input, base_w = _epilogue_operands()
        bias_sum = torch.randn(base_w.shape[0], dtype=torch.float64).to("cuda")
        out = qkan_epilogue_forward(
            postacts, eff_pw, None, base_input, base_w, bias_sum
        )
        ref = (postacts * eff_pw).sum(dim=2) + F.linear(base_input, base_w, bias_sum)
    torch.testing.assert_close(out, ref, **F64)


@needs_gpu
@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
def test_float64_layer_with_fused_epilogue_matches_eager(train):
    torch.manual_seed(0)
    layer = QKANLayer(
        8,
        4,
        device="cuda",
        c_dtype=torch.complex128,
        p_dtype=torch.float64,
        postact_weight_trainable=True,
        postact_bias_trainable=True,
    )
    with torch.no_grad():
        layer.postact_weights.normal_()
        layer.postact_bias.normal_()
    layer.train(train)
    x = torch.randn(33, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    weight = torch.randn(33, 4, device="cuda", dtype=torch.float64)
    leaves = [x, *(p for p in layer.parameters() if p.requires_grad)]

    results = []
    for fused in (False, True):
        layer.set_fused_epilogue(fused)
        y = layer(x)
        assert y.dtype == torch.float64
        result = [y.detach()]
        if train:  # the eval route's fused kernel is a plain forward
            result += torch.autograd.grad((y * weight).sum(), leaves)
        results.append(result)
    for fused, eager in zip(results[1], results[0]):
        torch.testing.assert_close(fused, eager, **F64)
