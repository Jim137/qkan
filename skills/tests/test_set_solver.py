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

"""set_solver switches the solver of a built QKANLayer or QKAN.

The solver function, c_dtype and the base activation are derived from the
solver when the layer is built, and ``layer.solver = "flash"`` changes none of
them. set_solver re-derives all three, and the plain assignment raises. The
reference throughout is a model built with the new solver and loaded with the
same state dict.
"""

import copy
import importlib
import io

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from qkan import QKAN, QKANLayer, compile_inference  # noqa: E402
from qkan.inference import (  # noqa: E402
    graph_submodules,
    make_graphed_inference,
    make_graphed_train_step,
)
from qkan.solver import (  # noqa: E402
    _CUTE_AVAILABLE,
    _CUTILE_AVAILABLE,
    _FLASH_AVAILABLE,
    _FLYDSL_AVAILABLE,
)

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
needs_triton = pytest.mark.skipif(not _FLASH_AVAILABLE, reason="needs Triton")
needs_flash = pytest.mark.skipif(
    not (_FLASH_AVAILABLE and CUDA), reason="needs Triton and CUDA"
)
RUNS_HERE = {
    "exact": True,
    "flash": _FLASH_AVAILABLE and CUDA,
    "cutile": _CUTILE_AVAILABLE and CUDA,
    "cute": _CUTE_AVAILABLE and CUDA,
}
SOLVERS = [
    pytest.param(name, marks=pytest.mark.skipif(not ok, reason=f"{name} not here"))
    for name, ok in RUNS_HERE.items()
]


def _model(solver, device=DEVICE, seed=0, **kwargs):
    return QKAN([3, 4, 2], reps=2, solver=solver, device=device, seed=seed, **kwargs)


def _rebuilt(model, solver, device=DEVICE, **kwargs):
    """What set_solver has to match: a model built with ``solver`` that loads
    the state dict."""
    ref = _model(solver, device, seed=1, **kwargs)
    ref.load_state_dict(model.state_dict())
    return ref.train(model.training)


def _grads(model, out):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.autograd.grad(out.sum(), params, allow_unused=True)


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "real"])
@pytest.mark.parametrize("new", SOLVERS)
@pytest.mark.parametrize("old", SOLVERS)
def test_matches_a_model_built_with_the_solver(old, new, ansatz, training):
    options = dict(ansatz=ansatz, preact_trainable=True, postact_bias_trainable=True)
    model = _model(old, **options).train(training)
    ref = _rebuilt(model, new, **options)

    model.set_solver(new)

    assert model.solver == new
    for layer, ref_layer in zip(model.layers, ref.layers):
        assert layer.solver == new
        assert layer.c_dtype == ref_layer.c_dtype
        assert layer._solver_fn is ref_layer._solver_fn
        assert type(layer.base_activation) is type(ref_layer.base_activation)
    x = torch.rand(8, 3, device=DEVICE)
    out, want = model(x), ref(x)
    assert torch.equal(out, want)
    # Equal bit for bit only because the last layer has two outputs: the
    # fused backward kernels round a wider one differently from run to run.
    for grad, ref_grad in zip(_grads(model, out), _grads(ref, want)):
        assert (grad is None) == (ref_grad is None)
        assert grad is None or torch.equal(grad, ref_grad)


@needs_flash
def test_a_flash_model_can_leave_the_gpu():
    model = _model("flash", "cuda")
    model.set_solver("exact")
    model.to("cpu")
    x = torch.rand(8, 3)
    assert torch.equal(model(x), _rebuilt(model, "exact", "cpu")(x))


# ---------------------------------------------------------------- c_dtype


@needs_triton
def test_c_dtype_is_resolved_as_at_construction():
    layer = QKANLayer(3, 2, solver="exact")
    assert layer.c_dtype == torch.complex64
    layer.set_solver("flash")  # real-valued kernels: complex64 means float32
    assert layer.c_dtype == torch.float32
    layer.set_solver("exact")  # the request was the default, not float32
    assert layer.c_dtype == torch.complex64
    layer.set_solver("exact", c_dtype=torch.complex128)
    assert layer.c_dtype == torch.complex128
    # Kept, and flash has no double precision; the message names the way out.
    with pytest.raises(ValueError, match=r"set_solver\(solver, c_dtype=\.\.\.\)"):
        layer.set_solver("flash")
    layer.set_solver("flash", c_dtype=torch.bfloat16)
    assert layer.c_dtype == torch.bfloat16


@needs_flash
@pytest.mark.parametrize("route", [["exact"], ["cutile", "flash", "exact"]])
def test_real_ansatz_keeps_a_complex_request(route):
    """flash stores complex64 as float32, which for ``real`` is another exact
    kernel than complex64: the switch has to go back to the request."""
    options = dict(ansatz="real", c_dtype=torch.complex64, preact_trainable=True)
    model = _model("flash", "cuda", **options)
    for solver in route:
        if RUNS_HERE[solver]:
            model.set_solver(solver)
    assert model.layers[0].c_dtype == torch.complex64
    x = torch.rand(8, 3, device="cuda")
    assert torch.equal(model(x), _rebuilt(model, "exact", "cuda", **options)(x))


@needs_flash
def test_c_dtype_argument_matches_a_model_built_with_it():
    model = _model("exact", "cuda")
    ref = _rebuilt(model, "flash", "cuda", c_dtype=torch.bfloat16)
    model.set_solver("flash", c_dtype=torch.bfloat16)
    x = torch.rand(8, 3, device="cuda")
    assert torch.equal(model(x), ref(x))


def test_model_c_dtype_changes_only_with_the_argument():
    model = _model("exact", "cpu")
    model.set_solver("exact")
    assert model.c_dtype is None
    model.set_solver("exact", c_dtype=torch.complex128)
    assert model.c_dtype == torch.complex128
    assert model.layers[0].c_dtype == torch.complex128
    # The model's own c_dtype is what refine() and pruning rebuild with.
    assert model.refine(3).layers[0].c_dtype == torch.complex128


# ---------------------------------------------------------------- refusals


def _custom_solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs):
    return torch.zeros(x.shape[0], 1, x.shape[1])


# (model options, set_solver arguments, message)
REFUSED = {
    "unknown name": ({}, ("nope",), "Unknown solver 'nope'"),
    "neither a name nor a callable": ({}, (None,), "must be a name or a callable"),
    "no kernel for the ansatz": (
        dict(ansatz="px_encoding", preact_trainable=True),
        ("flash",),
        "has no kernel for ansatz 'px_encoding'",
    ),
    "no path for the c_dtype": (
        dict(c_dtype=torch.complex128),
        ("mlx",),  # would also turn the fused epilogue on
        "does not support c_dtype",
    ),
    "no path for the c_dtype argument": (
        {},
        ("flash", torch.complex128),
        "does not support c_dtype",
    ),
    "too deep": (dict(reps=12), ("cutn",), "supports reps <= 11"),
    "qml with trainable preacts": (
        dict(preact_trainable=True),
        ("qml",),
        "preact_trainable=True is not supported",
    ),
    "a callable for a named solver": (
        {},
        (_custom_solver,),
        "cannot switch between a named solver and a callable",
    ),
}


def _snapshot(model):
    """Everything set_solver may touch, on the model and on each layer."""
    layers = [m for m in model.layers if isinstance(m, QKANLayer)]
    return [
        (m.solver, m.c_dtype, m.base_activation, m.solver_kwargs)
        for m in (model, *layers)
    ] + [(m._solver_fn, m._c_dtype_request, m._use_fused_epilogue) for m in layers]


@pytest.mark.parametrize("case", REFUSED)
def test_a_refused_switch_changes_nothing(case):
    options, arguments, message = REFUSED[case]
    options = {"reps": 2, **options}
    model = QKAN([3, 4, 2], solver="exact", solver_kwargs={"marker": 1}, **options)
    x = torch.rand(8, 3)
    state, out = _snapshot(model), model(x)
    with pytest.raises(ValueError, match=message) as refusal:
        model.set_solver(*arguments)
    # The hint goes with a c_dtype that was kept, and with nothing else.
    hinted = "set_solver(solver, c_dtype=...)" in str(refusal.value)
    assert hinted == (case == "no path for the c_dtype")
    with pytest.raises(ValueError, match=message):
        model.layers[0].set_solver(*arguments)
    assert _snapshot(model) == state
    assert torch.equal(model(x), out)


@needs_triton
def test_one_refusing_layer_keeps_the_whole_model():
    model = _model("exact", "cpu")
    model.layers[1].set_solver("exact", c_dtype=torch.complex128)
    state = _snapshot(model)
    with pytest.raises(ValueError, match="does not support c_dtype"):
        model.set_solver("flash")  # layer 0 could switch, layer 1 cannot
    assert _snapshot(model) == state


@pytest.mark.skipif(_FLYDSL_AVAILABLE, reason="flydsl is installed")
@pytest.mark.parametrize("named", [0, 1, 2], ids=["model", "layer 0", "layer 1"])
def test_a_missing_backend_changes_nothing(named):
    """Whichever module holds the one activation built from a name, its
    kernel is found missing before anything is switched."""
    model = _model("exact", "cpu")
    modules = [model, *model.layers]
    for module in modules[:named] + modules[named + 1 :]:
        module.base_activation = nn.Tanh()
    state = _snapshot(model)
    for module in {model, modules[named]}:
        with pytest.raises(ImportError):
            module.set_solver("flydsl")
        assert _snapshot(model) == state


def test_the_check_comes_before_the_backend():
    layer = QKANLayer(3, 2, solver="exact")
    with pytest.raises(ValueError, match="does not support c_dtype"):
        layer.set_solver("flydsl", c_dtype=torch.complex128)


def test_frozen_preacts_can_go_to_qml():
    layer = QKANLayer(3, 3, solver="exact")
    layer.set_solver("qml")
    assert layer.solver == "qml" and layer._solver_fn is None


@pytest.mark.parametrize("case", [c for c in REFUSED if "a callable" not in c])
def test_a_model_without_qkan_layers_is_checked_too(case):
    options, arguments, message = REFUSED[case]
    model = QKAN([3], **options)
    with pytest.raises(ValueError, match=message) as refusal:
        model.set_solver(*arguments)
    hinted = "set_solver(solver, c_dtype=...)" in str(refusal.value)
    assert hinted == (case == "no path for the c_dtype")
    assert model.solver == "exact"


def test_a_model_without_qkan_layers_is_switched():
    model = QKAN([3])
    model.set_solver("cutn")
    assert model.solver == "cutn"


def test_map_layers_are_left_alone():
    model = QKAN([3, 4, 2], reps=2, is_map=True, hidden=4, c_dtype=torch.float32)
    x = torch.rand(8, 3)
    out = model(x)
    model.set_solver("exact", c_dtype=torch.complex64)  # float32 means complex64
    assert torch.equal(model(x), out)


def test_without_c_dtype_the_model_and_each_layer_keep_their_own():
    model = _model("exact", "cpu", c_dtype=torch.float32)
    model.layers[1].set_solver("exact", c_dtype=torch.complex128)
    model.set_solver("cutn")
    assert model.c_dtype == torch.float32
    assert [layer.c_dtype for layer in model.layers] == [
        torch.float32,
        torch.complex128,
    ]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "switch",
    [dict(solver="cutn"), dict(solver="exact", c_dtype=torch.complex128)],
    ids=["solver", "c_dtype"],
)
def test_refine_and_pruning_refuse_a_layer_switched_on_its_own(index, switch):
    """They rebuild from the model's own solver and c_dtype."""
    model = _model("exact", "cpu", save_act=True)
    model(torch.rand(8, 3))
    model.layers[index].set_solver(**switch)
    for rebuild in (
        lambda: model.refine(3),
        lambda: model.prune_node(),
        lambda: model.prune_input(active_inputs=[0, 1]),
    ):
        with pytest.raises(ValueError, match=r"call model.set_solver\("):
            rebuild()
    model.set_solver("cutn", c_dtype=torch.complex128)
    assert model.refine(3).layers[index].solver == "cutn"


@pytest.mark.parametrize(
    "options",
    [dict(ansatz="real"), dict(c_dtype=torch.float32), dict(is_map=True, hidden=4)],
    ids=["real", "float32", "map"],
)
def test_pruning_accepts_layers_that_were_not_switched(options):
    model = _model("exact", "cpu", save_act=True, **options)
    model(torch.rand(8, 3))
    assert isinstance(model.prune_input(active_inputs=[0, 1]), QKAN)


@needs_flash
def test_a_pruned_model_is_switched_like_an_unpruned_one():
    """get_subset builds the pruned layers: they keep the c_dtype as it was
    requested, not the float32 that flash stores for it."""
    model = _model("flash", "cuda", save_act=True)
    model(torch.rand(8, 3, device="cuda"))
    pruned = model.prune_node(
        active_neurons_id=[[0, 1, 2], [0, 1, 2, 3], [0, 1]], mode="manual"
    )
    pruned.set_solver("exact")
    assert [layer.c_dtype for layer in pruned.layers] == [torch.complex64] * 2
    assert isinstance(pruned.refine(3), QKAN)


@needs_triton
def test_a_layer_without_a_recorded_request_keeps_its_c_dtype():
    """As unpickled from a model that was saved whole before set_solver."""
    layer = QKANLayer(3, 2, solver="flash", c_dtype=torch.bfloat16)
    del layer._c_dtype_request
    layer.set_solver("mlx")
    assert layer.c_dtype == layer._c_dtype_request == torch.bfloat16


# ---------------------------------------------------------------- callables


def _recording_solver(calls):
    def solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs):
        calls.append(kwargs)
        return theta.sum() * x[:, None, :].expand(-1, 2, -1)

    return solver


def test_solver_kwargs_are_those_of_the_new_solver():
    first, second = [], []
    layer = QKANLayer(
        3, 2, solver=_recording_solver(first), theta_size=[1], solver_kwargs={"a": 1}
    )
    x = torch.rand(4, 3)
    layer(x)
    assert first[-1]["a"] == 1
    layer.set_solver(_recording_solver(second), solver_kwargs={"b": 2})
    layer(x)
    assert second[-1]["b"] == 2 and "a" not in second[-1] and len(first) == 1
    layer.set_solver(_recording_solver(first))
    layer(x)
    assert "a" not in first[-1] and "b" not in first[-1]


def test_qkan_hands_one_solver_kwargs_dict_to_every_layer():
    calls = []
    model = QKAN([3, 2, 2], solver=_recording_solver(calls), theta_size=[1])
    model.set_solver(_recording_solver(calls), solver_kwargs={"b": 2})
    assert all(layer.solver_kwargs is model.solver_kwargs for layer in model.layers)
    model.set_solver(_recording_solver(calls))
    assert model.solver_kwargs == {} == model.layers[0].solver_kwargs


def test_named_and_callable_solvers_do_not_mix():
    custom = QKANLayer(3, 2, solver=_custom_solver, theta_size=[1])
    with pytest.raises(ValueError, match="between a named solver and a callable"):
        custom.set_solver("exact")  # theta is (1,), not the pz layout
    assert custom.solver is _custom_solver
    # With a callable ansatz theta follows theta_size under either kind.
    layer = QKANLayer(3, 2, solver="exact", ansatz=lambda *a, **k: None, theta_size=[1])
    layer.set_solver(_custom_solver)
    assert layer(torch.rand(4, 3)).shape == (4, 2)


class _ModuleSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x, theta, preacts_weight, preacts_bias, reps, **kwargs):
        return self.scale * x[:, None, :].expand(-1, 2, -1)


@pytest.mark.parametrize(
    "build",
    [
        lambda solver: QKANLayer(3, 2, solver=solver, theta_size=[1]),
        lambda solver: QKAN([3, 2], solver=solver, theta_size=[1]),
    ],
    ids=["layer", "model"],
)
def test_a_module_solver_is_registered_and_released(build):
    model = build(_ModuleSolver())
    assert "solver.scale" in model.state_dict()
    with pytest.raises(AttributeError, match=r"call set_solver\(\)"):
        model.solver = _ModuleSolver()
    model.set_solver(_custom_solver)
    assert model.solver is _custom_solver
    assert not any("scale" in key for key in model.state_dict())
    module = _ModuleSolver()
    model.set_solver(module)
    assert model.solver is module and "solver.scale" in model.state_dict()
    model(torch.rand(4, 3)).sum().backward()
    assert module.scale.grad is not None


@needs_cuda
def test_data_parallel_replicates_a_module_solver():
    model = QKAN([3, 2], solver=_ModuleSolver(), theta_size=[1]).to("cuda")
    out = nn.DataParallel(model, device_ids=[0, 0])(torch.rand(8, 3, device="cuda"))
    assert out.shape == (8, 2)


# ---------------------------------------------------------------- activation


@needs_triton
def test_a_named_base_activation_follows_the_solver():
    from qkan.solver.flash.triton_activation import TritonActivation

    model = _model("flash", "cpu", base_activation="gelu")
    model.set_solver("exact")
    for module in (model, *model.layers):
        assert isinstance(module.base_activation, nn.GELU)
        assert module.base_activation.approximate == "none"
        assert module.base_activation.training
    model.eval().set_solver("flash")
    for module in (model, *model.layers):
        assert isinstance(module.base_activation, TritonActivation)
        assert module.base_activation.kind == "gelu_exact"
        # One module for the model and its layers, in the model's mode.
        assert module.base_activation is model.base_activation
        assert not module.base_activation.training


def _triton_silu():
    from qkan.solver.flash.triton_activation import TritonActivation

    return TritonActivation("silu")


@needs_triton
@pytest.mark.parametrize(
    "make",
    [nn.Tanh, lambda: torch.sin, _triton_silu],
    ids=["torch", "function", "kernel"],
)
def test_a_base_activation_passed_in_is_kept(make):
    activation = make()
    model = _model("flash", "cpu", base_activation=activation)
    model.set_solver("exact")
    assert all(m.base_activation is activation for m in (model, *model.layers))


KERNEL_ACTIVATIONS = {
    "flash": ("qkan.solver.flash.triton_activation", "TritonActivation"),
    "cute": ("qkan.solver.cute.cute_activation", "CuTeActivation"),
    "cutile": ("qkan.solver.cutile.cutile_activation", "CuTileActivation"),
    "flydsl": ("qkan.solver.flydsl.flydsl_activation", "FlyDSLActivation"),
}


class _StandInActivation(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind


@pytest.mark.parametrize("solver", KERNEL_ACTIVATIONS)
def test_every_kernel_activation_built_from_a_name_follows(solver, monkeypatch):
    module, name = KERNEL_ACTIVATIONS[solver]
    monkeypatch.setattr(importlib.import_module(module), name, _StandInActivation)
    layer = QKANLayer(3, 2, solver=solver, base_activation="tanh")
    assert isinstance(layer.base_activation, _StandInActivation)
    layer.set_solver("exact")
    assert isinstance(layer.base_activation, nn.Tanh)


DERIVED = {
    "deepcopy": copy.deepcopy,
    "refine": lambda model: model.refine(3),
    "get_subset": lambda model: model.layers[0].get_subset([0, 1], [0]),
}


@needs_triton
@pytest.mark.parametrize("derive", DERIVED)
def test_the_activation_of_a_derived_model_still_follows_the_solver(derive):
    derived = DERIVED[derive](_model("flash", "cpu"))
    derived.set_solver("exact")
    layers = derived.layers if isinstance(derived, QKAN) else [derived]
    assert all(isinstance(layer.base_activation, nn.SiLU) for layer in layers)


# ---------------------------------------------------------------- epilogue


def test_fused_epilogue_follows_the_default_of_the_solver(monkeypatch):
    monkeypatch.delenv("QKAN_FUSED_EPILOGUE", raising=False)
    assert QKANLayer(3, 2, solver="mlx")._use_fused_epilogue is True
    layer = QKANLayer(3, 2, solver="exact")
    assert layer._use_fused_epilogue is False
    layer.set_solver("mlx")  # on by default there
    assert layer._use_fused_epilogue is True
    layer.set_solver("exact")
    assert layer._use_fused_epilogue is False
    monkeypatch.setenv("QKAN_FUSED_EPILOGUE", "1")  # the default of a build now
    layer.set_solver("cutn")
    assert layer._use_fused_epilogue is True
    layer.set_solver("mlx")
    assert layer._use_fused_epilogue is True
    monkeypatch.setenv("QKAN_FUSED_EPILOGUE", "0")
    layer.set_solver("mlx")
    assert layer._use_fused_epilogue is False
    monkeypatch.setenv("QKAN_FUSED_EPILOGUE", "true")  # only "1" turns it on
    layer.set_solver("mlx")
    assert layer._use_fused_epilogue is False


@pytest.mark.parametrize("choice", [True, False])
def test_a_set_fused_epilogue_choice_survives_every_switch(choice):
    layer = QKANLayer(3, 2, solver="exact")
    layer.set_fused_epilogue(choice)
    for solver in ("mlx", "exact", "cutn", "mlx"):
        layer.set_solver(solver)
        assert layer._use_fused_epilogue is choice


# ---------------------------------------------------------------- assignment


def _saved_and_loaded(model):
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


@pytest.mark.parametrize(
    "copied",
    [lambda m: m, copy.deepcopy, _saved_and_loaded],
    ids=["built", "deepcopy", "pickle"],
)
def test_assigning_solver_raises(copied):
    model = copied(_model("exact", "cpu"))
    x = torch.rand(8, 3)
    out = model(x)
    for module in (model, model.layers[0]):
        with pytest.raises(AttributeError, match=r"call set_solver\(\)"):
            module.solver = "cutn"
        assert module.solver == "exact"
    assert torch.equal(model(x), out)
    model.set_solver("exact", c_dtype=torch.float32)
    assert torch.equal(model(x), out)


# ---------------------------------------------------------------- inference


@needs_flash
def test_compiled_inference_drops_graphs_of_the_old_solver():
    model = _model("exact", "cuda").eval()
    compiled = compile_inference(model).eval()
    captures = []
    capture = compiled._capture
    compiled._capture = lambda x: captures.append(x.shape) or capture(x)
    inputs = [torch.rand(n, 3, device="cuda") for n in (8, 5)]
    with torch.no_grad():
        before = [compiled(x).clone() for x in inputs]
        assert len(captures) == 2  # both with the exact solver
        model.set_solver("flash", c_dtype=torch.bfloat16)
        for x, old in zip(inputs, before):
            want = model(x)
            assert not torch.equal(old, want)
            assert torch.equal(compiled(x), want)
            assert torch.equal(compiled(x), want)  # the replay of the new graph
    assert len(captures) == 4


@needs_flash
def test_graphed_callables_refuse_to_replay_the_old_solver():
    model = _model("exact", "cuda")
    x = torch.rand(8, 3, device="cuda")
    y = torch.rand(8, 2, device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train_step = make_graphed_train_step(model, x, y, nn.MSELoss(), optimizer)
    replay = make_graphed_inference(model.eval(), x)
    replay(x), train_step(x, y)
    model.set_solver("flash")
    for stale in (lambda: replay(x), lambda: train_step(x, y)):
        with pytest.raises(RuntimeError, match="capture again"):
            stale()
    with torch.no_grad():
        assert torch.equal(make_graphed_inference(model, x)(x), model(x))


@needs_flash
def test_layers_wrapped_for_graph_capture_are_switched():
    model = _model("exact", "cuda").eval()
    ref = _rebuilt(model, "flash", "cuda")
    x = torch.rand(8, 3, device="cuda")
    graph_submodules(model, x, lambda m: isinstance(m, QKANLayer))
    assert not isinstance(model.layers[0], QKANLayer)  # wrapped
    model.set_solver("flash")
    layers = [m for m in model.modules() if isinstance(m, QKANLayer)]
    assert [layer.solver for layer in layers] == ["flash", "flash"]
    with torch.no_grad():
        assert torch.equal(model(x), ref(x))


@needs_flash
def test_a_layer_switched_on_its_own_makes_graphs_stale():
    layer = QKANLayer(3, 2, solver="exact", device="cuda").eval()
    x = torch.rand(8, 3, device="cuda")
    replay = make_graphed_inference(layer, x)
    layer.set_solver("flash")
    with pytest.raises(RuntimeError, match="capture again"):
        replay(x)


def test_a_subclass_with_a_solver_class_attribute_can_be_built():
    class Layer(QKANLayer):
        solver = "exact"

    assert Layer(3, 2, solver="cutn").solver == "cutn"
