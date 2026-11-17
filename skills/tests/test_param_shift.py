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

"""Parameter-shift gradients of the real-device solvers (#63).

The qiskit and cudaq solvers returned no gradient for the layer input, so only
the last layer of a model trained, and a wrong one for ``preacts_weight``.
"""

import importlib

import pytest
import torch

from qkan import QKAN
from qkan.solver._param_shift import ParamShift
from qkan.solver.exact import torch_exact_solver

# ansatz, preacts_trainable: every circuit the solvers build. rpz reads
# w * x + b whether or not the pre-activations are trainable.
CIRCUITS = [
    ("pz_encoding", False),
    ("pz_encoding", True),
    ("rpz_encoding", False),
    ("rpz", False),
    ("real", False),
    ("real", True),
]
BATCH, OUT_DIM, IN_DIM = 3, 3, 2
NAMES = ("x", "theta", "preacts_weight", "preacts_bias")


def exact(x, theta, preacts_weight, preacts_bias, reps, config):
    """<Z> of every circuit from the exact solver, in double precision."""
    return torch_exact_solver(
        x,
        theta,
        preacts_weight,
        preacts_bias,
        reps,
        fast_measure=False,
        dtype=torch.float64,
        **config,
    )


class Device:
    """A stand-in device: the exact expectations, without a gradient."""

    def __init__(self):
        self.runs = 0

    def __call__(self, *args):
        self.runs += 1
        with torch.no_grad():
            return exact(*args)


def shifted(*args):
    return ParamShift.apply(Device(), *args)


def circuits(ansatz, preacts_trainable, reps, group=(OUT_DIM, IN_DIM), in_dim=IN_DIM):
    """x, theta, preacts_weight and preacts_bias of one layer in double, all
    requiring a gradient, and the solver config."""
    generator = torch.Generator().manual_seed(reps)
    last = 2 if ansatz == "pz_encoding" else 1
    axis = reps if ansatz == "real" else reps + 1
    shapes = ((BATCH, in_dim), (*group, axis, last), (*group, reps), (*group, reps))
    inputs = [
        torch.empty(shape, dtype=torch.float64)
        .uniform_(-1.5, 1.5, generator=generator)
        .requires_grad_()
        for shape in shapes
    ]
    config = {
        "ansatz": ansatz,
        "preacts_trainable": preacts_trainable,
        "out_dim": OUT_DIM,
    }
    return inputs, reps, config


def gradients(evaluate, inputs, reps, config):
    """Gradients of a loss that weights every circuit differently."""
    out = evaluate(*inputs, reps, config)
    weights = torch.linspace(0.5, 2.0, out.numel(), dtype=out.dtype).reshape(out.shape)
    return torch.autograd.grad((out * weights).sum(), inputs, allow_unused=True)


def assert_close(got, want, name, atol=1e-10):
    """torch's assert_close, with ``name`` in front of its report."""
    torch.testing.assert_close(
        got, want, atol=atol, rtol=0, msg=lambda report: f"{name}: {report}"
    )


def assert_gradients_match(inputs, reps, config):
    """The shift rules give what autograd gives the exact solver."""
    expected = gradients(exact, inputs, reps, config)
    assert expected[0] is not None and expected[1] is not None
    for name, got, want in zip(
        NAMES, gradients(shifted, inputs, reps, config), expected
    ):
        if want is None:  # the circuits do not read this tensor
            assert got is None, name
        else:
            assert_close(got, want, name)


@pytest.mark.parametrize("group", [(OUT_DIM, IN_DIM), (1,)], ids=["per-edge", "shared"])
@pytest.mark.parametrize("reps", [1, 2, 3])
@pytest.mark.parametrize("ansatz, preacts_trainable", CIRCUITS)
def test_gradients_match_autograd(ansatz, preacts_trainable, reps, group):
    inputs, reps, config = circuits(ansatz, preacts_trainable, reps, group)
    assert_close(
        shifted(*inputs, reps, config), exact(*inputs, reps, config), "forward", 1e-12
    )
    assert_gradients_match(inputs, reps, config)


def test_group_tiled_over_the_inputs():
    """A group of 2 is tiled over 3 inputs (g0, g1, g0) for every output."""
    assert_gradients_match(*circuits("rpz_encoding", False, 2, group=(2,), in_dim=3))


@pytest.mark.parametrize(
    "ansatz, preacts_trainable, needs, runs_per_rep",
    [
        # circuits that encode x itself: its gates share one angle
        ("pz_encoding", False, "x", 2),
        ("real", False, "x theta", 2),
        # w * x + b: one pair per rep gives x, the weight and the bias together
        ("pz_encoding", True, "x theta preacts_weight preacts_bias", 2),
        ("pz_encoding", True, "preacts_bias", 2),
        ("pz_encoding", True, "preacts_weight", 2),
        ("rpz_encoding", False, "x", 2),
        # nothing to differentiate in the encoding
        ("pz_encoding", False, "theta", 0),
        ("pz_encoding", True, "theta", 0),
        # a weight and a bias the circuit does not read
        ("pz_encoding", False, "preacts_weight preacts_bias", 0),
    ],
)
def test_circuit_runs(ansatz, preacts_trainable, needs, runs_per_rep):
    """The backward runs the circuits twice per scalar theta and twice per rep
    for the encoding, and not at all for what needs no gradient."""
    inputs, reps, config = circuits(ansatz, preacts_trainable, 3)
    needs = needs.split()
    for name, tensor in zip(NAMES, inputs):
        tensor.requires_grad_(name in needs)
    device = Device()
    ParamShift.apply(device, *inputs, reps, config).sum().backward()
    theta_runs = 2 * inputs[1].numel() if "theta" in needs else 0
    assert device.runs == 1 + theta_runs + runs_per_rep * reps
    # the chain rule runs on copies: a frozen tensor stays frozen
    assert [name for name, t in zip(NAMES, inputs) if t.requires_grad] == needs


def test_backward_in_inference_mode():
    """The chain rule through w * x + b records a graph of its own."""
    inputs, reps, config = circuits("pz_encoding", True, 2)
    expected = torch.autograd.grad(exact(*inputs, reps, config).sum(), inputs)
    loss = shifted(*inputs, reps, config).sum()
    with torch.inference_mode():
        loss.backward()
    for name, tensor, want in zip(NAMES, inputs, expected):
        assert_close(tensor.grad, want, name)


@pytest.mark.parametrize("ansatz, preacts_trainable", CIRCUITS)
def test_second_derivatives_raise(ansatz, preacts_trainable):
    """A device gives first-order gradients: differentiating one in the
    circuits' inputs raises instead of treating it as a constant."""
    inputs, reps, config = circuits(ansatz, preacts_trainable, 2)
    expected = torch.autograd.grad(
        exact(*inputs, reps, config).sum(), inputs, allow_unused=True
    )
    for index, (name, want) in enumerate(zip(NAMES, expected)):
        if want is None:  # the circuits do not read this tensor
            continue
        # this tensor alone needs a gradient, as theta does in a first layer
        alone = [t.detach().requires_grad_(i == index) for i, t in enumerate(inputs)]
        out = shifted(*alone, reps, config)
        (grad,) = torch.autograd.grad(out.sum(), alone[index], create_graph=True)
        assert_close(grad, want, name)
        with pytest.raises(RuntimeError, match="no second derivatives"):
            grad.sum().backward()


@pytest.mark.parametrize("ansatz, preacts_trainable", CIRCUITS)
def test_jacobian_vector_product(ansatz, preacts_trainable):
    """J v by double backward differentiates the gradients in grad_output
    alone, which they are linear in: first order, so it works."""
    inputs, reps, config = circuits(ansatz, preacts_trainable, 2)
    inputs = tuple(t.detach() for t in inputs)
    vectors = tuple(
        torch.linspace(-1, 1, t.numel()).reshape(t.shape).double() for t in inputs
    )
    jvp = torch.autograd.functional.jvp
    _, want = jvp(lambda *a: exact(*a, reps, config), inputs, vectors)
    _, got = jvp(lambda *a: shifted(*a, reps, config), inputs, vectors)
    assert want.abs().max() > 1e-2
    assert_close(got, want, "J v")


# ---------------------------------------------------------------------------
# The solvers themselves, on their noiseless simulators
# ---------------------------------------------------------------------------


@pytest.fixture(params=["qiskit", "cudaq"])
def solver(request):
    pytest.importorskip(request.param)
    return request.param


def build(solver, ansatz, preact_trainable, **solver_kwargs):
    """A two-layer model; the seed gives every solver the same parameters."""
    torch.manual_seed(0)
    return QKAN(
        [1, 2, 1],
        reps=2,
        solver=solver,
        ansatz=ansatz,
        preact_trainable=preact_trainable,
        fast_measure=False,  # a device measures <Z>
        c_dtype=torch.float32 if ansatz == "real" else torch.complex64,
        solver_kwargs=solver_kwargs,
    )


def model_gradients(model, input_grad=True):
    """The gradient of every trainable parameter, and of the input."""
    x = torch.tensor([[-0.9], [0.4]], requires_grad=input_grad)
    model(x).pow(2).sum().backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    return {"x": x.grad, **grads} if input_grad else grads


def assert_model_gradients_match(
    solver, ansatz, preact_trainable, freeze=(), input_grad=True, **solver_kwargs
):
    """``solver`` gives every tensor the gradient the exact solver gives it."""
    reference = build("exact", ansatz, preact_trainable)
    model = build(solver, ansatz, preact_trainable, **solver_kwargs)
    for name in freeze:
        reference.get_parameter(name).requires_grad_(False)
        model.get_parameter(name).requires_grad_(False)
    expected = model_gradients(reference, input_grad)
    got = model_gradients(model, input_grad)
    assert expected.keys() == got.keys()
    for name, want in expected.items():
        assert want.abs().max() > 1e-2, name  # nothing is compared with zero
        assert_close(got[name], want, name, atol=1e-5)


@pytest.mark.parametrize(
    "ansatz, preact_trainable, solver_kwargs",
    [
        ("pz_encoding", True, {}),
        ("rpz_encoding", False, {}),
        ("real", False, {}),
        ("pz_encoding", False, {"parallel_qubits": 3}),
    ],
)
def test_model_gradients_match_exact(solver, ansatz, preact_trainable, solver_kwargs):
    """Both layers of a model get the gradients the exact solver gives them."""
    assert_model_gradients_match(solver, ansatz, preact_trainable, **solver_kwargs)


@pytest.mark.parametrize(
    "ansatz, freeze, input_grad",
    [
        # a layer with nothing to train still gives its input a gradient
        ("real", ["layers.1.theta"], True),
        # rpz keeps a trainable bias with preact_trainable=False: in the first
        # layer it is then the only tensor that needs a gradient
        ("rpz_encoding", ["layers.0.theta", "layers.1.theta"], False),
    ],
)
def test_frozen_theta(solver, ansatz, freeze, input_grad):
    assert_model_gradients_match(solver, ansatz, False, freeze, input_grad)


def test_backward_runs_the_circuits_of_its_solver(solver, monkeypatch):
    """The shifted circuits run where the forward ran."""
    module = importlib.import_module(f"qkan.solver.{solver}")
    evaluate = getattr(module, f"_{solver}_evaluate")
    runs = []

    def counted(*args):
        runs.append(args)
        return evaluate(*args)

    monkeypatch.setattr(module, f"_{solver}_evaluate", counted)
    model_gradients(build(solver, "real", False))
    # per layer: the forward, 2 for each of its 4 thetas, 2 per rep for its input
    assert len(runs) == 2 * (1 + 2 * 4 + 2 * 2)
