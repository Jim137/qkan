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

"""refine() and initialize_from_another_model() for every built-in ansatz (#55)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan import QKAN  # noqa: E402
from qkan.solver import _FLASH_AVAILABLE  # noqa: E402

# ansatz -> (final variational layer: 1 extra theta slice, params per slice)
THETA = {
    "pz_encoding": (1, 2),
    "rpz_encoding": (1, 1),
    "px_encoding": (1, 1),
    "mix": (1, 2),
    "real": (0, 1),
}
ANSATZES = list(THETA)

# (ansatz, preact_trainable) for which more reps compute the same function.
# Left out: px, for which no identity is set up, and real with frozen preacts,
# where every added rep applies Ry(x) and raises the frequency of the
# activation.
PRESERVING = [
    ("pz_encoding", False),
    ("pz_encoding", True),
    ("rpz_encoding", False),
    ("rpz_encoding", True),
    ("mix", False),
    ("mix", True),
    ("real", True),
]


def make_model(ansatz, reps, p_dim=4, preact_trainable=True, width=(2, 3, 1), **kwargs):
    model = QKAN(
        list(width),
        reps=reps,
        ansatz=ansatz,
        preact_trainable=preact_trainable,
        postact_weight_trainable=True,
        postact_bias_trainable=True,
        ba_trainable=True,
        c_dtype=torch.float32 if ansatz == "real" else torch.complex64,
        p_dim=p_dim,
        **kwargs,
    )
    # Stand-in for training: no parameter keeps its reset value.  The preacts
    # keep |w * x + b| < 1 on [-0.8, 0.8], the domain of px's acos.
    with torch.no_grad():
        for layer in model.layers:
            layer.theta.normal_()
            layer.preacts_weight.uniform_(0.2, 0.7)
            layer.preacts_bias.uniform_(-0.2, 0.2)
            layer.postact_weights.normal_()
            layer.postact_bias.normal_()
            layer.base_weight.normal_()
    return model


def inputs(device="cpu"):
    return torch.rand(64, 2, device=device) * 1.6 - 0.8


def assert_same_function(model, reference, x):
    out = model(x)
    torch.testing.assert_close(out, reference(x), rtol=1e-5, atol=1e-5)
    out.sum().backward()  # and training can go on from there
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize("preact_trainable", [False, True])
@pytest.mark.parametrize("p_dim", [2, 4])
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_refine_copies_every_rep(ansatz, p_dim, preact_trainable):
    """real has no final theta slice; copying one raised IndexError."""
    torch.manual_seed(0)
    model = make_model(ansatz, 2, p_dim, preact_trainable)
    refined = model.refine(4)
    final, per_slice = THETA[ansatz]
    for src, dst in zip(model.layers, refined.layers):
        theta, new_theta = src._theta_natural(), dst._theta_natural()
        assert new_theta.shape == (dst.out_dim, dst.in_dim, 4 + final, per_slice)
        assert torch.equal(new_theta[..., :2, :], theta[..., :2, :])
        if final and ansatz != "mix":  # mix re-bases its old final layer
            assert torch.equal(new_theta[..., 2, :], theta[..., 2, :])
        assert torch.equal(dst._preacts_w_natural()[..., :2], src._preacts_w_natural())
        assert torch.equal(dst._preacts_b_natural()[..., :2], src._preacts_b_natural())
        for name in ("postact_weights", "postact_bias", "base_weight"):
            assert torch.equal(getattr(dst, name), getattr(src, name))


@pytest.mark.parametrize("fast_measure", [True, False])
@pytest.mark.parametrize("p_dim", [2, 4])
@pytest.mark.parametrize("ansatz, preact_trainable", PRESERVING)
def test_refine_preserves_function(ansatz, preact_trainable, p_dim, fast_measure):
    torch.manual_seed(0)
    model = make_model(ansatz, 2, p_dim, preact_trainable, fast_measure=fast_measure)
    x = inputs()
    for new_reps in (3, 4, 5):
        assert_same_function(model.refine(new_reps), model, x)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _FLASH_AVAILABLE),
    reason="needs CUDA and triton",
)
@pytest.mark.parametrize("p_dim", [2, 4])
@pytest.mark.parametrize(
    "ansatz, preact_trainable", [case for case in PRESERVING if case[0] != "mix"]
)
def test_refine_preserves_function_flash(ansatz, preact_trainable, p_dim):
    """mix has no fused kernel."""
    torch.manual_seed(0)
    model = make_model(
        ansatz, 2, p_dim, preact_trainable, solver="flash", device="cuda"
    )
    x = inputs("cuda")
    for new_reps in (3, 4, 5):
        assert_same_function(model.refine(new_reps), model, x)


@pytest.mark.parametrize("p_dims", [(2, 4), (4, 2)], ids=["2to4", "4to2"])
@pytest.mark.parametrize("ansatz, preact_trainable", PRESERVING)
def test_initialize_from_another_model(ansatz, preact_trainable, p_dims):
    """Called directly, across p_dim, on a target whose parameters are not at
    their reset values."""
    torch.manual_seed(0)
    src = make_model(ansatz, 2, p_dims[0], preact_trainable)
    dst = make_model(ansatz, 3, p_dims[1], preact_trainable)
    assert dst.initialize_from_another_model(src) is dst
    assert_same_function(dst, src, inputs())


@pytest.mark.parametrize("p_dims", [(2, 4), (4, 2)], ids=["2to4", "4to2"])
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_same_reps_is_a_plain_copy(ansatz, p_dims):
    """Weight transfer to a model of equal reps (e.g. onto a QPU solver)."""
    torch.manual_seed(0)
    # one layer for px: the next one would be fed values outside acos's domain
    width = (2, 3) if ansatz == "px_encoding" else (2, 3, 1)
    src = make_model(ansatz, 2, p_dims[0], width=width)
    dst = make_model(ansatz, 2, p_dims[1], width=width)
    dst.initialize_from_another_model(src)
    for a, b in zip(src.layers, dst.layers):
        assert torch.equal(b._theta_natural(), a._theta_natural())
        assert torch.equal(b._preacts_w_natural(), a._preacts_w_natural())
        assert torch.equal(b._preacts_b_natural(), a._preacts_b_natural())
    assert_same_function(dst, src, inputs())


@pytest.mark.parametrize("new_reps", [2, 3])
@pytest.mark.parametrize("group", [2, -1])
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding"])
def test_shared_theta_spreads_over_groups(ansatz, group, new_reps):
    """A theta shared by all edges (group=1) broadcasts onto every group or
    edge of the target."""
    torch.manual_seed(0)
    src = make_model(ansatz, 2, group=1)
    dst = make_model(ansatz, new_reps, group=group)
    dst.initialize_from_another_model(src)
    assert_same_function(dst, src, inputs())


def test_frozen_postact_bias_is_kept():
    """A bias that came with a checkpoint into a layer that does not train it:
    the values were copied, but the forward went on without them."""
    torch.manual_seed(0)
    trained = make_model("pz_encoding", 2)
    model = QKAN([2, 3, 1], reps=2, ba_trainable=True)
    model.load_state_dict(trained.state_dict())
    assert_same_function(model.refine(3), model, inputs())


def test_fewer_reps_than_the_source_is_rejected():
    src = make_model("pz_encoding", 3)
    dst = make_model("pz_encoding", 2)
    with pytest.raises(AssertionError, match="more reps"):
        dst.initialize_from_another_model(src)


def pz_like_solver(x, theta, preacts_weight, preacts_bias, reps, **kwargs):
    """A user solver with the pz layout: theta (out, in, reps + 1, 2)."""
    from qkan.solver.exact import torch_exact_solver

    return torch_exact_solver(
        x, theta, preacts_weight, preacts_bias, reps, out_dim=theta.shape[0]
    )


def custom_model(reps, theta_size, **kwargs):
    model = QKAN([2, 3], reps=reps, theta_size=theta_size, **kwargs)
    with torch.no_grad():
        model.layers[0].theta.normal_()
    return model


def test_custom_theta_grown_by_the_user():
    """Layer extension of a custom solver: refine() cannot grow theta_size, so
    the caller builds the larger model. Added Rz data gates are the identity."""
    src = custom_model(1, [3, 2, 2, 2], solver=pz_like_solver)
    dst = custom_model(3, [3, 2, 4, 2], solver=pz_like_solver)
    dst.layers[0].theta.data.zero_()
    dst.initialize_from_another_model(src)
    assert_same_function(dst, src, inputs())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(theta_size=[7], ansatz=lambda theta: theta),
        dict(theta_size=[3, 2, 4, 2], ansatz=lambda theta: theta),
        # a name says nothing about what a custom solver does with theta
        dict(theta_size=[3, 2, 4, 2], ansatz="real", solver=pz_like_solver),
        dict(theta_size=[3, 2, 4, 2], ansatz="mix", solver=pz_like_solver),
    ],
    ids=["1d", "4d", "solver-real", "solver-mix"],
)
def test_custom_theta_is_copied_untouched(kwargs):
    model = custom_model(1, **kwargs)
    refined = model.refine(2)
    assert torch.equal(refined.layers[0].theta, model.layers[0].theta)


def test_custom_theta_of_another_rank_is_rejected():
    src = custom_model(1, [7], ansatz=lambda theta: theta)
    dst = custom_model(1, [3, 7], ansatz=lambda theta: theta)
    with pytest.raises(AssertionError, match="another rank"):
        dst.initialize_from_another_model(src)
