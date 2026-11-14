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
# px needs trainable preacts to run at all. Left out: real with frozen
# preacts, where every added rep applies Ry(x) and raises the frequency of the
# activation; it is rejected.
PRESERVING = [
    ("pz_encoding", False),
    ("pz_encoding", True),
    ("rpz_encoding", False),
    ("rpz_encoding", True),
    ("px_encoding", True),
    ("mix", False),
    ("mix", True),
    ("real", True),
]
ALIASES = {"pz": "pz_encoding", "rpz": "rpz_encoding", "px": "px_encoding"}


def make_model(ansatz, reps, p_dim=4, preact_trainable=True, width=None, **kwargs):
    if width is None:
        # one layer for px: the next one would be fed values outside acos's domain
        width = (2, 3) if ansatz in ("px_encoding", "px") else (2, 3, 1)
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
            if layer.is_batchnorm:
                layer.bn.weight.uniform_(0.5, 1.5)
                layer.bn.bias.normal_()
                layer.bn.running_mean.normal_()
                layer.bn.running_var.uniform_(0.5, 1.5)
                layer.bn.num_batches_tracked.fill_(3)
    return model


def inputs(device="cpu"):
    return torch.rand(64, 2, device=device) * 1.6 - 0.8


def close(out, expected):
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def assert_same_function(model, reference, x):
    out = model(x)
    close(out, reference(x))
    out.sum().backward()  # and training can go on from there
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize("p_dim", [2, 4])
@pytest.mark.parametrize("ansatz, preact_trainable", PRESERVING)
def test_refine_copies_every_rep(ansatz, p_dim, preact_trainable):
    """real has no final theta slice; copying one raised IndexError."""
    torch.manual_seed(0)
    model = make_model(ansatz, 2, p_dim, preact_trainable)
    refined = model.refine(4)
    final, per_slice = THETA[ansatz]
    # the reps of the source: the first of the target, for px the last
    old = slice(2, 4) if ansatz == "px_encoding" else slice(0, 2)
    for src, dst in zip(model.layers, refined.layers):
        theta, new_theta = src._theta_natural(), dst._theta_natural()
        assert new_theta.shape == (dst.out_dim, dst.in_dim, 4 + final, per_slice)
        assert torch.equal(new_theta[..., old, :], theta[..., :2, :])
        if final and ansatz != "mix":  # mix re-bases its old final layer
            assert torch.equal(new_theta[..., old.stop, :], theta[..., 2, :])
        assert torch.equal(dst._preacts_w_natural()[..., old], src._preacts_w_natural())
        assert torch.equal(dst._preacts_b_natural()[..., old], src._preacts_b_natural())
        if ansatz == "px_encoding":  # the added reps, in front
            assert not new_theta[..., :2, :].any()
            assert not dst._preacts_w_natural()[..., :2].any()
            assert not dst._preacts_b_natural()[..., :2].any()
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
    "ansatz, preact_trainable",
    [case for case in PRESERVING if case[0] not in ("mix", "px_encoding")],
)
def test_refine_preserves_function_flash(ansatz, preact_trainable, p_dim):
    """mix and px have no fused kernel."""
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


@pytest.mark.parametrize("alias", ALIASES)
def test_ansatz_aliases(alias):
    torch.manual_seed(0)
    model = make_model(alias, 2)
    x = inputs()
    for new_reps in (3, 4):
        refined = model.refine(new_reps)
        assert_same_function(refined, model, x)
        named = make_model(ALIASES[alias], new_reps)
        named.load_state_dict(refined.state_dict())
        assert_same_function(named, model, x)


@pytest.mark.parametrize("p_dims", [(2, 4), (4, 2)], ids=["2to4", "4to2"])
@pytest.mark.parametrize("ansatz, preact_trainable", PRESERVING + [("real", False)])
def test_same_reps_is_a_plain_copy(ansatz, preact_trainable, p_dims):
    """Weight transfer to a model of equal reps (e.g. onto a QPU solver)."""
    torch.manual_seed(0)
    src = make_model(ansatz, 2, p_dims[0], preact_trainable)
    dst = make_model(ansatz, 2, p_dims[1], preact_trainable)
    dst.initialize_from_another_model(src)
    for a, b in zip(src.layers, dst.layers):
        assert torch.equal(b._theta_natural(), a._theta_natural())
        assert torch.equal(b._preacts_w_natural(), a._preacts_w_natural())
        assert torch.equal(b._preacts_b_natural(), a._preacts_b_natural())
    assert_same_function(dst, src, inputs())


def test_px_added_reps_stay_inside_the_domain_of_acos():
    """Inputs beyond 1, where the reset weight of 1 would leave it."""
    torch.manual_seed(0)
    model = make_model("px_encoding", 2)
    with torch.no_grad():
        model.layers[0].preacts_weight.uniform_(0.2, 0.5)
    x = torch.rand(64, 2) * 2.8 - 1.4
    assert model(x).isfinite().all()
    for new_reps in (3, 4, 5):
        assert_same_function(model.refine(new_reps), model, x)


@pytest.mark.parametrize("new_reps", [2, 3])
@pytest.mark.parametrize("group", [2, -1])
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_shared_theta_spreads_over_groups(ansatz, group, new_reps):
    """A theta shared by all edges (group=1) broadcasts onto every group or
    edge of the target."""
    torch.manual_seed(0)
    src = make_model(ansatz, 2, group=1)
    dst = make_model(ansatz, new_reps, group=group)
    dst.initialize_from_another_model(src)
    assert_same_function(dst, src, inputs())


def unchanged(model, state):
    return all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("p_dims", [(2, 4), (4, 2)], ids=["2to4", "4to2"])
def test_real_with_frozen_preacts_cannot_add_reps(p_dims):
    """Neither a changed function nor a half-initialized target."""
    torch.manual_seed(0)
    src = make_model("real", 2, p_dims[0], preact_trainable=False)
    with torch.no_grad():  # frozen preacts are still what they were built with
        for layer in src.layers:
            layer.preacts_weight.fill_(1)
            layer.preacts_bias.zero_()
    with pytest.raises(ValueError, match="frozen preacts"):
        src.refine(3)

    dst = make_model("real", 3, p_dims[1], preact_trainable=False)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    with pytest.raises(ValueError, match="preact_trainable=True"):
        dst.initialize_from_another_model(src)
    assert unchanged(dst, before)

    # what the message says to do
    trainable = make_model("real", 3, p_dims[1], preact_trainable=True)
    trainable.initialize_from_another_model(src)
    assert_same_function(trainable, src, inputs())


@pytest.mark.parametrize("source_has_it", [True, False])
def test_batchnorm_on_one_side_only_is_rejected(source_has_it):
    src = make_model("pz_encoding", 2, is_batchnorm=source_has_it)
    dst = make_model("pz_encoding", 3, is_batchnorm=not source_has_it)
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    with pytest.raises(ValueError, match="layer 0 has is_batchnorm"):
        dst.initialize_from_another_model(src)
    assert unchanged(dst, before)


def test_another_depth_is_rejected():
    src = make_model("pz_encoding", 2, width=(2, 3, 1))
    for width in ((2, 3), (2, 3, 1, 1)):
        dst = make_model("pz_encoding", 3, width=width)
        before = {k: v.clone() for k, v in dst.state_dict().items()}
        with pytest.raises(AssertionError, match="different width"):
            dst.initialize_from_another_model(src)
        assert unchanged(dst, before)


def pruned_model(p_dim=4, **kwargs):
    """Edges removed in every layer, as prune_edge() does it, with one
    threshold per layer."""
    model = make_model("pz_encoding", 2, p_dim, save_act=True, **kwargs)
    model(inputs())
    model.attribute()
    for l, layer in enumerate(model.layers):
        scores = model.edge_scores[l]
        keep = (scores > scores.median()).to(layer.mask.dtype)
        layer.mask.data.copy_(keep.reshape(layer.mask.shape))
        layer._mask_is_identity = False
    return model


@pytest.mark.parametrize("p_dims", [(2, 2), (4, 4), (2, 4), (4, 2)])
def test_pruning_mask_is_kept(p_dims):
    torch.manual_seed(0)
    model = pruned_model(p_dims[0])
    x = inputs()
    refined = make_model("pz_encoding", 3, p_dims[1], save_act=True)
    refined.initialize_from_another_model(model)
    for src, dst in zip(model.layers, refined.layers):
        assert torch.equal(dst._as_oi(dst.mask), src._as_oi(src.mask))
        assert not dst.mask.all() and dst.mask.any()
    assert_same_function(refined, model, x)
    # and in eval mode, which folds the mask into the weights
    close(refined.eval()(x), model.eval()(x))


@pytest.mark.parametrize("pruned", [True, False], ids=["mask", "identity-mask"])
def test_a_target_in_eval_mode_folds_the_copied_values(pruned):
    """The mask, and the postact bias when the mask is the identity."""
    torch.manual_seed(0)
    model = pruned_model() if pruned else make_model("pz_encoding", 2)
    x = inputs()
    target = make_model("pz_encoding", 3).eval()
    target.initialize_from_another_model(model)
    assert not target.training
    assert all(not m.training for m in target.modules())
    assert all(layer._eff_pw is not None for layer in target.layers)
    close(target(x), model.eval()(x))


def test_a_target_in_train_mode_stays_trainable():
    torch.manual_seed(0)
    model = make_model("pz_encoding", 2)
    for refined in (
        model.refine(3),
        make_model("pz_encoding", 3).initialize_from_another_model(model),
    ):
        assert refined.training and all(m.training for m in refined.modules())
        assert all(layer._eff_pw is None for layer in refined.layers)
        refined(inputs()).sum().backward()
        for name, param in refined.named_parameters():
            assert not param.requires_grad or param.grad is not None, name


@pytest.mark.parametrize("is_batchnorm", [False, True])
def test_refine_keeps_the_mode(is_batchnorm):
    """In eval mode a batchnorm layer uses the running statistics it was given."""
    torch.manual_seed(0)
    model = make_model("pz_encoding", 2, is_batchnorm=is_batchnorm)
    x = inputs()
    refined = model.eval().refine(3)
    assert not refined.training
    close(refined(x), model(x))
    if is_batchnorm:
        for src, dst in zip(model.layers, refined.layers):
            for name, value in src.bn.state_dict().items():
                assert torch.equal(dst.bn.state_dict()[name], value), name
    assert model.train().refine(3).training


def test_a_mask_the_source_does_not_apply_is_not_applied():
    """Written behind the layer's back: the source runs without it."""
    torch.manual_seed(0)
    model = make_model("pz_encoding", 2)
    x = inputs()
    unmasked = model(x)
    model.layers[0].mask.data[0, 0] = 0
    close(model(x), unmasked)
    assert_same_function(model.refine(3), model, x)


PRUNINGS = {
    "prune_node": lambda m: m.prune_node(
        mode="manual", active_neurons_id=[[0, 1], [0, 2], [0]]
    ),
    "prune_input": lambda m: m.prune_input(active_inputs=[0, 1]),
    "prune": lambda m: m.prune(node_th=-1.0, edge_th=-1.0),
}


@pytest.mark.parametrize("is_batchnorm", [False, True])
@pytest.mark.parametrize("prune_edges", [False, True])
@pytest.mark.parametrize("pruning", PRUNINGS)
def test_refine_of_a_pruned_model(pruning, prune_edges, is_batchnorm):
    """A pruned model keeps its mask and its batchnorm layer, and so does
    its refinement."""
    torch.manual_seed(0)
    model = make_model(
        "pz_encoding", 2, width=(2, 4, 1), save_act=True, is_batchnorm=is_batchnorm
    )
    x = inputs()
    model(x)
    if prune_edges:
        model.attribute()
        model.prune_edge(model.edge_scores[0].median().item())
    pruned = PRUNINGS[pruning](model)
    if pruning == "prune_input":  # nothing was removed: still the model
        close(pruned.eval()(x), model.eval()(x))
        pruned.train()
    refined = pruned.refine(3)
    for src, dst in zip(pruned.layers, refined.layers):
        assert dst.is_batchnorm == src.is_batchnorm == is_batchnorm
        assert torch.equal(dst._as_oi(dst.mask), src._as_oi(src.mask))
    assert_same_function(refined, pruned, x)  # train mode: batch statistics
    close(refined.eval()(x), pruned.eval()(x))


def test_constructor_options_are_kept():
    model = make_model(
        "pz_encoding", 2, solver_kwargs={"some_option": 1}, checkpoint_reps=True
    )
    refined = model.refine(3)
    assert refined.solver_kwargs == {"some_option": 1}
    assert refined.solver_kwargs is not model.solver_kwargs
    assert refined.checkpoint_reps is True
    for layer in refined.layers:
        assert layer.solver_kwargs == {"some_option": 1}
        assert layer.checkpoint_reps is True


def test_input_selection_is_kept():
    """A model from prune_input() reads its inputs out of the full x."""
    torch.manual_seed(0)
    model = make_model("pz_encoding", 2, width=(3, 3, 1), save_act=True)
    x = torch.rand(64, 3) * 1.6 - 0.8
    model(x)  # prune_input() hands the cached batch on
    pruned = model.prune_input(active_inputs=[0, 2])
    refined = pruned.refine(3)
    assert torch.equal(refined.input_id, pruned.input_id)
    assert_same_function(refined, pruned, x)


def test_frozen_postact_bias_is_kept():
    """A bias that came with a checkpoint into a layer that does not train it:
    the values were copied, but the forward went on without them."""
    torch.manual_seed(0)
    trained = make_model("pz_encoding", 2)
    model = QKAN([2, 3, 1], reps=2, ba_trainable=True, save_act=True)
    model.load_state_dict(trained.state_dict())
    x = inputs()
    assert_same_function(model.refine(3), model, x)
    # and through a pruning that removes nothing
    pruned = model.prune_input(active_inputs=[0, 1])
    assert_same_function(pruned, model, x)
    assert_same_function(pruned.refine(3), model, x)


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
        dict(theta_size=[3, 2, 4, 2], ansatz="px", solver=pz_like_solver),
    ],
    ids=["1d", "4d", "solver-real", "solver-mix", "solver-px"],
)
def test_custom_theta_is_copied_untouched(kwargs):
    model = custom_model(1, **kwargs)
    with torch.no_grad():
        model.layers[0].preacts_weight.uniform_(0.2, 0.7)
    refined = model.refine(2)
    assert torch.equal(refined.layers[0].theta, model.layers[0].theta)
    # and the preacts fill the leading block, whatever the name
    assert torch.equal(
        refined.layers[0]._preacts_w_natural()[..., :1],
        model.layers[0]._preacts_w_natural(),
    )


def test_custom_theta_of_another_rank_is_rejected():
    src = custom_model(1, [7], ansatz=lambda theta: theta)
    dst = custom_model(1, [3, 7], ansatz=lambda theta: theta)
    with pytest.raises(AssertionError, match="another rank"):
        dst.initialize_from_another_model(src)
