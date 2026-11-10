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


"""Regressions from the PR #19 review (CPU-only, exact solver)."""

import pytest

torch = pytest.importorskip("torch")

from qkan import QKAN  # noqa: E402
from qkan.qkan import QKANLayer  # noqa: E402


def _dataset(n_var=2, n=32):
    x = torch.rand(n, n_var) * 2 - 1
    y = x.sum(dim=1, keepdim=True)
    return {
        "train_input": x,
        "train_label": y,
        "test_input": x.clone(),
        "test_label": y.clone(),
    }


def test_pruned_mask_survives_state_dict_load():
    # A pruned checkpoint's mask must stay effective after load — the
    # identity fast path previously skipped the `* mask` multiplies.
    src = QKANLayer(2, 2, reps=2, solver="exact", seed=0)
    src._as_oi(src.mask).data[0, 0] = 0.0
    src._mask_is_identity = False

    dst = QKANLayer(2, 2, reps=2, solver="exact", seed=1)
    dst.load_state_dict(src.state_dict())
    assert dst._mask_is_identity is False

    x = torch.rand(8, 2)
    for mode in (True, False):
        src.train(mode)
        dst.train(mode)
        assert torch.allclose(src(x), dst(x), atol=1e-6)

    # The reverse: an identity-mask checkpoint restores the fast path.
    clean = QKANLayer(2, 2, reps=2, solver="exact", seed=0)
    dst.load_state_dict(clean.state_dict())
    assert dst._mask_is_identity is True


def test_frozen_nonzero_postact_bias_is_honored():
    # A checkpoint trained with postact_bias_trainable=True loaded into a
    # frozen-bias layer must not drop the stored bias values.
    src = QKANLayer(2, 2, reps=2, solver="exact", seed=0, postact_bias_trainable=True)
    src.postact_bias.data.fill_(0.5)

    dst = QKANLayer(2, 2, reps=2, solver="exact", seed=0, postact_bias_trainable=False)
    dst.load_state_dict(src.state_dict())
    assert dst._pb_is_nonzero is True

    x = torch.rand(8, 2)
    for mode in (True, False):
        src.train(mode)
        dst.train(mode)
        assert torch.allclose(src(x), dst(x), atol=1e-6)


def test_grouped_layer_autoprune_does_not_raise():
    model = QKAN([3, 2], reps=2, group=1, solver="exact", seed=0)
    model.train_(
        _dataset(n_var=3),
        steps=1,
        prune_every=1,
        prune_threshold=1e9,
        verbose=False,
    )
    # Shared theta has no per-edge magnitude — mask must be untouched.
    assert model.layers[0]._mask_is_identity is True


def test_adam_then_lbfgs_through_train_():
    from qkan.optim import adam_then_lbfgs

    model = QKAN([2, 2, 1], reps=2, solver="exact", seed=0)
    opt = adam_then_lbfgs(model, total_steps=4, pct_adam=0.5)
    results = model.train_(_dataset(), steps=4, optimizer=opt, verbose=False)
    assert len(results["train_loss"]) == 4  # crossed the LBFGS switch


CASTS = {
    "to(dtype=)": lambda m: m.to(dtype=torch.bfloat16),
    "bfloat16()": lambda m: m.bfloat16(),
    "parent.to()": lambda m: torch.nn.Sequential(m).to(torch.bfloat16)[0],
}


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("cast", CASTS.values(), ids=CASTS.keys())
@pytest.mark.parametrize("p_dim", [4, 2])
def test_cached_state_follows_a_cast(cast, p_dim, training):
    """Only QKANLayer.to() refreshed p_dtype, the p_dim=2 views and the eval
    folds; .bfloat16() or a parent's .to() left them stale (#42)."""
    layer = QKANLayer(
        2, 2, reps=2, solver="exact", seed=0, p_dim=p_dim, postact_bias_trainable=True
    )
    layer = cast(layer.train(training))
    assert layer.p_dtype == torch.bfloat16
    # Solver outputs are cast to p_dtype; float32 postacts would promote the
    # output to float32.
    assert layer(torch.randn(3, 2, dtype=torch.bfloat16)).dtype == torch.bfloat16


@pytest.mark.parametrize("p_dim", [4, 2])
def test_load_state_dict_assign_refreshes_cached_state(p_dim):
    """assign=True replaces the parameters; the forward kept reading the old
    ones, so the new ones never got a gradient (#42)."""
    torch.manual_seed(0)
    src = QKANLayer(2, 2, reps=2, solver="exact", p_dim=p_dim)
    layer = QKANLayer(2, 2, reps=2, solver="exact", p_dim=p_dim)
    layer.load_state_dict(src.state_dict(), assign=True)
    x = torch.randn(3, 2)
    torch.testing.assert_close(layer(x), src(x))
    layer(x).sum().backward()
    assert layer.theta.grad is not None


def test_refine_keeps_the_cast_dtype():
    model = QKAN([2, 3, 1], reps=2, solver="exact", seed=0).bfloat16()
    assert next(model.refine(3).parameters()).dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_follows_a_parent_move():
    layer = torch.nn.Sequential(QKANLayer(2, 2, reps=2, solver="exact")).cuda()[0]
    assert layer.device.type == "cuda"
    model = QKAN([2, 3, 1], reps=2, solver="exact").cuda()
    assert next(model.refine(3).parameters()).is_cuda


def test_adabelief_eager_fallback_mixed_state_dtype():
    from qkan.optim import TritonAdaBelief

    for p_dtype, s_dtype in (
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ):
        p = torch.nn.Parameter(torch.ones(4, dtype=p_dtype))
        opt = TritonAdaBelief([p], lr=1e-2, state_dtype=s_dtype)
        for _ in range(2):
            p.grad = torch.full_like(p, 0.1)
            opt.step()  # CPU -> eager fallback; must not raise
        state = opt.state[p]
        assert state["m"].dtype == s_dtype


def test_graphed_param_swap_migrates_optimizer_names():
    from qkan.inference import _reallocate_params_on_stream
    from qkan.optim import QKANMuon

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.randn(3, 3))
            self.fc = torch.nn.Linear(3, 3)

    model = Toy()
    opt = QKANMuon(model.named_parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()  # populate the partition cache

    _reallocate_params_on_stream(model, opt)
    names = set(opt._param_names.values())
    assert "base_weight" in names and "fc.weight" in names
    live = {id(p) for p in model.parameters()}
    assert {id(p) for p in opt.param_groups[0]["params"]} <= live
    # A step after the swap must update the live parameters.
    before = model.fc.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.1)
    opt.step()
    assert not torch.equal(before, model.fc.weight.detach())


@pytest.mark.parametrize("solver", ["flash", "cute", "cutile", "flydsl"])
def test_complex_c_dtype_kept_for_exact_fallback(solver):
    # Kernel-backed ansatzes run in real f32; the rest fall back to
    # torch_exact_solver, whose Rz/Rx gates fail on a real dtype.
    def c_dtype(ansatz):
        return QKANLayer(2, 2, reps=2, solver=solver, ansatz=ansatz).c_dtype

    assert c_dtype("pz") == torch.float32
    assert c_dtype("rpz") == torch.float32
    assert c_dtype("mix") == torch.complex64
    assert c_dtype("px") == torch.complex64
