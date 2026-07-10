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


def test_dtype_cast_flag_refreshes_on_to():
    layer = QKANLayer(2, 2, reps=2, solver="exact", seed=0)
    assert layer._needs_dtype_cast is (layer.c_dtype != layer.p_dtype)
    layer.to(dtype=torch.bfloat16)  # keyword path
    assert layer.p_dtype == torch.bfloat16
    assert layer._needs_dtype_cast is (layer.c_dtype != torch.bfloat16)


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
