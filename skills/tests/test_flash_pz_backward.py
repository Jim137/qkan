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

"""GPU checks for the flash PZ backward and its launch policies.

With no policy named, the backward must accept every call it accepted before
launch policies existed.  A named policy may only change how the kernel is
launched: never what it computes, and never which calls succeed.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from qkan import QKAN, QKANLayer  # noqa: E402
from qkan.solver import _FLASH_AVAILABLE  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton"),
]


def _input(batch, dim, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, dim, generator=g) * 2 - 1).to("cuda")


def _grads(width, x, *, out_seed=2, **kwargs):
    """Gradients of a fixed random projection of a freshly seeded QKAN."""

    torch.manual_seed(0)
    model = QKAN(list(width), device="cuda", seed=0, **kwargs)
    x = x.to(model.layers[0].p_dtype).clone().requires_grad_(True)
    y = model(x)
    g = torch.Generator().manual_seed(out_seed)
    weight = torch.rand(y.shape, generator=g).to(y.device, y.dtype)
    (y * weight).sum().backward()
    grads = {"x": x.grad}
    grads.update({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    return grads


def _assert_close(got, ref, rel):
    assert got.keys() == ref.keys()
    for name, expected in ref.items():
        expected = expected.double()
        scale = expected.abs().max().clamp_min(1e-6)
        error = (got[name].double() - expected).abs().max()
        assert error <= rel * scale, (
            f"{name}: |error| {error:.3g} > {rel} * {scale:.3g}"
        )


# ── The default launch keeps every call master accepted ─────────────────────


@pytest.mark.parametrize(
    "c_dtype, rel",
    [(torch.complex128, 1e-5), (torch.float16, 2e-2)],
    ids=["float64", "float16"],
)
def test_pz_backward_runs_in_float64_and_float16(c_dtype, rel):
    # QKANLayer maps complex128 to float64 for flash; float16 is passed as is.
    x = _input(16, 4)
    got = _grads((4, 3), x, solver="flash", c_dtype=c_dtype, p_dtype=torch.float32)
    _assert_close(got, _grads((4, 3), x, solver="exact"), rel)


def test_pz_backward_accepts_an_empty_batch():
    model = QKAN([4, 3], solver="flash", device="cuda", seed=0)
    x = torch.empty(0, 4, device="cuda", requires_grad=True)
    model(x).sum().backward()
    assert x.grad.shape == (0, 4)
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not p.grad.any(), name


def test_pz_backward_matches_exact_with_zero_reps():
    x = _input(16, 4)
    got = _grads((4, 3), x, solver="flash", reps=0)
    _assert_close(got, _grads((4, 3), x, solver="exact", reps=0), 1e-5)


def test_pz_backward_accepts_a_layer_without_outputs():
    layer = QKANLayer(4, 0, device="cuda", solver="flash", seed=0)
    x = _input(8, 4).requires_grad_(True)
    layer(x).sum().backward()
    assert not x.grad.any()


def test_torch_compile_trains_the_default_launch_in_one_graph():
    torch._dynamo.reset()
    x = _input(64, 8)
    eager = _grads((8, 4, 2), x, solver="flash", reps=3)

    torch.manual_seed(0)
    model = QKAN([8, 4, 2], reps=3, device="cuda", solver="flash", seed=0)
    compiled = torch.compile(model, fullgraph=True)
    xc = x.clone().requires_grad_(True)
    y = compiled(xc)
    weight = torch.rand(y.shape, generator=torch.Generator().manual_seed(2))
    (y * weight.to(y.device)).sum().backward()
    got = {"x": xc.grad}
    got.update({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    _assert_close(got, eager, 1e-5)


def test_default_launch_never_reads_a_policy(monkeypatch):
    from qkan.solver.flash import fused_ops, launch_policy

    def forbidden(*args, **kwargs):
        raise AssertionError("the default launch consulted the policy machinery")

    for module in (launch_policy, fused_ops):
        for name in ("load_launch_policy", "select_pz_backward_policy"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)

    x = _input(16, 4)
    for solver_kwargs in ({}, {"pz_launch_policy": "default"}):
        _grads((4, 3), x, solver="flash", solver_kwargs=solver_kwargs)


# ── FP8 ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("preact_trainable", [False, True])
def test_fp8_pz_gradients_track_fp32(preact_trainable):
    x = _input(64, 8)
    common = dict(solver="flash", preact_trainable=preact_trainable)
    ref = _grads((8, 8, 4), x, c_dtype=torch.float32, p_dtype=torch.float32, **common)
    got = _grads(
        (8, 8, 4), x, c_dtype=torch.float8_e4m3fn, p_dtype=torch.bfloat16, **common
    )
    for name, expected in ref.items():
        cosine = torch.nn.functional.cosine_similarity(
            got[name].float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine > 0.99, f"{name}: cosine {cosine.item():.4f}"


# ── Named policies ──────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_policy(monkeypatch):
    """Register a policy tuned for this GPU and one exact PZ shape."""

    from qkan.solver.flash import fused_ops, launch_policy

    real_load = launch_policy.load_launch_policy
    loaded = {}

    def load(name):
        return loaded[name] if name in loaded else real_load(name)

    def make(name, *, n_oi, batch, block_b, num_warps=2, num_stages=1):
        index = torch.cuda.current_device()
        document = {
            "schema_version": 1,
            "name": name,
            "operation": "pz_backward",
            "architecture": {
                "kind": "exact",
                "value": fused_ops._device_architecture("cuda", index),
            },
            "match": {
                "kind": "exact",
                "n_oi": n_oi,
                "batch": batch,
                "reps": 3,
                "preacts_trainable": False,
                "fast_measure": True,
            },
            "block_selector": {"kind": "fixed", "block_b": block_b},
            "precisions": {
                precision: {
                    "num_warps": num_warps,
                    "waves_per_eu": 0,
                    "num_stages": num_stages,
                }
                for precision in ("fp32", "bf16", "fp8")
            },
        }
        loaded[name] = launch_policy.validate_launch_policy_document(name, document)
        return name

    monkeypatch.setattr(launch_policy, "load_launch_policy", load)
    fused_ops._resolve_pz_launch.cache_clear()
    launch_policy._warn_policy_fallback.cache_clear()
    yield make
    fused_ops._resolve_pz_launch.cache_clear()
    launch_policy._warn_policy_fallback.cache_clear()


def test_named_policy_changes_the_launch_not_the_gradients(synthetic_policy):
    from qkan.solver.flash import fused_ops

    name = synthetic_policy("synthetic-n32-batch64", n_oi=32, batch=64, block_b=64)
    x = _input(64, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the 4 -> 2 layer falls back
        got = _grads(
            (8, 4, 2),
            x,
            solver="flash",
            reps=3,
            solver_kwargs={"pz_launch_policy": name},
        )
    launch = fused_ops.resolve_pz_launch(
        name,
        torch.device("cuda"),
        n_oi=32,
        batch=64,
        reps=3,
        c_dtype=torch.float32,
        preacts_trainable=False,
        fast_measure=True,
    )
    assert launch is not None and launch.block_b == 64
    assert dict(launch.options) == {"num_warps": 2, "num_stages": 1}
    _assert_close(got, _grads((8, 4, 2), x, solver="flash", reps=3), 1e-5)


def test_named_policy_trains_through_unmatched_layers_and_batches(synthetic_policy):
    # One solver_kwargs dict reaches every layer, and the last batch of an
    # epoch is short: neither may fail mid-training.
    name = synthetic_policy("synthetic-n256-batch512", n_oi=256, batch=512, block_b=512)
    torch.manual_seed(0)
    model = QKAN(
        [16, 16, 1],
        device="cuda",
        solver="flash",
        seed=0,
        solver_kwargs={"pz_launch_policy": name},
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for batch in (512, 76, 512):
            loss = model(_input(batch, 16, seed=batch)).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    fallbacks = [w for w in caught if name in str(w.message)]
    assert len(fallbacks) == 1, [str(w.message) for w in caught]


def test_torch_compile_trains_with_a_named_policy(synthetic_policy):
    torch._dynamo.reset()
    name = synthetic_policy("synthetic-n32-batch64", n_oi=32, batch=64, block_b=64)
    x = _input(64, 8)
    eager = _grads((8, 4, 2), x, solver="flash", reps=3)

    torch.manual_seed(0)
    model = QKAN(
        [8, 4, 2],
        reps=3,
        device="cuda",
        solver="flash",
        seed=0,
        solver_kwargs={"pz_launch_policy": name},
    )
    compiled = torch.compile(model)
    xc = x.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = compiled(xc)
    weight = torch.rand(y.shape, generator=torch.Generator().manual_seed(2))
    (y * weight.to(y.device)).sum().backward()
    got = {"x": xc.grad}
    got.update({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    _assert_close(got, eager, 1e-5)


def _packaged_policy_names():
    from importlib import resources

    package = resources.files("qkan.solver.flash.policies")
    return sorted(
        path.name.removesuffix(".json")
        for path in package.iterdir()
        if path.name.endswith(".json")
    )


@pytest.mark.parametrize("name", _packaged_policy_names())
def test_packaged_policy_for_this_gpu_keeps_the_gradients(name):
    from qkan.solver.flash import fused_ops, launch_policy

    policy = launch_policy.load_launch_policy(name)
    device = torch.device("cuda", torch.cuda.current_device())
    architecture = fused_ops._device_architecture("cuda", device.index)
    if policy.architecture.value != architecture:
        pytest.skip(
            f"tuned for {policy.architecture.value}, this GPU is {architecture}"
        )

    match = policy.match
    in_dim = max(d for d in range(1, int(match.n_oi**0.5) + 1) if match.n_oi % d == 0)
    out_dim = match.n_oi // in_dim
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.rand(match.batch, in_dim, generator=g, device="cuda") * 2 - 1
    theta = torch.randn(out_dim, in_dim, match.reps + 1, 2, generator=g, device="cuda")
    weight = torch.ones(out_dim, in_dim, match.reps, device="cuda")
    bias = torch.zeros(out_dim, in_dim, match.reps, device="cuda")
    grad_out = torch.randn(match.batch, out_dim, in_dim, generator=g, device="cuda")
    for c_dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn):
        launch = fused_ops.resolve_pz_launch(
            name,
            device,
            n_oi=match.n_oi,
            batch=match.batch,
            reps=match.reps,
            c_dtype=c_dtype,
            preacts_trainable=match.preacts_trainable,
            fast_measure=match.fast_measure,
        )
        assert launch is not None, c_dtype
        args = (x, theta, weight, bias, grad_out, False, True)
        got = fused_ops.triton_pz_backward(*args, c_dtype=c_dtype, launch=launch)
        ref = fused_ops.triton_pz_backward(*args, c_dtype=c_dtype)
        names = ("grad_x", "grad_theta")
        _assert_close(dict(zip(names, got[:2])), dict(zip(names, ref[:2])), 1e-4)
