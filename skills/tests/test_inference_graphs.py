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

"""A captured CUDA graph is not replayed once the state it froze has changed:
parameter storage (#47) and the torch.autocast state (#53)."""

import contextlib
import copy
import re

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from qkan import (  # noqa: E402
    QKAN,
    CompiledInference,
    compile_inference,
    graph_submodules,
    make_graphed_inference,
    make_graphed_train_step,
)
from qkan.solver import _FLASH_AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need CUDA"
)

MODELS = [
    "linear",
    "qkan-exact",
    pytest.param(
        "qkan-flash",
        marks=pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton"),
    ),
]


def make_model(kind, seed=0):
    torch.manual_seed(seed)
    if kind == "linear":
        return nn.Linear(8, 8, device="cuda")
    return QKAN([8, 8], solver=kind.removeprefix("qkan-"), device="cuda")


def make_input():
    return torch.randn(16, 8, device="cuda")


# ---------------------------------------------------------------------------
# #47: parameter storage
# ---------------------------------------------------------------------------


def new_state(kind, prefix):
    """Other values for a wrapped ``make_model(kind)``."""
    return {prefix + k: v for k, v in make_model(kind, seed=1).state_dict().items()}


# Each one gives the parameters new storage, or lets the wrapped module rebuild
# what it derives from them, while the input (the cache key) stays the same.
STORAGE_CHANGES = {
    "cast": lambda ci, kind: ci.double().float(),
    "move": lambda ci, kind: ci.cpu().cuda(),
    "parent cast": lambda ci, kind: nn.Sequential(ci).double().float(),
    "load": lambda ci, kind: ci.load_state_dict(new_state(kind, "module.")),
    "assign load": lambda ci, kind: ci.load_state_dict(
        new_state(kind, "module."), assign=True
    ),
    "parent load": lambda ci, kind: nn.Sequential(ci).load_state_dict(
        new_state(kind, "0.module.")
    ),
}


@pytest.mark.parametrize("change", STORAGE_CHANGES)
@pytest.mark.parametrize("kind", MODELS)
def test_storage_change_drops_the_graphs(kind, change):
    model = make_model(kind)
    ci = compile_inference(model).eval()
    x = make_input()
    with torch.no_grad():
        ci(x)
        assert len(ci._cache) == 1
        STORAGE_CHANGES[change](ci, kind)
        assert len(ci._cache) == 0
        torch.testing.assert_close(ci(x), model(x))
        assert len(ci._cache) == 1


def test_cast_round_trip_does_not_read_freed_storage():
    """#47 as reported: another tensor takes the block the cast frees, and the
    graph captured before the cast reads that tensor instead of the weight."""
    lin = nn.Linear(256, 256, device="cuda")
    ci = CompiledInference(lin).eval()
    x = torch.randn(64, 256, device="cuda")
    with torch.no_grad():
        ref = lin(x).clone()
        ci(x)
        ci.double()
        _squatter = torch.full((256, 256), 7.0, device="cuda")
        ci.float()
        torch.testing.assert_close(ci(x), ref)


@pytest.mark.parametrize("kind", MODELS)
def test_graph_submodules_wrappers_follow_the_parent(kind):
    """The wrappers sit inside the model, so casts and loads reach them only
    through the parent."""
    target = make_model(kind)
    net = nn.Sequential(target, nn.Linear(8, 8, device="cuda")).eval()
    x = make_input()
    with torch.no_grad():
        ref = net(x).clone()
    graph_submodules(net, x, predicate=lambda m: m is target)
    wrapper = net[0]
    assert isinstance(wrapper, CompiledInference) and len(wrapper._cache) == 1

    for change in (
        lambda: net.double().float(),
        lambda: net.load_state_dict(net.state_dict()),
    ):
        change()
        assert len(wrapper._cache) == 0
        with torch.no_grad():
            torch.testing.assert_close(net(x), ref)
        assert len(wrapper._cache) == 1


# ---------------------------------------------------------------------------
# #53: torch.autocast state
# ---------------------------------------------------------------------------


def autocast(dtype):
    """``torch.autocast`` in ``dtype``; None is outside autocast."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def wrong_state(captured, current):
    where = [
        "outside torch.autocast" if d is None else f"under torch.autocast(dtype={d})"
        for d in (captured, current)
    ]
    return re.escape(
        f"this graph was captured {where[0]} and replays those kernels: it "
        f"cannot run {where[1]}"
    )


# (state at capture, another state)
OTHER_STATE = [
    pytest.param(None, torch.bfloat16, id="outside-bf16"),
    pytest.param(torch.bfloat16, None, id="bf16-outside"),
    pytest.param(torch.bfloat16, torch.float16, id="bf16-fp16"),
]


@pytest.mark.parametrize(
    "order",
    [(None, torch.bfloat16), (torch.bfloat16, None)],
    ids=["outside first", "autocast first"],
)
@pytest.mark.parametrize("kind", MODELS)
def test_compiled_inference_keeps_one_graph_per_autocast_state(kind, order):
    model = make_model(kind)
    ci = compile_inference(model).eval()
    x = make_input()
    with torch.no_grad():
        for state in order + order:  # capture in each state, then replay
            with autocast(state):
                got, want = ci(x), model(x)
            assert got.dtype == want.dtype == (state or torch.float32)
            torch.testing.assert_close(got, want)
    assert len(ci._cache) == 2


def test_compiled_inference_keys_on_the_cuda_autocast_dtype():
    ci = compile_inference(make_model("linear")).eval()
    x = make_input()
    with torch.no_grad():
        for dtype in (torch.bfloat16, torch.float16):
            with torch.autocast("cuda", dtype=dtype):
                assert ci(x).dtype == dtype
        assert len(ci._cache) == 2
        # CPU autocast leaves CUDA kernels alone: same graph as outside
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert ci(x).dtype == torch.float32
        assert ci(x).dtype == torch.float32
    assert len(ci._cache) == 3


@pytest.mark.parametrize("captured, other", OTHER_STATE)
@pytest.mark.parametrize("kind", MODELS)
def test_graphed_inference_raises_in_another_autocast_state(kind, captured, other):
    model = make_model(kind)
    x = make_input()
    with autocast(captured):
        fn = make_graphed_inference(model, x)
    with torch.no_grad():
        with autocast(captured):
            got, want = fn(x), model(x)
        assert got.dtype == want.dtype == (captured or torch.float32)
        torch.testing.assert_close(got, want)
        with autocast(other):
            with pytest.raises(RuntimeError, match=wrong_state(captured, other)):
                fn(torch.zeros_like(x))
        torch.testing.assert_close(got, want)  # the refused call wrote nothing


@pytest.mark.parametrize("captured, other", OTHER_STATE)
@pytest.mark.parametrize("kind", MODELS)
def test_graphed_train_step_raises_in_another_autocast_state(kind, captured, other):
    model = make_model(kind)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    x, y = make_input(), make_input()
    with autocast(captured):
        train_step = make_graphed_train_step(model, x, y, loss_fn, optimizer)
        loss = train_step(x, y).clone()
        want = loss_fn(model(x), y)
    torch.testing.assert_close(loss, want)
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    with autocast(other):
        with pytest.raises(RuntimeError, match=wrong_state(captured, other)):
            train_step(torch.zeros_like(x), y)
    # the refused call left the gradients of the last step alone
    for grad, p in zip(grads, (p for p in model.parameters() if p.grad is not None)):
        assert torch.equal(grad, p.grad)


# torch.autocast caches its casts of the weights until the context exits. A
# graph that captured a cached cast replays without the cast kernel: it reads
# a tensor that is freed with the context and never sees an optimizer step.


class DecoratedForward(nn.Module):
    """An autocast built ahead of time switches the cache on at every entry."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, x):
        return self.lin(x)


def fused_qkan(solver):
    """Its backward casts base_weight on the autograd thread."""
    model = QKAN([8, 8, 8], solver=solver, device="cuda", ba_trainable=True)
    for layer in model.layers:
        layer.set_fused_epilogue(True)
    return model


TRAINED = {
    "linear": lambda: nn.Linear(8, 8),
    "decorated forward": DecoratedForward,
    "fused qkan-exact": lambda: fused_qkan("exact"),
    "fused qkan-flash": lambda: fused_qkan("flash"),
}


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            kind,
            marks=pytest.mark.skipif(
                "flash" in kind and not _FLASH_AVAILABLE, reason="no triton"
            ),
        )
        for kind in TRAINED
    ],
)
def test_train_step_captured_under_autocast_follows_the_optimizer(kind):
    torch.manual_seed(0)
    eager = TRAINED[kind]().cuda()
    graphed = copy.deepcopy(eager)
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.3) for m in (eager, graphed)]
    loss_fn = nn.MSELoss()
    x, y = make_input(), make_input()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        train_step = make_graphed_train_step(graphed, x, y, loss_fn, optimizers[1])
    for _ in range(4):
        optimizers[0].zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_fn(eager(x), y).backward()
            train_step(x, y)
        for want, got in zip(eager.parameters(), graphed.parameters()):
            if want.grad is not None:
                torch.testing.assert_close(got.grad, want.grad, atol=1e-3, rtol=0)
        for optimizer in optimizers:
            optimizer.step()


CAPTURES = {
    "low-level": make_graphed_inference,
    "compiled": lambda model, x: compile_inference(model).eval(),
}


@pytest.mark.parametrize("capture", CAPTURES.values(), ids=CAPTURES.keys())
def test_autocast_graph_outlives_its_autocast_context(capture):
    lin = nn.Linear(256, 256, device="cuda")
    x = torch.randn(64, 256, device="cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fn = capture(lin, x)
            fn(x)
            want = lin(x).clone()
        # torch hands streams out of a pool of 32 and the allocator keeps a
        # free list per stream: take the blocks the context freed on each one
        held = []
        for _ in range(33):
            with torch.cuda.stream(torch.cuda.Stream()):
                for shape in ((256, 256), (256,)):
                    held.append(
                        torch.full(shape, 7.0, dtype=torch.bfloat16, device="cuda")
                    )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(fn(x), want)


def test_capture_leaves_no_cast_behind_for_eager_code():
    """The casts cached while capturing hold nothing until the first replay."""
    lin = nn.Linear(256, 256, device="cuda")
    x = torch.randn(64, 256, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        want = lin(x).clone()
        make_graphed_inference(lin, x)  # captured, never replayed
        torch.testing.assert_close(lin(x), want)
