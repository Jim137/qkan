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

"""QKANLayer output dtype follows the dtype of its own F.linear term (#23)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from qkan import QKAN, QKANLayer  # noqa: E402
from qkan.solver import _CUTE_AVAILABLE, _FLASH_AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="autocast dtype checks need CUDA"
)

SOLVERS = [
    "exact",
    pytest.param(
        "flash", marks=pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton")
    ),
    pytest.param(
        "cute",
        marks=pytest.mark.skipif(not _CUTE_AVAILABLE, reason="no CuTe extension"),
    ),
]
DTYPES = [torch.float32, torch.bfloat16]
AUTOCAST = dict(device_type="cuda", dtype=torch.bfloat16)


def make_layer(solver, dtype, fused=False, **kwargs):
    torch.manual_seed(0)
    layer = QKANLayer(12, 10, reps=2, solver=solver, device="cuda", **kwargs)
    layer = layer.to(dtype)
    layer.set_fused_epilogue(fused)  # independent of QKAN_FUSED_EPILOGUE
    x = torch.randn(48, 12, device="cuda", dtype=dtype, requires_grad=True)
    return layer, x


@pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_train_route(solver, dtype, fused):
    """Outside autocast the output keeps the parameter dtype; under bf16
    autocast it is bf16 -- for fp32 parameters too, as nn.Linear behaves."""
    layer, x = make_layer(solver, dtype, fused)
    assert layer(x).dtype == dtype
    with torch.autocast(**AUTOCAST):
        y = layer(x)
    assert y.dtype == torch.bfloat16
    y.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == dtype


@pytest.mark.parametrize("fused", [False, True], ids=["eager", "fused"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("solver", SOLVERS)
def test_eval_route(solver, dtype, fused):
    layer, x = make_layer(solver, dtype, fused, postact_bias_trainable=True)
    layer.eval()
    with torch.no_grad():
        assert layer(x).dtype == dtype
        with torch.autocast(**AUTOCAST):
            assert layer(x).dtype == torch.bfloat16


def test_eval_cache_built_under_autocast_works_outside_it():
    """.eval() folds the postact bias into a cached F.linear bias with a
    `sum`; built under autocast that cache came out fp32, so a later forward
    outside autocast failed with "self and mat2 must have the same dtype"."""
    layer, x = make_layer("exact", torch.bfloat16, postact_bias_trainable=True)
    with torch.no_grad():
        layer.postact_bias.normal_()
        with torch.autocast(**AUTOCAST):
            layer.eval()
        assert layer(x).dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", DTYPES)
def test_stacked_network(dtype):
    """Layers hand each other the autocast dtype, and the stack still trains."""
    torch.manual_seed(0)
    net = QKAN(width=[12, 8, 6], reps=2, solver="exact", device="cuda").to(dtype)
    x = torch.randn(48, 12, device="cuda", dtype=dtype, requires_grad=True)
    with torch.autocast(**AUTOCAST):
        y = net(x)
    assert y.dtype == torch.bfloat16
    y.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_epilogue_stack_matches_eager(dtype):
    """Fused layers return the autocast dtype like eager ones (#33), so the
    second layer gets bf16 input against its own weights in backward."""
    grads = []
    for fused in (False, True):
        torch.manual_seed(0)
        net = nn.Sequential(
            QKANLayer(12, 8, reps=2, solver="exact", device="cuda"),
            QKANLayer(8, 6, reps=2, solver="exact", device="cuda"),
        ).to(dtype)
        for layer in net:
            layer.set_fused_epilogue(fused)
        x = torch.randn(48, 12, device="cuda", dtype=dtype, requires_grad=True)
        with torch.autocast(**AUTOCAST):
            y = net(x)
        assert y.dtype == torch.bfloat16
        y.float().sum().backward()
        grads.append(x.grad)
    torch.testing.assert_close(grads[1], grads[0], rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _CUTE_AVAILABLE, reason="no CuTe extension")
def test_compiled_output_follows_autocast(run_snippet):
    """The crash #23 reported (inductor folding main + F.linear into an addmm
    whose operands disagree) does not reproduce on torch 2.11, which compiles
    but returned fp32; this pins the dtype.  Its own interpreter keeps dynamo
    state out of the other tests."""
    result = run_snippet(
        """
        from qkan import QKANLayer

        torch.manual_seed(0)
        layer = QKANLayer(16, 16, reps=1, solver="cute", device="cuda")
        layer = layer.to(torch.bfloat16)
        x = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            print("OUT", torch.compile(layer)(x).dtype)
        """,
        preamble="import torch\n",
        timeout=900,
    )
    assert "OUT torch.bfloat16" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
