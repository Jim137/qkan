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

"""TELinear: an nn.Linear drop-in on Transformer Engine (NVFP4 / FP8).
Its torch.autocast parity with nn.Linear is in test_linear_autocast.py."""

import math

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

te = pytest.importorskip("transformer_engine.pytorch")
TELinear = pytest.importorskip("qkan.solver.te_linear").TELinear

from transformer_engine.common import recipe  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not te.is_nvfp4_available(),
    reason="needs a GPU with NVFP4 support",
)


def pair(k, n):
    """An nn.Linear and a TELinear made from it."""
    torch.manual_seed(0)
    ref = nn.Linear(k, n, device="cuda", dtype=torch.bfloat16)
    return TELinear.from_linear(ref), ref


def run(module, x, rec=None):
    module.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    with te.autocast(enabled=rec is not None, recipe=rec):
        y = module(x)
    y.float().pow(2).mean().backward()
    return y.detach(), x.grad, module.weight.grad


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_init_matches_nn_linear():
    """te.Linear draws its weight from a normal and zeroes its bias."""
    lin = TELinear(768, 2304)
    bound = 1 / math.sqrt(768)
    for _ in range(2):  # at construction and after reset_parameters()
        assert lin.weight.abs().max() <= bound  # kaiming_uniform(a=sqrt(5))
        assert abs(lin.weight.std().item() - bound / math.sqrt(3)) < 1e-3
        assert 0 < lin.bias.abs().max() <= bound
        lin.reset_parameters()


@pytest.mark.parametrize(
    "rec, max_err",
    [
        (None, 1e-2),
        (recipe.Float8CurrentScaling(), 0.1),
        (recipe.NVFP4BlockScaling(), 0.25),
    ],
    ids=["no-recipe", "fp8", "nvfp4"],
)
def test_matches_nn_linear(rec, max_err):
    """Output, input grad and weight grad against bf16 nn.Linear; a zero
    weight grad (TE #3281 on sm_120) would fail too."""
    lin, ref = pair(768, 2304)
    x = torch.randn(1024, 768, device="cuda", dtype=torch.bfloat16)
    for out, ref_out in zip(run(lin, x, rec), run(ref, x)):
        assert rel(out, ref_out) < max_err


def test_nvfp4_under_torch_autocast_with_fp32_activations():
    """LayerNorm under torch.autocast hands the next layer fp32; TE's NVFP4
    input quantizer (with its Hadamard transform) only accepts bf16."""
    torch.manual_seed(0)
    lin = TELinear(768, 2304)  # fp32 parameters, as with AMP
    x = torch.randn(1024, 768, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with te.autocast(recipe=recipe.NVFP4BlockScaling()):
            y = lin(x)
    y.float().sum().backward()
    assert y.dtype == torch.bfloat16 and x.grad.dtype == torch.float32
    assert torch.isfinite(lin.weight.grad).all()


@pytest.mark.parametrize(
    "m, k, n",
    [(1024, 768, 10), (1024, 10, 768), (1000, 768, 2304)],
    ids=["out-features", "in-features", "tokens"],
)
def test_unaligned_shapes_run_unquantized(m, k, n):
    """Plain te.Linear raises on these under an NVFP4 recipe."""
    lin, _ = pair(k, n)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    for out, plain in zip(run(lin, x, recipe.NVFP4BlockScaling()), run(lin, x)):
        torch.testing.assert_close(out, plain)
