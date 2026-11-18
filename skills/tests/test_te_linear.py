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
    return y.detach(), x.grad, module.weight.grad, module.bias.grad


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


# Lowest and highest error against bf16 nn.Linear. bf16 itself is at 3e-3, so
# a quantized GEMM is above the lower bound; a broken one is above the upper.
PLAIN = (None, 0, 1e-2)
FP8 = (recipe.Float8CurrentScaling(), 1e-2, 0.1)
NVFP4 = (recipe.NVFP4BlockScaling(), 5e-2, 0.25)


def check(lin, ref, x, rec, min_err, max_err):
    """Output and the input, weight and bias gradients; a zero weight grad
    (TE #3281 on sm_120) would fail too."""
    for out, ref_out in zip(run(lin, x, rec), run(ref, x)):
        assert out.shape == ref_out.shape
        assert min_err <= rel(out, ref_out) < max_err


@pytest.mark.parametrize(
    "rec, min_err, max_err", [PLAIN, FP8, NVFP4], ids=["no-recipe", "fp8", "nvfp4"]
)
def test_matches_nn_linear(rec, min_err, max_err):
    lin, ref = pair(768, 2304)
    x = torch.randn(1024, 768, device="cuda", dtype=torch.bfloat16)
    check(lin, ref, x, rec, min_err, max_err)


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


def test_from_linear_keeps_frozen_parameters():
    ref = nn.Linear(64, 64, device="cuda").requires_grad_(False)
    assert not any(p.requires_grad for p in TELinear.from_linear(ref).parameters())


def test_compiled_inference_follows_te_autocast():
    """A CUDA graph captured outside te.autocast must not answer inside it."""
    from qkan import compile_inference

    lin, _ = pair(768, 768)
    compiled = compile_inference(lin).eval()
    x = torch.randn(64, 768, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        plain = compiled(x).clone()
        with te.autocast(recipe=recipe.NVFP4BlockScaling()):
            quantized, eager = compiled(x).clone(), lin(x)
        assert rel(quantized, plain) > 5e-2
        torch.testing.assert_close(quantized, eager)
        torch.testing.assert_close(compiled(x), plain)


def test_follows_a_change_of_recipe():
    """te.Linear keeps the quantizers of the first recipe of a class (TE #3157)."""
    (lin, _), (fresh, _) = pair(768, 768), pair(768, 768)
    x = torch.randn(64, 768, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        with te.autocast(recipe=recipe.NVFP4BlockScaling()):
            lin(x)
        with te.autocast(recipe=recipe.NVFP4BlockScaling(disable_2d_quantization=True)):
            torch.testing.assert_close(lin(x), fresh(x), rtol=0, atol=0)


@pytest.mark.parametrize("rec, min_err, max_err", [FP8, NVFP4], ids=["fp8", "nvfp4"])
@pytest.mark.parametrize(
    "shape", [(1000, 768), (3, 197, 768), (1, 1, 768)], ids=["2d", "3d", "one-token"]
)
def test_any_token_count_is_quantized(shape, rec, min_err, max_err):
    """Plain te.Linear raises on token counts that aren't multiples of 32 (16
    for FP8); zero rows pad them instead of running them unquantized."""
    lin, ref = pair(768, 768)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    check(lin, ref, x, rec, min_err, max_err)


@pytest.mark.parametrize("k, n", [(768, 10), (10, 768)], ids=["out", "in"])
def test_unaligned_features_raise(k, n):
    """Transformer Engine can't quantize them, and nothing else is run instead."""
    lin, _ = pair(k, n)
    x = torch.randn(1024, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible"):
        run(lin, x, recipe.NVFP4BlockScaling())
