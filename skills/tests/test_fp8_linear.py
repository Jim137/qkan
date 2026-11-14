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

"""Fp8Linear runs the fp8 GEMM at every size (#54) and survives dtype casts (#48)."""

import copy
import io

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from qkan.solver.fp8 import (  # noqa: E402
    Fp8Linear,
    dequantize_from_fp8,
    quantize_to_fp8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 GEMM needs CUDA with capability >= (8, 9)",
)

E4M3 = torch.float8_e4m3fn
# unaligned on both sides, on one side, and aligned
SIZES = [(10, 20), (768, 10), (10, 768), (32, 16)]
# Relative error against the float32 GEMM over the operands the fp8 GEMM sees.
# The fp8 GEMM is off by the rounding of its output (at most 2^-8 for bf16);
# a bf16 F.linear on the unquantized input is off by the input's e4m3 rounding,
# about 3e-2.
TOL = 5e-3


@pytest.fixture
def scaled_mm_calls(monkeypatch):
    """(mat1's shape, mat2's address) of every torch._scaled_mm call.

    Not the operands: mat2 is a view of the weight, and a live view blocks
    a swap-mode conversion.
    """
    calls = []
    real = torch._scaled_mm

    def counting(mat1, mat2, *args, **kwargs):
        calls.append((mat1.shape, mat2.data_ptr()))
        return real(mat1, mat2, *args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", counting)
    return calls


@pytest.fixture(params=[False, True], ids=["set_data", "swap_tensors"])
def swap_params(request):
    old = torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(request.param)
    yield
    torch.__future__.set_swap_module_params_on_conversion(old)


def make(in_features, out_features, bias=True, leading=(64,), dtype=torch.float32):
    torch.manual_seed(0)
    layer = Fp8Linear(in_features, out_features, bias=bias, device="cuda")
    x = torch.randn(*leading, in_features, device="cuda", dtype=dtype)
    return layer, x


def unpadded(layer):
    """The (out, in) weight and (out,) bias inside the stored, padded ones."""
    weight = layer.weight[: layer.out_features, : layer.in_features]
    bias = None if layer.bias is None else layer.bias[: layer.out_features]
    return weight, bias


def reference(layer, x_fp8, x_scale):
    """float32 GEMM over the quantized operands."""
    weight, bias = unpadded(layer)
    weight = dequantize_from_fp8(weight, layer.weight_scale, torch.float32)
    bias = None if bias is None else bias.float()
    return F.linear(dequantize_from_fp8(x_fp8, x_scale, torch.float32), weight, bias)


def rel_err(y, ref):
    return ((y.float() - ref).norm() / ref.norm()).item()


def saved_and_loaded(obj, **load_kwargs):
    """torch.save(obj) and back."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False, **load_kwargs)


def fp8_state(layer):
    """Bits of the e4m3 weight, and its scale."""
    return layer.weight.view(torch.uint8).cpu(), layer.weight_scale.cpu()


def assert_fp8_state(layer, state):
    assert layer.weight.dtype == E4M3
    assert layer.weight_scale.dtype == torch.float32
    assert all(torch.equal(a, b) for a, b in zip(fp8_state(layer), state))


# --- #54: every size runs the fp8 GEMM ------------------------------------


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize(
    "in_features, out_features", [(768, 10), (10, 768), (10, 10), (17, 33), (768, 768)]
)
def test_every_size_runs_the_fp8_gemm(in_features, out_features, bias, scaled_mm_calls):
    layer, x = make(in_features, out_features, bias)
    y = layer(x)
    assert len(scaled_mm_calls) == 1
    assert y.shape == (64, out_features) and y.dtype == torch.bfloat16
    assert y.is_contiguous()

    ref = reference(layer, *quantize_to_fp8(x))
    assert rel_err(y, ref) < TOL
    # The tolerance tells the fp8 GEMM from a bf16 F.linear on the float input.
    weight, bias_ = unpadded(layer)
    weight = dequantize_from_fp8(weight, layer.weight_scale)
    assert rel_err(F.linear(x.bfloat16(), weight, bias_), ref) > 2 * TOL

    # No size copies the weight per call: the GEMM reads the stored one.
    assert scaled_mm_calls[0] == (
        (64, in_features + -in_features % 16),
        layer.weight.data_ptr(),
    )


@pytest.mark.parametrize("leading", [(1,), (5, 7), (2, 3, 4)], ids=["row", "3d", "4d"])
@pytest.mark.parametrize("in_features, out_features", [(768, 10), (10, 768)])
def test_leading_dims(in_features, out_features, leading, scaled_mm_calls):
    layer, x = make(in_features, out_features, leading=leading)
    y = layer(x)
    assert len(scaled_mm_calls) == 1
    assert y.shape == (*leading, out_features) and y.is_contiguous()
    assert rel_err(y, reference(layer, *quantize_to_fp8(x))) < TOL


@pytest.mark.parametrize("dtype", [E4M3, torch.float8_e5m2], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("in_features, out_features", [(768, 10), (10, 768)])
def test_fp8_input_with_scale(in_features, out_features, dtype, scaled_mm_calls):
    layer, x = make(in_features, out_features, leading=(2, 32))
    x_fp8, x_scale = quantize_to_fp8(x, dtype)
    y = layer(x_fp8, input_scale=x_scale)
    assert len(scaled_mm_calls) == 1
    assert y.shape == (2, 32, out_features)
    assert rel_err(y, reference(layer, x_fp8, x_scale)) < TOL


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_padding_is_the_aligned_gemm_on_zero_padded_operands(bias):
    """Bit for bit, bias included."""
    small, x = make(10, 20, bias)
    big = Fp8Linear(16, 32, bias=bias, device="cuda")
    big.weight.zero_()
    big.weight[:20, :10] = small.weight[:20, :10]
    big.weight_scale.copy_(small.weight_scale)
    if bias:
        big.bias.zero_()
        big.bias[:20] = small.bias[:20]
    assert torch.equal(small(x), big(F.pad(x, (0, 6)))[:, :20])


def test_operands_are_stored_padded_with_zeros():
    layer, _ = make(10, 20)
    assert layer.weight.shape == (32, 16) and layer.weight.dtype == E4M3
    assert layer.bias.shape == (32,)
    bits = layer.weight.view(torch.uint8)
    assert not bits[20:].any() and not bits[:, 10:].any() and bits[:20, :10].any()
    assert not layer.bias[20:].any()


@pytest.mark.parametrize("in_features, out_features", SIZES)
def test_state_dict_keeps_the_out_in_layout(in_features, out_features):
    """As views: a write through the state dict reaches the layer."""
    layer, x = make(in_features, out_features)
    state = layer.state_dict()
    assert list(state) == ["weight", "bias", "weight_scale"]
    assert state["weight"].shape == (out_features, in_features)
    assert state["weight"].dtype == E4M3
    assert state["bias"].shape == (out_features,)
    assert state["weight_scale"].dtype == torch.float32
    state["bias"].zero_()
    assert not layer.bias.any()
    assert rel_err(layer(x), reference(layer, *quantize_to_fp8(x))) < TOL


@pytest.mark.parametrize("assign", [False, True], ids=["copy", "assign"])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("in_features, out_features", SIZES)
def test_state_dict_round_trip(in_features, out_features, bias, device, assign):
    """Through a file, as a checkpoint is: the layout of every version."""
    source, x = make(in_features, out_features, bias)
    state = saved_and_loaded(nn.Sequential(source).state_dict(), map_location=device)
    assert state["0.weight"].shape == (out_features, in_features)

    net = nn.Sequential(Fp8Linear(in_features, out_features, bias=bias, device="cuda"))
    net.load_state_dict(state, assign=assign)
    net.cuda()
    assert net[0].weight.shape == source.weight.shape
    assert torch.equal(net(x), source(x))


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "source, target",
    [
        ((32, 16), (20, 10)),  # its (16, 32) weight is what (20, 10) stores
        ((20, 10), (32, 16)),
        ((768, 12), (768, 10)),
        ((12, 768), (10, 768)),
        ((10, 20), (11, 21)),
    ],
)
def test_a_checkpoint_of_another_size_is_rejected(source, target, strict):
    """Also within one multiple of 16, where the stored shapes agree."""
    state = nn.Sequential(Fp8Linear(*source, device="cuda")).state_dict()
    layer, _ = make(*target)
    before = layer.weight.clone()
    with pytest.raises(RuntimeError, match="size mismatch for 0.weight"):
        nn.Sequential(layer).load_state_dict(state, strict=strict)
    state = {k.removeprefix("0."): v for k, v in state.items()}
    with pytest.raises(RuntimeError, match="size mismatch for weight") as error:
        layer.load_state_dict(state, strict=strict)
    assert f"current model is torch.Size([{target[1]}, {target[0]}])" in str(
        error.value
    )
    assert "Missing key" not in str(error.value)
    assert same_bits(layer.weight, before)

    if target[0] % 16 or target[1] % 16:
        # the stored layout is not a checkpoint's
        padded = layer.state_dict() | {"weight": layer.weight}
        with pytest.raises(RuntimeError, match="size mismatch for weight"):
            layer.load_state_dict(padded, strict=strict)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("in_features, out_features", SIZES)
def test_a_module_pickled_with_unpadded_operands_runs(in_features, out_features, bias):
    """torch.save(model) from before the operands were stored padded."""
    layer, x = make(in_features, out_features, bias)
    y = layer(x)
    # two layers that share their parameters, to see that they stay tied
    twin = Fp8Linear(in_features, out_features, bias=bias, device="cuda")
    twin.weight_scale.copy_(layer.weight_scale)
    for name, value in zip(("weight", "bias"), unpadded(layer)):
        if value is not None:
            shared = nn.Parameter(value.clone(), requires_grad=False)
            layer._parameters[name] = twin._parameters[name] = shared
    old = saved_and_loaded(nn.Sequential(layer, twin))
    assert old[0].weight is old[1].weight and old[0].bias is old[1].bias
    assert old[0].weight.shape == (
        out_features + -out_features % 16,
        in_features + -in_features % 16,
    )
    assert torch.equal(old[0](x), y) and torch.equal(old[1](x), y)

    for copy_ in (copy.deepcopy(old), saved_and_loaded(old)):
        assert copy_[0].weight is copy_[1].weight
        assert copy_[0].weight.shape == old[0].weight.shape
        assert torch.equal(copy_[0](x), y)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("in_features, out_features", SIZES)
def test_functional_call_with_a_state_dict(in_features, out_features, bias):
    """The (out, in) operands of a state dict are padded on the way in."""
    layer, x = make(in_features, out_features, bias)
    other = Fp8Linear(in_features, out_features, bias=bias, device="cuda")
    assert torch.equal(
        torch.func.functional_call(layer, other.state_dict(), x), other(x)
    )
    params = dict(other.named_parameters()) | dict(other.named_buffers())
    assert torch.equal(torch.func.functional_call(layer, params, x), other(x))
    if bias:  # one operand alone
        only_bias = {"bias": other.state_dict()["bias"]}
        expected = Fp8Linear(in_features, out_features, device="cuda")
        expected.load_state_dict(layer.state_dict() | only_bias)
        assert torch.equal(torch.func.functional_call(layer, only_bias, x), expected(x))


@pytest.mark.parametrize("in_features, out_features", [(10, 20), (768, 10), (32, 16)])
def test_vmap_over_the_layer(in_features, out_features):
    layer, x = make(in_features, out_features, leading=(3, 8))
    y = torch.func.vmap(layer)(x)
    for i in range(3):
        assert torch.equal(y[i], layer(x[i]))


def test_compiles_as_one_graph():
    layer, x = make(10, 20)
    # the eager backend: only the trace is at stake
    compiled = torch.compile(layer, fullgraph=True, backend="eager")
    assert rel_err(compiled(x), reference(layer, *quantize_to_fp8(x))) < TOL


EDITS = {
    "bias in place": lambda layer, other: layer.bias.copy_(other.bias),
    "bias through .data": lambda layer, other: layer.bias.data.copy_(other.bias),
    "new bias": lambda layer, other: setattr(layer, "bias", other.bias),
    "weight in place": lambda layer, other: (
        layer.weight.copy_(other.weight),
        layer.weight_scale.copy_(other.weight_scale),
    ),
    "new weight": lambda layer, other: (
        setattr(layer, "weight", other.weight),
        layer.weight_scale.copy_(other.weight_scale),
    ),
}


@pytest.mark.parametrize("edit", EDITS)
@pytest.mark.parametrize(
    "in_features, out_features", [(10, 768), (768, 10), (768, 768)]
)
def test_edits_after_a_forward_reach_the_gemm(in_features, out_features, edit):
    """No padded copy may outlive the parameters it was made from."""
    layer, x = make(in_features, out_features)
    other = Fp8Linear(in_features, out_features, device="cuda")
    layer(x)
    with torch.no_grad():
        EDITS[edit](layer, other)
    assert rel_err(layer(x), reference(layer, *quantize_to_fp8(x))) < TOL


def amax_recipe(x, dtype=E4M3):
    """``x / (max|x| / 448)`` in float32, rounded to fp8."""
    amax = x.detach().abs().amax().to(torch.float32).clamp_min(1e-12)
    scale = amax / 448.0
    return (x.to(torch.float32) / scale).to(dtype), scale


def same_bits(a, b):
    def bits(t):
        return t.contiguous().reshape(-1).view(torch.uint8)

    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


QUANTIZE_INPUTS = {
    "randn": lambda: torch.randn(64, 48),
    "tiny": lambda: torch.randn(64, 48) * 1e-30,
    "small": lambda: torch.randn(64, 48) * 1e-3,
    "large": lambda: torch.randn(64, 48) * 1e4,
    "zeros": lambda: torch.zeros(4, 16),
    "inf": lambda: torch.tensor([[1.0, float("inf"), -2.0, 0.5]]),
    "nan": lambda: torch.tensor([[1.0, float("nan"), -2.0, 0.5]]),
    "strided": lambda: torch.randn(48, 64).t()[::2, 1::3],
    "3d": lambda: torch.randn(3, 5, 16),
    "0d": lambda: torch.tensor(-3.7),
}


@pytest.mark.parametrize("target", [E4M3, torch.float8_e5m2], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64]
)
@pytest.mark.parametrize("data", QUANTIZE_INPUTS)
def test_quantize_keeps_the_amax_recipe(data, dtype, device, target):
    """Bit for bit, with and without a gradient to keep."""
    torch.manual_seed(0)
    x = QUANTIZE_INPUTS[data]().to(device, dtype)
    before = x.clone()
    expected, expected_scale = amax_recipe(x, target)

    for requires_grad in (False, True):
        x_fp8, scale = quantize_to_fp8(x.requires_grad_(requires_grad), target)
        assert same_bits(x_fp8, expected)
        assert scale.dtype == torch.float32 and scale.dim() == 0
        assert not scale.requires_grad and x_fp8.requires_grad == requires_grad
        # NaN != NaN, and the payload is not part of the recipe
        assert torch.equal(scale, expected_scale) or (
            scale.isnan() and expected_scale.isnan()
        )
    assert same_bits(x.detach(), before)  # the input is not written


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize_is_differentiable(dtype):
    """quantize -> dequantize passes the gradient straight through, up to the
    fp8 rounding of the scale: the gradient of the recipe."""
    x = torch.randn(8, 16, device="cuda", dtype=dtype, requires_grad=True)
    grads = []
    for quantize in (quantize_to_fp8, amax_recipe):
        x_fp8, scale = quantize(x)
        out = dequantize_from_fp8(x_fp8, scale, torch.float32).sum()
        grads.append(torch.autograd.grad(out, x)[0])
    assert same_bits(*grads)
    assert grads[0].isfinite().all() and (grads[0] != 0).all()


def test_quantize_under_function_transforms():
    """vmap gives one scale per sample; forward mode keeps the tangent."""
    x = torch.randn(4, 8, 16, device="cuda")
    x_fp8, scale = torch.func.vmap(quantize_to_fp8)(x)
    assert scale.shape == (4,)
    for i in range(4):
        expected, expected_scale = amax_recipe(x[i])
        assert same_bits(x_fp8[i], expected) and scale[i] == expected_scale

    def round_trip(t):
        return dequantize_from_fp8(*quantize_to_fp8(t), torch.float32)

    _, tangent = torch.func.jvp(round_trip, (x,), (torch.ones_like(x),))
    assert tangent.isfinite().all() and (tangent != 0).all()


# --- #48: dtype casts leave the fp8 weight alone --------------------------


def test_parent_cast_to_bf16():
    """The reproduction of #48."""
    f = Fp8Linear(256, 256, device="cuda")
    net = nn.Sequential(f).bfloat16()
    y = net(torch.randn(64, 256, device="cuda", dtype=torch.bfloat16))
    assert y.dtype == torch.bfloat16


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize(
    "cast, dtype",
    [
        pytest.param(lambda m: m.bfloat16(), torch.bfloat16, id="bfloat16"),
        pytest.param(lambda m: m.half(), torch.float16, id="half"),
        pytest.param(lambda m: m.float(), torch.float32, id="float"),
        pytest.param(lambda m: m.double(), torch.float64, id="double"),
        pytest.param(lambda m: m.to(torch.float16), torch.float16, id="to-dtype"),
    ],
)
def test_dtype_cast(cast, dtype, bias, swap_params, scaled_mm_calls):
    layer, x = make(10, 24, bias, dtype=dtype)
    layer(x)
    state = fp8_state(layer)

    net = cast(nn.Sequential(layer))
    assert cast(layer) is layer
    assert_fp8_state(layer, state)
    assert layer.activation_dtype == dtype
    assert layer.bias is None or layer.bias.dtype == dtype

    # torch._scaled_mm (torch 2.11) writes bf16 and float16, and float32
    # without a bias. The other dtypes raise: no other route serves them.
    tol = {torch.bfloat16: TOL, torch.float16: 5e-4, torch.float32: 1e-6}
    if dtype not in tol or (dtype == torch.float32 and bias):
        with pytest.raises(RuntimeError):
            net(x)
        return
    scaled_mm_calls.clear()
    y = net(x)
    assert len(scaled_mm_calls) == 1
    assert y.dtype == dtype
    # each output dtype at its own rounding: a relabelled bf16 result fails
    assert rel_err(y, reference(layer, *quantize_to_fp8(x))) < tol[dtype]


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_device_moves(bias, swap_params):
    layer, x = make(10, 24, bias)
    net = nn.Sequential(layer)
    y = net(x)
    state = fp8_state(layer)

    def devices():
        return {t.device.type for t in layer.state_dict().values()}

    net.cpu()
    assert devices() == {"cpu"}
    assert_fp8_state(layer, state)
    net.to("cuda")
    assert devices() == {"cuda"}
    assert_fp8_state(layer, state)
    assert torch.equal(net(x), y)

    net.to("cpu", torch.float16)  # a move and a cast in one call
    assert devices() == {"cpu"}
    assert_fp8_state(layer, state)
    assert layer.activation_dtype == torch.float16
    assert layer.bias is None or layer.bias.dtype == torch.float16
    out = net.cuda()(x)
    assert devices() == {"cuda"}
    assert out.dtype == torch.float16
    assert rel_err(out, reference(layer, *quantize_to_fp8(x))) < TOL
