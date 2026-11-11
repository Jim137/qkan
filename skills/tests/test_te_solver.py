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

"""solver="te": the QKAN layer as one Fourier-feature GEMM."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qkan import QKAN, QKANLayer  # noqa: E402
from qkan.solver import _FLASH_AVAILABLE  # noqa: E402

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    _NVFP4 = torch.cuda.is_available() and te.is_nvfp4_available()
except ImportError:
    _NVFP4 = False

needs_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and _FLASH_AVAILABLE), reason="needs CUDA + triton"
)
needs_nvfp4 = pytest.mark.skipif(
    not _NVFP4, reason="needs Transformer Engine and an NVFP4 GPU"
)

ANSATZES = ["pz_encoding", "rpz_encoding", "real", "mix"]
# Everything but preacts_weight, which the rewrite needs fixed at 1.
PARAMS = ("theta", "preacts_bias", "base_weight", "postact_weights", "postact_bias")


def pair(
    ref_solver, ansatz="pz_encoding", reps=2, dtype=torch.float64, device=None, **kwargs
):
    """A te layer and a reference layer with the same random parameters."""
    device = device or ("cpu" if dtype == torch.float64 else "cuda")
    c_dtype = torch.complex128 if dtype == torch.float64 else torch.float32
    layers = [
        QKANLayer(
            5,
            4,
            reps=reps,
            solver=solver,
            ansatz=ansatz,
            fast_measure=False,
            postact_weight_trainable=True,
            postact_bias_trainable=True,
            device=device,
            p_dtype=dtype,
            c_dtype=c_dtype,
            **kwargs,
        )
        for solver in ("te", ref_solver)
    ]
    layers[1].c_dtype = c_dtype  # "real" forces bf16 at construction (#37)
    torch.manual_seed(0)
    state = layers[0].state_dict()
    for name in PARAMS:
        state[name] = torch.randn_like(state[name])
    state["mask"] = (torch.rand_like(state["mask"]) > 0.3).to(dtype)
    for layer in layers:
        layer.load_state_dict(state)
    x = torch.randn(64, 5, device=device, dtype=dtype) * 2
    return layers, x


def run(layer, x):
    layer.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out = layer(x)
    out.sin().sum().backward()
    grads = {name: getattr(layer, name).grad for name in PARAMS}
    return out.detach(), x.grad, grads


@pytest.mark.parametrize("reps", [1, 2, 3])
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_matches_exact(ansatz, reps):
    """Train route, eval route and every gradient, in fp64."""
    (layer, ref), x = pair("exact", ansatz, reps)
    torch.testing.assert_close(run(layer, x), run(ref, x), rtol=0, atol=1e-11)
    layer.eval()
    ref.eval()
    torch.testing.assert_close(layer(x), ref(x), rtol=0, atol=1e-11)


def test_float32_input_to_float64_layer():
    """The features are computed in the wider of the two dtypes."""
    (layer, ref), x = pair("exact")
    x = x.float()
    torch.testing.assert_close(layer(x), ref(x.double()), rtol=0, atol=1e-11)


@pytest.mark.parametrize(
    "kwargs", [dict(group=3), dict(is_batchnorm=True)], ids=["group", "batchnorm"]
)
def test_layer_options(kwargs):
    (layer, ref), x = pair("exact", **kwargs)
    torch.testing.assert_close(run(layer, x)[:2], run(ref, x)[:2], rtol=0, atol=1e-11)


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, pytest.param(torch.float32, marks=needs_cuda)],
    ids=["cpu-fp64", "cuda-fp32"],
)
def test_postacts_match_exact(dtype):
    """forward_no_sum (plotting, pruning) still gets the circuit's postacts;
    on CUDA "mix" goes to the exact solver, which flash doesn't cover."""
    (layer, ref), x = pair("exact", "mix", dtype=dtype)
    ref.c_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    torch.testing.assert_close(layer.forward_no_sum(x), ref.forward_no_sum(x))


def test_prune_keeps_fast_measure():
    """Pruning rebuilds the layers with fast_measure=False, so a te model can
    be pruned and still matches its exact-solver twin."""
    torch.manual_seed(0)
    models = [
        QKAN([4, 6, 2], reps=2, solver=solver, fast_measure=False, save_act=True)
        for solver in ("te", "exact")
    ]
    models[1].load_state_dict(models[0].state_dict())
    x = torch.randn(32, 4)
    pruned = []
    for model in models:
        model(x)
        pruned.append(
            model.prune_node(mode="manual", active_neurons_id=[[], [0, 2, 5]])
        )
    torch.testing.assert_close(pruned[0](x), pruned[1](x), rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fast_measure=True),
        dict(fast_measure=False, preact_trainable=True),
        dict(fast_measure=False, preact_init=True),
        dict(fast_measure=False, ansatz="px_encoding"),
    ],
    ids=["fast_measure", "preact_trainable", "preact_init", "px"],
)
def test_rejects_non_fourier_layers(kwargs):
    with pytest.raises(ValueError, match="solver='te'"):
        QKANLayer(5, 4, solver="te", **kwargs)


def test_rejects_trained_preact_weights():
    """A checkpoint trained with preact_trainable=True computes another function."""
    layer = QKANLayer(5, 4, solver="te", fast_measure=False)
    state = layer.state_dict()
    state["preacts_weight"] = torch.randn_like(state["preacts_weight"])
    with pytest.raises(ValueError, match="preacts_weight"):
        layer.load_state_dict(state)


def graph_nodes(layer, x):
    """Names of the autograd nodes behind the layer's output."""
    seen, stack = set(), [layer(x.clone().requires_grad_()).grad_fn]
    while stack:
        fn = stack.pop()
        if fn is not None and fn not in seen:
            seen.add(fn)
            stack += [parent for parent, _ in fn.next_functions]
    return {type(fn).__name__ for fn in seen}


KERNELS = {"_FourierCoefficientsBackward", "_FourierFeaturesBackward"}


class UserActivation(torch.nn.Module):
    """Not the kernel's relu, whatever its ``kind`` says."""

    kind = "relu"

    def forward(self, x):
        return F.leaky_relu(x, 0.2)


@needs_cuda
@pytest.mark.parametrize(
    "kwargs, kernels",
    [
        (dict(), KERNELS),
        (dict(group=3), KERNELS),  # coefficients of 3 edges, broadcast to 4x5
        (dict(base_activation=UserActivation()), {"_FourierCoefficientsBackward"}),
    ],
    ids=["kernels", "group", "user-activation"],
)
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "real"])
def test_matches_flash(ansatz, kwargs, kernels):
    """fp32 on CUDA, where the coefficients and the features come from the
    Triton kernels; an activation the user passed is called as given."""
    (layer, ref), x = pair("flash", ansatz, dtype=torch.float32, **kwargs)
    assert graph_nodes(layer, x) & KERNELS == kernels
    torch.testing.assert_close(run(layer, x), run(ref, x), rtol=1e-4, atol=1e-4)


COEFFICIENTS = {"_FourierCoefficientsBackward", "_FlashFunctionBackward"}


@needs_cuda
@pytest.mark.parametrize(
    "reps, node", [(2, "_FourierCoefficientsBackward"), (3, "_FlashFunctionBackward")]
)
def test_kernel_reps_limit(monkeypatch, reps, node):
    """Up to MAX_REPS the coefficients come from the kernel, past it from
    flash samples."""
    from qkan.solver import te_ops_coefficients

    monkeypatch.setattr(te_ops_coefficients, "MAX_REPS", 2)
    (layer, ref), x = pair("flash", reps=reps, dtype=torch.float32)
    assert graph_nodes(layer, x) & COEFFICIENTS == {node}
    torch.testing.assert_close(run(layer, x), run(ref, x), rtol=1e-4, atol=1e-4)


@needs_cuda
def test_mix_on_cuda():
    """No kernel covers "mix": its coefficients are sampled, the features fuse."""
    (layer, ref), x = pair("exact", "mix", dtype=torch.float32)
    ref.c_dtype = torch.complex64
    assert graph_nodes(layer, x) & KERNELS == {"_FourierFeaturesBackward"}
    torch.testing.assert_close(run(layer, x), run(ref, x), rtol=1e-4, atol=1e-4)


@needs_cuda
@pytest.mark.parametrize("ansatz", ANSATZES)
def test_float64_on_cuda(ansatz):
    """The kernels compute in float32, so float64 takes neither and keeps its
    precision. torch.autocast leaves float64 alone."""
    (layer, ref), x = pair("exact", ansatz, device="cuda")
    assert not graph_nodes(layer, x) & KERNELS
    torch.testing.assert_close(run(layer, x), run(ref, x), rtol=0, atol=1e-11)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(layer(x), ref(x), rtol=0, atol=1e-11)


@needs_cuda
def test_bf16_parameters():
    """bf16 parameters take both kernels and reach the GEMM in bf16."""
    (layer, ref), x = pair("flash", dtype=torch.float32)
    layer.to(torch.bfloat16)
    ref.load_state_dict(layer.state_dict())
    assert graph_nodes(layer, x) & KERNELS == KERNELS
    out = layer(x)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref(x), rtol=2e-2, atol=5e-2)


@needs_cuda
@pytest.mark.parametrize(
    "dtype, tol", [(torch.float32, 1e-3), (torch.float64, 1e-9)], ids=["fp32", "fp64"]
)
@pytest.mark.parametrize("ansatz", ["pz_encoding", "rpz_encoding", "real"])
def test_second_derivatives(ansatz, dtype, tol):
    """A PINN loss on du/dx trains like the exact solver: through the kernels
    in fp32, through torch ops in fp64."""
    (layer, ref), x = pair("exact", ansatz, dtype=dtype, device="cuda")
    if dtype == torch.float32:
        ref.c_dtype = torch.complex64
    grads = []
    for model in (layer, ref):
        model.zero_grad(set_to_none=True)
        xg = x.clone().requires_grad_()
        (du_dx,) = torch.autograd.grad(model(xg).sin().sum(), xg, create_graph=True)
        du_dx.square().sum().backward()
        grads.append(
            (du_dx.detach(), xg.grad, model.theta.grad, model.base_weight.grad)
        )
    torch.testing.assert_close(grads[0], grads[1], rtol=tol, atol=tol)


@needs_cuda
@pytest.mark.parametrize("amp", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_follows_torch_autocast(amp, monkeypatch):
    """Under torch.autocast the features are written straight in its dtype."""
    from qkan.solver import te_ops_features

    seen = []
    kernel = te_ops_features.fourier_features
    monkeypatch.setattr(
        te_ops_features,
        "fourier_features",
        lambda *args: seen.append(args[3]) or kernel(*args),
    )
    (layer, ref), x = pair("flash", dtype=torch.float32)
    with torch.autocast("cuda", dtype=amp):
        out = layer(x)
    assert out.dtype == amp and seen == [amp]
    torch.testing.assert_close(out.float(), ref(x), rtol=2e-2, atol=5e-2)


@needs_cuda
def test_cpu_layer_inside_cuda_autocast():
    """torch.autocast("cuda") leaves CPU tensors alone, and so does the layer."""
    layer = QKANLayer(5, 4, solver="te", fast_measure=False)
    x = torch.randn(8, 5)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x)
    torch.testing.assert_close(out, layer(x))


@needs_cuda
@pytest.mark.parametrize("nvfp4", [False, True], ids=["fp8", "nvfp4"])
@pytest.mark.parametrize("act", ["silu", torch.nn.SiLU()], ids=["kernel", "torch"])
def test_te_branch_without_te(monkeypatch, act, nvfp4):
    """The te.autocast branch with a plain GEMM standing in for Transformer
    Engine's: both operands arrive padded to 32, in bf16 for NVFP4, and the
    padding is sliced off the output."""
    from qkan.solver import te as te_solver

    def gemm(features, w, recipe, dtype):
        assert features.shape[0] % 32 == features.shape[1] % 32 == 0
        assert w.shape[0] % 32 == 0 and w.shape[1] == features.shape[1]
        assert features.dtype == w.dtype == (torch.bfloat16 if nvfp4 else dtype)
        return F.linear(features.float(), w.float()).to(dtype)

    recipe = type("Recipe", (), {"nvfp4": lambda self: nvfp4})()
    (layer, ref), x = pair("flash", dtype=torch.float32, base_activation=act)
    expected = ref(x[:37])  # 37 rows, 25 features, 4 outputs: all padded
    monkeypatch.setattr(te_solver, "_te_recipe", lambda: recipe)
    monkeypatch.setattr(te_solver._TEGemm, "apply", staticmethod(gemm))
    tol = 5e-2 if nvfp4 else 1e-4
    torch.testing.assert_close(layer(x[:37]), expected, rtol=tol, atol=tol)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@needs_nvfp4
@pytest.mark.parametrize("name", ["Float8CurrentScaling", "NVFP4BlockScaling"])
def test_gemm_matches_te_linear(name):
    """The solver's GEMM quantizes like te.Linear. FP8 current scaling is
    deterministic, so everything matches exactly; NVFP4 rounds gradients
    stochastically, so only its forward does."""
    from qkan.solver.te import _TEGemm

    torch.manual_seed(0)
    rec = getattr(recipe, name)()
    x = torch.randn(256, 640, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 640, device="cuda", dtype=torch.bfloat16) * 0.05
    grad = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    lin = te.Linear(640, 64, bias=False, params_dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(w)
    x_ref, x, w = x.clone().requires_grad_(), x.requires_grad_(), w.requires_grad_()
    with te.autocast(recipe=rec):
        ref = lin(x_ref)
        out = _TEGemm.apply(x, w, rec, torch.bfloat16)
    ref.backward(grad)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    if name == "Float8CurrentScaling":
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
        torch.testing.assert_close(w.grad, lin.weight.grad, rtol=0, atol=0)


@needs_nvfp4
@pytest.mark.parametrize("name", ["Float8CurrentScaling", "NVFP4BlockScaling"])
def test_input_without_grad(name):
    """A first layer's input needs no gradient, but the weights still train."""
    torch.manual_seed(0)
    layer = QKANLayer(64, 32, reps=1, solver="te", fast_measure=False, device="cuda")
    with te.autocast(recipe=getattr(recipe, name)()):
        layer(torch.randn(64, 64, device="cuda")).sum().backward()
    assert layer.theta.grad.abs().sum() > 0


@needs_nvfp4
def test_torch_activation_under_te():
    """An activation the user passed takes the torch feature map, which
    still hands NVFP4 the bf16 operand its Hadamard transform needs."""
    outs = []
    for act in ("silu", torch.nn.SiLU()):
        torch.manual_seed(0)
        layer = QKANLayer(
            64, 32, solver="te", fast_measure=False, device="cuda", base_activation=act
        )
        with te.autocast(recipe=recipe.NVFP4BlockScaling()):
            outs.append(layer(torch.randn(64, 64, device="cuda")))
    torch.testing.assert_close(outs[0], outs[1], rtol=0.1, atol=0.1)


@needs_nvfp4
def test_no_second_derivatives_under_te():
    """Transformer Engine's GEMM is opaque to autograd: a create_graph backward
    raises instead of silently dropping the second-order terms."""
    layer = QKANLayer(64, 32, reps=1, solver="te", fast_measure=False, device="cuda")
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    with te.autocast(recipe=recipe.NVFP4BlockScaling()):
        out = layer(x)
        with pytest.raises(RuntimeError, match="second derivatives"):
            torch.autograd.grad(out.sum(), x, create_graph=True)


@needs_nvfp4
def test_rejects_delayed_scaling():
    """DelayedScaling needs amax history kept across steps, which a solver
    call doesn't have."""
    layer = QKANLayer(64, 32, reps=1, solver="te", fast_measure=False, device="cuda")
    with te.autocast(recipe=recipe.DelayedScaling()), pytest.raises(ValueError):
        layer(torch.randn(64, 64, device="cuda"))


@needs_nvfp4
@pytest.mark.parametrize(
    "name, max_err",
    [("Float8CurrentScaling", 0.05), ("NVFP4BlockScaling", 0.1)],
    ids=["fp8", "nvfp4"],
)
@pytest.mark.parametrize(
    "shape", [(256, 128, 64), (100, 30, 50)], ids=["aligned", "padded"]
)
def test_te_recipe(name, max_err, shape):
    """A quantized layer stays close to its bf16 run, padded shapes included.
    Its gradients are noisier (up to 0.21 for FP8 and 0.41 for NVFP4 here,
    from E5M2 / stochastically rounded wgrads), so they are only checked to
    be correlated: a zero gradient (TE #3281 on sm_120) has error 1."""
    batch, in_dim, out_dim = shape
    torch.manual_seed(0)
    layer = QKANLayer(
        in_dim, out_dim, reps=2, solver="te", fast_measure=False, device="cuda"
    )
    x = torch.randn(batch, in_dim, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref_out, ref_grad_x, ref_grads = run(layer, x)
        with te.autocast(recipe=getattr(recipe, name)()):
            assert "_TEGemmBackward" in graph_nodes(layer, x)
            out, grad_x, grads = run(layer, x)
    assert rel(out, ref_out) < max_err
    assert rel(grad_x, ref_grad_x) < 0.75
    for param in ("theta", "base_weight"):
        assert rel(grads[param], ref_grads[param]) < 0.75
