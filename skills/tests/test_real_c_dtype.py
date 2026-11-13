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

"""The real ansatz honours an explicit c_dtype (#37)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan import QKAN, QKANLayer  # noqa: E402
from qkan.solver import (  # noqa: E402
    _CUTE_AVAILABLE,
    _CUTILE_AVAILABLE,
    _FLASH_AVAILABLE,
)

FUSED = [
    pytest.param(
        "flash", marks=pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton")
    ),
    pytest.param(
        "cute",
        marks=pytest.mark.skipif(not _CUTE_AVAILABLE, reason="no CuTe extension"),
    ),
    pytest.param(
        "cutile",
        marks=pytest.mark.skipif(not _CUTILE_AVAILABLE, reason="no cuda-tile"),
    ),
]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_explicit_c_dtype_is_kept():
    """reset_parameters() used to force bf16 back, and pruning dropped it."""
    layer = QKANLayer(3, 2, ansatz="real", c_dtype=torch.complex64)
    assert layer.c_dtype == torch.complex64
    layer.reset_parameters()
    assert layer.c_dtype == torch.complex64
    assert layer.get_subset([0, 2], [1]).c_dtype == torch.complex64

    model = QKAN([3, 4, 2], ansatz="real", c_dtype=torch.complex64, save_act=True)
    model(torch.randn(16, 3))
    for pruned in (model.prune_node(threshold=0.0), model.prune_input(threshold=0.0)):
        assert all(layer.c_dtype == torch.complex64 for layer in pruned.layers)


def run_real(solver, c_dtype, batch, state_dict=None):
    """Forward and backward of a 3 -> 2 real layer; returns its parameters,
    output, grad_x and grad_theta."""
    torch.manual_seed(0)
    x = torch.randn(batch, 3, device="cuda", requires_grad=True)
    layer = QKANLayer(
        3, 2, reps=2, ansatz="real", solver=solver, device="cuda", c_dtype=c_dtype
    )
    if state_dict is None:
        with torch.no_grad():
            layer.theta.normal_()
    else:
        layer.load_state_dict(state_dict)
    y = layer(x)
    y.sum().backward()
    return layer.state_dict(), (y.detach(), x.grad, layer.theta.grad)


@needs_cuda
@pytest.mark.parametrize("batch", [64, 65536 + 7])
@pytest.mark.parametrize("solver", FUSED)
def test_fp32_matches_exact(solver, batch):
    """Forced to bf16, the fused real kernels differed from exact by ~5e-3.
    flash and cuTile also launched fp32 real one batch row per grid.y block,
    and CUDA caps grid.y at 65535."""
    params, ref = run_real("exact", torch.complex64, batch)
    _, out = run_real(solver, torch.float32, batch, params)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@needs_cuda
@pytest.mark.parametrize("solver", FUSED)
def test_fp8_matches_bf16(solver):
    """fp8 only changes the backward's state checkpoints.  flash's real
    forward didn't compile for fp8, and its backward dropped the fp8
    prescale on most checkpoints."""
    params, ref = run_real(solver, torch.bfloat16, 256)
    _, out = run_real(solver, torch.float8_e4m3fn, 256, params)
    assert torch.equal(out[0], ref[0])
    for grad, grad_ref in zip(out[1:], ref[1:]):  # fp8 checkpoints cost ~2%
        assert (grad - grad_ref).abs().max() <= 0.1 * grad_ref.abs().max()
