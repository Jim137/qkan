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

"""DARUAN's device follows its parameters (#46)."""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from qkan import DARUAN  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Ways a module ends up on another device without DARUAN being told.
MOVES = {
    "cuda": lambda m: m.cuda(),
    "to": lambda m: m.to("cuda"),
    "parent": lambda m: nn.Sequential(m).cuda()[0],
}
moves = pytest.mark.parametrize("move", MOVES.values(), ids=MOVES.keys())


def test_device_cannot_be_assigned():
    """A stored copy is what went stale: the device is only ever derived."""
    m = DARUAN(4, 3)
    with pytest.raises(AttributeError):
        m.device = "cuda"


def test_device_is_read_from_the_parameters():
    """Also where no hook could refresh a copy: parameters swapped in for one
    call. The meta device needs no GPU."""
    m = DARUAN(4, 3)
    params = {k: v.detach().to("meta") for k, v in m.named_parameters()}
    y = torch.func.functional_call(m, params, (torch.randn(8, 4),))
    assert y.device.type == "meta" and m.device.type == "cpu"
    assert m.to("meta").device.type == "meta"


@needs_cuda
@moves
def test_forward_after_move_to_cuda(move):
    x = torch.randn(8, 4)
    m = DARUAN(4, 3)
    ref = m(x)

    m = move(m)
    assert m.device.type == "cuda"
    for inp in (x.cuda(), x):  # forward moves the input to the module
        y = m(inp)
        assert y.device.type == "cuda"
        assert torch.allclose(y.cpu(), ref, atol=1e-5)


@needs_cuda
def test_forward_after_move_back_to_cpu():
    x = torch.randn(8, 4)
    m = DARUAN(4, 3, device="cuda")
    ref = m(x.cuda()).cpu()

    m.cpu()
    for inp in (x, x.cuda()):
        y = m(inp)
        assert y.device.type == "cpu"
        assert torch.allclose(y, ref, atol=1e-5)


@needs_cuda
@pytest.mark.parametrize("src_device, dst_device", [("cuda", "cpu"), ("cpu", "cuda")])
def test_load_state_dict_assign_from_other_device(src_device, dst_device):
    x = torch.randn(8, 4)
    src = DARUAN(4, 3, device=src_device, seed=1)
    dst = DARUAN(4, 3, device=dst_device)

    dst.load_state_dict(src.state_dict(), assign=True)
    y = dst(x)
    assert y.device.type == dst.device.type == src_device
    assert torch.allclose(y, src(x), atol=1e-6)
