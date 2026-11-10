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

"""The nn.Linear drop-ins follow torch.autocast like nn.Linear (#43)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import qkan.solver.linear  # noqa: E402
from qkan.solver import _CUTILE_AVAILABLE, _FLASH_AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

LINEARS = [
    pytest.param(
        "TritonLinear",
        marks=pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton"),
    ),
    pytest.param(
        "CuTileLinear",
        marks=pytest.mark.skipif(not _CUTILE_AVAILABLE, reason="no cuda-tile"),
    ),
    pytest.param(
        "TELinear",
        marks=pytest.mark.skipif(
            not hasattr(qkan.solver.linear, "TELinear"), reason="no transformer_engine"
        ),
    ),
]


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "m, k, n",
    [(64, 256, 256), (8192, 1024, 1024)],  # below / above TritonLinear's F.linear path
)
@pytest.mark.parametrize("name", LINEARS)
def test_matches_nn_linear_under_autocast(name, m, k, n, param_dtype, amp_dtype):
    torch.manual_seed(0)
    lin = getattr(qkan.solver.linear, name)(k, n, device="cuda").to(param_dtype)
    ref = nn.Linear(k, n, device="cuda").to(param_dtype)
    lin.load_state_dict(ref.state_dict())  # every drop-in takes nn.Linear checkpoints
    x = torch.randn(m, k, device="cuda")

    def run(module):
        with torch.autocast("cuda", dtype=amp_dtype):
            y = module(x)
            with torch.no_grad():
                assert module(x).dtype == amp_dtype
        y.float().sum().backward()
        return y, module.weight.grad

    # assert_close also checks the dtypes: amp_dtype out, param_dtype grad.
    # bf16 GEMMs summing thousands of terms differ between backends by ~1e-3
    # of the largest value, not of each element.
    for out, ref_out in zip(run(lin), run(ref)):
        atol = max(2e-1, 1e-2 * ref_out.abs().max().item())
        torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=atol)
