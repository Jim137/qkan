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

"""Batch-scaled offsets past 2^31 elements (flash #31; CuTe #21/#22).

One launch past the limit has to agree with the same batch done in chunks that
each stay far below it.  A match is not proof on its own: a wrapped store and
the load that reads it back use the same wrong address, so if that memory
happens to be mapped the round trip can survive.  Confirm a new test of this
kind fails on unfixed code first (ideally under compute-sanitizer).
"""

import os

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from qkan.solver import (  # noqa: E402
    _CUTE_AVAILABLE,
    _CUTILE_AVAILABLE,
    _FLASH_AVAILABLE,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        os.environ.get("QKAN_LARGE_TESTS") != "1",
        reason="allocates 3-5 GB; set QKAN_LARGE_TESTS=1",
    ),
]

SOLVERS = [
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


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
    reason="fp8 states need sm_89+",
)
@pytest.mark.parametrize("solver", SOLVERS)
def test_backward_states_past_int32(run_snippet, solver):
    # pz at 128x128, reps=3 keeps 12 states x 4 components per (row, out, in):
    # ~786k elements a row, so the states buffer passes 2^31 near 2.7k rows.
    # fp8 states are the cheapest way there -- the limit counts elements.
    result = run_snippet(
        f"""
        import torch
        from qkan import QKANLayer

        torch.manual_seed(0)
        layer = QKANLayer(
            128, 128, reps=3, solver="{solver}", device="cuda",
            c_dtype=torch.float8_e4m3fn,
        )
        x = torch.randn(3072, 128, device="cuda")

        def theta_grad(rows):
            layer.zero_grad()
            for chunk in torch.split(x, rows):
                layer(chunk).sum().backward()
            torch.cuda.synchronize()
            return layer.theta.grad.clone()

        full = theta_grad(len(x))  # one launch, states past 2^31 elements
        chunked = theta_grad(512)  # every launch far below the limit
        torch.testing.assert_close(full, chunked, rtol=1e-3, atol=1e-3)
        print("OK")
        """,
        preamble="",
        timeout=1800,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )


@pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton")
def test_flash_forward_output_past_int32(run_snippet):
    # The output offset is b_offs * (out_dim * in_dim): at 1024x1024 it passes
    # 2^31 from row 2048.  Called directly, so no layer epilogue temporaries.
    result = run_snippet(
        """
        import torch
        from qkan.solver.flash.fused_ops import triton_pz_forward

        torch.manual_seed(0)
        out_dim = in_dim = 1024
        x = torch.randn(2049, in_dim, device="cuda")
        theta = torch.randn(out_dim, in_dim, 2, 2, device="cuda")  # reps=1
        pw = torch.ones(out_dim, in_dim, 1, device="cuda")
        pb = torch.zeros(out_dim, in_dim, 1, device="cuda")

        def forward(rows):
            return triton_pz_forward(rows, theta, pw, pb, False, c_dtype=torch.bfloat16)

        with torch.no_grad():
            full = forward(x)  # one launch past 2^31 output elements
            for start in range(0, len(x), 256):
                part = forward(x[start : start + 256])
                assert torch.equal(full[start : start + 256], part), start
        print("OK")
        """,
        preamble="",
        timeout=1800,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
