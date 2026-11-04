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

"""Batch-scaled offsets past 2^31 elements (flash #31; CuTe #21/#22; #35).

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
        reason="allocates up to 13 GB of GPU memory; set QKAN_LARGE_TESTS=1",
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


# ---------------------------------------------------------------------------
# #35: the same int32 offsets outside the solver kernels.  Each kernel is
# called directly, so the test needs only its own inputs, not a solver run.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton")
def test_fused_epilogue_past_int32(run_snippet):
    # postacts is (B, O, I): its batch offset b * O*I passes 2^31 at the last
    # row here.  The forward reads it, and the backward reads it for
    # grad_eff_pw and writes grad_postacts the same way.
    result = run_snippet(
        """
        import torch
        from qkan.qkan_epilogue import QKANEpilogue

        torch.manual_seed(0)
        B, O, I = 32769, 256, 256  # B * O * I = 2^31 + 65536
        dt = torch.bfloat16
        postacts = torch.randn(B, O, I, device="cuda", dtype=dt)
        eff_pw = torch.randn(O, I, device="cuda", dtype=dt, requires_grad=True)
        base_input = torch.randn(B, I, device="cuda", dtype=dt)
        base_w = torch.randn(O, I, device="cuda", dtype=dt)

        def epilogue(rows):
            p = postacts[rows].detach().requires_grad_()
            y = QKANEpilogue.apply(p, eff_pw, None, base_input[rows], base_w)
            grad_p, grad_pw = torch.autograd.grad(y.float().sum(), (p, eff_pw))
            return y.detach(), grad_p, grad_pw

        full_y, full_grad, grad_pw = epilogue(slice(None))  # one launch past 2^31
        for start in range(0, B, 8192):  # each launch far below it
            rows = slice(start, start + 8192)
            y, grad, _ = epilogue(rows)
            assert torch.equal(full_y[rows], y), ("forward", start)
            assert torch.equal(full_grad[rows], grad), ("backward", start)
        # grad_out is all ones, so grad_eff_pw is postacts summed over the
        # batch, up to bf16 rounding of the result.
        ref_pw = postacts.sum(0, dtype=torch.float32)
        torch.testing.assert_close(grad_pw.float(), ref_pw, rtol=2**-8, atol=1e-3)
        print("OK")
        """,
        preamble="",
        timeout=1800,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )


@pytest.mark.parametrize(
    "module, fn",
    [
        pytest.param(
            "flash.triton_linear",
            "triton_linear_forward",
            marks=pytest.mark.skipif(not _FLASH_AVAILABLE, reason="no triton"),
            id="triton",
        ),
        # Already passes before #35 (cuTile addresses tiles in 64 bits); it
        # guards the cuTile twin of TritonLinear at the same sizes.
        pytest.param(
            "cutile.cutile_linear",
            "cutile_linear_forward",
            marks=pytest.mark.skipif(not _CUTILE_AVAILABLE, reason="no cuda-tile"),
            id="cutile",
        ),
    ],
)
@pytest.mark.parametrize(
    "m, k, n",
    [
        (2**19 + 1, 4096, 64),  # input offset m * K passes 2^31
        (2**21 + 1, 16, 1024),  # output offset m * N passes 2^31
    ],
)
def test_linear_past_int32(run_snippet, module, fn, m, k, n):
    result = run_snippet(
        f"""
        import torch
        import torch.nn.functional as F
        from qkan.solver.{module} import {fn} as linear

        torch.manual_seed(0)
        x = torch.randn({m}, {k}, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn({n}, {k}, device="cuda") / {k} ** 0.5).to(torch.bfloat16)
        y = linear(x, w, None)  # past 2^31 elements
        # Compare in row blocks so assert_close's temporaries stay small.
        for x_rows, y_rows in zip(x.split(2**18), y.split(2**18)):
            ref = F.linear(x_rows, w)
            torch.testing.assert_close(y_rows, ref, rtol=2e-2, atol=5e-2)
        print("OK")
        """,
        preamble="",
        timeout=1800,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )


@pytest.mark.skipif(not _CUTILE_AVAILABLE, reason="no cuda-tile")
def test_cutile_activation_past_int32(run_snippet):
    # The activations flatten their input into one 1-D array, and cuTile
    # rejects an array past 2^31 elements ("shape is too big").
    result = run_snippet(
        """
        import torch
        from qkan.solver.cutile.cutile_activations import (
            cutile_activation_backward,
            cutile_activation_forward,
        )

        torch.manual_seed(0)
        dt = torch.bfloat16
        n, chunk = 2**31 + 4096, 2**29

        def check(fn, *tensors):
            full = fn(*tensors, "silu", dt)  # the whole tensor in one call
            for start in range(0, n, chunk):
                rows = slice(start, start + chunk)
                part = fn(*(t[rows] for t in tensors), "silu", dt)
                assert torch.equal(full[rows], part), (fn.__name__, start)

        x = torch.randn(n, device="cuda", dtype=dt)
        check(cutile_activation_forward, x)
        check(cutile_activation_backward, x, x)  # x doubles as grad_y
        print("OK")
        """,
        preamble="",
        timeout=1800,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
