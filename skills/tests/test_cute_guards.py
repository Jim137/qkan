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

"""Host-side guard tests for the CuTe launchers.

The kernels keep parameter-tensor offsets in int32, so the launchers must
reject the shapes that would wrap or produce an invalid launch config.
Several of these failures poison the CUDA context for the whole process
(illegal memory access), so every case runs in its own subprocess.
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CuTe kernels require CUDA"
)


# ---------------------------------------------------------------------------
# Empty batch — batch == 0 yields dim3 grid(n_oi, 0), which is an invalid
# launch configuration, so the dispatch macro has to skip the launch.
# ---------------------------------------------------------------------------


FORWARD_CALLS = {
    # each launcher has its own trailing flags
    "pz": "C.pz_forward(x, theta, pw, pb, False, True, False)",
    "rpz": "C.rpz_forward(x, theta, pw, pb, True, False)",
    "real": "C.real_forward(x, theta, pw, pb, False, True, True, False)",
}


@pytest.mark.parametrize("ansatz", list(FORWARD_CALLS))
def test_empty_batch_forward_returns_empty(run_snippet, ansatz):
    result = run_snippet(f"""
        out_dim, in_dim, reps = 4, 3, 2
        x = torch.empty(0, in_dim, device="cuda")
        theta = theta_for(out_dim, in_dim, reps, "{ansatz}")
        pw, pb = preacts_for(out_dim, in_dim, reps)
        y = {FORWARD_CALLS[ansatz]}
        assert y.shape == (0, out_dim, in_dim), y.shape
        print("OK")
    """)
    assert "OK" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_empty_batch_backward_returns_grads(run_snippet):
    result = run_snippet("""
        out_dim, in_dim, reps = 4, 3, 2
        x = torch.empty(0, in_dim, device="cuda")
        theta = theta_for(out_dim, in_dim, reps)
        pw, pb = preacts_for(out_dim, in_dim, reps)
        grad_out = torch.empty(0, out_dim, in_dim, device="cuda")
        grad_x, grad_theta = C.pz_backward(
            x, theta, pw, pb, grad_out, False, True, 32
        )[:2]
        assert grad_x.shape == (0, in_dim), grad_x.shape
        assert grad_theta.shape == theta.shape, grad_theta.shape
        assert torch.count_nonzero(grad_theta) == 0
        print("OK")
    """)
    assert "OK" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"


# ---------------------------------------------------------------------------
# Oversized batch — the grid.y guard has to run before the output allocation,
# or the launcher OOMs on a 32 GiB tensor instead of saying what went wrong.
# ---------------------------------------------------------------------------


def test_oversized_batch_reports_grid_limit_not_oom(run_snippet):
    result = run_snippet("""
        out_dim, in_dim, reps = 64, 8, 2
        x = torch.empty(17_000_000, in_dim, device="cuda")
        theta = theta_for(out_dim, in_dim, reps)
        pw, pb = preacts_for(out_dim, in_dim, reps)
        try:
            C.pz_forward(x, theta, pw, pb, False, True, False)
        except Exception as error:
            print("RAISED:", type(error).__name__, str(error)[:200])
        else:
            print("NO ERROR")
    """)
    assert "grid.y blocks" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# reps has to be bounded both ways: theta.size(2) == 0 gives reps == -1, and
# n_states plus the smem byte count derive from reps and are int32.
# ---------------------------------------------------------------------------


def test_degenerate_reps_rejected(run_snippet):
    result = run_snippet("""
        out_dim, in_dim = 4, 3
        x = torch.randn(8, in_dim, device="cuda")
        theta = torch.empty(out_dim, in_dim, 0, 2, device="cuda")
        pw = torch.empty(out_dim, in_dim, 0, device="cuda")
        pb = torch.empty(out_dim, in_dim, 0, device="cuda")
        try:
            C.pz_forward(x, theta, pw, pb, False, True, False)
        except Exception as error:
            print("RAISED:", type(error).__name__, str(error)[:200])
        else:
            print("NO ERROR")
    """)
    assert "reps" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "illegal memory access" not in result.stdout


def test_zero_reps_real_ansatz_accepted(run_snippet):
    """real derives reps from theta.size(2) directly, so reps == 0 is valid.

    pz/rpz take theta.size(2) - 1, so the same axis length means reps == -1
    there; the bound belongs on the derived reps, not on the axis.
    """
    result = run_snippet("""
        from qkan import QKANLayer

        layer = QKANLayer(6, 4, reps=0, ansatz="real", solver="cute", device="cuda")
        assert layer.theta.shape[2] == 0, layer.theta.shape
        y = layer(torch.randn(8, 6, device="cuda"))
        assert torch.isfinite(y).all(), y
        print("OK", tuple(y.shape))
    """)
    assert "OK" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_reps_exceeding_shared_memory_rejected(run_snippet):
    result = run_snippet("""
        out_dim, in_dim, reps = 2, 2, 100_000
        x = torch.randn(8, in_dim, device="cuda")
        theta = theta_for(out_dim, in_dim, reps)
        pw, pb = preacts_for(out_dim, in_dim, reps)
        try:
            C.pz_forward(x, theta, pw, pb, False, True, False)
        except Exception as error:
            print("RAISED:", type(error).__name__, str(error)[:200])
        else:
            print("NO ERROR")
    """)
    assert "shared memory" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# pb is never validated, yet grad_pb is allocated from pb.sizes() and written
# with the same raw int32 index as grad_pw.
# ---------------------------------------------------------------------------


def test_mismatched_preacts_bias_rejected(run_snippet):
    result = run_snippet("""
        out_dim, in_dim, reps = 4, 3, 2
        x = torch.randn(8, in_dim, device="cuda")
        theta = theta_for(out_dim, in_dim, reps)
        pw = torch.randn(out_dim, in_dim, reps, device="cuda")
        pb = torch.randn(1, device="cuda")
        try:
            C.pz_forward(x, theta, pw, pb, True, True, False)
        except Exception as error:
            print("RAISED:", type(error).__name__, str(error)[:200])
        else:
            print("NO ERROR")
    """)
    assert "preacts" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "NO ERROR" not in result.stdout


# ---------------------------------------------------------------------------
# The kernels build theta's int32 strides from x.size(1), so the bound the
# numel checks establish is not the quantity they index with unless the two
# agree.  At the kernel boundary the layer always passes theta as
# (out_dim, in_dim, reps..., ...), whatever the layer's own grouping.
# ---------------------------------------------------------------------------


def test_theta_in_dim_mismatch_rejected(run_snippet):
    result = run_snippet("""
        out_dim, in_dim, reps = 4, 3, 2
        # x five columns wider than theta was built for
        x = torch.randn(8, in_dim + 5, device="cuda")
        theta = theta_for(out_dim, in_dim, reps)
        pw, pb = preacts_for(out_dim, in_dim, reps)
        try:
            C.pz_forward(x, theta, pw, pb, False, True, False)
        except Exception as error:
            print("RAISED:", type(error).__name__, str(error)[:200])
        else:
            print("NO ERROR")
    """)
    assert "in_dim" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "NO ERROR" not in result.stdout


# ---------------------------------------------------------------------------
# pz backward's state stride (n_states * 4 * BLOCK_B) is derived from reps.
# It is 64-bit, but the value itself only stays in int32 range because the
# shared-memory guard bounds reps -- pin that coupling so raising either the
# smem limit or BLOCK_B cannot silently reintroduce the overflow.
# ---------------------------------------------------------------------------


def test_smem_guard_bounds_pz_state_stride_below_int32(run_snippet):
    result = run_snippet("""
        BLOCK_B_MAX = 256  # select_block_b's largest tile

        def accepted(reps):
            x = torch.randn(8, 2, device="cuda")
            theta = theta_for(2, 2, reps)
            pw, pb = preacts_for(2, 2, reps)
            go = torch.randn(8, 2, 2, device="cuda")
            try:
                C.pz_backward(x, theta, pw, pb, go, False, True, 32)
                return True
            except RuntimeError as error:
                if "shared memory" in str(error):
                    return False
                raise

        lo, hi = 1, 1 << 20
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if accepted(mid):
                lo = mid
            else:
                hi = mid - 1
        # batch 8 selects the smallest tile, so this reps ceiling is the
        # loosest one the guard ever allows -- worst case for the stride.
        n_states = 3 * lo + 3
        stride = n_states * 4 * BLOCK_B_MAX
        print(f"max_reps={lo} worst_stride={stride} int32_max={2**31 - 1}")
        assert stride < 2**31, stride
        print("OK")
    """)
    assert "OK" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"
