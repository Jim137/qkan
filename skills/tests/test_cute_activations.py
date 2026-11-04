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

"""CuTe pointwise activation kernels on inputs the vectorised path can't take.

The vectorised kernels issue 16-byte loads, but a contiguous *view* keeps its
storage offset in data_ptr(), so ``x[1:]`` or the second chunk of
``torch.chunk(x, k)`` is contiguous yet not 16-byte aligned.  That reached the
int4 load and killed the CUDA context with ``misaligned address`` (#26).
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CuTe kernels require CUDA"
)

DTYPES = ["float32", "bfloat16"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_on_misaligned_view_matches_aligned(run_snippet, dtype):
    result = run_snippet(f"""
        x = torch.randn(4097, device="cuda", dtype=torch.{dtype})
        view = x[1:]
        assert view.is_contiguous() and view.data_ptr() % 16 != 0
        # same kernel math on an aligned copy is the exact reference
        expected = C.activation_forward(view.clone(), 0)  # silu
        got = C.activation_forward(view, 0)
        torch.cuda.synchronize()
        print("EQUAL:", torch.equal(got, expected))
    """)
    assert "EQUAL: True" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-1500:]}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_on_misaligned_views_matches_aligned(run_snippet, dtype):
    result = run_snippet(f"""
        x = torch.randn(4097, device="cuda", dtype=torch.{dtype})[1:]
        grad_y = torch.randn(4099, device="cuda", dtype=torch.{dtype})[3:]
        assert x.data_ptr() % 16 != 0 and grad_y.data_ptr() % 16 != 0
        expected = C.activation_backward(grad_y.clone(), x.clone(), 0)
        got = C.activation_backward(grad_y, x, 0)
        torch.cuda.synchronize()
        print("EQUAL:", torch.equal(got, expected))
    """)
    assert "EQUAL: True" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-1500:]}"
    )


def test_layer_on_batch_chunks_matches_full_batch(run_snippet):
    """The real-world trigger: micro-batching a layer whose in_dim * 4 bytes
    is not a multiple of 16, so chunk boundaries land off 16-byte alignment."""
    result = run_snippet("""
        from qkan import QKANLayer

        torch.manual_seed(0)
        layer = QKANLayer(7, 4, reps=2, solver="cute", device="cuda")
        x = torch.randn(64, 7, device="cuda")
        full = layer(x)
        chunks = torch.chunk(x, 3)
        assert any(c.data_ptr() % 16 for c in chunks)
        parts = [layer(c) for c in chunks]
        sum(p.sum() for p in parts).backward()
        torch.cuda.synchronize()
        print("MATCH:", torch.allclose(torch.cat(parts), full, atol=1e-6))
    """)
    assert "MATCH: True" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-1500:]}"
    )


# ---------------------------------------------------------------------------
# The element count comes from x alone, so a smaller grad_y is read past its
# end — silently, unless the allocator happens to leave it unmapped (#24).
# ---------------------------------------------------------------------------


def test_backward_rejects_grad_y_shape_mismatch(run_snippet):
    result = run_snippet("""
        try:
            C.activation_backward(
                torch.randn(8, device="cuda"), torch.randn(1_000_000, device="cuda"), 0
            )
        except Exception as error:
            print("RAISED:", type(error).__name__, str(error)[:200])
        else:
            print("NO ERROR")
    """)
    assert "grad_y shape" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
