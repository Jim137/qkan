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

"""Launch-error handling for the CuTe extension.

#22 added ``C10_CUDA_KERNEL_LAUNCH_CHECK`` to the dispatch macros, which is
what made silently-skipped launches visible.  Two consequences need pinning:
the check must not report faults it did not cause (#27), and the newly
reachable arch-mismatch path must not rebuild inside a training step (#28).
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CuTe kernels require CUDA"
)


def test_unrelated_async_fault_not_attributed_to_cute(run_snippet):
    """A fault left pending by other code must not surface as a CuTe error."""
    result = run_snippet("""
        from torch.utils.cpp_extension import load_inline

        # Stand-in for a third-party extension that launches without ever
        # calling cudaGetLastError, leaving the failure pending on the thread.
        probe = load_inline(
            name="qkan_unchecked_launch_probe",
            cpp_sources="void bad_launch();",
            cuda_sources=(
                "#include <torch/extension.h>\\n"
                "__global__ void noop_kernel() {}\\n"
                "void bad_launch() { noop_kernel<<<1, 100000>>>(); }\\n"
            ),
            functions=["bad_launch"],
            verbose=False,
        )

        out_dim, in_dim, reps = 4, 3, 2
        x = torch.randn(8, in_dim, device="cuda")
        theta = theta_for(out_dim, in_dim, reps)
        pw, pb = preacts_for(out_dim, in_dim, reps)
        reference = C.pz_forward(x, theta, pw, pb, False, True, False)

        probe.bad_launch()  # leaves cudaErrorInvalidConfiguration pending

        try:
            y = C.pz_forward(x, theta, pw, pb, False, True, False)
        except Exception as error:
            print("RAISED:", type(error).__name__, str(error)[:120])
        else:
            print("CLEAN:", torch.equal(y, reference))
    """)
    assert "CLEAN: True" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )


def test_arch_mismatch_does_not_rebuild_inside_forward(run_snippet):
    """An unusable prebuilt wheel must fail fast, not JIT-rebuild mid-step."""
    result = run_snippet("""
        import qkan.solver.cute.cute_ops as co

        co._ext = None
        co._supports_current_device = lambda ext: False

        def boom():
            raise AssertionError("JIT rebuild attempted inside forward")

        co._load_jit = boom

        try:
            co._get_ext()
        except AssertionError:
            raise
        except Exception as error:
            print("RAISED:", type(error).__name__)
            print("MESSAGE:", str(error)[:200])
        else:
            print("NO ERROR")
    """)
    assert "JIT rebuild attempted" not in result.stderr, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
    # ImportError is what cute.py's solver-selection guards already catch
    assert "RAISED: ImportError" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
    # the message has to say what to do, not just that something failed
    assert "QKAN_FORCE_BUILD" in result.stdout, result.stdout


def test_arch_mismatch_is_not_cached_as_usable(run_snippet):
    """A retry after the mismatch must not get the rejected module back.

    _load_prebuilt() assigns the module to the _ext cache as a side effect, so
    failing without clearing it makes the next call return the incompatible
    extension straight from cache and skip the probe entirely.
    """
    result = run_snippet("""
        import qkan.solver.cute.cute_ops as co

        co._ext = None
        co._supports_current_device = lambda ext: False
        co._load_jit = lambda: (_ for _ in ()).throw(
            AssertionError("JIT rebuild attempted")
        )

        for attempt in (1, 2):
            try:
                ext = co._get_ext()
            except ImportError:
                print(f"attempt {attempt}: ImportError")
            else:
                print(f"attempt {attempt}: RETURNED {ext!r}")
    """)
    assert result.stdout.count("ImportError") == 2, (
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
    assert "RETURNED" not in result.stdout, result.stdout
