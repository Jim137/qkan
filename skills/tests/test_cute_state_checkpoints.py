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

"""State-checkpoint precision has to change what the backward allocates.

The states buffer is the largest allocation in a CuTe backward, and choosing
a narrower checkpoint dtype exists to shrink it.  cute_real_backward had no
bf16 allocation arm, so bf16 states were written into an fp32-sized buffer
and the saving never happened — on the default path, since QKANLayer runs
the real ansatz with c_dtype=bfloat16 by default (#25).

Each launcher still picks the buffer dtype and the kernel's state type in two
separate if/else chains, so every (ansatz, precision) pair is checked here
rather than only the one that regressed.
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CuTe kernels require CUDA"
)

BACKWARDS = {
    "pz": "C.pz_backward(x, theta, pw, pb, go, True, True, bits)",
    "rpz": "C.rpz_backward(x, theta, pw, pb, go, True, bits)",
    "real": "C.real_backward(x, theta, pw, pb, go, True, True, True, bits)",
}

# state_bits -> the largest peak / fp32-peak ratio that still shows the narrow
# dtype was honoured (states dominate, so the ideal is 1/2 and 1/4).
NARROW = {16: 0.6, 8: 0.35}


@pytest.mark.parametrize("bits", list(NARROW))
@pytest.mark.parametrize("ansatz", list(BACKWARDS))
def test_narrow_states_shrink_the_states_buffer(run_snippet, ansatz, bits):
    result = run_snippet(f"""
        out_dim = in_dim = 16
        reps, batch = 4, 256
        x = torch.randn(batch, in_dim, device="cuda")
        theta = theta_for(out_dim, in_dim, reps, "{ansatz}")
        pw, pb = preacts_for(out_dim, in_dim, reps)
        go = torch.randn(batch, out_dim, in_dim, device="cuda")

        def peak(bits):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            {BACKWARDS[ansatz]}
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base

        print("PEAK", peak(32), peak({bits}))
    """)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("PEAK")), "")
    assert line, f"stdout={result.stdout}\nstderr={result.stderr}"
    fp32, narrow = map(int, line.split()[1:])
    assert narrow / fp32 < NARROW[bits], (
        f"{bits}-bit states peak {narrow / 2**20:.2f} MiB vs fp32 "
        f"{fp32 / 2**20:.2f} MiB (ratio {narrow / fp32:.3f})"
    )
