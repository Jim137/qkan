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


"""Shared fixtures for the CuTe launcher tests.

Several of the failures under test poison the CUDA context for the whole
process (illegal memory access, misaligned address), so each case runs in its
own interpreter.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

# theta layout is per-ansatz: pz (o, i, reps+1, 2), rpz (o, i, reps+1, 1),
# real (o, i, reps, 1).  Kept next to the runner so both test modules build
# shapes the same way.
PREAMBLE = """
import torch
import qkan._C as C


def theta_for(out_dim, in_dim, reps, ansatz="pz", device="cuda"):
    last = 2 if ansatz == "pz" else 1
    axis = reps if ansatz == "real" else reps + 1
    return torch.randn(out_dim, in_dim, axis, last, device=device)


def preacts_for(out_dim, in_dim, reps, device="cuda"):
    w = torch.randn(out_dim, in_dim, reps, device=device)
    b = torch.randn(out_dim, in_dim, reps, device=device)
    return w, b
"""


@pytest.fixture(scope="session")
def run_snippet():
    """Run a snippet against this checkout in a fresh interpreter."""

    def _run(body, *, timeout=300):
        return subprocess.run(
            [sys.executable, "-c", PREAMBLE + textwrap.dedent(body)],
            capture_output=True,
            text=True,
            # inherit the environment: snippets that JIT-compile a probe
            # extension need ninja and the CUDA toolchain on PATH.
            env=dict(os.environ, PYTHONPATH=str(SRC)),
            timeout=timeout,
        )

    return _run
