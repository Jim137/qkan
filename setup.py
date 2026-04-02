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

"""
Build script for QKAN — handles optional CuTe CUDA extension.

The CuTe solver compiles when:
  1. ``pip install -e .[cute]`` is used, AND
  2. CUDA toolkit is available, AND
  3. CUTLASS headers are found (CUTLASS_PATH env var or standard locations)

If any condition fails, the build proceeds without the extension (JIT fallback).

Environment variables:
  CUTLASS_PATH   — root of CUTLASS checkout (containing include/cute/tensor.hpp)
  QKAN_CUDA_ARCHS — semicolon-separated SM list (default: auto-detect)
  NVCC_THREADS   — parallel nvcc compilation threads (default: 4)
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup

# ---------------------------------------------------------------------------
# Helpers — modelled after flash-attention/setup.py
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
NVCC_THREADS = os.getenv("NVCC_THREADS", "4")


def get_cuda_bare_metal_version(cuda_dir):
    """Return (major, minor) of the nvcc at *cuda_dir*."""
    nvcc = os.path.join(cuda_dir, "bin", "nvcc")
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
    except Exception:
        return None
    for line in out.split("\n"):
        if "release" in line:
            # e.g. "Cuda compilation tools, release 12.6, V12.6.77"
            parts = line.split("release")[-1].strip().split(",")[0].split(".")
            return (int(parts[0]), int(parts[1]))
    return None


def find_cuda_home():
    """Return CUDA_HOME or None."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isdir(cuda_home):
        return cuda_home
    # Try well-known locations
    for candidate in ["/usr/local/cuda", "/usr/local/cuda-12"]:
        if os.path.isdir(candidate):
            return candidate
    return None


def find_cutlass_include():
    """Return the CUTLASS include/ directory or None."""
    candidates = [
        os.environ.get("CUTLASS_PATH", ""),
        str(THIS_DIR.parent / "cutlass"),  # sibling checkout
        os.path.expanduser("~/cutlass"),
        "/usr/local/cutlass",
    ]
    for base in candidates:
        inc = os.path.join(base, "include")
        if os.path.isfile(os.path.join(inc, "cute", "tensor.hpp")):
            return inc
    return None


def get_cuda_archs():
    """Return list of SM architectures to compile for."""
    env = os.getenv("QKAN_CUDA_ARCHS", "")
    if env:
        return [a.strip() for a in env.split(";") if a.strip()]
    # Auto-detect from current GPU
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            sm = f"{cap[0]}{cap[1]}"
            return [sm]
    except Exception:
        pass
    return ["80"]  # safe default


def make_gencode_flags(archs, cuda_version):
    """Build -gencode flags for the given architectures."""
    flags = []
    major, minor = cuda_version
    ver = major * 10 + minor  # e.g. 126 for 12.6

    for sm in archs:
        sm_int = int(sm)
        if sm_int >= 120 and ver < 128:
            continue  # Blackwell needs CUDA >= 12.8
        if sm_int >= 100 and ver < 128:
            continue
        if sm_int >= 90 and ver < 118:
            continue  # Hopper needs CUDA >= 11.8

        compute = f"compute_{sm}"
        # Family-specific flag for Blackwell (CUDA >= 12.9)
        if sm_int >= 100 and ver >= 129:
            compute = f"compute_{sm}f"

        flags += ["-gencode", f"arch={compute},code=sm_{sm}"]

    # Embed PTX for newest arch (forward compatibility)
    if archs:
        newest = max(archs, key=int)
        flags += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return flags


# ---------------------------------------------------------------------------
# Conditional CuTe extension
# ---------------------------------------------------------------------------

ext_modules = []

# Only attempt if CUDA and CUTLASS are present
CUDA_HOME = find_cuda_home()
CUTLASS_INC = find_cutlass_include()
BUILD_CUTE = CUDA_HOME is not None and CUTLASS_INC is not None

# Allow explicit disable
if os.getenv("QKAN_SKIP_CUDA_BUILD", "0") == "1":
    BUILD_CUTE = False

if BUILD_CUTE:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cuda_version = get_cuda_bare_metal_version(CUDA_HOME)
    if cuda_version is None:
        print("WARNING: Could not detect CUDA version, skipping CuTe build")
        BUILD_CUTE = False

if BUILD_CUTE:
    archs = get_cuda_archs()
    cc_flags = make_gencode_flags(archs, cuda_version)

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        "--threads",
        NVCC_THREADS,
    ]

    ext_modules.append(
        CUDAExtension(
            name="qkan._C",
            sources=["src/qkan/csrc/cute_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags + cc_flags,
            },
            include_dirs=[CUTLASS_INC],
        )
    )

    print(f"[qkan] Building CuTe extension: CUDA {cuda_version}, archs={archs}")
    print(f"[qkan] CUTLASS include: {CUTLASS_INC}")

# ---------------------------------------------------------------------------
# setup() — pyproject.toml handles metadata; setup.py only adds ext_modules
# ---------------------------------------------------------------------------

cmdclass = {}
if BUILD_CUTE:
    cmdclass["build_ext"] = BuildExtension

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
