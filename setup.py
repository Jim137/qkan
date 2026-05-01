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
Build script for QKAN with optional CuTe CUDA extension.

Follows the flash-attention build pattern:
  1. Try downloading a pre-built wheel from GitHub releases
  2. Fall back to local compilation if download fails
  3. Skip CUDA build entirely when QKAN_SKIP_CUDA_BUILD=TRUE

Environment variables:
  CUTLASS_PATH        — root of CUTLASS checkout (containing include/cute/tensor.hpp)
  QKAN_CUDA_ARCHS     — semicolon-separated SM list (default: auto-detect)
  QKAN_FORCE_BUILD    — TRUE to skip pre-built wheel download and compile locally
  QKAN_SKIP_CUDA_BUILD — TRUE to skip CUDA compilation entirely (sdist builds)
  QKAN_LOCAL_VERSION  — local version suffix (e.g. "cu126torch2.6")
  NVCC_THREADS        — parallel nvcc compilation threads (default: 4)

Usage:
  # Standard install (downloads pre-built wheel or compiles)
  CUTLASS_PATH=/path/to/cutlass pip install --no-build-isolation -e .[cute]

  # Force local build
  QKAN_FORCE_BUILD=TRUE CUTLASS_PATH=/path/to/cutlass pip install --no-build-isolation -e .[cute]

  # Build wheel for distribution
  QKAN_LOCAL_VERSION=cu126torch2.6 python setup.py bdist_wheel
"""

import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from setuptools import setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = "qkan"
NVCC_THREADS = os.getenv("NVCC_THREADS", "4")
FORCE_BUILD = os.getenv("QKAN_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("QKAN_SKIP_CUDA_BUILD", "FALSE") == "TRUE"

# GitHub releases URL for pre-built wheels
WHEEL_BASE_URL = "https://github.com/Jim137/qkan/releases/download"


# ---------------------------------------------------------------------------
# Version helpers (flash-attention pattern)
# ---------------------------------------------------------------------------

def get_package_version():
    """Read version from qkan/__init__.py, optionally append local version."""
    init_path = THIS_DIR / "src" / "qkan" / "__init__.py"
    with open(init_path, "r") as f:
        version_match = re.search(r'^__version__\s*=\s*["\'](.+?)["\']', f.read(), re.MULTILINE)
    if version_match is None:
        raise RuntimeError("Cannot find __version__ in qkan/__init__.py")
    public_version = version_match.group(1)
    local_version = os.environ.get("QKAN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    return public_version


def get_wheel_url():
    """Construct the URL and filename for a pre-built wheel."""
    import torch

    version = get_package_version()
    # Strip any existing local version for the base
    base_version = version.split("+")[0]

    cuda_version = torch.version.cuda
    if cuda_version is not None:
        cuda_short = cuda_version.replace(".", "")[:3]  # "12.6" → "126"
    else:
        return None, None

    torch_version = torch.__version__.split("+")[0]  # "2.6.0" → "2.6.0"
    torch_short = ".".join(torch_version.split(".")[:2])  # "2.6"

    # C++11 ABI flag
    cxx11_abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"

    # Python version tag
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Platform tag
    plat = platform.machine()
    platform_name = f"linux_{plat}" if platform.system() == "Linux" else f"{platform.system().lower()}_{plat}"

    local_tag = f"cu{cuda_short}torch{torch_short}cxx11abi{cxx11_abi}"
    wheel_filename = (
        f"{PACKAGE_NAME}-{base_version}+{local_tag}-"
        f"{python_version}-{python_version}-{platform_name}.whl"
    )
    wheel_url = f"{WHEEL_BASE_URL}/v{base_version}/{wheel_filename}"

    return wheel_url, wheel_filename


# ---------------------------------------------------------------------------
# CUDA detection helpers
# ---------------------------------------------------------------------------

def get_cuda_bare_metal_version(cuda_dir):
    """Return (major, minor) of the nvcc at *cuda_dir*."""
    nvcc = os.path.join(cuda_dir, "bin", "nvcc")
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
    except Exception:
        return None
    for line in out.split("\n"):
        if "release" in line:
            parts = line.split("release")[-1].strip().split(",")[0].split(".")
            return (int(parts[0]), int(parts[1]))
    return None


def find_cuda_home():
    """Return CUDA_HOME or None."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isdir(cuda_home):
        return cuda_home
    for candidate in ["/usr/local/cuda", "/usr/local/cuda-12"]:
        if os.path.isdir(candidate):
            return candidate
    return None


def find_cutlass_include():
    """Return the CUTLASS include/ directory, auto-downloading if needed."""
    candidates = [
        os.environ.get("CUTLASS_PATH", ""),
        str(THIS_DIR.parent / "cutlass"),
        os.path.expanduser("~/cutlass"),
        "/usr/local/cutlass",
        str(THIS_DIR / ".cutlass"),  # auto-downloaded location
    ]
    for base in candidates:
        inc = os.path.join(base, "include")
        if os.path.isfile(os.path.join(inc, "cute", "tensor.hpp")):
            return inc

    # Auto-download CUTLASS headers (sparse checkout — only include/)
    if os.getenv("QKAN_NO_CUTLASS_DOWNLOAD", "0") == "1":
        return None

    print("[qkan] CUTLASS headers not found — downloading automatically...")
    dl_dir = str(THIS_DIR / ".cutlass")
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/NVIDIA/cutlass.git", dl_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "sparse-checkout", "set", "include/"],
            cwd=dl_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        inc = os.path.join(dl_dir, "include")
        if os.path.isfile(os.path.join(inc, "cute", "tensor.hpp")):
            print(f"[qkan] CUTLASS downloaded to {dl_dir}")
            return inc
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[qkan] WARNING: Failed to download CUTLASS. Install git or set CUTLASS_PATH.")
    return None


def get_cuda_archs():
    """Return list of SM architectures to compile for."""
    env = os.getenv("QKAN_CUDA_ARCHS", "")
    if env:
        return [a.strip() for a in env.split(";") if a.strip()]
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return [f"{cap[0]}{cap[1]}"]
    except Exception:
        pass
    return ["80"]


def make_gencode_flags(archs, cuda_version):
    """Build -gencode flags for the given architectures."""
    flags = []
    major, minor = cuda_version
    ver = major * 10 + minor

    for sm in archs:
        sm_int = int(sm)
        if sm_int >= 120 and ver < 128:
            continue
        if sm_int >= 100 and ver < 128:
            continue
        if sm_int >= 90 and ver < 118:
            continue

        compute = f"compute_{sm}"
        if sm_int >= 100 and ver >= 129:
            compute = f"compute_{sm}f"

        flags += ["-gencode", f"arch={compute},code=sm_{sm}"]

    if archs:
        newest = max(archs, key=int)
        flags += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return flags


# ---------------------------------------------------------------------------
# Cached wheel command (flash-attention pattern)
# ---------------------------------------------------------------------------

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class CachedWheelsCommand(_bdist_wheel):
        """Try downloading a pre-built wheel before building from source."""

        def run(self):
            if FORCE_BUILD:
                return super().run()

            try:
                wheel_url, wheel_filename = get_wheel_url()
            except Exception:
                print("[qkan] Could not determine wheel URL, building from source")
                return super().run()

            if wheel_url is None:
                return super().run()

            print(f"[qkan] Checking for pre-built wheel: {wheel_url}")
            try:
                urllib.request.urlretrieve(wheel_url, wheel_filename)

                if not os.path.exists(self.dist_dir):
                    os.makedirs(self.dist_dir)

                impl_tag, abi_tag, plat_tag = self.get_tag()
                archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
                wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
                os.rename(wheel_filename, wheel_path)
                print(f"[qkan] Using pre-built wheel: {wheel_path}")
            except (urllib.error.HTTPError, urllib.error.URLError):
                print("[qkan] Pre-built wheel not found, building from source...")
                try:
                    super().run()
                except Exception as e:
                    print(f"[qkan] CUDA build failed ({e}), building pure-Python wheel")
                    # Remove CUDA extension and rebuild without it
                    self.distribution.ext_modules = []
                    super().run()

except ImportError:
    CachedWheelsCommand = None


# ---------------------------------------------------------------------------
# CuTe CUDA extension
# ---------------------------------------------------------------------------

ext_modules = []
BUILD_CUTE = False

if not SKIP_CUDA_BUILD:
    CUDA_HOME = find_cuda_home()
    CUTLASS_INC = find_cutlass_include() if CUDA_HOME else None
    BUILD_CUTE = CUDA_HOME is not None and CUTLASS_INC is not None

    if BUILD_CUTE:
        try:
            from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        except ImportError:
            print("[qkan] torch not found, skipping CuTe build (install torch first)")
            BUILD_CUTE = False

    if BUILD_CUTE:
        cuda_version = get_cuda_bare_metal_version(CUDA_HOME)
        if cuda_version is None:
            print("[qkan] WARNING: Could not detect CUDA version, skipping CuTe build")
            BUILD_CUTE = False

    # Check for CUDA version mismatch (common in build-isolation environments
    # where pip installs a torch with a different CUDA than the system nvcc).
    # Under QKAN_FORCE_BUILD=TRUE (CI), we trust the caller: a newer nvcc can
    # target the architectures torch was built against. Only warn in that case,
    # don't skip. Without FORCE_BUILD, skip → pure-Python fallback, which is
    # the right behaviour for `pip install qkan` where torch's CUDA runtime
    # and the system nvcc may legitimately differ.
    if BUILD_CUTE:
        try:
            import torch
            torch_cuda = torch.version.cuda
            if torch_cuda is not None:
                torch_ver = int(torch_cuda.split(".")[0]) * 10 + int(torch_cuda.split(".")[1])
                sys_ver = cuda_version[0] * 10 + cuda_version[1]
                if torch_ver != sys_ver:
                    msg = (f"[qkan] System CUDA {cuda_version[0]}.{cuda_version[1]} "
                           f"vs torch CUDA {torch_cuda}")
                    if FORCE_BUILD:
                        print(f"{msg} — proceeding (QKAN_FORCE_BUILD=TRUE)")
                    else:
                        print(f"{msg} — skipping CuTe build "
                              f"(use --no-build-isolation to match)")
                        BUILD_CUTE = False
        except Exception:
            pass

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
            sources=[
                "csrc/cute_kernels.cu",
                "csrc/cute_activations.cu",
                "csrc/cute_linear.cu",
            ],
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
# setup()
# ---------------------------------------------------------------------------

cmdclass = {}
if BUILD_CUTE:
    cmdclass["build_ext"] = BuildExtension
if CachedWheelsCommand is not None:
    cmdclass["bdist_wheel"] = CachedWheelsCommand

setup(
    version=get_package_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
