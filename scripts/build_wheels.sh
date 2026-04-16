#!/bin/bash
# Build QKAN wheels with CuTe CUDA extension for distribution.
#
# Follows flash-attention's wheel-building pattern:
# - Embeds CUDA version + PyTorch version + CXX11 ABI in wheel name
# - Requires CUTLASS_PATH to be set
# - Builds for the current GPU architecture by default
#
# Usage:
#   # Build wheel for current environment
#   ./scripts/build_wheels.sh
#
#   # Build for specific architectures
#   QKAN_CUDA_ARCHS="80;90;120" ./scripts/build_wheels.sh
#
#   # Build source distribution (no CUDA)
#   QKAN_SKIP_CUDA_BUILD=TRUE ./scripts/build_wheels.sh sdist
#
# Output: dist/*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── Check prerequisites ──────────────────────────────────────────────

if [ -z "${CUTLASS_PATH:-}" ]; then
    if [ -f "../cutlass/include/cute/tensor.hpp" ]; then
        export CUTLASS_PATH="$(cd ../cutlass && pwd)"
        echo "[build] Auto-detected CUTLASS_PATH=$CUTLASS_PATH"
    else
        echo "[build] CUTLASS_PATH not set — setup.py will auto-download headers"
    fi
fi

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" || {
    echo "ERROR: PyTorch not found. Install PyTorch first."
    exit 1
}

# ── Determine version tag ────────────────────────────────────────────

CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.','')[:3])")
TORCH_VERSION=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')")
CXX11_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")

export QKAN_LOCAL_VERSION="cu${CUDA_VERSION}torch${TORCH_VERSION}cxx11abi${CXX11_ABI}"

echo "[build] Building QKAN wheel"
echo "  CUDA:     ${CUDA_VERSION}"
echo "  PyTorch:  ${TORCH_VERSION}"
echo "  CXX11ABI: ${CXX11_ABI}"
echo "  Version:  $(python -c "from setup import get_package_version; print(get_package_version())")"
echo "  CUTLASS:  ${CUTLASS_PATH}"
echo "  Archs:    ${QKAN_CUDA_ARCHS:-auto-detect}"

# ── Build ─────────────────────────────────────────────────────────────

if [ "${1:-wheel}" = "sdist" ]; then
    echo "[build] Building source distribution..."
    QKAN_SKIP_CUDA_BUILD=TRUE python setup.py sdist --dist-dir=dist
else
    echo "[build] Building wheel..."
    QKAN_FORCE_BUILD=TRUE python setup.py bdist_wheel --dist-dir=dist
fi

echo ""
echo "[build] Done. Output:"
ls -lh dist/*.whl 2>/dev/null || ls -lh dist/*.tar.gz 2>/dev/null
