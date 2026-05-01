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
CuTe DSL CUDA kernels for QKAN quantum circuit simulation.

Uses CUTLASS CuTe tensor abstractions, __sincosf intrinsics, shared-memory
trig caching, and warp-shuffle reductions.

Loading strategy (flash-attention pattern):
  1. Try importing the pre-built extension ``qkan._C`` (compiled by setup.py)
  2. Fall back to JIT compilation via ``torch.utils.cpp_extension.load()``
  3. If neither works, ``_CUTE_KERNELS_AVAILABLE`` stays False

Build the extension with::

    CUTLASS_PATH=/path/to/cutlass pip install -e .[cute]
"""

from __future__ import annotations

import os
import pathlib

import torch

# ---------------------------------------------------------------------------
# Extension loading: pre-built → JIT fallback
# ---------------------------------------------------------------------------

_ext = None
_CUTE_KERNELS_AVAILABLE = False


def _find_cutlass_include() -> str | None:
    """Locate CUTLASS include/ directory."""
    # Project root: src/qkan/cute_ops.py → parents[2] = project root
    project_root = pathlib.Path(__file__).resolve().parents[2]
    candidates = [
        os.environ.get("CUTLASS_PATH", ""),
        str(project_root.parent / "cutlass"),  # sibling checkout
        str(project_root / ".cutlass"),  # auto-downloaded by setup.py
        "/usr/local/cutlass",
        os.path.expanduser("~/cutlass"),
    ]
    for base in candidates:
        inc = os.path.join(base, "include")
        if os.path.isfile(os.path.join(inc, "cute", "tensor.hpp")):
            return inc
    return None


def _load_prebuilt():
    """Try importing the extension compiled by setup.py."""
    global _ext, _CUTE_KERNELS_AVAILABLE
    try:
        import qkan._C as _prebuilt  # type: ignore[import-not-found]

        _ext = _prebuilt
        _CUTE_KERNELS_AVAILABLE = True
        return _ext
    except ImportError:
        return None


def _load_jit():
    """Fall back to JIT compilation (requires CUTLASS headers at runtime)."""
    global _ext, _CUTE_KERNELS_AVAILABLE

    cutlass_inc = _find_cutlass_include()
    if cutlass_inc is None:
        raise ImportError(
            "CuTe extension not pre-built and CUTLASS headers not found.\n"
            "Either:\n"
            "  CUTLASS_PATH=/path/to/cutlass pip install -e .[cute]\n"
            "or:\n"
            "  export CUTLASS_PATH=/path/to/cutlass  (for JIT fallback)"
        )

    from torch.utils.cpp_extension import load

    csrc_dir = pathlib.Path(__file__).resolve().parents[2] / "csrc"
    sources = [
        str(csrc_dir / "cute_kernels.cu"),
        str(csrc_dir / "cute_activations.cu"),
        str(csrc_dir / "cute_linear.cu"),
    ]
    _ext = load(
        name="qkan_cute_ops",
        sources=sources,
        extra_include_paths=[cutlass_inc],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-lineinfo",
        ],
        verbose=False,
    )
    _CUTE_KERNELS_AVAILABLE = True
    return _ext


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext
    # Try pre-built first (instant), then JIT (slow first time)
    _ext = _load_prebuilt()
    if _ext is None:
        _ext = _load_jit()
    return _ext


# ---------------------------------------------------------------------------
# Eagerly check availability (don't compile, just probe)
# ---------------------------------------------------------------------------

try:
    import qkan._C  # type: ignore[import-not-found]  # noqa: F401

    _CUTE_KERNELS_AVAILABLE = True
except ImportError:
    try:
        if _find_cutlass_include() is not None:
            _CUTE_KERNELS_AVAILABLE = True  # headers found → JIT will work
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Python wrappers matching the fused_ops.py interface
# ---------------------------------------------------------------------------


def _state_bits(c_dtype: torch.dtype) -> int:
    """Map c_dtype to state checkpoint precision bits (32=f32, 16=bf16, 8=nvfp8)."""
    if c_dtype == torch.float8_e4m3fn:
        return 8
    if c_dtype == torch.bfloat16:
        return 16
    return 32


def _use_bf16(c_dtype: torch.dtype) -> bool:
    """Whether to use bf16 I/O for forward kernels (halves memory bandwidth)."""
    return c_dtype in (torch.bfloat16, torch.float8_e4m3fn)


def _resolve_io_dtype(c_dtype: torch.dtype) -> torch.dtype:
    """Map QKAN compute dtype to I/O dtype for CuTe kernels.

    complex64/128 → float32 (QKAN uses complex to mean 'full precision float')
    float8_e4m3fn → bfloat16 (bf16 I/O with fp8 checkpoints)
    """
    if c_dtype in (torch.complex64, torch.complex128, torch.float32):
        return torch.float32
    if c_dtype == torch.float8_e4m3fn:
        return torch.bfloat16
    return c_dtype  # bfloat16 etc.


def cute_pz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """CuTe PZ-encoding forward.  Returns (batch, out_dim, in_dim)."""
    ext = _get_ext()
    bf16 = _use_bf16(c_dtype)
    return ext.pz_forward(
        x, theta, preacts_w, preacts_b, preacts_trainable, fast_measure, bf16
    )


def cute_pz_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    grad_output: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    ext = _get_ext()
    io_dtype = _resolve_io_dtype(c_dtype)
    results = ext.pz_backward(
        x.to(io_dtype).contiguous(),
        theta.to(io_dtype).contiguous(),
        preacts_w.to(io_dtype).contiguous() if preacts_trainable else preacts_w,
        preacts_b.to(io_dtype).contiguous() if preacts_trainable else preacts_b,
        grad_output.contiguous(),
        preacts_trainable,
        fast_measure,
        _state_bits(c_dtype),
    )
    return tuple(results)


def cute_rpz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    ext = _get_ext()
    bf16 = _use_bf16(c_dtype)
    return ext.rpz_forward(
        x.contiguous(),
        theta.contiguous(),
        preacts_w.contiguous(),
        preacts_b.contiguous(),
        fast_measure,
        bf16,
    )


def cute_rpz_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    grad_output: torch.Tensor,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _get_ext()
    io_dtype = _resolve_io_dtype(c_dtype)
    results = ext.rpz_backward(
        x.to(io_dtype).contiguous(),
        theta.to(io_dtype).contiguous(),
        preacts_w.to(io_dtype).contiguous(),
        preacts_b.to(io_dtype).contiguous(),
        grad_output.contiguous(),
        fast_measure,
        _state_bits(c_dtype),
    )
    return tuple(results)


def cute_real_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    ext = _get_ext()
    compute_bf16 = (
        c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        or _resolve_io_dtype(c_dtype) == torch.bfloat16
    )
    bf16 = _use_bf16(c_dtype)
    return ext.real_forward(
        x.contiguous(),
        theta.contiguous(),
        preacts_w.contiguous(),
        preacts_b.contiguous(),
        preacts_trainable,
        fast_measure,
        compute_bf16,
        bf16,
    )


def cute_real_backward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    grad_output: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    ext = _get_ext()
    compute_bf16 = (
        c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        or _resolve_io_dtype(c_dtype) == torch.bfloat16
    )
    io_dtype = _resolve_io_dtype(c_dtype)
    results = ext.real_backward(
        x.to(io_dtype).contiguous(),
        theta.to(io_dtype).contiguous(),
        preacts_w.to(io_dtype).contiguous(),
        preacts_b.to(io_dtype).contiguous(),
        grad_output.contiguous(),
        preacts_trainable,
        fast_measure,
        compute_bf16,
        _state_bits(c_dtype),
    )
    return tuple(results)


# ---------------------------------------------------------------------------
# Pointwise activation kernels (used by QKANLayer's base path)
# ---------------------------------------------------------------------------

# Must match the ActKind enum in csrc/cute_activations.cu.
_ACTIVATION_KIND_MAP: dict[str, int] = {
    "silu": 0,
    "gelu_exact": 1,
    "gelu_tanh": 2,
    "relu": 3,
    "tanh": 4,
    "sigmoid": 5,
}


def _activation_kind_int(kind: str) -> int:
    try:
        return _ACTIVATION_KIND_MAP[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CuTe activation kind '{kind}'. "
            f"Supported: {sorted(_ACTIVATION_KIND_MAP)}"
        ) from exc


def cute_activation_forward(
    x: torch.Tensor,
    kind: str,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """CuTe pointwise activation forward.

    Args:
        x: input tensor (CUDA, contiguous-ish; will be made contiguous if not).
            Computed in-kernel as f32; I/O in f32 or bf16 depending on ``x.dtype``.
        kind: one of ``"silu"``, ``"gelu_exact"``, ``"gelu_tanh"``, ``"relu"``,
            ``"tanh"``, ``"sigmoid"``.
        c_dtype: compute dtype hint (kept for parity with the rest of cute_ops;
            unused — the kernel always computes in f32 and matches I/O dtype to
            ``x.dtype``).

    Returns:
        Tensor with the same shape and dtype as ``x``.
    """
    del c_dtype  # accepted for API parity; kernel computes in f32 internally
    ext = _get_ext()
    return ext.activation_forward(x, _activation_kind_int(kind))


def cute_activation_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    kind: str,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """CuTe pointwise activation backward (returns grad_x).

    Args:
        grad_y: upstream gradient with the same shape/dtype as ``x``.
        x: forward input.
        kind: see :func:`cute_activation_forward`.
        c_dtype: compute dtype hint (unused, see :func:`cute_activation_forward`).
    """
    del c_dtype
    ext = _get_ext()
    return ext.activation_backward(grad_y, x, _activation_kind_int(kind))


# ---------------------------------------------------------------------------
# Fused HQKAN module: Linear → QKAN → Linear with minimal dispatch overhead
# ---------------------------------------------------------------------------


class FusedHQKAN(torch.nn.Module):
    """Fused Linear→QKAN→Linear as used in HQKANsformer MLP blocks.

    Convenience wrapper replacing ``nn.Sequential(Linear(d, els), QKAN(...), Linear(els, d))``
    with named sub-modules for easier parameter access and serialization.

    Usage::

        block = FusedHQKAN(768, 10, reps=1)
    """

    def __init__(
        self,
        in_features: int,
        qkan_width: int,
        reps: int = 1,
        solver: str = "cute",
        ansatz: str = "pz_encoding",
        c_dtype: torch.dtype = torch.bfloat16,
        p_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        from qkan import QKAN

        self.down = torch.nn.Linear(
            in_features, qkan_width, device=device, dtype=p_dtype
        )
        self.qkan = QKAN(
            width=[qkan_width, qkan_width],
            reps=reps,
            ba_trainable=True,
            device=device,
            solver=solver,  # type: ignore[arg-type]
            ansatz=ansatz,
            c_dtype=c_dtype,
            p_dtype=p_dtype,
        )
        self.up = torch.nn.Linear(qkan_width, in_features, device=device, dtype=p_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.qkan(self.down(x)))
