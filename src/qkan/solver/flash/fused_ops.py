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
Triton-fused kernels for QKAN quantum circuit simulation.

Implements pz_encoding, rpz_encoding, and real ansatz forward passes as fused
Triton kernels, avoiding materialization of intermediate complex state vectors.
"""

import re
import warnings
from functools import lru_cache
from typing import NamedTuple

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from triton.compiler import make_backend  # type: ignore
from triton.runtime import driver  # type: ignore

from qkan._kernel_utils import _select_block_b
from qkan.solver.flash.launch_policy import (
    LaunchConfig,
    LaunchPolicyMismatchError,
    PrecisionKind,
    normalized_architecture,
    select_pz_backward_policy,
)

# Dtypes a launch policy can carry a tuned entry for.  Every other c_dtype
# (float64, float16) keeps the built-in launch.
_PRECISION_KINDS = {
    torch.float32: PrecisionKind.FP32,
    torch.bfloat16: PrecisionKind.BF16,
    torch.float8_e4m3fn: PrecisionKind.FP8,
}

# From Triton 3.6 the HIP backend emits waves_per_eu=N as
# amdgpu-waves-per-eu="N, N", pinning occupancy, with 0 meaning no limit.
# Before 3.6 it emitted "N", a minimum only, defaulting to 1.
_WAVES_PER_EU_PINS_OCCUPANCY = (3, 6)


class PzLaunch(NamedTuple):
    """A PZ backward launch resolved from a policy, as plain constants.

    ``options`` holds the Triton launch keywords as sorted (name, value)
    pairs, so the whole launch is hashable and can travel through
    ``torch.compile`` as a constant.
    """

    block_b: int
    options: tuple[tuple[str, int], ...]


@lru_cache(maxsize=None)
def _device_architecture(device_type: str, device_index: int) -> str:
    """The architecture name launch policies match: ``gfx942``, ``sm120``.

    CUDA builds of torch fill ``gcnArchName`` with the product name (for
    example "NVIDIA GeForce RTX 5090"), so it is read only on ROCm, and CUDA
    devices are named by compute capability instead.
    """

    if device_type != "cuda":
        return "unknown"
    properties = torch.cuda.get_device_properties(device_index)
    if torch.version.hip is not None:
        return normalized_architecture(getattr(properties, "gcnArchName", None))
    return f"sm{properties.major}{properties.minor}"


@lru_cache(maxsize=None)
def _triton_version() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", triton.__version__)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


@lru_cache(maxsize=None)
def _supports_waves_per_eu() -> bool:
    """Report whether the active Triton backend accepts ``waves_per_eu``.

    ``waves_per_eu`` is AMD/HIP-only and Triton's JIT raises ``KeyError`` for
    any launch keyword missing from its backend's options object, so ask that
    backend rather than inferring the vendor from ``torch.version``. Testing
    ``__dict__`` mirrors the JIT's own check and so also covers fields set
    outside the dataclass declaration. The accepted set belongs to the backend
    vendor, of which a process resolves exactly one, so caching it process-
    wide stays correct across devices.
    """

    try:
        target = driver.active.get_current_target()
        options = make_backend(target).parse_options({})
    except Exception:  # pragma: no cover - requires an unloadable driver
        return torch.version.hip is not None
    return "waves_per_eu" in options.__dict__


@lru_cache(maxsize=None)
def _warn_waves_per_eu_dropped(waves_per_eu: int) -> None:
    """Warn once per value that a policy's ``waves_per_eu`` was not applied."""

    warnings.warn(
        f"launch policy requested waves_per_eu={waves_per_eu}, which the "
        "active Triton backend does not accept; launching without it",
        RuntimeWarning,
        stacklevel=3,
    )


def _launch_options(launch: LaunchConfig) -> dict[str, int]:
    """Build the Triton launch keywords a policy asks for on this backend.

    ``num_warps`` and ``num_stages`` are portable.  ``waves_per_eu`` follows
    the Triton 3.6 contract, under which the packaged policies were measured:
    N pins occupancy at N waves per EU and 0 sets no limit.  Older Triton only
    takes N as a minimum, so a non-zero value cannot be reproduced there and
    the policy is refused; 0 is simply omitted.  Backends without the option
    drop it, and only a non-zero request diverges from the tuned policy.
    """

    options = {"num_warps": launch.num_warps, "num_stages": launch.num_stages}
    if not _supports_waves_per_eu():
        if launch.waves_per_eu:
            _warn_waves_per_eu_dropped(launch.waves_per_eu)
        return options
    if _triton_version() >= _WAVES_PER_EU_PINS_OCCUPANCY:
        options["waves_per_eu"] = launch.waves_per_eu
    elif launch.waves_per_eu:
        installed = ".".join(map(str, _triton_version()))
        raise LaunchPolicyMismatchError(
            f"launch policy sets waves_per_eu={launch.waves_per_eu}, which pins "
            f"occupancy only from Triton 3.6; Triton {installed} treats it as a "
            "minimum, so the tuned launch cannot be reproduced"
        )
    return options


@lru_cache(maxsize=None)
def _resolve_pz_launch(
    policy_name: str,
    device_type: str,
    device_index: int,
    n_oi: int,
    batch: int,
    reps: int,
    c_dtype: torch.dtype,
    preacts_trainable: bool,
    fast_measure: bool,
) -> PzLaunch | None:
    policy = select_pz_backward_policy(
        n_oi=n_oi,
        batch=batch,
        architecture=_device_architecture(device_type, device_index),
        precision=_PRECISION_KINDS.get(c_dtype),
        reps=reps,
        preacts_trainable=preacts_trainable,
        fast_measure=fast_measure,
        policy_name=policy_name,
    )
    if policy is None:
        return None
    options = tuple(sorted(_launch_options(policy.launch).items()))
    return PzLaunch(policy.launch.block_b, options)


@torch.compiler.disable
def resolve_pz_launch(
    policy_name: str,
    device: torch.device,
    *,
    n_oi: int,
    batch: int,
    reps: int,
    c_dtype: torch.dtype,
    preacts_trainable: bool,
    fast_measure: bool,
) -> PzLaunch | None:
    """Resolve a named policy for one PZ backward call, or ``None``.

    ``None`` keeps the built-in launch (see ``select_pz_backward_policy``).
    Results are cached per call shape, so steady-state training pays one
    dictionary lookup per forward.  Policy documents are file reads that
    ``torch.compile`` cannot trace, so this runs eagerly; only named policies
    reach it, and the default launch compiles as one graph.
    """

    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
    else:
        index = 0
    return _resolve_pz_launch(
        policy_name,
        device.type,
        index,
        n_oi,
        batch,
        reps,
        c_dtype,
        preacts_trainable,
        fast_measure,
    )


@triton.jit
def _pz_encoding_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 2]
    pw_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts weights)
    pb_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts bias)
    out_ptr,  # [Batch, Out_Dim, In_Dim]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for output
    stride_o_b,
    stride_o_o,
    stride_o_i,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
    BLOCK_B: tl.constexpr = 1,
):
    """
    Fused QKAN pz_encoding forward kernel with batch tiling.

    Grid: (out_dim * in_dim, cdiv(batch, BLOCK_B)).
    Circuit: H|0> -> [Rz(t0) Ry(t1) Rz(enc)]×reps -> Rz(t0_final) Ry(t1_final) -> measure Z
    """
    pid_oi = tl.program_id(0)
    pid_b = tl.program_id(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim
    if idx_o >= out_dim:
        return

    # Batch-scaled offsets pass 2^31 elements at realistic sizes, but
    # tl.program_id is int32, as is any int argument below 2^31 (#31).  b_offs
    # itself stays small and int32; only its products with a stride are
    # widened, so no 64-bit vector stays live across the reps loops.  Same
    # rule in every kernel here.
    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < batch_size

    x_vals = tl.load(
        x_ptr + b_offs.to(tl.int64) * stride_x_b + idx_i * stride_x_i,
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_B], dtype=tl.float32)
    r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i1 = tl.zeros([BLOCK_B], dtype=tl.float32)

    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i

    for layer in range(reps):
        # Rz(t0) — scalar theta trig, broadcast to batch tile
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p).to(tl.float32)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Ry(t1) — scalar theta trig
        t1 = tl.load(theta_base + layer * stride_t_r + 1 * stride_t_p).to(tl.float32)
        a = t1 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(enc) — vectorized data trig
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            ).to(tl.float32)
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            ).to(tl.float32)
            enc = w * x_vals + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Rz(t0), Ry(t1)
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p).to(tl.float32)
    t1 = tl.load(theta_base + reps * stride_t_r + 1 * stride_t_p).to(tl.float32)

    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    a = t1 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if FAST_MEASURE:
        result = tl.sqrt(r0 * r0 + i0 * i0) - tl.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    out_offs = (
        b_offs.to(tl.int64) * stride_o_b + idx_o * stride_o_o + idx_i * stride_o_i
    )
    tl.store(out_ptr + out_offs, result, mask=b_mask)


def triton_pz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Launch the Triton pz_encoding kernel.

    Args:
        x: (batch, in_dim) on CUDA
        theta: (out_dim, in_dim, reps+1, 2) on CUDA
        preacts_w, preacts_b: (out_dim, in_dim, reps)
        preacts_trainable: whether preacts are used
        fast_measure: True for |alpha|-|beta|, False for |alpha|^2-|beta|^2
        c_dtype: compute/storage dtype (torch.float32 or torch.bfloat16)

    Returns:
        (batch, out_dim, in_dim) in c_dtype
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=io_dtype)

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    grid = (n_oi, triton.cdiv(batch, BLOCK_B))

    if preacts_trainable:
        preacts_w = preacts_w.to(io_dtype).contiguous()
        preacts_b = preacts_b.to(io_dtype).contiguous()
        pw_strides = (preacts_w.stride(0), preacts_w.stride(1), preacts_w.stride(2))
        pb_strides = (preacts_b.stride(0), preacts_b.stride(1), preacts_b.stride(2))
    else:
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)

    _pz_encoding_kernel[grid](
        x,
        theta,
        preacts_w,
        preacts_b,
        output,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
        BLOCK_B=BLOCK_B,
    )

    return output


# ── rpz_encoding kernel ─────────────────────────────────────────────────────


@triton.jit
def _rpz_encoding_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 1]
    pw_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts weights, always loaded)
    pb_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts bias, always loaded)
    out_ptr,  # [Batch, Out_Dim, In_Dim]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for output
    stride_o_b,
    stride_o_o,
    stride_o_i,
    # Compile-time constants
    FAST_MEASURE: tl.constexpr,
    BLOCK_B: tl.constexpr = 1,
):
    """
    Fused rpz_encoding forward kernel with batch tiling.

    Grid: (out_dim * in_dim, cdiv(batch, BLOCK_B)).
    Circuit: H|0> -> [Ry(theta[l,0]) Rz(w*x+b)]×reps -> Ry(theta[L,0]) -> measure Z
    """
    pid_oi = tl.program_id(0)
    pid_b = tl.program_id(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim
    if idx_o >= out_dim:
        return

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < batch_size

    x_vals = tl.load(
        x_ptr + b_offs.to(tl.int64) * stride_x_b + idx_i * stride_x_i,
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_B], dtype=tl.float32)
    r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i1 = tl.zeros([BLOCK_B], dtype=tl.float32)

    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    pw_base = pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i
    pb_base = pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i

    for layer in range(reps):
        # Ry(theta) — scalar trig, broadcast
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p).to(tl.float32)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(w*x+b) — vectorized data trig
        w = tl.load(pw_base + layer * stride_pw_r).to(tl.float32)
        b = tl.load(pb_base + layer * stride_pb_r).to(tl.float32)
        enc = w * x_vals + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Ry(theta[reps, 0])
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p).to(tl.float32)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if FAST_MEASURE:
        result = tl.sqrt(r0 * r0 + i0 * i0) - tl.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    out_offs = (
        b_offs.to(tl.int64) * stride_o_b + idx_o * stride_o_o + idx_i * stride_o_i
    )
    tl.store(out_ptr + out_offs, result, mask=b_mask)


def triton_rpz_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Launch the Triton rpz_encoding kernel.

    Args:
        x, theta, preacts_w, preacts_b: tensors on CUDA
        fast_measure: measurement mode
        c_dtype: compute/storage dtype (torch.float32 or torch.bfloat16)

    Returns:
        (batch, out_dim, in_dim) in c_dtype
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    preacts_w = preacts_w.to(io_dtype).contiguous()
    preacts_b = preacts_b.to(io_dtype).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=io_dtype)
    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    grid = (n_oi, triton.cdiv(batch, BLOCK_B))

    _rpz_encoding_kernel[grid](
        x,
        theta,
        preacts_w,
        preacts_b,
        output,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        preacts_w.stride(0),
        preacts_w.stride(1),
        preacts_w.stride(2),
        preacts_b.stride(0),
        preacts_b.stride(1),
        preacts_b.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        FAST_MEASURE=fast_measure,
        BLOCK_B=BLOCK_B,
    )

    return output


@triton.jit
def _rpz_encoding_backward_kernel(
    # Pointers
    x_ptr,
    theta_ptr,
    pw_ptr,
    pb_ptr,
    grad_out_ptr,
    states_ptr,
    grad_theta_ptr,
    grad_x_ptr,
    grad_pw_ptr,
    grad_pb_ptr,
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for grad_output
    stride_go_b,
    stride_go_o,
    stride_go_i,
    # Strides for states buffer (n_programs, n_states, BLOCK_B, 4)
    stride_s_n,
    stride_s_s,
    stride_s_b,
    stride_s_c,
    # Strides for grad_theta
    stride_gt_o,
    stride_gt_i,
    stride_gt_r,
    stride_gt_p,
    # Strides for grad_x
    stride_gx_b,
    stride_gx_i,
    # Strides for grad_pw/pb
    stride_gpw_o,
    stride_gpw_i,
    stride_gpw_r,
    stride_gpb_o,
    stride_gpb_i,
    stride_gpb_r,
    # Compile-time constants
    FAST_MEASURE: tl.constexpr,
    FP8_PRESCALE: tl.constexpr = 1.0,
    BLOCK_B: tl.constexpr = 1,
):
    """Backward kernel for rpz_encoding ansatz with batch tiling."""
    pid_oi = tl.program_id(0)
    pid_b = tl.program_id(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim
    if idx_o >= out_dim:
        return

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < batch_size
    b_range = tl.arange(0, BLOCK_B)

    x_vals = tl.load(
        x_ptr + b_offs.to(tl.int64) * stride_x_b + idx_i * stride_x_i,
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)
    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    pw_base = pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i
    pb_base = pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i
    program_idx = pid_oi.to(tl.int64) * tl.cdiv(batch_size, BLOCK_B) + pid_b
    FP8_INV = 1.0 / FP8_PRESCALE
    states_base = states_ptr + program_idx * stride_s_n

    # ── Phase 1: Forward recompute, saving states ──
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_B], dtype=tl.float32)
    r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i1 = tl.zeros([BLOCK_B], dtype=tl.float32)

    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 0 * stride_s_c,
        r0 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 1 * stride_s_c,
        i0 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 2 * stride_s_c,
        r1 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 3 * stride_s_c,
        i1 * FP8_PRESCALE,
        mask=b_mask,
    )

    state_idx = 1
    for layer in range(reps):
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p).to(tl.float32)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            r0,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            i0,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            r1,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            i1,
            mask=b_mask,
        )
        state_idx += 1

        w = tl.load(pw_base + layer * stride_pw_r).to(tl.float32)
        b = tl.load(pb_base + layer * stride_pb_r).to(tl.float32)
        enc = w * x_vals + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            r0,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            i0,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            r1,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            i1,
            mask=b_mask,
        )
        state_idx += 1

    # Final Ry(theta[reps,0])
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p).to(tl.float32)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = tl.load(
        grad_out_ptr
        + b_offs.to(tl.int64) * stride_go_b
        + idx_o * stride_go_o
        + idx_i * stride_go_i,
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)

    if FAST_MEASURE:
        alpha_norm = tl.sqrt(r0 * r0 + i0 * i0)
        beta_norm = tl.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = tl.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = tl.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
        ar0 = go * r0 * inv_alpha
        ai0 = go * i0 * inv_alpha
        ar1 = -go * r1 * inv_beta
        ai1 = -go * i1 * inv_beta
    else:
        ar0 = 2.0 * go * r0
        ai0 = 2.0 * go * i0
        ar1 = -2.0 * go * r1
        ai1 = -2.0 * go * i1

    # ── Phase 3: Backward sweep ──
    grad_x_local = tl.zeros([BLOCK_B], dtype=tl.float32)
    gt_base = grad_theta_ptr + idx_o * stride_gt_o + idx_i * stride_gt_i

    # Backward through final Ry(theta[reps,0])
    state_idx -= 1
    sr0 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    si0 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    sr1 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    si1 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )

    t0 = (
        tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p).to(tl.float32)
        * FP8_INV
    )
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)

    grad_t0_vec = 0.5 * (
        ar0 * (-s * sr0 - c * sr1)
        + ai0 * (-s * si0 - c * si1)
        + ar1 * (c * sr0 - s * sr1)
        + ai1 * (c * si0 - s * si1)
    )
    tl.atomic_add(
        gt_base + reps * stride_gt_r + 0 * stride_gt_p,
        tl.sum(tl.where(b_mask, grad_t0_vec, 0.0)),
    )

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    for layer in range(reps - 1, -1, -1):
        # Backward through Rz(enc)
        state_idx -= 1
        sr0 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        si0 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        sr1 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        si1 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)

        w = tl.load(pw_base + layer * stride_pw_r).to(tl.float32)
        b = tl.load(pb_base + layer * stride_pb_r).to(tl.float32)
        enc = w * x_vals + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_enc = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        tl.atomic_add(
            grad_pw_ptr
            + idx_o * stride_gpw_o
            + idx_i * stride_gpw_i
            + layer * stride_gpw_r,
            tl.sum(tl.where(b_mask, grad_enc * x_vals, 0.0)),
        )
        tl.atomic_add(
            grad_pb_ptr
            + idx_o * stride_gpb_o
            + idx_i * stride_gpb_i
            + layer * stride_gpb_r,
            tl.sum(tl.where(b_mask, grad_enc, 0.0)),
        )
        grad_x_local += grad_enc * w

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(theta[l,0])
        state_idx -= 1
        sr0 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        si0 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        sr1 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        si1 = tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)

        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p).to(tl.float32)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_t0_vec = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        tl.atomic_add(
            gt_base + layer * stride_gt_r + 0 * stride_gt_p,
            tl.sum(tl.where(b_mask, grad_t0_vec, 0.0)),
        )

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    gx_offs = grad_x_ptr + b_offs.to(tl.int64) * stride_gx_b + idx_i * stride_gx_i
    tl.atomic_add(gx_offs, grad_x_local, mask=b_mask)


def triton_rpz_backward(
    x, theta, pw, pb, grad_output, fast_measure, c_dtype=torch.float32
):
    """Launch rpz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    x = x.to(c_dtype).contiguous()
    theta = theta.to(c_dtype).contiguous()
    pw = pw.to(c_dtype).contiguous()
    pb = pb.to(c_dtype).contiguous()
    grad_output = grad_output.contiguous()

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    n_states = 2 * reps + 2
    n_b_blocks = triton.cdiv(batch, BLOCK_B)
    n_programs = n_oi * n_b_blocks
    compute_fp8 = c_dtype == torch.float8_e4m3fn
    if compute_fp8:
        states_dtype = torch.float8_e4m3fn
    elif c_dtype == torch.bfloat16:
        states_dtype = torch.bfloat16
    else:
        states_dtype = torch.float32
    states = torch.empty(
        n_programs, n_states, 4, BLOCK_B, device=x.device, dtype=states_dtype
    )

    grad_theta = torch.zeros(theta.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)
    grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
    grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)

    grid = (out_dim * in_dim, n_b_blocks)
    _rpz_encoding_backward_kernel[grid](
        x,
        theta,
        pw,
        pb,
        grad_output,
        states,
        grad_theta,
        grad_x,
        grad_pw,
        grad_pb,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        pw.stride(0),
        pw.stride(1),
        pw.stride(2),
        pb.stride(0),
        pb.stride(1),
        pb.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(3),  # stride_s_b: BLOCK_B dim (contiguous)
        states.stride(2),  # stride_s_c: component dim
        grad_theta.stride(0),
        grad_theta.stride(1),
        grad_theta.stride(2),
        grad_theta.stride(3),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_pw.stride(0),
        grad_pw.stride(1),
        grad_pw.stride(2),
        grad_pb.stride(0),
        grad_pb.stride(1),
        grad_pb.stride(2),
        FAST_MEASURE=fast_measure,
        BLOCK_B=BLOCK_B,
    )

    return grad_x, grad_theta, grad_pw, grad_pb


# ── real ansatz kernel ───────────────────────────────────────────────────────


@triton.jit
def _round_to_int(x):
    """Round scalar to nearest integer, returned as same dtype. bf16-safe."""
    return tl.where(x >= 0, (x + 0.5).to(tl.int32), (x - 0.5).to(tl.int32)).to(
        tl.float32
    )


@triton.jit
def _approx_cos(x):
    """Polynomial cosine approximation with range reduction. Works with any dtype including bf16."""
    # Range reduce to [-π, π]: k = round(x / 2π)
    INV_TWO_PI: tl.constexpr = 0.15915494309189535
    TWO_PI: tl.constexpr = 6.283185307179586
    k = _round_to_int(x * INV_TWO_PI)
    x = x - k * TWO_PI
    # cos(x) via Horner form: 1 - x²/2 + x⁴/24 - x⁶/720 + x⁸/40320
    x2 = x * x
    c = x2 * 2.48015873e-05 - 1.38888889e-03
    c = c * x2 + 4.16666667e-02
    c = c * x2 - 5.0e-01
    c = c * x2 + 1.0
    return c


@triton.jit
def _approx_sin(x):
    """Polynomial sine approximation with range reduction. Works with any dtype including bf16."""
    # Range reduce to [-π, π]: k = round(x / 2π)
    INV_TWO_PI: tl.constexpr = 0.15915494309189535
    TWO_PI: tl.constexpr = 6.283185307179586
    k = _round_to_int(x * INV_TWO_PI)
    x = x - k * TWO_PI
    # sin(x) via Horner form: x*(1 - x²/6 + x⁴/120 - x⁶/5040 + x⁸/362880)
    x2 = x * x
    s = x2 * 2.75573192e-06 - 1.98412698e-04
    s = s * x2 + 8.33333333e-03
    s = s * x2 - 1.66666667e-01
    s = s * x2 + 1.0
    return s * x


@triton.jit
def _real_encoding_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps, 1]  (no final layer)
    pw_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts weights)
    pb_ptr,  # [Out_Dim, In_Dim, Reps]  (preacts bias)
    out_ptr,  # [Batch, Out_Dim, In_Dim]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for output
    stride_o_b,
    stride_o_o,
    stride_o_i,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
    COMPUTE_BF16: tl.constexpr = False,
    BLOCK_B: tl.constexpr = 1,
):
    """
    Fused real ansatz forward kernel.

    Circuit: H|0> -> [X, Ry(theta[l,0]), Z, Ry(enc_x)]×reps -> measure Z
    No final rotation. Data encoding uses Ry (not Rz).
    When COMPUTE_BF16=True, uses real-only fast path (no imaginary components).
    Tiles over the batch dimension for better throughput: theta loads/trig
    are amortized across the batch tile.
    Grid: (out_dim * in_dim, cdiv(batch, BLOCK_B)).
    """
    pid_oi = tl.program_id(0)
    pid_b = tl.program_id(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    if idx_o >= out_dim:
        return

    # Batch tile offsets
    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < batch_size

    # Load x values for the batch tile [BLOCK_B]
    x_vals = tl.load(
        x_ptr + b_offs.to(tl.int64) * stride_x_b + idx_i * stride_x_i,
        mask=b_mask,
        other=0.0,
    )

    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    INV_SQRT2: tl.constexpr = 0.7071067811865476

    if COMPUTE_BF16:
        # Real-only vectorized path: imaginary components always zero.
        # Theta trig is scalar (shared across batch tile), data trig is vectorized.
        r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
        r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)

        for layer in range(reps):
            r0, r1 = r1, r0  # X gate

            # Theta: scalar load + scalar trig, broadcast to batch tile
            t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
            a = t0.to(tl.float32) * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            nr1 = s * r0 + c * r1
            r0, r1 = nr0, nr1

            r1 = -r1  # Z gate

            # Data encoding: vectorized trig over batch tile
            enc = x_vals.to(tl.float32)
            if PREACTS_TRAINABLE:
                w = tl.load(
                    pw_ptr
                    + idx_o * stride_pw_o
                    + idx_i * stride_pw_i
                    + layer * stride_pw_r
                )
                b = tl.load(
                    pb_ptr
                    + idx_o * stride_pb_o
                    + idx_i * stride_pb_i
                    + layer * stride_pb_r
                )
                enc = w.to(tl.float32) * x_vals.to(tl.float32) + b.to(tl.float32)

            a = enc * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            nr1 = s * r0 + c * r1
            r0, r1 = nr0, nr1

        if FAST_MEASURE:
            result = tl.abs(r0) - tl.abs(r1)
        else:
            result = r0 * r0 - r1 * r1
    else:
        # Full complex vectorized path
        r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
        i0 = tl.zeros([BLOCK_B], dtype=tl.float32)
        r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
        i1 = tl.zeros([BLOCK_B], dtype=tl.float32)

        for layer in range(reps):
            r0, i0, r1, i1 = r1, i1, r0, i0  # X gate

            t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
            a = t0 * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            ni0 = c * i0 - s * i1
            nr1 = s * r0 + c * r1
            ni1 = s * i0 + c * i1
            r0, i0, r1, i1 = nr0, ni0, nr1, ni1

            r1 = -r1  # Z gate
            i1 = -i1

            enc = x_vals
            if PREACTS_TRAINABLE:
                w = tl.load(
                    pw_ptr
                    + idx_o * stride_pw_o
                    + idx_i * stride_pw_i
                    + layer * stride_pw_r
                )
                b = tl.load(
                    pb_ptr
                    + idx_o * stride_pb_o
                    + idx_i * stride_pb_i
                    + layer * stride_pb_r
                )
                enc = w * x_vals + b

            a = enc * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            ni0 = c * i0 - s * i1
            nr1 = s * r0 + c * r1
            ni1 = s * i0 + c * i1
            r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        if FAST_MEASURE:
            result = tl.sqrt(r0 * r0 + i0 * i0) - tl.sqrt(r1 * r1 + i1 * i1)
        else:
            result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    # Vectorized store for batch tile
    out_offs = (
        b_offs.to(tl.int64) * stride_o_b + idx_o * stride_o_o + idx_i * stride_o_i
    )
    tl.store(out_ptr + out_offs, result, mask=b_mask)


# ── Backward kernels ───────────────────────────────────────────────────────


@triton.jit
def _pz_encoding_backward_kernel(
    # Pointers
    x_ptr,  # [Batch, In_Dim]
    theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 2]
    pw_ptr,  # [Out_Dim, In_Dim, Reps]
    pb_ptr,  # [Out_Dim, In_Dim, Reps]
    grad_out_ptr,  # [Batch, Out_Dim, In_Dim]
    states_ptr,  # [batch*out*in, n_states, 4]
    grad_theta_ptr,  # [Out_Dim, In_Dim, Reps+1, 2]
    grad_x_ptr,  # [Batch, In_Dim]
    grad_pw_ptr,  # [Out_Dim, In_Dim, Reps]
    grad_pb_ptr,  # [Out_Dim, In_Dim, Reps]
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for grad_output
    stride_go_b,
    stride_go_o,
    stride_go_i,
    # Strides for states buffer (n_programs, n_states, BLOCK_B, 4)
    stride_s_n,
    stride_s_s,
    stride_s_b,
    stride_s_c,
    # Strides for grad_theta
    stride_gt_o,
    stride_gt_i,
    stride_gt_r,
    stride_gt_p,
    # Strides for grad_x
    stride_gx_b,
    stride_gx_i,
    # Strides for grad_pw/pb
    stride_gpw_o,
    stride_gpw_i,
    stride_gpw_r,
    stride_gpb_o,
    stride_gpb_i,
    stride_gpb_r,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
    FP8_PRESCALE: tl.constexpr = 1.0,
    BLOCK_B: tl.constexpr = 1,
):
    """Backward kernel for pz_encoding ansatz with batch tiling."""
    pid_oi = tl.program_id(0)
    pid_b = tl.program_id(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim
    if idx_o >= out_dim:
        return

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < batch_size
    b_range = tl.arange(0, BLOCK_B)

    x_vals = tl.load(
        x_ptr + b_offs.to(tl.int64) * stride_x_b + idx_i * stride_x_i,
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)
    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    program_idx = pid_oi.to(tl.int64) * tl.cdiv(batch_size, BLOCK_B) + pid_b
    # FP8 states are stored x FP8_PRESCALE so that small amplitudes stay in
    # e4m3's normal range.  Every state store scales and every state load
    # unscales; theta and the in-register state are never scaled.
    FP8_INV = 1.0 / FP8_PRESCALE
    states_base = states_ptr + program_idx * stride_s_n

    # ── Phase 1: Forward recompute, saving states ──
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i0 = tl.zeros([BLOCK_B], dtype=tl.float32)
    r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
    i1 = tl.zeros([BLOCK_B], dtype=tl.float32)

    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 0 * stride_s_c,
        r0 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 1 * stride_s_c,
        i0 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 2 * stride_s_c,
        r1 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + 0 * stride_s_s + b_range * stride_s_b + 3 * stride_s_c,
        i1 * FP8_PRESCALE,
        mask=b_mask,
    )

    state_idx = 1
    for layer in range(reps):
        # Rz(t0) — scalar trig
        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p).to(tl.float32)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            r0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            i0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            r1 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            i1 * FP8_PRESCALE,
            mask=b_mask,
        )
        state_idx += 1

        # Ry(t1) — scalar trig
        t1 = tl.load(theta_base + layer * stride_t_r + 1 * stride_t_p).to(tl.float32)
        a = t1 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            r0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            i0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            r1 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            i1 * FP8_PRESCALE,
            mask=b_mask,
        )
        state_idx += 1

        # Rz(enc) — vectorized data trig
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            ).to(tl.float32)
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            ).to(tl.float32)
            enc = w * x_vals + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            r0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            i0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            r1 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            i1 * FP8_PRESCALE,
            mask=b_mask,
        )
        state_idx += 1

    # Final Rz(t0)
    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p).to(tl.float32)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    tl.store(
        states_base + state_idx * stride_s_s + b_range * stride_s_b + 0 * stride_s_c,
        r0 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + state_idx * stride_s_s + b_range * stride_s_b + 1 * stride_s_c,
        i0 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + state_idx * stride_s_s + b_range * stride_s_b + 2 * stride_s_c,
        r1 * FP8_PRESCALE,
        mask=b_mask,
    )
    tl.store(
        states_base + state_idx * stride_s_s + b_range * stride_s_b + 3 * stride_s_c,
        i1 * FP8_PRESCALE,
        mask=b_mask,
    )
    state_idx += 1

    # Final Ry(t1)
    t1 = tl.load(theta_base + reps * stride_t_r + 1 * stride_t_p).to(tl.float32)
    a = t1 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = tl.load(
        grad_out_ptr
        + b_offs.to(tl.int64) * stride_go_b
        + idx_o * stride_go_o
        + idx_i * stride_go_i,
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)

    if FAST_MEASURE:
        alpha_norm = tl.sqrt(r0 * r0 + i0 * i0)
        beta_norm = tl.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = tl.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = tl.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
        ar0 = go * r0 * inv_alpha
        ai0 = go * i0 * inv_alpha
        ar1 = -go * r1 * inv_beta
        ai1 = -go * i1 * inv_beta
    else:
        ar0 = 2.0 * go * r0
        ai0 = 2.0 * go * i0
        ar1 = -2.0 * go * r1
        ai1 = -2.0 * go * i1

    # ── Phase 3: Backward sweep ──
    grad_x_local = tl.zeros([BLOCK_B], dtype=tl.float32)
    gt_base = grad_theta_ptr + idx_o * stride_gt_o + idx_i * stride_gt_i

    # Backward through final Ry(t1_final)
    state_idx -= 1
    sr0 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    si0 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    sr1 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    si1 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )

    t1 = tl.load(theta_base + reps * stride_t_r + 1 * stride_t_p).to(tl.float32)
    a = t1 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)

    grad_t1_vec = 0.5 * (
        ar0 * (-s * sr0 - c * sr1)
        + ai0 * (-s * si0 - c * si1)
        + ar1 * (c * sr0 - s * sr1)
        + ai1 * (c * si0 - s * si1)
    )
    tl.atomic_add(
        gt_base + reps * stride_gt_r + 1 * stride_gt_p,
        tl.sum(tl.where(b_mask, grad_t1_vec, 0.0)),
    )

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # Backward through final Rz(t0_final)
    state_idx -= 1
    sr0 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 0 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    si0 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 1 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    sr1 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 2 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )
    si1 = (
        tl.load(
            states_base
            + state_idx * stride_s_s
            + b_range * stride_s_b
            + 3 * stride_s_c,
            mask=b_mask,
            other=0.0,
        ).to(tl.float32)
        * FP8_INV
    )

    t0 = tl.load(theta_base + reps * stride_t_r + 0 * stride_t_p).to(tl.float32)
    a = t0 * 0.5
    c = tl.cos(a)
    s = tl.sin(a)

    grad_t0_vec = 0.5 * (
        -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
        + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
    )
    tl.atomic_add(
        gt_base + reps * stride_gt_r + 0 * stride_gt_p,
        tl.sum(tl.where(b_mask, grad_t0_vec, 0.0)),
    )

    nar0 = c * ar0 - s * ai0
    nai0 = s * ar0 + c * ai0
    nar1 = c * ar1 + s * ai1
    nai1 = -s * ar1 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    for layer in range(reps - 1, -1, -1):
        # Backward through Rz(enc)
        state_idx -= 1
        sr0 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        si0 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        sr1 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 2 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        si1 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 3 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )

        enc = x_vals
        if PREACTS_TRAINABLE:
            w = tl.load(
                pw_ptr + idx_o * stride_pw_o + idx_i * stride_pw_i + layer * stride_pw_r
            ).to(tl.float32)
            b = tl.load(
                pb_ptr + idx_o * stride_pb_o + idx_i * stride_pb_i + layer * stride_pb_r
            ).to(tl.float32)
            enc = w * x_vals + b

        a = enc * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_enc = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        if PREACTS_TRAINABLE:
            tl.atomic_add(
                grad_pw_ptr
                + idx_o * stride_gpw_o
                + idx_i * stride_gpw_i
                + layer * stride_gpw_r,
                tl.sum(tl.where(b_mask, grad_enc * x_vals, 0.0)),
            )
            tl.atomic_add(
                grad_pb_ptr
                + idx_o * stride_gpb_o
                + idx_i * stride_gpb_i
                + layer * stride_gpb_r,
                tl.sum(tl.where(b_mask, grad_enc, 0.0)),
            )
            grad_x_local += grad_enc * w
        else:
            grad_x_local += grad_enc

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(t1)
        state_idx -= 1
        sr0 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        si0 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        sr1 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 2 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        si1 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 3 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )

        t1 = tl.load(theta_base + layer * stride_t_r + 1 * stride_t_p).to(tl.float32)
        a = t1 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_t1_vec = 0.5 * (
            ar0 * (-s * sr0 - c * sr1)
            + ai0 * (-s * si0 - c * si1)
            + ar1 * (c * sr0 - s * sr1)
            + ai1 * (c * si0 - s * si1)
        )
        tl.atomic_add(
            gt_base + layer * stride_gt_r + 1 * stride_gt_p,
            tl.sum(tl.where(b_mask, grad_t1_vec, 0.0)),
        )

        nar0 = c * ar0 + s * ar1
        nai0 = c * ai0 + s * ai1
        nar1 = -s * ar0 + c * ar1
        nai1 = -s * ai0 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Rz(t0)
        state_idx -= 1
        sr0 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        si0 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        sr1 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 2 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )
        si1 = (
            tl.load(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 3 * stride_s_c,
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)
            * FP8_INV
        )

        t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p).to(tl.float32)
        a = t0 * 0.5
        c = tl.cos(a)
        s = tl.sin(a)

        grad_t0_vec = 0.5 * (
            -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )
        tl.atomic_add(
            gt_base + layer * stride_gt_r + 0 * stride_gt_p,
            tl.sum(tl.where(b_mask, grad_t0_vec, 0.0)),
        )

        nar0 = c * ar0 - s * ai0
        nai0 = s * ar0 + c * ai0
        nar1 = c * ar1 + s * ai1
        nai1 = -s * ar1 + c * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    gx_offs = grad_x_ptr + b_offs.to(tl.int64) * stride_gx_b + idx_i * stride_gx_i
    tl.atomic_add(gx_offs, grad_x_local, mask=b_mask)


def triton_pz_backward(
    x,
    theta,
    pw,
    pb,
    grad_output,
    preacts_trainable,
    fast_measure,
    c_dtype=torch.float32,
    launch: PzLaunch | None = None,
):
    """Launch pz_encoding backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb).

    ``launch=None`` is the built-in launch: ``_select_block_b`` picks BLOCK_B
    and the backend picks every launch option.  A ``PzLaunch`` from
    ``resolve_pz_launch`` overrides both.
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2] - 1

    compute_fp8 = c_dtype == torch.float8_e4m3fn
    io_dtype = torch.bfloat16 if compute_fp8 else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    grad_output = grad_output.contiguous()

    n_oi = out_dim * in_dim
    if launch is None:
        BLOCK_B = _select_block_b(n_oi, batch)
        launch_options: dict[str, int] = {}
    else:
        BLOCK_B = launch.block_b
        launch_options = dict(launch.options)
    n_states = 3 * reps + 3  # H state + 3 per layer + after final Rz
    n_b_blocks = triton.cdiv(batch, BLOCK_B)
    n_programs = n_oi * n_b_blocks
    if compute_fp8:
        states_dtype = torch.float8_e4m3fn
    elif c_dtype == torch.bfloat16:
        states_dtype = torch.bfloat16
    else:
        states_dtype = torch.float32
    states = torch.empty(
        n_programs, n_states, 4, BLOCK_B, device=x.device, dtype=states_dtype
    )

    grad_theta = torch.zeros(theta.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)

    if preacts_trainable:
        pw = pw.to(io_dtype).contiguous()
        pb = pb.to(io_dtype).contiguous()
        grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)
        pw_strides = (pw.stride(0), pw.stride(1), pw.stride(2))
        pb_strides = (pb.stride(0), pb.stride(1), pb.stride(2))
        gpw_strides = (grad_pw.stride(0), grad_pw.stride(1), grad_pw.stride(2))
        gpb_strides = (grad_pb.stride(0), grad_pb.stride(1), grad_pb.stride(2))
    else:
        grad_pw = torch.zeros(1, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(1, device=x.device, dtype=torch.float32)
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)
        gpw_strides = (0, 0, 0)
        gpb_strides = (0, 0, 0)

    grid = (out_dim * in_dim, n_b_blocks)
    _pz_encoding_backward_kernel[grid](
        x,
        theta,
        pw,
        pb,
        grad_output,
        states,
        grad_theta,
        grad_x,
        grad_pw,
        grad_pb,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(3),  # stride_s_b: BLOCK_B dim (contiguous)
        states.stride(2),  # stride_s_c: component dim
        grad_theta.stride(0),
        grad_theta.stride(1),
        grad_theta.stride(2),
        grad_theta.stride(3),
        grad_x.stride(0),
        grad_x.stride(1),
        *gpw_strides,
        *gpb_strides,
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
        FP8_PRESCALE=224.0 if compute_fp8 else 1.0,
        BLOCK_B=BLOCK_B,
        **launch_options,
    )

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )


def triton_real_forward(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
    c_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Launch the Triton real ansatz kernel.

    Args:
        x: (batch, in_dim)
        theta: (out_dim, in_dim, reps, 1), already expanded (no +1 layer)
        preacts_w: (out_dim, in_dim, reps), or any tensor if not trainable
        preacts_b: (out_dim, in_dim, reps), or any tensor if not trainable
        preacts_trainable: whether preacts are used
        fast_measure: measurement mode
        c_dtype: compute dtype (torch.bfloat16 or torch.float32)

    Returns:
        (batch, out_dim, in_dim) in c_dtype
    """
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2]  # No +1 for real ansatz

    io_dtype = torch.bfloat16 if c_dtype == torch.float8_e4m3fn else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    preacts_w = preacts_w.to(io_dtype).contiguous()
    preacts_b = preacts_b.to(io_dtype).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=io_dtype)

    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    grid = (n_oi, triton.cdiv(batch, BLOCK_B))

    if preacts_trainable:
        pw_strides = (preacts_w.stride(0), preacts_w.stride(1), preacts_w.stride(2))
        pb_strides = (preacts_b.stride(0), preacts_b.stride(1), preacts_b.stride(2))
    else:
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)

    _real_encoding_kernel[grid](
        x,
        theta,
        preacts_w,
        preacts_b,
        output,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
        COMPUTE_BF16=compute_bf16,
        BLOCK_B=BLOCK_B,
    )

    return output


@triton.jit
def _real_encoding_backward_kernel(
    # Pointers
    x_ptr,
    theta_ptr,
    pw_ptr,
    pb_ptr,
    grad_out_ptr,
    states_ptr,
    grad_theta_ptr,
    grad_x_ptr,
    grad_pw_ptr,
    grad_pb_ptr,
    # Shapes
    batch_size,
    in_dim,
    out_dim,
    reps,
    # Strides for x
    stride_x_b,
    stride_x_i,
    # Strides for theta
    stride_t_o,
    stride_t_i,
    stride_t_r,
    stride_t_p,
    # Strides for preacts weight
    stride_pw_o,
    stride_pw_i,
    stride_pw_r,
    # Strides for preacts bias
    stride_pb_o,
    stride_pb_i,
    stride_pb_r,
    # Strides for grad_output
    stride_go_b,
    stride_go_o,
    stride_go_i,
    # Strides for states buffer: (n_programs, n_states, BLOCK_B, n_components)
    stride_s_n,
    stride_s_s,
    stride_s_b,
    stride_s_c,
    # Strides for grad_theta
    stride_gt_o,
    stride_gt_i,
    stride_gt_r,
    stride_gt_p,
    # Strides for grad_x
    stride_gx_b,
    stride_gx_i,
    # Strides for grad_pw/pb
    stride_gpw_o,
    stride_gpw_i,
    stride_gpw_r,
    stride_gpb_o,
    stride_gpb_i,
    stride_gpb_r,
    # Compile-time constants
    PREACTS_TRAINABLE: tl.constexpr,
    FAST_MEASURE: tl.constexpr,
    COMPUTE_BF16: tl.constexpr = False,
    FP8_PRESCALE: tl.constexpr = 1.0,
    BLOCK_B: tl.constexpr = 1,
):
    """Backward kernel for real ansatz with batch tiling.

    Grid: (out_dim * in_dim, cdiv(batch, BLOCK_B)).
    States: (n_programs, n_states, BLOCK_B, n_components).
    Theta grads are accumulated locally over the batch tile before atomic_add,
    reducing contention by BLOCK_B×.
    """
    pid_oi = tl.program_id(0)
    pid_b = tl.program_id(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    if idx_o >= out_dim:
        return

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < batch_size

    x_vals = tl.load(
        x_ptr + b_offs.to(tl.int64) * stride_x_b + idx_i * stride_x_i,
        mask=b_mask,
        other=0.0,
    )
    theta_base = theta_ptr + idx_o * stride_t_o + idx_i * stride_t_i
    # states_base indexes into (n_programs, n_states, BLOCK_B, n_components)
    program_idx = pid_oi.to(tl.int64) * tl.cdiv(batch_size, BLOCK_B) + pid_b
    FP8_INV = 1.0 / FP8_PRESCALE
    states_base = states_ptr + program_idx * stride_s_n
    b_range = tl.arange(0, BLOCK_B)
    INV_SQRT2: tl.constexpr = 0.7071067811865476

    if COMPUTE_BF16:
        # ── Real-only batch-tiled path: states are (..., BLOCK_B, 2) ──

        # Phase 1: Forward recompute, saving [r0, r1] per batch element
        r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
        r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)

        tl.store(
            states_base + 0 * stride_s_s + b_range * stride_s_b + 0 * stride_s_c,
            r0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base + 0 * stride_s_s + b_range * stride_s_b + 1 * stride_s_c,
            r1 * FP8_PRESCALE,
            mask=b_mask,
        )

        state_idx = 1
        for layer in range(reps):
            r0, r1 = r1, r0  # X gate

            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                r0 * FP8_PRESCALE,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                r1 * FP8_PRESCALE,
                mask=b_mask,
            )
            state_idx += 1

            # Theta: scalar load + scalar trig, broadcast to batch tile
            t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
            a = t0.to(tl.float32) * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            nr1 = s * r0 + c * r1
            r0, r1 = nr0, nr1

            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                r0 * FP8_PRESCALE,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                r1 * FP8_PRESCALE,
                mask=b_mask,
            )
            state_idx += 1

            r1 = -r1  # Z gate

            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                r0 * FP8_PRESCALE,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                r1 * FP8_PRESCALE,
                mask=b_mask,
            )
            state_idx += 1

            # Data encoding: vectorized over batch tile
            enc = x_vals.to(tl.float32)
            if PREACTS_TRAINABLE:
                w = tl.load(
                    pw_ptr
                    + idx_o * stride_pw_o
                    + idx_i * stride_pw_i
                    + layer * stride_pw_r
                )
                b = tl.load(
                    pb_ptr
                    + idx_o * stride_pb_o
                    + idx_i * stride_pb_i
                    + layer * stride_pb_r
                )
                enc = w.to(tl.float32) * x_vals.to(tl.float32) + b.to(tl.float32)

            a = enc * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            nr1 = s * r0 + c * r1
            r0, r1 = nr0, nr1

        # Phase 2: Measurement gradient (vectorized over batch)
        go = tl.load(
            grad_out_ptr
            + b_offs.to(tl.int64) * stride_go_b
            + idx_o * stride_go_o
            + idx_i * stride_go_i,
            mask=b_mask,
            other=0.0,
        )

        if FAST_MEASURE:
            alpha_norm = tl.abs(r0)
            beta_norm = tl.abs(r1)
            inv_alpha = tl.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
            inv_beta = tl.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
            ar0 = go * r0 * inv_alpha
            ar1 = -go * r1 * inv_beta
        else:
            ar0 = 2.0 * go * r0
            ar1 = -2.0 * go * r1

        # Phase 3: Backward sweep (real-only, batch-tiled)
        grad_x_local = tl.zeros([BLOCK_B], dtype=tl.float32)
        gt_base = grad_theta_ptr + idx_o * stride_gt_o + idx_i * stride_gt_i

        for layer in range(reps - 1, -1, -1):
            # Backward through Ry(enc)
            state_idx -= 1
            sr0 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 0 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            sr1 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 1 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )

            enc = x_vals.to(tl.float32)
            if PREACTS_TRAINABLE:
                w = tl.load(
                    pw_ptr
                    + idx_o * stride_pw_o
                    + idx_i * stride_pw_i
                    + layer * stride_pw_r
                )
                b = tl.load(
                    pb_ptr
                    + idx_o * stride_pb_o
                    + idx_i * stride_pb_i
                    + layer * stride_pb_r
                )
                enc = w.to(tl.float32) * x_vals.to(tl.float32) + b.to(tl.float32)

            a = enc * 0.5
            c = tl.cos(a)
            s = tl.sin(a)

            # grad_enc is [BLOCK_B] vector
            grad_enc = 0.5 * (ar0 * (-s * sr0 - c * sr1) + ar1 * (c * sr0 - s * sr1))

            if PREACTS_TRAINABLE:
                # Accumulate locally over batch tile, single atomic_add
                tl.atomic_add(
                    grad_pw_ptr
                    + idx_o * stride_gpw_o
                    + idx_i * stride_gpw_i
                    + layer * stride_gpw_r,
                    tl.sum(tl.where(b_mask, grad_enc * x_vals, 0.0)),
                )
                tl.atomic_add(
                    grad_pb_ptr
                    + idx_o * stride_gpb_o
                    + idx_i * stride_gpb_i
                    + layer * stride_gpb_r,
                    tl.sum(tl.where(b_mask, grad_enc, 0.0)),
                )
                grad_x_local += grad_enc * w
            else:
                grad_x_local += grad_enc

            nar0 = c * ar0 + s * ar1
            nar1 = -s * ar0 + c * ar1
            ar0, ar1 = nar0, nar1

            # Backward through Z gate
            ar1 = -ar1

            # Backward through Ry(theta[l,0])
            state_idx -= 2
            sr0 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 0 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            sr1 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 1 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )

            # Theta: scalar trig, vectorized grad accumulation
            t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
            a = t0.to(tl.float32) * 0.5
            c = tl.cos(a)
            s = tl.sin(a)

            # grad_t0 per batch element, sum locally before atomic_add
            grad_t0_vec = 0.5 * (ar0 * (-s * sr0 - c * sr1) + ar1 * (c * sr0 - s * sr1))
            tl.atomic_add(
                gt_base + layer * stride_gt_r + 0 * stride_gt_p,
                tl.sum(tl.where(b_mask, grad_t0_vec, 0.0)),
            )

            nar0 = c * ar0 + s * ar1
            nar1 = -s * ar0 + c * ar1
            ar0, ar1 = nar0, nar1

            # Backward through X gate
            ar0, ar1 = ar1, ar0

        # grad_x: vectorized masked atomic_add
        gx_offs = grad_x_ptr + b_offs.to(tl.int64) * stride_gx_b + idx_i * stride_gx_i
        tl.atomic_add(gx_offs, grad_x_local, mask=b_mask)
    else:
        # ── Full complex state path: states are (..., BLOCK_B, 4) ──

        # Phase 1: Forward recompute, saving states
        r0 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
        i0 = tl.zeros([BLOCK_B], dtype=tl.float32)
        r1 = tl.full([BLOCK_B], INV_SQRT2, dtype=tl.float32)
        i1 = tl.zeros([BLOCK_B], dtype=tl.float32)

        tl.store(
            states_base + 0 * stride_s_s + b_range * stride_s_b + 0 * stride_s_c,
            r0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base + 0 * stride_s_s + b_range * stride_s_b + 1 * stride_s_c,
            i0 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base + 0 * stride_s_s + b_range * stride_s_b + 2 * stride_s_c,
            r1 * FP8_PRESCALE,
            mask=b_mask,
        )
        tl.store(
            states_base + 0 * stride_s_s + b_range * stride_s_b + 3 * stride_s_c,
            i1 * FP8_PRESCALE,
            mask=b_mask,
        )

        state_idx = 1
        for layer in range(reps):
            r0, i0, r1, i1 = r1, i1, r0, i0  # X gate

            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                r0,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                i0,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 2 * stride_s_c,
                r1,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 3 * stride_s_c,
                i1,
                mask=b_mask,
            )
            state_idx += 1

            t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
            a = t0 * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            ni0 = c * i0 - s * i1
            nr1 = s * r0 + c * r1
            ni1 = s * i0 + c * i1
            r0, i0, r1, i1 = nr0, ni0, nr1, ni1

            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                r0,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                i0,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 2 * stride_s_c,
                r1,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 3 * stride_s_c,
                i1,
                mask=b_mask,
            )
            state_idx += 1

            r1 = -r1  # Z gate
            i1 = -i1

            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 0 * stride_s_c,
                r0,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 1 * stride_s_c,
                i0,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 2 * stride_s_c,
                r1,
                mask=b_mask,
            )
            tl.store(
                states_base
                + state_idx * stride_s_s
                + b_range * stride_s_b
                + 3 * stride_s_c,
                i1,
                mask=b_mask,
            )
            state_idx += 1

            enc = x_vals
            if PREACTS_TRAINABLE:
                w = tl.load(
                    pw_ptr
                    + idx_o * stride_pw_o
                    + idx_i * stride_pw_i
                    + layer * stride_pw_r
                )
                b = tl.load(
                    pb_ptr
                    + idx_o * stride_pb_o
                    + idx_i * stride_pb_i
                    + layer * stride_pb_r
                )
                enc = w * x_vals + b

            a = enc * 0.5
            c = tl.cos(a)
            s = tl.sin(a)
            nr0 = c * r0 - s * r1
            ni0 = c * i0 - s * i1
            nr1 = s * r0 + c * r1
            ni1 = s * i0 + c * i1
            r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Phase 2: Measurement gradient
        go = tl.load(
            grad_out_ptr
            + b_offs.to(tl.int64) * stride_go_b
            + idx_o * stride_go_o
            + idx_i * stride_go_i,
            mask=b_mask,
            other=0.0,
        )

        if FAST_MEASURE:
            alpha_norm = tl.sqrt(r0 * r0 + i0 * i0)
            beta_norm = tl.sqrt(r1 * r1 + i1 * i1)
            inv_alpha = tl.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
            inv_beta = tl.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
            ar0 = go * r0 * inv_alpha
            ai0 = go * i0 * inv_alpha
            ar1 = -go * r1 * inv_beta
            ai1 = -go * i1 * inv_beta
        else:
            ar0 = 2.0 * go * r0
            ai0 = 2.0 * go * i0
            ar1 = -2.0 * go * r1
            ai1 = -2.0 * go * i1

        # Phase 3: Backward sweep
        grad_x_local = tl.zeros([BLOCK_B], dtype=tl.float32)
        gt_base = grad_theta_ptr + idx_o * stride_gt_o + idx_i * stride_gt_i

        for layer in range(reps - 1, -1, -1):
            # Backward through Ry(enc)
            state_idx -= 1
            sr0 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 0 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            si0 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 1 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            sr1 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 2 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            si1 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 3 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )

            enc = x_vals
            if PREACTS_TRAINABLE:
                w = tl.load(
                    pw_ptr
                    + idx_o * stride_pw_o
                    + idx_i * stride_pw_i
                    + layer * stride_pw_r
                )
                b = tl.load(
                    pb_ptr
                    + idx_o * stride_pb_o
                    + idx_i * stride_pb_i
                    + layer * stride_pb_r
                )
                enc = w * x_vals + b

            a = enc * 0.5
            c = tl.cos(a)
            s = tl.sin(a)

            grad_enc = 0.5 * (
                ar0 * (-s * sr0 - c * sr1)
                + ai0 * (-s * si0 - c * si1)
                + ar1 * (c * sr0 - s * sr1)
                + ai1 * (c * si0 - s * si1)
            )

            if PREACTS_TRAINABLE:
                tl.atomic_add(
                    grad_pw_ptr
                    + idx_o * stride_gpw_o
                    + idx_i * stride_gpw_i
                    + layer * stride_gpw_r,
                    tl.sum(tl.where(b_mask, grad_enc * x_vals, 0.0)),
                )
                tl.atomic_add(
                    grad_pb_ptr
                    + idx_o * stride_gpb_o
                    + idx_i * stride_gpb_i
                    + layer * stride_gpb_r,
                    tl.sum(tl.where(b_mask, grad_enc, 0.0)),
                )
                grad_x_local += grad_enc * w
            else:
                grad_x_local += grad_enc

            nar0 = c * ar0 + s * ar1
            nai0 = c * ai0 + s * ai1
            nar1 = -s * ar0 + c * ar1
            nai1 = -s * ai0 + c * ai1
            ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

            # Backward through Z gate
            ar1 = -ar1
            ai1 = -ai1

            # Backward through Ry(theta[l,0])
            state_idx -= 2
            sr0 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 0 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            si0 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 1 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            sr1 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 2 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )
            si1 = (
                tl.load(
                    states_base
                    + state_idx * stride_s_s
                    + b_range * stride_s_b
                    + 3 * stride_s_c,
                    mask=b_mask,
                    other=0.0,
                ).to(tl.float32)
                * FP8_INV
            )

            t0 = tl.load(theta_base + layer * stride_t_r + 0 * stride_t_p)
            a = t0 * 0.5
            c = tl.cos(a)
            s = tl.sin(a)

            grad_t0_vec = 0.5 * (
                ar0 * (-s * sr0 - c * sr1)
                + ai0 * (-s * si0 - c * si1)
                + ar1 * (c * sr0 - s * sr1)
                + ai1 * (c * si0 - s * si1)
            )
            tl.atomic_add(
                gt_base + layer * stride_gt_r + 0 * stride_gt_p,
                tl.sum(tl.where(b_mask, grad_t0_vec, 0.0)),
            )

            nar0 = c * ar0 + s * ar1
            nai0 = c * ai0 + s * ai1
            nar1 = -s * ar0 + c * ar1
            nai1 = -s * ai0 + c * ai1
            ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

            # Backward through X gate
            ar0, ai0, ar1, ai1 = ar1, ai1, ar0, ai0

        gx_offs = grad_x_ptr + b_offs.to(tl.int64) * stride_gx_b + idx_i * stride_gx_i
        tl.atomic_add(gx_offs, grad_x_local, mask=b_mask)


def triton_real_backward(
    x,
    theta,
    pw,
    pb,
    grad_output,
    preacts_trainable,
    fast_measure,
    c_dtype: torch.dtype = torch.bfloat16,
):
    """Launch real ansatz backward kernel. Returns (grad_x, grad_theta, grad_pw, grad_pb)."""
    batch, in_dim = x.shape
    out_dim = theta.shape[0]
    reps = theta.shape[2]  # No +1 for real ansatz

    compute_fp8 = c_dtype == torch.float8_e4m3fn
    compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
    io_dtype = torch.bfloat16 if compute_fp8 else c_dtype
    x = x.to(io_dtype).contiguous()
    theta = theta.to(io_dtype).contiguous()
    pw = pw.to(io_dtype).contiguous()
    pb = pb.to(io_dtype).contiguous()
    grad_output = grad_output.contiguous()
    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)

    n_states = 3 * reps + 1  # H state + 3 per layer (after X, Ry_theta, Z)
    n_b_blocks = triton.cdiv(batch, BLOCK_B)
    n_programs = n_oi * n_b_blocks
    # Real-only bf16 path stores 2 components (r0, r1); full path stores 4
    n_components = 2 if compute_bf16 else 4
    # States: (n_programs, n_states, n_components, BLOCK_B) — coalesced layout
    states = torch.empty(
        n_programs,
        n_states,
        n_components,
        BLOCK_B,
        device=x.device,
        dtype=torch.float32,
    )

    # Gradient accumulation tensors use float32 for efficient atomic operations
    grad_theta = torch.zeros(theta.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)

    if preacts_trainable:
        grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)
        pw_strides = (pw.stride(0), pw.stride(1), pw.stride(2))
        pb_strides = (pb.stride(0), pb.stride(1), pb.stride(2))
        gpw_strides = (grad_pw.stride(0), grad_pw.stride(1), grad_pw.stride(2))
        gpb_strides = (grad_pb.stride(0), grad_pb.stride(1), grad_pb.stride(2))
    else:
        grad_pw = torch.zeros(1, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(1, device=x.device, dtype=torch.float32)
        pw_strides = (0, 0, 0)
        pb_strides = (0, 0, 0)
        gpw_strides = (0, 0, 0)
        gpb_strides = (0, 0, 0)

    grid = (out_dim * in_dim, n_b_blocks)
    _real_encoding_backward_kernel[grid](
        x,
        theta,
        pw,
        pb,
        grad_output,
        states,
        grad_theta,
        grad_x,
        grad_pw,
        grad_pb,
        batch,
        in_dim,
        out_dim,
        reps,
        x.stride(0),
        x.stride(1),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        theta.stride(3),
        *pw_strides,
        *pb_strides,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(3),  # stride_s_b: BLOCK_B dim (contiguous)
        states.stride(2),  # stride_s_c: component dim
        grad_theta.stride(0),
        grad_theta.stride(1),
        grad_theta.stride(2),
        grad_theta.stride(3),
        grad_x.stride(0),
        grad_x.stride(1),
        *gpw_strides,
        *gpb_strides,
        PREACTS_TRAINABLE=preacts_trainable,
        FAST_MEASURE=fast_measure,
        COMPUTE_BF16=compute_bf16,
        FP8_PRESCALE=224.0 if compute_fp8 else 1.0,
        BLOCK_B=BLOCK_B,
    )

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )
