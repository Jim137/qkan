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
Fused Triton epilogue for QKANLayer.

Computes, in a single kernel launch:

    out[b, o] = sum_i (postacts[b, o, i] + pb[o, i]) * eff_pw[o, i]
              + sum_i base_input[b, i] * base_w[o, i]
              + (bias_sum[o] if provided)

Replaces the 4-5 dispatches done by the eager epilogue:
    (postacts + pb) * eff_pw            -> add + mul
    .sum(dim=2)                          -> reduction
    F.linear(base_input, base_w[, b])    -> matmul (+ bias add)
    main + base                          -> add

Backward stays in eager PyTorch — the forward win covers both eval and train
hot paths, which is what the latency profile flagged. ``QKANEpilogue.apply``
saves the inputs and recomputes the (cheap) gradients with torch ops.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #


if _TRITON_AVAILABLE:

    @triton.jit
    def _epilogue_fwd_kernel(
        postacts_ptr,  # [B, O, I]
        eff_pw_ptr,  # [O, I]
        pb_ptr,  # [O, I]   (may be unused; only read when HAS_PB)
        base_input_ptr,  # [B, I]
        base_w_ptr,  # [O, I]
        bias_sum_ptr,  # [O]      (may be unused; only read when HAS_BIAS_SUM)
        out_ptr,  # [B, O]
        # Shapes
        B,
        O,
        I,
        # Strides — postacts
        sp_b,
        sp_o,
        sp_i,
        # Strides — eff_pw / pb / base_w (all (O, I))
        spw_o,
        spw_i,
        spb_o,
        spb_i,
        sbw_o,
        sbw_i,
        # Strides — base_input
        sbi_b,
        sbi_i,
        # Strides — out
        so_b,
        so_o,
        # Flags
        HAS_PB: tl.constexpr,
        HAS_BIAS_SUM: tl.constexpr,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """One program -> [BLOCK_M, BLOCK_N] tile of out[b, o].

        Grid: (cdiv(B, BLOCK_M), cdiv(O, BLOCK_N))

        Inside: stream over I in BLOCK_I chunks accumulating in fp32.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # batch tile
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # out tile

        m_mask = offs_m < B
        n_mask = offs_n < O

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for i_start in range(0, I, BLOCK_I):
            offs_i = i_start + tl.arange(0, BLOCK_I)
            i_mask = offs_i < I

            # postacts[b, o, i]: [BLOCK_M, BLOCK_N, BLOCK_I]
            p_ptrs = (
                postacts_ptr
                + offs_m[:, None, None] * sp_b
                + offs_n[None, :, None] * sp_o
                + offs_i[None, None, :] * sp_i
            )
            p_mask = (
                m_mask[:, None, None] & n_mask[None, :, None] & i_mask[None, None, :]
            )
            postacts_tile = tl.load(p_ptrs, mask=p_mask, other=0.0).to(tl.float32)

            # eff_pw[o, i]: [BLOCK_N, BLOCK_I]
            pw_ptrs = eff_pw_ptr + offs_n[:, None] * spw_o + offs_i[None, :] * spw_i
            pw_mask = n_mask[:, None] & i_mask[None, :]
            eff_pw_tile = tl.load(pw_ptrs, mask=pw_mask, other=0.0).to(tl.float32)

            if HAS_PB:
                pb_ptrs = pb_ptr + offs_n[:, None] * spb_o + offs_i[None, :] * spb_i
                pb_tile = tl.load(pb_ptrs, mask=pw_mask, other=0.0).to(tl.float32)
                # (postacts + pb) * eff_pw — broadcast over batch dim
                term = (postacts_tile + pb_tile[None, :, :]) * eff_pw_tile[None, :, :]
            else:
                term = postacts_tile * eff_pw_tile[None, :, :]

            # base_input[b, i]: [BLOCK_M, BLOCK_I]
            bi_ptrs = base_input_ptr + offs_m[:, None] * sbi_b + offs_i[None, :] * sbi_i
            bi_mask = m_mask[:, None] & i_mask[None, :]
            base_input_tile = tl.load(bi_ptrs, mask=bi_mask, other=0.0).to(tl.float32)

            # base_w[o, i]: [BLOCK_N, BLOCK_I]
            bw_ptrs = base_w_ptr + offs_n[:, None] * sbw_o + offs_i[None, :] * sbw_i
            bw_tile = tl.load(bw_ptrs, mask=pw_mask, other=0.0).to(tl.float32)

            # base_input * base_w broadcast: [BLOCK_M, BLOCK_N, BLOCK_I]
            base_term = base_input_tile[:, None, :] * bw_tile[None, :, :]

            acc += tl.sum(term + base_term, axis=2)

        if HAS_BIAS_SUM:
            bs = tl.load(bias_sum_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bs[None, :]

        out_ptrs = out_ptr + offs_m[:, None] * so_b + offs_n[None, :] * so_o
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# --------------------------------------------------------------------------- #
# Python launcher
# --------------------------------------------------------------------------- #


def _select_blocks(B: int, O: int, I: int) -> tuple[int, int, int]:
    """Heuristic tile sizes for the fused epilogue.

    The kernel materializes a [BLOCK_M, BLOCK_N, BLOCK_I] intermediate so the
    tile product must fit comfortably in registers (target <= 4096 fp32).
    Empirically a thin-N tile (BLOCK_N=8) with moderate BLOCK_M and small
    BLOCK_I beats fatter shapes on the (B=128, O=64, I=128) regime.
    """

    def _po2_clip(x: int, lo: int, hi: int) -> int:
        v = 1
        while v < x and v < hi:
            v *= 2
        return max(lo, min(v, hi))

    block_m = _po2_clip(B, 8, 32)
    block_n = _po2_clip(O, 8, 8)  # thin-N tile is fastest empirically
    # Small BLOCK_I keeps the 3D intermediate in registers; the I loop unrolls
    # cheaply.
    if I <= 16:
        block_i = _po2_clip(I, 8, 16)
    else:
        block_i = 16
    return block_m, block_n, block_i


def qkan_epilogue_forward(
    postacts: torch.Tensor,  # (B, O, I)
    eff_pw: torch.Tensor,  # (O, I)
    pb: Optional[torch.Tensor],  # (O, I) or None
    base_input: torch.Tensor,  # (B, I)
    base_w: torch.Tensor,  # (O, I)
    bias_sum: Optional[torch.Tensor] = None,  # (O,) or None — eval-path fold
) -> torch.Tensor:
    """Fused (B, O, I) epilogue forward returning (B, O).

    All tensors must be CUDA + contiguous in their natural layout (the kernel
    reads via strides, but contiguity gives the best throughput).
    """
    if not _TRITON_AVAILABLE:
        raise ImportError("Triton is required for qkan_epilogue_forward.")
    assert postacts.is_cuda, "qkan_epilogue_forward requires CUDA tensors"
    assert postacts.dim() == 3
    B, O, I = postacts.shape
    assert eff_pw.shape == (O, I), f"eff_pw shape {tuple(eff_pw.shape)} != ({O},{I})"
    assert base_input.shape == (B, I)
    assert base_w.shape == (O, I)
    if pb is not None:
        assert pb.shape == (O, I)
    if bias_sum is not None:
        assert bias_sum.shape == (O,)

    out_dtype = postacts.dtype
    out = torch.empty((B, O), device=postacts.device, dtype=out_dtype)

    block_m, block_n, block_i = _select_blocks(B, O, I)
    grid = (triton.cdiv(B, block_m), triton.cdiv(O, block_n))

    # Triton needs a real pointer even for unused arms.
    pb_for_kernel = pb if pb is not None else postacts
    bs_for_kernel = bias_sum if bias_sum is not None else out

    _epilogue_fwd_kernel[grid](
        postacts,
        eff_pw,
        pb_for_kernel,
        base_input,
        base_w,
        bs_for_kernel,
        out,
        B,
        O,
        I,
        postacts.stride(0),
        postacts.stride(1),
        postacts.stride(2),
        eff_pw.stride(0),
        eff_pw.stride(1),
        pb_for_kernel.stride(0),
        pb_for_kernel.stride(1) if pb_for_kernel.dim() >= 2 else 0,
        base_w.stride(0),
        base_w.stride(1),
        base_input.stride(0),
        base_input.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_PB=pb is not None,
        HAS_BIAS_SUM=bias_sum is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_I=block_i,
    )
    return out


def qkan_epilogue_backward(
    grad_out: torch.Tensor,  # (B, O)
    postacts: torch.Tensor,  # (B, O, I)
    eff_pw: torch.Tensor,  # (O, I)
    pb: Optional[torch.Tensor],  # (O, I) or None
    base_input: torch.Tensor,  # (B, I)
    base_w: torch.Tensor,  # (O, I)
    needs: tuple[bool, bool, bool, bool, bool],
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Eager backward for the fused epilogue.

    Computes gradients with torch ops. Each of these calls is a single CUDA
    dispatch, so total backward cost is dominated by ~5 kernels — comparable
    to the eager forward we replaced. Acceptable given the priority on the
    fwd path (eval has no backward at all).

    Returns (grad_postacts, grad_eff_pw, grad_pb, grad_base_input, grad_base_w),
    each ``None`` when the corresponding flag in ``needs`` is False.
    """
    need_p, need_pw, need_pb, need_bi, need_bw = needs
    # main term: out_main = sum_i (postacts + pb) * eff_pw
    #   d/d postacts[b,o,i] = grad_out[b,o] * eff_pw[o,i]
    #   d/d eff_pw[o,i]     = sum_b grad_out[b,o] * (postacts[b,o,i] + pb[o,i])
    #   d/d pb[o,i]         = sum_b grad_out[b,o] * eff_pw[o,i]
    # base term: out_base = base_input @ base_w^T
    #   d/d base_input[b,i] = grad_out @ base_w
    #   d/d base_w[o,i]     = grad_out^T @ base_input

    grad_postacts: Optional[torch.Tensor] = None
    grad_eff_pw: Optional[torch.Tensor] = None
    grad_pb: Optional[torch.Tensor] = None
    grad_base_input: Optional[torch.Tensor] = None
    grad_base_w: Optional[torch.Tensor] = None

    if need_p:
        # (B, O, 1) * (1, O, I) -> (B, O, I)
        grad_postacts = grad_out.unsqueeze(-1) * eff_pw.unsqueeze(0)
    if need_pw:
        # sum_b grad_out[b,o] * (postacts[b,o,i] + pb[o,i])
        x = postacts if pb is None else (postacts + pb)
        grad_eff_pw = (grad_out.unsqueeze(-1) * x).sum(dim=0)
    if need_pb:
        # sum_b grad_out[b,o] * eff_pw[o,i]
        grad_pb = grad_out.sum(dim=0).unsqueeze(-1) * eff_pw
    if need_bi:
        grad_base_input = grad_out @ base_w  # (B,O)@(O,I) -> (B,I)
    if need_bw:
        grad_base_w = grad_out.transpose(0, 1) @ base_input  # (O,B)@(B,I) -> (O,I)

    return grad_postacts, grad_eff_pw, grad_pb, grad_base_input, grad_base_w


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #


class QKANEpilogue(torch.autograd.Function):
    """Autograd wrapper for the fused epilogue.

    Saves all inputs (cheap — they're already live in the train graph) and
    returns gradients via :func:`qkan_epilogue_backward`. The eval path
    bypasses this Function entirely (it calls ``qkan_epilogue_forward``
    directly under ``torch.no_grad``).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        postacts: torch.Tensor,
        eff_pw: torch.Tensor,
        pb: Optional[torch.Tensor],
        base_input: torch.Tensor,
        base_w: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(postacts, eff_pw, pb, base_input, base_w)
        ctx.has_pb = pb is not None
        return qkan_epilogue_forward(postacts, eff_pw, pb, base_input, base_w, None)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        postacts, eff_pw, pb, base_input, base_w = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        needs = (
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.has_pb and ctx.needs_input_grad[2],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[4],
        )
        gp, gpw, gpb, gbi, gbw = qkan_epilogue_backward(
            grad_out, postacts, eff_pw, pb, base_input, base_w, needs
        )
        # When pb is None, the slot is also None.
        return gp, gpw, gpb, gbi, gbw


__all__ = [
    "QKANEpilogue",
    "qkan_epilogue_forward",
    "qkan_epilogue_backward",
]
