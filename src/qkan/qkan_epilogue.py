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

The backward is also fused (see ``qkan_epilogue_backward``). One Triton
kernel (``_epilogue_bwd_kernel``) covers the broadcast + reduction
gradients (``grad_postacts``, ``grad_eff_pw``, ``grad_pb``) by sharing
loads of ``grad_out`` / ``eff_pw`` / ``postacts`` across the three
outputs; the two matmul-shaped gradients (``grad_base_input`` and
``grad_base_w``) stay on cuBLAS — a Triton tile can't beat cuBLAS gemm
here. This replaces the ~7 eager dispatches with ~4 launches.
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

    # ------------------------------------------------------------------ #
    # Backward kernel                                                    #
    # ------------------------------------------------------------------ #
    #
    # Single fused kernel for the cheap (broadcast + reduction) gradients:
    #
    #   grad_postacts[b,o,i] = grad_out[b,o] * eff_pw[o,i]
    #   grad_eff_pw[o,i]     = sum_b grad_out[b,o]*postacts[b,o,i] + pb[o,i]*S[o]
    #   grad_pb[o,i]         = eff_pw[o,i] * S[o]
    #
    # where S[o] = sum_b grad_out[b,o] is precomputed eagerly (one cheap
    # dispatch). The two matmul-shaped gradients (grad_base_input,
    # grad_base_w) stay in eager — cuBLAS beats a hand-rolled Triton tile.
    #
    # Grid: (cdiv(O, BLOCK_N), cdiv(I, BLOCK_I)). Streams B in BLOCK_B chunks.
    # Per tile we load grad_out[B_chunk, O_tile] and eff_pw[O_tile, I_tile]
    # once; the eff_pw load is shared by the grad_postacts write and the
    # grad_pb write so we save a kernel launch vs. the original 3-kernel
    # design.

    @triton.jit
    def _epilogue_bwd_kernel(
        grad_out_ptr,  # [B, O]
        postacts_ptr,  # [B, O, I]
        eff_pw_ptr,  # [O, I]
        pb_ptr,  # [O, I]   (only read if HAS_PB)
        sum_go_ptr,  # [O]      (precomputed sum_b grad_out)
        grad_postacts_ptr,  # [B, O, I]
        grad_eff_pw_ptr,  # [O, I]
        grad_pb_ptr,  # [O, I]   (only written if HAS_PB and NEED_PB)
        B,
        O,
        I,
        # grad_out strides
        sgo_b,
        sgo_o,
        # postacts strides
        sp_b,
        sp_o,
        sp_i,
        # eff_pw / pb / grad_eff_pw / grad_pb strides (all OxI)
        spw_o,
        spw_i,
        spb_o,
        spb_i,
        sge_o,
        sge_i,
        sgpb_o,
        sgpb_i,
        # grad_postacts strides
        sgp_b,
        sgp_o,
        sgp_i,
        HAS_PB: tl.constexpr,
        NEED_POSTACTS: tl.constexpr,
        NEED_EFF_PW: tl.constexpr,
        NEED_PB: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_i = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        n_mask = offs_n < O
        i_mask = offs_i < I
        oi_mask = n_mask[:, None] & i_mask[None, :]

        # eff_pw[O, I] tile — used by grad_postacts (every B chunk) and grad_pb.
        pw_ptrs = eff_pw_ptr + offs_n[:, None] * spw_o + offs_i[None, :] * spw_i
        pw_tile = tl.load(pw_ptrs, mask=oi_mask, other=0.0).to(tl.float32)

        acc_eff_pw = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float32)

        for b_start in range(0, B, BLOCK_B):
            offs_b = b_start + tl.arange(0, BLOCK_B)
            b_mask = offs_b < B

            # grad_out[B, O] tile -> [BLOCK_B, BLOCK_N]
            go_ptrs = grad_out_ptr + offs_b[:, None] * sgo_b + offs_n[None, :] * sgo_o
            go_mask = b_mask[:, None] & n_mask[None, :]
            go_tile = tl.load(go_ptrs, mask=go_mask, other=0.0).to(tl.float32)

            if NEED_EFF_PW:
                # postacts[B, O, I] tile
                p_ptrs = (
                    postacts_ptr
                    + offs_b[:, None, None] * sp_b
                    + offs_n[None, :, None] * sp_o
                    + offs_i[None, None, :] * sp_i
                )
                p_mask = (
                    b_mask[:, None, None]
                    & n_mask[None, :, None]
                    & i_mask[None, None, :]
                )
                p_tile = tl.load(p_ptrs, mask=p_mask, other=0.0).to(tl.float32)
                # acc[o,i] += sum_b grad_out[b,o] * postacts[b,o,i]
                acc_eff_pw += tl.sum(go_tile[:, :, None] * p_tile, axis=0)

            if NEED_POSTACTS:
                # grad_postacts[b,o,i] = grad_out[b,o] * eff_pw[o,i]
                gp = go_tile[:, :, None] * pw_tile[None, :, :]
                gp_ptrs = (
                    grad_postacts_ptr
                    + offs_b[:, None, None] * sgp_b
                    + offs_n[None, :, None] * sgp_o
                    + offs_i[None, None, :] * sgp_i
                )
                gp_mask = (
                    b_mask[:, None, None]
                    & n_mask[None, :, None]
                    & i_mask[None, None, :]
                )
                tl.store(
                    gp_ptrs,
                    gp.to(grad_postacts_ptr.dtype.element_ty),
                    mask=gp_mask,
                )

        if NEED_EFF_PW or NEED_PB:
            s_o = tl.load(sum_go_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

        if NEED_EFF_PW:
            if HAS_PB:
                pb_ptrs = pb_ptr + offs_n[:, None] * spb_o + offs_i[None, :] * spb_i
                pb_tile = tl.load(pb_ptrs, mask=oi_mask, other=0.0).to(tl.float32)
                acc_eff_pw = acc_eff_pw + pb_tile * s_o[:, None]
            ge_ptrs = (
                grad_eff_pw_ptr + offs_n[:, None] * sge_o + offs_i[None, :] * sge_i
            )
            tl.store(
                ge_ptrs,
                acc_eff_pw.to(grad_eff_pw_ptr.dtype.element_ty),
                mask=oi_mask,
            )

        if NEED_PB:
            gpb = pw_tile * s_o[:, None]
            gpb_ptrs = grad_pb_ptr + offs_n[:, None] * sgpb_o + offs_i[None, :] * sgpb_i
            tl.store(gpb_ptrs, gpb.to(grad_pb_ptr.dtype.element_ty), mask=oi_mask)


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


def _qkan_epilogue_backward_eager(
    grad_out: torch.Tensor,
    postacts: torch.Tensor,
    eff_pw: torch.Tensor,
    pb: Optional[torch.Tensor],
    base_input: torch.Tensor,
    base_w: torch.Tensor,
    needs: tuple[bool, bool, bool, bool, bool],
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Reference eager backward (fallback for CPU / no-Triton)."""
    need_p, need_pw, need_pb, need_bi, need_bw = needs

    grad_postacts: Optional[torch.Tensor] = None
    grad_eff_pw: Optional[torch.Tensor] = None
    grad_pb: Optional[torch.Tensor] = None
    grad_base_input: Optional[torch.Tensor] = None
    grad_base_w: Optional[torch.Tensor] = None

    if need_p:
        grad_postacts = grad_out.unsqueeze(-1) * eff_pw.unsqueeze(0)
    if need_pw:
        x = postacts if pb is None else (postacts + pb)
        grad_eff_pw = (grad_out.unsqueeze(-1) * x).sum(dim=0)
    if need_pb:
        grad_pb = grad_out.sum(dim=0).unsqueeze(-1) * eff_pw
    if need_bi:
        grad_base_input = grad_out @ base_w
    if need_bw:
        grad_base_w = grad_out.transpose(0, 1) @ base_input

    return grad_postacts, grad_eff_pw, grad_pb, grad_base_input, grad_base_w


def _select_bwd_blocks(B: int, O: int, I: int) -> tuple[int, int, int]:
    """Heuristic tile sizes for the single fused backward kernel.

    Returns (BLOCK_N, BLOCK_I, BLOCK_B). Empirically a (16, 16, 16/32) tile
    is fastest on the (B=128, O=64, I=128) regime on H100; the kernel
    materializes a [BLOCK_B, BLOCK_N, BLOCK_I] postacts tile so we keep all
    three modest.
    """

    def _po2_clip(x: int, lo: int, hi: int) -> int:
        v = 1
        while v < x and v < hi:
            v *= 2
        return max(lo, min(v, hi))

    bn = _po2_clip(O, 8, 16)
    bi = _po2_clip(I, 8, 16)
    bb = _po2_clip(B, 8, 32)
    return bn, bi, bb


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
    """Hybrid Triton + cuBLAS backward for the epilogue.

    Replaces the ~7 eager dispatches with:
      * one fused Triton kernel for grad_postacts, grad_eff_pw, grad_pb
        (broadcast + reductions that share grad_out / eff_pw / postacts loads),
      * one ``grad_out.sum(0)`` (cheap), and
      * cuBLAS matmuls for grad_base_input (grad_out @ base_w) and grad_base_w
        (grad_out.T @ base_input) — Triton can't beat cuBLAS here.

    Falls back to the pure eager implementation on CPU or when Triton is
    missing.

    Returns ``(grad_postacts, grad_eff_pw, grad_pb, grad_base_input,
    grad_base_w)`` — each ``None`` when ``needs`` says it isn't required.
    """
    need_p, need_pw, need_pb, need_bi, need_bw = needs

    if not (_TRITON_AVAILABLE and grad_out.is_cuda):
        return _qkan_epilogue_backward_eager(
            grad_out, postacts, eff_pw, pb, base_input, base_w, needs
        )

    B, O, I = postacts.shape

    grad_postacts: Optional[torch.Tensor] = None
    grad_eff_pw: Optional[torch.Tensor] = None
    grad_pb: Optional[torch.Tensor] = None
    grad_base_input: Optional[torch.Tensor] = None
    grad_base_w: Optional[torch.Tensor] = None

    # Issue the cuBLAS matmuls FIRST so they can overlap with the Triton
    # kernel on cuBLAS's queue. Empirically reduces total backward time.
    if need_bi:
        grad_base_input = grad_out @ base_w
    if need_bw:
        grad_base_w = grad_out.transpose(0, 1) @ base_input

    # ----- Triton-fused: grad_postacts / grad_eff_pw / grad_pb ----- #
    if need_p or need_pw or need_pb:
        sum_go = (
            grad_out.sum(dim=0) if (need_pw and pb is not None) or need_pb else None
        )

        if need_p:
            grad_postacts = torch.empty_like(postacts)
        if need_pw:
            grad_eff_pw = torch.empty_like(eff_pw)
        if need_pb:
            grad_pb = torch.empty_like(eff_pw)

        # Stand-in pointers for unused outputs / inputs (Triton needs reals).
        gp_for_kernel = grad_postacts if grad_postacts is not None else postacts
        ge_for_kernel = grad_eff_pw if grad_eff_pw is not None else eff_pw
        gpb_for_kernel = grad_pb if grad_pb is not None else eff_pw
        pb_for_kernel = pb if pb is not None else eff_pw
        sum_go_for_kernel = sum_go if sum_go is not None else eff_pw

        bn, bi, bb = _select_bwd_blocks(B, O, I)
        grid = (triton.cdiv(O, bn), triton.cdiv(I, bi))
        _epilogue_bwd_kernel[grid](
            grad_out,
            postacts,
            eff_pw,
            pb_for_kernel,
            sum_go_for_kernel,
            gp_for_kernel,
            ge_for_kernel,
            gpb_for_kernel,
            B,
            O,
            I,
            grad_out.stride(0),
            grad_out.stride(1),
            postacts.stride(0),
            postacts.stride(1),
            postacts.stride(2),
            eff_pw.stride(0),
            eff_pw.stride(1),
            pb_for_kernel.stride(0),
            pb_for_kernel.stride(1),
            ge_for_kernel.stride(0),
            ge_for_kernel.stride(1),
            gpb_for_kernel.stride(0),
            gpb_for_kernel.stride(1),
            gp_for_kernel.stride(0),
            gp_for_kernel.stride(1),
            gp_for_kernel.stride(2),
            HAS_PB=pb is not None,
            NEED_POSTACTS=need_p,
            NEED_EFF_PW=need_pw,
            NEED_PB=need_pb,
            BLOCK_N=bn,
            BLOCK_I=bi,
            BLOCK_B=bb,
        )

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
