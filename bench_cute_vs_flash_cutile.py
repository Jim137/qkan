#!/usr/bin/env python3
"""Benchmark: CuTe DSL solver vs Flash (Triton) vs cuTile.

Measures forward + backward throughput, peak memory, and gradient accuracy
across solvers, ansatzes, dtypes, and problem sizes.

Usage:
    CUTLASS_PATH=/path/to/cutlass python bench_cute_vs_flash_cutile.py
"""

import os
import time

import torch

# Ensure CUTLASS path is set
if "CUTLASS_PATH" not in os.environ:
    os.environ["CUTLASS_PATH"] = os.path.expanduser("~/git/vibe/cutlass")

from qkan.cute_ops import (
    cute_pz_backward,
    cute_pz_forward,
    cute_rpz_backward,
    cute_rpz_forward,
    cute_real_backward,
    cute_real_forward,
)
from qkan.fused_ops import (
    triton_pz_backward,
    triton_pz_forward,
    triton_rpz_backward,
    triton_rpz_forward,
    triton_real_backward,
    triton_real_forward,
)

try:
    from qkan.cutile_ops import (
        cutile_pz_backward,
        cutile_pz_forward,
        cutile_rpz_backward,
        cutile_rpz_forward,
        cutile_real_backward,
        cutile_real_forward,
    )

    HAS_CUTILE = True
except ImportError:
    HAS_CUTILE = False


def bench_fn(fn, *args, warmup=5, iters=50):
    """Benchmark a function with CUDA events.  Returns median ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn(*args)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]  # median


def bench_fwd_bwd(fwd_fn, bwd_fn, x, theta, pw, pb, go, fwd_kw, bwd_kw):
    """Benchmark forward + backward combined."""

    def run():
        out = fwd_fn(x, theta, pw, pb, **fwd_kw)
        bwd_fn(x, theta, pw, pb, go, **bwd_kw)

    return bench_fn(run)


def peak_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


# ─── Solver dispatch tables ───────────────────────────────────────────

SOLVERS = {
    "cute": {
        "pz": (cute_pz_forward, cute_pz_backward),
        "rpz": (cute_rpz_forward, cute_rpz_backward),
        "real": (cute_real_forward, cute_real_backward),
    },
    "flash": {
        "pz": (triton_pz_forward, triton_pz_backward),
        "rpz": (triton_rpz_forward, triton_rpz_backward),
        "real": (triton_real_forward, triton_real_backward),
    },
}
if HAS_CUTILE:
    SOLVERS["cutile"] = {
        "pz": (cutile_pz_forward, cutile_pz_backward),
        "rpz": (cutile_rpz_forward, cutile_rpz_backward),
        "real": (cutile_real_forward, cutile_real_backward),
    }


def make_fwd_bwd_kwargs(ansatz, preacts_trainable, fast_measure, c_dtype):
    """Build solver-agnostic fwd/bwd kwargs for each ansatz."""
    if ansatz == "pz":
        fwd_kw = dict(
            preacts_trainable=preacts_trainable,
            fast_measure=fast_measure,
            c_dtype=c_dtype,
        )
        bwd_kw = dict(
            preacts_trainable=preacts_trainable,
            fast_measure=fast_measure,
            c_dtype=c_dtype,
        )
    elif ansatz == "rpz":
        fwd_kw = dict(fast_measure=fast_measure, c_dtype=c_dtype)
        bwd_kw = dict(fast_measure=fast_measure, c_dtype=c_dtype)
    elif ansatz == "real":
        compute_bf16 = c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        fwd_kw = dict(
            preacts_trainable=preacts_trainable,
            fast_measure=fast_measure,
            c_dtype=c_dtype,
        )
        bwd_kw = dict(
            preacts_trainable=preacts_trainable,
            fast_measure=fast_measure,
            c_dtype=c_dtype,
        )
    return fwd_kw, bwd_kw


def make_tensors(batch, in_dim, out_dim, reps, ansatz, device="cuda"):
    x = torch.randn(batch, in_dim, device=device)
    if ansatz in ("pz",):
        theta = torch.randn(out_dim, in_dim, reps + 1, 2, device=device)
    elif ansatz in ("rpz",):
        theta = torch.randn(out_dim, in_dim, reps + 1, 1, device=device)
    elif ansatz in ("real",):
        theta = torch.randn(out_dim, in_dim, reps, 1, device=device)
    pw = torch.randn(out_dim, in_dim, reps, device=device)
    pb = torch.randn(out_dim, in_dim, reps, device=device)
    go = torch.randn(batch, out_dim, in_dim, device=device)
    return x, theta, pw, pb, go


def run_benchmark():
    print("=" * 85)
    print(f"QKAN Solver Benchmark: CuTe DSL vs Flash (Triton) vs cuTile")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Solvers: {', '.join(SOLVERS.keys())}")
    print("=" * 85)

    configs = [
        # (batch, in_dim, out_dim, reps, ansatz, preacts, c_dtype, label)
        (64, 8, 16, 3, "pz", False, torch.float32, "pz f32 small"),
        (256, 16, 32, 3, "pz", False, torch.float32, "pz f32 medium"),
        (256, 16, 32, 3, "pz", False, torch.bfloat16, "pz bf16 medium"),
        (256, 16, 32, 3, "pz", False, torch.float8_e4m3fn, "pz fp8 medium"),
        (512, 32, 64, 3, "pz", False, torch.float32, "pz f32 large"),
        (512, 32, 64, 3, "pz", False, torch.float8_e4m3fn, "pz fp8 large"),
        (256, 16, 32, 3, "rpz", False, torch.float32, "rpz f32 medium"),
        (256, 16, 32, 3, "rpz", False, torch.float8_e4m3fn, "rpz fp8 medium"),
        (256, 16, 32, 3, "real", False, torch.bfloat16, "real bf16 medium"),
        (256, 16, 32, 3, "real", False, torch.float8_e4m3fn, "real fp8 medium"),
    ]

    header = f"{'Config':<22} | {'Solver':<8} | {'Fwd+Bwd ms':>10} | {'Speedup':>8} | {'Mem MB':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for batch, in_dim, out_dim, reps, ansatz, pt, c_dtype, label in configs:
        x, theta, pw, pb, go = make_tensors(batch, in_dim, out_dim, reps, ansatz)
        fwd_kw, bwd_kw = make_fwd_bwd_kwargs(ansatz, pt, True, c_dtype)

        results = {}
        for solver_name, solver_fns in SOLVERS.items():
            if ansatz not in solver_fns:
                continue
            fwd_fn, bwd_fn = solver_fns[ansatz]
            try:
                torch.cuda.reset_peak_memory_stats()
                ms = bench_fwd_bwd(fwd_fn, bwd_fn, x, theta, pw, pb, go, fwd_kw, bwd_kw)
                mem = peak_mem_mb()
                results[solver_name] = (ms, mem)
            except Exception as e:
                results[solver_name] = (None, None)

        # Find baseline for speedup (flash)
        base_ms = results.get("flash", (None,))[0]

        for solver_name, (ms, mem) in results.items():
            if ms is None:
                speedup_str = "ERR"
                ms_str = "ERR"
                mem_str = "ERR"
            else:
                ms_str = f"{ms:.3f}"
                mem_str = f"{mem:.1f}"
                if base_ms and ms > 0:
                    speedup_str = f"{base_ms / ms:.2f}x"
                else:
                    speedup_str = "1.00x"
            print(f"{label:<22} | {solver_name:<8} | {ms_str:>10} | {speedup_str:>8} | {mem_str:>8}")

        print()

    # ─── Accuracy comparison ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Gradient Accuracy (max |cute - flash| over all elements)")
    print("=" * 70)

    torch.manual_seed(42)
    for ansatz in ["pz", "rpz", "real"]:
        for c_dtype, dlabel in [
            (torch.float32, "f32"),
            (torch.float8_e4m3fn, "fp8"),
        ]:
            x, theta, pw, pb, go = make_tensors(128, 8, 16, 3, ansatz)
            fwd_kw, bwd_kw = make_fwd_bwd_kwargs(ansatz, False, True, c_dtype)

            try:
                cute_fwd, cute_bwd = SOLVERS["cute"][ansatz]
                flash_fwd, flash_bwd = SOLVERS["flash"][ansatz]
                cg = cute_bwd(x, theta, pw, pb, go, **bwd_kw)
                fg = flash_bwd(x, theta, pw, pb, go, **bwd_kw)
                dx = (cg[0] - fg[0]).abs().max().item()
                dt = (cg[1] - fg[1]).abs().max().item()
                print(f"  {ansatz:<5} {dlabel:<4}: grad_x={dx:.2e}  grad_theta={dt:.2e}")
            except Exception as e:
                print(f"  {ansatz:<5} {dlabel:<4}: ERROR ({e})")

    print()


if __name__ == "__main__":
    run_benchmark()
