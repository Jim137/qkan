#!/usr/bin/env python
# Copyright (c) 2026, Jiun-Cheng Jiang and the QKAN contributors. All rights reserved.
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

"""Benchmark the MLX solver against the PyTorch exact solver on Apple silicon.

Times a full training step (forward, MSE loss, backward, Adam update) and an
inference forward of a QKAN for each ``solver@device`` pair, e.g.::

    python scripts/bench_mlx_solver.py
    python scripts/bench_mlx_solver.py --widths 784,64,10 --batch 256 --ansatz pz
    python scripts/bench_mlx_solver.py --runs mlx@mps mlx@cpu   # skip the baselines
    QKAN_FUSED_EPILOGUE=0 python scripts/bench_mlx_solver.py --runs mlx@mps mlx@cpu

The last form times the mlx solver without its fused sum over inputs, i.e. the
kernel returning ``(batch, out, in)`` postacts to a PyTorch epilogue.

The exact solver materialises complex ``(2, 2, batch, out, in)`` gate tensors
for every repetition, so it needs gigabytes for the larger shapes; a baseline
whose estimate exceeds ``--max-exact-gb`` is skipped rather than left to swap.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

from qkan import QKAN

DEFAULT_CASES = [
    # (widths, batch)
    ([4, 8, 2], 64),
    ([16, 32, 16], 256),
    ([64, 64, 64], 512),
    ([784, 64, 10], 64),
    ([784, 64, 10], 256),
    ([100, 100, 100], 1000),
]


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _release_memory() -> None:
    """Drop the buffer pools both frameworks keep, so one run's large buffers
    do not squeeze the next one on a machine short of memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _exact_gb(widths: list[int], batch: int, reps: int) -> float:
    """Rough peak of the exact solver's training step, in GB.

    Autograd keeps about three complex64 ``(batch, out, in, 2)`` states per
    repetition of every layer, and the backward pass roughly doubles that.
    """
    edges = sum(a * b for a, b in zip(widths[:-1], widths[1:]))
    return batch * edges * 16 * 3 * (reps + 1) * 2 / 1e9


def _median_ms(fn, device: str, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def bench(run: str, widths: list[int], batch: int, args) -> tuple[float, float]:
    """(train step ms, inference forward ms) for one ``solver@device`` run."""
    solver, device = run.split("@")
    model = QKAN(
        widths,
        reps=args.reps,
        solver=solver,
        device=device,
        ansatz=args.ansatz,
        preact_trainable=args.preacts,
        seed=0,
    )
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(batch, widths[0], generator=gen).to(device)
    y = torch.randn(batch, widths[-1], generator=gen).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    train_ms = _median_ms(train_step, device, args.iters, args.warmup)
    model.eval()

    def infer():
        with torch.no_grad():
            model(x)

    infer_ms = _median_ms(infer, device, args.iters, args.warmup)
    return train_ms, infer_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--widths", help="comma-separated layer widths, e.g. 784,64,10")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--ansatz", default="pz")
    parser.add_argument("--preacts", action="store_true", help="preact_trainable=True")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["exact@mps", "exact@cpu", "mlx@mps", "mlx@cpu"],
        help="solver@device pairs; the first one is the speed-up baseline",
    )
    parser.add_argument(
        "--max-exact-gb",
        type=float,
        default=4.0,
        help="skip exact-solver runs estimated to need more memory than this",
    )
    args = parser.parse_args()

    if args.widths:
        cases = [([int(w) for w in args.widths.split(",")], args.batch)]
    else:
        cases = DEFAULT_CASES

    print(
        f"torch {torch.__version__}, ansatz={args.ansatz}, reps={args.reps}, "
        f"preacts={args.preacts}; median of {args.iters} (ms)"
    )
    header = f"{'widths':>16} {'batch':>6} | " + " | ".join(
        f"{run + ' train/infer':>24}" for run in args.runs
    )
    print(header + f" | speed-up vs {args.runs[0]} (train, infer)")
    for widths, batch in cases:
        results: dict[str, tuple[float, float] | None] = {}
        for run in args.runs:
            if run.startswith("exact") and (
                _exact_gb(widths, batch, args.reps) > args.max_exact_gb
            ):
                results[run] = None
                continue
            results[run] = bench(run, widths, batch, args)
            _release_memory()
        cells = [
            f"{'skipped':>24}" if r is None else f"{r[0]:>11.2f} /{r[1]:>10.2f} "
            for r in results.values()
        ]
        base = results[args.runs[0]]
        gains = []
        for run in args.runs[1:]:
            r = results[run]
            if base is None or r is None or not run.startswith("mlx"):
                continue
            gains.append(f"{run}: {base[0] / r[0]:.1f}x, {base[1] / r[1]:.1f}x")
        name = "-".join(str(w) for w in widths)
        print(
            f"{name:>16} {batch:>6} | " + " | ".join(cells) + " | " + "; ".join(gains)
        )


if __name__ == "__main__":
    main()
