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

"""Tune a flash PZ backward launch policy on the local GPU.

For one PZ shape (``n_oi = out_dim * in_dim`` and a batch) this sweeps
BLOCK_B, num_warps and num_stages in fp32, bf16 and fp8, rejects any launch
whose gradients differ from the built-in launch beyond summation-order noise,
and times the whole backward with CUDA graphs, so the numbers are GPU time
without host overhead.  BLOCK_B is kept within the padded-row budget of the
built-in launch, so a policy never needs more ``states`` memory.

A precision keeps the built-in launch unless a candidate beats it by
``--min-gain`` in a confirmation run that alternates the two.  If one
precision gains, the policy document is written to stdout, ready for
``src/qkan/solver/flash/policies/<name>.json``; otherwise nothing is.

    python scripts/tune_pz_launch_policy.py --out-dim 16 --in-dim 16 --batch 512

The GPU must be otherwise idle: other processes skew the timings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys

import torch
from triton.testing import do_bench_cudagraph  # type: ignore

from qkan._kernel_utils import _select_block_b
from qkan.solver.flash import fused_ops
from qkan.solver.flash.launch_policy import (
    padded_rows,
    validate_launch_policy_document,
)

PRECISIONS = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}
BLOCKS = (16, 32, 64, 128, 256, 512, 1024)
WARPS = (1, 2, 4, 8)
STAGES = (1, 2, 3, 4)
# Triton's CUDA defaults, which the built-in launch uses.
DEFAULT_WARPS, DEFAULT_STAGES = 4, 3


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def make_inputs(out_dim, in_dim, batch, reps, c_dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    io_dtype = torch.float32 if c_dtype == torch.float32 else torch.bfloat16
    x = (torch.rand(batch, in_dim, generator=g, device="cuda") * 2 - 1).to(io_dtype)
    theta = torch.randn(out_dim, in_dim, reps + 1, 2, generator=g, device="cuda")
    pw = torch.ones(out_dim, in_dim, reps, device="cuda")
    pb = torch.zeros(out_dim, in_dim, reps, device="cuda")
    grad_out = torch.randn(batch, out_dim, in_dim, generator=g, device="cuda")
    return x, theta.to(io_dtype), pw, pb, grad_out.to(io_dtype)


def backward(inputs, c_dtype, launch):
    x, theta, pw, pb, grad_out = inputs
    return fused_ops.triton_pz_backward(
        x, theta, pw, pb, grad_out, False, True, c_dtype=c_dtype, launch=launch
    )


def same_gradients(got, ref, rel=1e-4):
    # Only the batch-block split changes, so only atomic summation order may.
    for a, b in zip(got[:2], ref[:2]):
        scale = b.abs().max().clamp_min(1e-12)
        if not torch.isfinite(a).all() or (a - b).abs().max() > rel * scale:
            return False
    return True


def bench(fn, rep):
    return do_bench_cudagraph(fn, rep=rep, return_mode="median")


def launch_for(block_b, num_warps, num_stages):
    options = (("num_stages", num_stages), ("num_warps", num_warps))
    return fused_ops.PzLaunch(block_b, options)


def tune_precision(name, c_dtype, args, n_oi):
    inputs = make_inputs(args.out_dim, args.in_dim, args.batch, args.reps, c_dtype)
    reference = backward(inputs, c_dtype, None)
    base_block = _select_block_b(n_oi, args.batch)
    budget = padded_rows(args.batch, base_block)
    blocks = [b for b in BLOCKS if padded_rows(args.batch, b) <= budget]

    def measure(block_b, num_warps, num_stages):
        launch = launch_for(block_b, num_warps, num_stages)
        try:
            if not same_gradients(backward(inputs, c_dtype, launch), reference):
                return None
            return bench(lambda: backward(inputs, c_dtype, launch), args.rep)
        except Exception as error:  # out of resources, for example
            log(f"  {name} {block_b}/{num_warps}/{num_stages}: {error}")
            return None

    base_ms = bench(lambda: backward(inputs, c_dtype, None), args.rep)
    log(f"{name}: built-in BLOCK_B={base_block} {base_ms * 1e3:.1f} us")

    # BLOCK_B x num_warps at the default stage count, then the stage count
    # for the three fastest pairs.
    timed = {}
    for block_b, num_warps in itertools.product(blocks, WARPS):
        timed[(block_b, num_warps, DEFAULT_STAGES)] = measure(
            block_b, num_warps, DEFAULT_STAGES
        )
    valid = sorted((ms, cfg) for cfg, ms in timed.items() if ms is not None)
    for _, (block_b, num_warps, _) in valid[:3]:
        for num_stages in STAGES:
            key = (block_b, num_warps, num_stages)
            if key not in timed:
                timed[key] = measure(*key)
    valid = sorted((ms, cfg) for cfg, ms in timed.items() if ms is not None)
    best_ms, best = valid[0]
    log(f"{name}: best {best} {best_ms * 1e3:.1f} us of {len(valid)} valid launches")

    # Confirm against the built-in launch, alternating to share any drift.
    best_launch = launch_for(*best)
    base_runs, best_runs = [], []
    for _ in range(args.confirm):
        base_runs.append(bench(lambda: backward(inputs, c_dtype, None), args.rep))
        best_runs.append(
            bench(lambda: backward(inputs, c_dtype, best_launch), args.rep)
        )
    base_ms, best_ms = statistics.median(base_runs), statistics.median(best_runs)
    gain = base_ms / best_ms - 1
    log(
        f"{name}: confirmed built-in {base_ms * 1e3:.1f} us, best {best_ms * 1e3:.1f}"
        f" us, gain {gain:+.1%}"
    )
    if gain < args.min_gain:
        return (base_block, DEFAULT_WARPS, DEFAULT_STAGES), base_ms, base_ms
    return best, base_ms, best_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dim", type=int, required=True)
    parser.add_argument("--in-dim", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rep", type=int, default=100, help="ms per timing")
    parser.add_argument("--confirm", type=int, default=7)
    parser.add_argument("--min-gain", type=float, default=0.05)
    args = parser.parse_args()

    torch.cuda.init()
    index = torch.cuda.current_device()
    architecture = fused_ops._device_architecture("cuda", index)
    n_oi = args.out_dim * args.in_dim
    log(
        f"{torch.cuda.get_device_name(index)} ({architecture}): n_oi={n_oi}, "
        f"batch={args.batch}, reps={args.reps}"
    )

    chosen = {}
    gained = False
    for name, c_dtype in PRECISIONS.items():
        config, base_ms, best_ms = tune_precision(name, c_dtype, args, n_oi)
        chosen[name] = config
        gained |= best_ms < base_ms
    if not gained:
        log("the built-in launch is already the fastest; no policy written")
        return 1

    blocks = {config[0] for config in chosen.values()}
    per_precision = len(blocks) > 1
    name = (
        f"pz-{architecture}-n{n_oi}-batch{args.batch}-r{args.reps}"
        "-fixed-preacts-fast-v1"
    )
    precisions = {}
    for precision, (block_b, num_warps, num_stages) in chosen.items():
        entry = {"num_warps": num_warps, "waves_per_eu": 0, "num_stages": num_stages}
        if per_precision:
            entry = {"block_b": block_b, **entry}
        precisions[precision] = entry
    document = {
        "schema_version": 1,
        "name": name,
        "operation": "pz_backward",
        "architecture": {"kind": "exact", "value": architecture},
        "match": {
            "kind": "exact",
            "n_oi": n_oi,
            "batch": args.batch,
            "reps": args.reps,
            "preacts_trainable": False,
            "fast_measure": True,
        },
        "block_selector": (
            {"kind": "per_precision"}
            if per_precision
            else {"kind": "fixed", "block_b": blocks.pop()}
        ),
        "precisions": precisions,
        "description": (
            f"PZ backward launch for the exact n_oi={n_oi}, batch={args.batch} "
            f"fixed-preacts fast-measure shape on {architecture}."
        ),
    }
    validate_launch_policy_document(name, document)
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
