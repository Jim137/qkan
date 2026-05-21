"""HQKANsformer optimizer head-to-head: AdamW vs QKANAdamMini vs cosine vs L-BFGS.

Builds a small HQKANsformer (GPT-2-style transformer with the MLP replaced by
Linear(d, els) -> QKAN([els, els], reps=1, flash) -> Linear(els, d), where
els = ceil(log2(d))) and trains it on TinyShakespeare with each optimizer.

The goal is to find out whether QKANAdamMini gives QKAN-bearing transformers
(a) a wall-clock speedup, (b) a memory saving at iso-loss, or (c) neither.

Run:
    python bench_hqkan_train.py --steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from qkan import (
    QKAN,
    AdaBelief,
    QKANAdamMini,
    QKANBeliefMini,
    QKANSpectralMini,
    TritonAdaBelief,
    adam_then_lbfgs,
)

# Try tiktoken; fall back to byte-level.
try:
    import tiktoken
    _TIKTOKEN = True
except ImportError:
    _TIKTOKEN = False


device = "cuda"
SEED = 42

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class HQKANMLP(nn.Module):
    """Linear(d->els) -> QKAN([els,els], reps=1) -> Linear(els->d)."""

    def __init__(self, n_embd: int):
        super().__init__()
        els = math.ceil(math.log2(n_embd))
        self.down = nn.Linear(n_embd, els)
        self.qkan = QKAN(
            width=[els, els],
            reps=1,
            ba_trainable=True,
            device=device,
            solver="flash",
            ansatz="pz_encoding",
        )
        self.up = nn.Linear(els, n_embd)

    def forward(self, x):
        s = x.shape
        x = x.reshape(-1, s[-1])
        x = self.up(self.qkan(self.down(x)))
        return x.view(*s)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = HQKANMLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class HQKANsformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        block_size: int = 256,
    ):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


TINY_SHAKE_PATH = "/home/jc/git/vibe/qkan/data/tinyshakespeare_input.txt"


def load_data(block_size: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
    """Returns (train_ids, val_ids, vocab_size, tokenizer_name)."""
    try:
        with open(TINY_SHAKE_PATH) as f:
            text = f.read()
        src = "tinyshakespeare"
    except OSError:
        # Synthetic fallback: random ints. Document this in the bench output.
        print("[WARN] tinyshakespeare not found; using synthetic random data")
        ids = torch.randint(0, 256, (200_000,), dtype=torch.long)
        n = int(0.9 * len(ids))
        return ids[:n], ids[n:], 256, "synthetic_byte"

    if _TIKTOKEN:
        enc = tiktoken.get_encoding("gpt2")
        ids = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)
        vocab_size = enc.n_vocab
        tok_name = f"tiktoken-gpt2 ({src})"
    else:
        # Byte-level fallback (chars in latin-1).
        ids = torch.tensor([ord(c) & 0xFF for c in text], dtype=torch.long)
        vocab_size = 256
        tok_name = f"byte-level ({src})"

    n_train = int(0.9 * len(ids))
    return ids[:n_train], ids[n_train:], vocab_size, tok_name


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ----------------------------------------------------------------------------
# Optimizer state inventory
# ----------------------------------------------------------------------------


def state_bytes(opt) -> int:
    total = 0
    states = opt.state if hasattr(opt, "state") else opt.early.state
    for st in states.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def state_numel(opt) -> int:
    total = 0
    states = opt.state if hasattr(opt, "state") else opt.early.state
    for st in states.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.numel()
    return total


def print_qkan_block_layout(opt: QKANAdamMini, model: nn.Module) -> None:
    """Show how Adam-mini partitioned each parameter."""
    print("  QKANAdamMini block layout:")
    state = opt.state
    n_show = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p not in state:
            continue
        st = state[p]
        v = st["exp_avg_sq_block"]
        ratio = (v.numel() / p.numel()) if p.numel() > 0 else 0
        marker = ""
        if ".theta" in name or "preacts_" in name:
            marker = "  <-- per-edge"
        elif "base_weight" in name or "postact_" in name:
            marker = "  <-- per-tensor"
        # Only print qkan-related params and a couple non-qkan for context.
        if "qkan" in name or n_show < 3:
            print(
                f"    {name:55s} param={tuple(p.shape)!s:18s} "
                f"v={tuple(v.shape)!s:12s} v/p={ratio:5.3f}{marker}"
            )
            if "qkan" not in name:
                n_show += 1


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------


@dataclass
class RunResult:
    name: str
    lr: float
    losses: list[float] = field(default_factory=list)
    val_loss: float = float("nan")
    step_ms: float = float("nan")
    peak_mem_bytes: int = 0
    opt_state_bytes: int = 0
    opt_state_numel: int = 0
    note: str = ""


def evaluate(model: nn.Module, data: torch.Tensor, block_size: int, batch_size: int,
             n_iters: int = 20, gen: Optional[torch.Generator] = None) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_iters):
            x, y = get_batch(data, block_size, batch_size, gen)
            _, loss = model(x, targets=y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def build_model(vocab_size: int, cfg: dict) -> HQKANsformer:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return HQKANsformer(
        vocab_size=vocab_size,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
    ).to(device)


def train_one(
    name: str,
    cfg: dict,
    vocab_size: int,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    optimizer_fn,
    n_steps: int,
    eval_every: int,
    use_cosine: bool = False,
    is_lbfgs: bool = False,
    cosine_min_ratio: float = 0.1,
) -> RunResult:
    """One full training run. optimizer_fn takes (model) -> optimizer."""
    print(f"\n=== Training: {name} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model(vocab_size, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params/1e6:.2f}M total, {n_train/1e6:.2f}M trainable")

    opt = optimizer_fn(model)

    # Initial step so optimizer materialises state (needed for accurate bytes count).
    gen = torch.Generator().manual_seed(SEED)
    x, y = get_batch(train_ids, cfg["block_size"], cfg["batch_size"], gen)
    if is_lbfgs:
        # LBFGSFinisher needs a closure even in early phase.
        def closure():
            opt.zero_grad() if hasattr(opt, "zero_grad") else None
            _, loss = model(x, targets=y)
            loss.backward()
            return loss
        loss0 = opt.step(closure).item()
    else:
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        loss0 = loss.item()

    # Print block layout for QKANAdamMini.
    if isinstance(opt, QKANAdamMini):
        print_qkan_block_layout(opt, model)
    print(f"  step 0 loss = {loss0:.4f}")

    # Now the main loop.
    result = RunResult(name=name, lr=cfg["lr"])
    result.opt_state_bytes = state_bytes(opt)
    result.opt_state_numel = state_numel(opt)

    # Cosine schedule helper.
    base_lr = cfg["lr"]

    def cosine_lr(step):
        # 50-step warmup, then cosine decay to base_lr * cosine_min_ratio.
        warmup = min(50, n_steps // 10)
        if step < warmup:
            return base_lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, n_steps - warmup)
        return base_lr * (cosine_min_ratio + (1 - cosine_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    gen_train = torch.Generator().manual_seed(SEED + 1)
    losses = []

    torch.cuda.synchronize()
    t_total = 0.0
    step_count = 0

    for step in range(n_steps):
        x, y = get_batch(train_ids, cfg["block_size"], cfg["batch_size"], gen_train)
        if use_cosine and not is_lbfgs:
            lr_now = cosine_lr(step)
            for g in opt.param_groups:
                g["lr"] = lr_now

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if is_lbfgs:
            def closure():
                opt.zero_grad() if hasattr(opt, "zero_grad") else None
                _, l = model(x, targets=y)
                l.backward()
                return l
            loss_t = opt.step(closure)
        else:
            opt.zero_grad()
            _, loss_t = model(x, targets=y)
            loss_t.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # Skip first 5 steps from timing (warmup).
        if step >= 5:
            t_total += dt
            step_count += 1
        losses.append(float(loss_t.item()))

        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            print(
                f"  step {step+1:4d}/{n_steps}  train_loss={losses[-1]:7.4f}  "
                f"step_ms={dt*1000:6.2f}"
            )

    result.losses = losses
    result.step_ms = (t_total / max(1, step_count)) * 1000
    result.peak_mem_bytes = torch.cuda.max_memory_allocated()

    # Final val loss.
    gen_val = torch.Generator().manual_seed(SEED + 99)
    result.val_loss = evaluate(model, val_ids, cfg["block_size"], cfg["batch_size"],
                                n_iters=20, gen=gen_val)
    print(f"  FINAL val_loss = {result.val_loss:.4f}  step_ms={result.step_ms:.2f}  "
          f"peak_mem={result.peak_mem_bytes/1e9:.2f}GB  "
          f"opt_state={result.opt_state_bytes/1e6:.2f}MB ({result.opt_state_numel} floats)")

    return result


# ----------------------------------------------------------------------------
# Main bench
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr_grid", type=str, default="3e-4,6e-4,1e-3")
    ap.add_argument("--quick", action="store_true",
                    help="Use 1 lr per opt (the middle of the grid) for fast iteration")
    ap.add_argument("--out_png", type=str, default="bench_hqkan_loss.png")
    ap.add_argument("--out_json", type=str, default="bench_hqkan_summary.json")
    ap.add_argument("--try_lbfgs", action="store_true",
                    help="Attempt adam_then_lbfgs (may OOM on the LM)")
    ap.add_argument("--size", type=str, default="tiny",
                    choices=("tiny", "gpt2_small", "gpt2_medium"),
                    help="tiny: 6L/6H/384D ~23M params; gpt2_small: 12L/12H/768D ~124M; gpt2_medium: 24L/16H/1024D ~355M")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    SIZE_CFGS = {
        "tiny":        dict(n_layer=6,  n_head=6,  n_embd=384,  block_size=256, batch_size=16),
        "gpt2_small":  dict(n_layer=12, n_head=12, n_embd=768,  block_size=256, batch_size=8),
        "gpt2_medium": dict(n_layer=24, n_head=16, n_embd=1024, block_size=256, batch_size=4),
    }
    cfg = dict(SIZE_CFGS[args.size])
    cfg["lr"] = 6e-4  # overridden per-run

    print("=" * 80)
    print("HQKANsformer optimizer bench")
    print("=" * 80)
    print(f"  Config: {cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}D, "
          f"els=ceil(log2({cfg['n_embd']}))={math.ceil(math.log2(cfg['n_embd']))}")
    print(f"  Batch: B={cfg['batch_size']}  T={cfg['block_size']}  steps={args.steps}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")

    train_ids, val_ids, vocab_size, tok_name = load_data(cfg["block_size"])
    print(f"  Data: {tok_name}, train_ids={len(train_ids)}, val_ids={len(val_ids)}, "
          f"vocab={vocab_size}")

    lr_grid = [float(s) for s in args.lr_grid.split(",")]
    if args.quick:
        lr_grid = [lr_grid[len(lr_grid) // 2]]
        print(f"  --quick: using single lr={lr_grid[0]}")

    eval_every = max(50, args.steps // 10)

    # The runs we'll do.
    all_runs: list[RunResult] = []
    best_per_opt: dict[str, RunResult] = {}

    OPTS = [
        "AdamW",
        "AdamW(fused)",
        "AdaBelief",
        "TritonAdaBelief",
        "TritonAdaBelief(bf16)",
        "QKANBeliefMini",
    ]
    if args.try_lbfgs:
        OPTS.append("adam_then_lbfgs")

    for opt_name in OPTS:
        for lr in lr_grid:
            run_cfg = dict(cfg)
            run_cfg["lr"] = lr

            label = f"{opt_name}@lr={lr:.0e}"

            if opt_name == "AdamW":
                fn = lambda m, lr=lr: torch.optim.AdamW(m.parameters(), lr=lr)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "AdamW(fused)":
                fn = lambda m, lr=lr: torch.optim.AdamW(m.parameters(), lr=lr, fused=True)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "TritonAdaBelief":
                fn = lambda m, lr=lr: TritonAdaBelief(m.parameters(), lr=lr)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "TritonAdaBelief(bf16)":
                fn = lambda m, lr=lr: TritonAdaBelief(
                    m.parameters(), lr=lr, state_dtype=torch.bfloat16
                )
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "AdamW+cosine":
                fn = lambda m, lr=lr: torch.optim.AdamW(m.parameters(), lr=lr)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every, use_cosine=True)
            elif opt_name == "QKANAdamMini":
                fn = lambda m, lr=lr: QKANAdamMini(m.named_parameters(), lr=lr)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "AdaBelief":
                fn = lambda m, lr=lr: AdaBelief(m.parameters(), lr=lr)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "QKANBeliefMini":
                fn = lambda m, lr=lr: QKANBeliefMini(m.named_parameters(), lr=lr)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "QKANSpectralMini":
                fn = lambda m, lr=lr: QKANSpectralMini(m.named_parameters(), lr=lr)
                res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                fn, args.steps, eval_every)
            elif opt_name == "adam_then_lbfgs":
                try:
                    fn = lambda m, lr=lr: adam_then_lbfgs(
                        m, total_steps=args.steps, lr_adam=lr,
                        pct_adam=0.7, use_adam_mini=True
                    )
                    res = train_one(label, run_cfg, vocab_size, train_ids, val_ids,
                                    fn, args.steps, eval_every, is_lbfgs=True)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"  [SKIP] adam_then_lbfgs failed: {e}")
                    res = RunResult(name=label, lr=lr, note=f"FAILED: {e}")
            else:
                raise ValueError(opt_name)

            all_runs.append(res)
            cur = best_per_opt.get(opt_name)
            if cur is None or (
                not math.isnan(res.val_loss) and res.val_loss < cur.val_loss
            ):
                best_per_opt[opt_name] = res

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS — best lr per optimizer")
    print("=" * 80)
    header = f"{'optimizer':<20} {'best_lr':>10} {'val_loss':>10} {'step_ms':>10} {'peak_GB':>8} {'opt_MB':>9} {'opt_numel':>12}"
    print(header)
    print("-" * len(header))
    for k, r in best_per_opt.items():
        print(f"{k:<20} {r.lr:>10.0e} {r.val_loss:>10.4f} {r.step_ms:>10.2f} "
              f"{r.peak_mem_bytes/1e9:>8.2f} {r.opt_state_bytes/1e6:>9.2f} {r.opt_state_numel:>12d}")

    # Plot the best curve per optimizer.
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, r in best_per_opt.items():
        if not r.losses:
            continue
        # Smooth with a running mean window of 10 for readability.
        L = torch.tensor(r.losses, dtype=torch.float32)
        w = 10
        if len(L) > w:
            smooth = torch.nn.functional.avg_pool1d(
                L.view(1, 1, -1), kernel_size=w, stride=1
            ).view(-1)
            xs = torch.arange(w - 1, len(L))
            ax.plot(xs.numpy(), smooth.numpy(),
                    label=f"{k} lr={r.lr:.0e} (val={r.val_loss:.3f})")
        else:
            ax.plot(L.numpy(), label=f"{k} lr={r.lr:.0e} (val={r.val_loss:.3f})")
    ax.set_xlabel("step")
    ax.set_ylabel("train cross-entropy (10-step EMA)")
    ax.set_title(f"HQKANsformer {cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}D — optimizer comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=120)
    print(f"\nSaved loss curves to {args.out_png}")

    # Dump JSON.
    summary = {
        "cfg": cfg,
        "n_steps": args.steps,
        "lr_grid": lr_grid,
        "tokenizer": tok_name,
        "vocab_size": vocab_size,
        "runs": [
            {
                "name": r.name,
                "lr": r.lr,
                "val_loss": r.val_loss,
                "final_train_loss": r.losses[-1] if r.losses else float("nan"),
                "step_ms": r.step_ms,
                "peak_mem_bytes": r.peak_mem_bytes,
                "opt_state_bytes": r.opt_state_bytes,
                "opt_state_numel": r.opt_state_numel,
                "note": r.note,
            }
            for r in all_runs
        ],
        "best_per_opt": {
            k: {
                "lr": r.lr, "val_loss": r.val_loss, "step_ms": r.step_ms,
                "peak_mem_bytes": r.peak_mem_bytes, "opt_state_bytes": r.opt_state_bytes,
                "opt_state_numel": r.opt_state_numel,
            } for k, r in best_per_opt.items()
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {args.out_json}")


if __name__ == "__main__":
    main()
