"""GPT-2 HQKANsformer benchmark: CuTe vs Flash vs cuTile.

Compares end-to-end training throughput on TinyShakespeare across:
  - Solvers: cute, flash, (cutile if available)
  - Ansatzes: pz_encoding, real
  - Dtypes: bf16, fp8
"""

import math
import os
import random
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

from qkan import QKAN

device = "cuda"

# ── Model (identical to bench_gpt2_precision.py) ─────────────────────────────


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class HQKANBlock(nn.Module):
    def __init__(self, config, solver, ansatz, c_dtype):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        els = math.ceil(math.log2(config.n_embd))
        p_dtype = (
            torch.bfloat16
            if c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
            else torch.float32
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, els),
            QKAN(
                width=[els, els],
                reps=1,
                ba_trainable=True,
                device=device,
                solver=solver,
                ansatz=ansatz,
                c_dtype=c_dtype,
                p_dtype=p_dtype,
            ),
            nn.Linear(els, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPTConfig:
    def __init__(self, **kw):
        self.block_size = 1024
        self.vocab_size = 50304
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.0
        self.bias = True
        for k, v in kw.items():
            setattr(self, k, v)


class GPTModel(nn.Module):
    def __init__(self, config, solver, ansatz, c_dtype):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        HQKANBlock(config, solver, ansatz, c_dtype)
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            return logits, F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return self.lm_head(x[:, [-1], :]), None


# ── Data ──────────────────────────────────────────────────────────────────────

TS_PATH = "./data/tinyshakespeare_input.txt"
if not os.path.isfile(TS_PATH):
    os.makedirs("./data", exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        TS_PATH,
    )
with open(TS_PATH, "r") as f:
    text = f.read()
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
data_all = torch.tensor([stoi[ch] for ch in text], dtype=torch.long, device=device)

BLOCK_SIZE = 1024
BATCH_SIZE = 1
N_WARMUP = 20
N_ITERS = 500


def get_batch():
    ix = torch.randint(len(data_all) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data_all[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data_all[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
    return x, y


# ── Benchmark ─────────────────────────────────────────────────────────────────


def run(label, solver, ansatz, c_dtype):
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config = GPTConfig(vocab_size=len(chars))
    model_dtype = (
        torch.bfloat16
        if c_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        else torch.float32
    )
    model = GPTModel(config, solver, ansatz, c_dtype).to(device)
    if model_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Warmup
    for _ in range(N_WARMUP):
        x, y = get_batch()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed
    torch.cuda.reset_peak_memory_stats()
    times = []
    final_loss = 0.0
    loss_at = {}
    for i in range(N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x, y = get_batch()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        final_loss = loss.item()
        if (i + 1) in (100, 250, 500):
            loss_at[i + 1] = final_loss

    times.sort()
    p50 = times[len(times) // 2]
    mean = sum(times) / len(times)
    peak = torch.cuda.max_memory_allocated() / 1024**2

    del model, optimizer
    torch.cuda.empty_cache()

    return dict(
        label=label,
        params=n_params,
        mean=mean,
        p50=p50,
        peak=peak,
        loss=final_loss,
        loss_at=loss_at,
    )


def main():
    # Check available solvers
    from qkan.solver import _CUTE_AVAILABLE, _CUTILE_AVAILABLE, _FLASH_AVAILABLE

    solvers = []
    if _FLASH_AVAILABLE:
        solvers.append("flash")
    if _CUTE_AVAILABLE:
        solvers.append("cute")
    if _CUTILE_AVAILABLE:
        solvers.append("cutile")

    print("=" * 90)
    print(
        f"GPT-2 HQKANsformer (12L, 768E, reps=1) — TinyShakespeare, batch={BATCH_SIZE}"
    )
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Solvers: {', '.join(solvers)}")
    print(f"Warmup: {N_WARMUP}, Timed: {N_ITERS} iters")
    print("=" * 90)

    configs = []
    for solver in solvers:
        for ansatz, alabel in [("pz_encoding", "pz"), ("real", "real")]:
            for c_dtype, dlabel in [
                (torch.bfloat16, "bf16"),
                (torch.float8_e4m3fn, "fp8"),
            ]:
                label = f"{solver}_{alabel}_{dlabel}"
                configs.append((label, solver, ansatz, c_dtype))

    header = f"{'Label':<22} {'Params':>8} {'Mean ms':>8} {'P50 ms':>8} {'Peak MiB':>9} {'Loss':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    results = []
    for label, solver, ansatz, c_dtype in configs:
        try:
            r = run(label, solver, ansatz, c_dtype)
            results.append(r)
            print(
                f"{r['label']:<22} {r['params']:>8,} {r['mean']:>8.1f} "
                f"{r['p50']:>8.1f} {r['peak']:>9.1f} {r['loss']:>7.4f}"
            )
        except Exception as e:
            print(f"{label:<22} FAILED: {e}")

    # Speedup table (vs first flash result)
    flash_results = [r for r in results if r["label"].startswith("flash")]
    if flash_results and len(results) > 1:
        print(f"\n{'Label':<22} {'P50 ms':>8} {'vs flash_pz_bf16':>16}")
        print("-" * 48)
        base = flash_results[0]["p50"]
        for r in results:
            sp = f"{base / r['p50']:.2f}x"
            print(f"{r['label']:<22} {r['p50']:>8.1f} {sp:>16}")

    # Loss progression
    steps = sorted({s for r in results for s in r.get("loss_at", {})})
    if steps and results:
        print(f"\nLoss progression:")
        hdr = f"{'Step':>6}" + "".join(f" {r['label']:>22}" for r in results)
        print(hdr)
        print("-" * len(hdr))
        for step in steps:
            row = f"{step:>6}"
            for r in results:
                val = r.get("loss_at", {}).get(step)
                row += f" {val:>22.4f}" if val is not None else f" {'—':>22}"
            print(row)


if __name__ == "__main__":
    main()
