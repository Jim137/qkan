"""
Benchmark: Solver x Ansatz comparison + MLP baseline

Compares:  exact(pz), exact(real), flash(pz), flash(real), cutile(pz), cutile(real), cutn(pz), cutn(real), Triton MLP, MLP

Benchmarks:
  1. README example  — QKAN [1,1] function fitting (no MLP)
  2. HQKAN CIFAR-100 — 50 epochs, HQKAN-44 vs 3-layer MLP (hqkan_cifar100.ipynb)
  3. HQKANsformer   — GPT-2, 2000 iters on TinyShakespeare (gqkan_gpt.ipynb setup)
  4. HQKANsformer   — GPT-2 batch=10, 200 iters on WebText (gqkan_gpt.ipynb setup)
  5. Extreme Synth   — QKAN [100,100] with batch=1000, 50 training steps

Results saved to bench_cu.md.
"""

import math
import os
import pickle
import random
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import triton
import triton.language as tl
from torch.utils.data import DataLoader

from torch.utils.flop_counter import FlopCounterMode

from qkan import KAN, QKAN, create_dataset

device = "cuda"
torch.backends.cudnn.benchmark = True

# Probe cuTile availability (requires CUDA driver >= 13.0)
_CUTILE_OK = False
try:
    import cuda.tile as _ct
    # Try a minimal launch to verify driver compatibility
    @_ct.kernel
    def _probe_kernel(out, N: _ct.Constant[int]):
        bid = _ct.bid(0)
        _ct.scatter(out, bid, 1.0)
    _probe_out = torch.zeros(1, device=device, dtype=torch.float32)
    _ct.launch(torch.cuda.current_stream(), (1, 1, 1), _probe_kernel, (_probe_out, 1))
    torch.cuda.synchronize()
    _CUTILE_OK = True
    print("cuTile: OK (driver compatible)")
    del _probe_out
except Exception as e:
    print(f"cuTile: SKIPPED ({e})")

# Solver x Ansatz variants for QKAN benchmarks
QKAN_VARIANTS = [
    ("exact_pz",   "exact",  "pz_encoding"),
    ("exact_real", "exact",  "real"),
    ("flash_pz",   "flash",  "pz_encoding"),
    ("flash_real", "flash",  "real"),
]
if _CUTILE_OK:
    QKAN_VARIANTS += [
        ("cutile_pz",  "cutile", "pz_encoding"),
        ("cutile_real","cutile", "real"),
    ]
QKAN_VARIANTS += [
    ("cutn_pz",    "cutn",   "pz_encoding"),
    ("cutn_real",  "cutn",   "real"),
]

def _real_kwargs(ansatz):
    """Return extra QKAN kwargs for real ansatz (bf16 compute/param dtype)."""
    if ansatz == "real":
        return dict(p_dtype=torch.bfloat16)
    return {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def warmup_cuda():
    a = torch.randn(256, 256, device=device)
    for _ in range(10):
        a = a @ a
    torch.cuda.synchronize()

def timed_init(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def bench_forward(model, x, n_warmup=10, n_iter=100):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000

def bench_gpt_forward(model, x, y, n_warmup=5, n_iter=20):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x, y)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000

def fmt_mem(b):
    """Format bytes as MiB."""
    return f"{b / 1024**2:.1f}"

def count_flops(model, *args):
    """Count FLOPs for a single forward pass."""
    model.eval()
    with torch.no_grad():
        counter = FlopCounterMode(display=False)
        with counter:
            model(*args)
    return counter.get_total_flops()

def fmt_flops(f):
    """Format FLOPs as human-readable string."""
    if f >= 1e12:
        return f"{f/1e12:.2f}T"
    if f >= 1e9:
        return f"{f/1e9:.2f}G"
    if f >= 1e6:
        return f"{f/1e6:.2f}M"
    if f >= 1e3:
        return f"{f/1e3:.2f}K"
    return f"{f:.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 1: README Function Fitting (no MLP)
# ══════════════════════════════════════════════════════════════════════════════

N_STEPS_1 = 100

print("=" * 70)
print(f"Benchmark 1: README Function Fitting  QKAN([1, 1], reps=3)")
print("=" * 70)

f_target = lambda x: torch.sin(20 * x) / x / 20
dataset = create_dataset(
    f_target, n_var=1, ranges=[0, 1], device=device,
    train_num=1000, test_num=1000, seed=0,
)
x_train, y_train = dataset["train_input"], dataset["train_label"]
x_test, y_test = dataset["test_input"], dataset["test_label"]
loss_fn_1 = nn.MSELoss()

readme_results = {}

for label, solver, ansatz in QKAN_VARIANTS:
    torch.manual_seed(0)
    warmup_cuda()

    def make_model(s=solver, a=ansatz):
        return QKAN(
            [1, 1], reps=3, device=device, seed=0,
            preact_trainable=True,
            postact_weight_trainable=True,
            postact_bias_trainable=True,
            ba_trainable=True,
            solver=s, ansatz=a,
            **_real_kwargs(a),
        )

    model, init_ms = timed_init(make_model)
    n_params = count_params(model)
    flops = count_flops(model, x_train)
    fwd_ms = bench_forward(model, x_train, n_warmup=20, n_iter=200)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(5):
        optimizer.zero_grad(); loss_fn_1(model(x_train), y_train).backward(); optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_samples = []
    t0 = time.perf_counter()
    for _ in range(N_STEPS_1):
        optimizer.zero_grad()
        loss = loss_fn_1(model(x_train), y_train)
        loss.backward(); optimizer.step()
        mem_samples.append(torch.cuda.memory_allocated())
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) * 1000
    peak_mem = torch.cuda.max_memory_allocated()
    avg_mem = sum(mem_samples) / len(mem_samples)

    model.eval()
    with torch.no_grad():
        test_loss = loss_fn_1(model(x_test), y_test).item()

    readme_results[label] = dict(
        init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
        step_ms=train_ms / N_STEPS_1, train_ms=train_ms, test_loss=test_loss,
        peak_mem_mib=peak_mem / 1024**2, avg_mem_mib=avg_mem / 1024**2,
    )
    print(f"  [{label:12s}] params:{n_params:5d} | flops:{fmt_flops(flops)} | init:{init_ms:7.1f}ms | fwd:{fwd_ms:.3f}ms | "
          f"step:{train_ms/N_STEPS_1:.3f}ms | {N_STEPS_1}steps:{train_ms:.1f}ms | "
          f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | test_loss:{test_loss:.4f}")
    del model, optimizer; torch.cuda.empty_cache()

# -- KAN baseline for benchmark 1 --
torch.manual_seed(0)
warmup_cuda()
model, init_ms = timed_init(lambda: KAN([1, 1], device=device, seed=0))
n_params = count_params(model)
flops = count_flops(model, x_train)
fwd_ms = bench_forward(model, x_train, n_warmup=20, n_iter=200)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()
for _ in range(5):
    optimizer.zero_grad(); loss_fn_1(model(x_train), y_train).backward(); optimizer.step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mem_samples = []
t0 = time.perf_counter()
for _ in range(N_STEPS_1):
    optimizer.zero_grad()
    loss = loss_fn_1(model(x_train), y_train)
    loss.backward(); optimizer.step()
    mem_samples.append(torch.cuda.memory_allocated())
torch.cuda.synchronize()
train_ms = (time.perf_counter() - t0) * 1000
peak_mem = torch.cuda.max_memory_allocated()
avg_mem = sum(mem_samples) / len(mem_samples)
model.eval()
with torch.no_grad():
    test_loss = loss_fn_1(model(x_test), y_test).item()
readme_results["kan"] = dict(
    init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
    step_ms=train_ms / N_STEPS_1, train_ms=train_ms, test_loss=test_loss,
    peak_mem_mib=peak_mem / 1024**2, avg_mem_mib=avg_mem / 1024**2,
)
print(f"  [{'kan':12s}] params:{n_params:5d} | flops:{fmt_flops(flops)} | init:{init_ms:7.1f}ms | fwd:{fwd_ms:.3f}ms | "
      f"step:{train_ms/N_STEPS_1:.3f}ms | {N_STEPS_1}steps:{train_ms:.1f}ms | "
      f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | test_loss:{test_loss:.4f}")
del model, optimizer; torch.cuda.empty_cache()

print()


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 2: HQKAN CIFAR-100 — 50 epochs
# ══════════════════════════════════════════════════════════════════════════════

N_EPOCHS_2 = 50

print("=" * 70)
print(f"Benchmark 2: HQKAN CIFAR-100 (HQKAN-44, {N_EPOCHS_2} epochs)")
print("=" * 70)


class CNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=3, device=device)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3, device=device)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, device=device)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.to(self.device)
        x = self.relu(self.cnn1(x)); x = self.maxpool1(x)
        x = self.relu(self.cnn2(x)); x = self.maxpool2(x)
        x = self.relu(self.cnn3(x)); x = self.maxpool3(x)
        return x.flatten(start_dim=1)


in_feat, out_feat = 256, 100
in_resize = int(np.ceil(np.log2(in_feat))) * 4  # 32
out_resize = int(np.ceil(np.log2(out_feat))) * 4  # 28

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=1000, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)
criterion_2 = nn.CrossEntropyLoss()
images_bench = next(iter(trainloader))[0].to(device)

cifar_results = {}

for label, solver, ansatz in QKAN_VARIANTS:
    torch.manual_seed(42)
    warmup_cuda()

    def make_model(s=solver, a=ansatz):
        return nn.Sequential(
            CNet(device=device),
            nn.Linear(in_feat, in_resize, device=device),
            QKAN([in_resize, out_resize], device=device, solver=s, ansatz=a, **_real_kwargs(a)),
            nn.Linear(out_resize, out_feat, device=device),
        )

    model, init_ms = timed_init(make_model)
    n_params = count_params(model)
    flops = count_flops(model, images_bench)
    fwd_ms = bench_forward(model, images_bench, n_warmup=10, n_iter=50)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    total_steps = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_samples = []
    t0 = time.perf_counter()
    for epoch in range(N_EPOCHS_2):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_2(output, labels)
            loss.backward(); optimizer.step()
            total_steps += 1
            mem_samples.append(torch.cuda.memory_allocated())
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) * 1000
    peak_mem = torch.cuda.max_memory_allocated()
    avg_mem = sum(mem_samples) / len(mem_samples)

    model.eval()
    test_loss = test_acc = test_top5 = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion_2(output, labels).item()
            test_acc += (output.argmax(1) == labels).float().mean().item()
            test_top5 += (output.topk(5, 1).indices == labels.unsqueeze(1)).any(1).float().mean().item()
    test_loss /= len(testloader); test_acc /= len(testloader); test_top5 /= len(testloader)

    cifar_results[label] = dict(
        init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
        step_ms=train_ms / total_steps, train_ms=train_ms, total_steps=total_steps,
        test_loss=test_loss, test_acc=test_acc, test_top5=test_top5,
        peak_mem_mib=peak_mem / 1024**2, avg_mem_mib=avg_mem / 1024**2,
    )
    print(f"  [{label:12s}] params:{n_params:6d} | flops:{fmt_flops(flops)} | fwd:{fwd_ms:.3f}ms | step:{train_ms/total_steps:.3f}ms | "
          f"{N_EPOCHS_2}ep:{train_ms/1000:.1f}s | "
          f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | "
          f"top1:{test_acc:.1%} top5:{test_top5:.1%}")
    del model, optimizer; torch.cuda.empty_cache()

# -- KAN baseline for CIFAR --
torch.manual_seed(42); warmup_cuda()
model, init_ms = timed_init(lambda: nn.Sequential(
    CNet(device=device),
    nn.Linear(in_feat, in_resize, device=device),
    KAN([in_resize, out_resize], device=device),
    nn.Linear(out_resize, out_feat, device=device),
))
n_params = count_params(model)
flops = count_flops(model, images_bench)
fwd_ms = bench_forward(model, images_bench, n_warmup=10, n_iter=50)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train(); total_steps = 0
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mem_samples = []
t0 = time.perf_counter()
for epoch in range(N_EPOCHS_2):
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); output = model(images)
        loss = criterion_2(output, labels); loss.backward(); optimizer.step()
        total_steps += 1; mem_samples.append(torch.cuda.memory_allocated())
torch.cuda.synchronize()
train_ms = (time.perf_counter() - t0) * 1000
peak_mem = torch.cuda.max_memory_allocated()
avg_mem = sum(mem_samples) / len(mem_samples)
model.eval(); test_loss = test_acc = test_top5 = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        test_loss += criterion_2(output, labels).item()
        test_acc += (output.argmax(1) == labels).float().mean().item()
        test_top5 += (output.topk(5, 1).indices == labels.unsqueeze(1)).any(1).float().mean().item()
test_loss /= len(testloader); test_acc /= len(testloader); test_top5 /= len(testloader)
cifar_results["kan"] = dict(
    init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
    step_ms=train_ms / total_steps, train_ms=train_ms, total_steps=total_steps,
    test_loss=test_loss, test_acc=test_acc, test_top5=test_top5,
    peak_mem_mib=peak_mem / 1024**2, avg_mem_mib=avg_mem / 1024**2,
)
print(f"  [{'kan':12s}] params:{n_params:6d} | flops:{fmt_flops(flops)} | fwd:{fwd_ms:.3f}ms | step:{train_ms/total_steps:.3f}ms | "
      f"{N_EPOCHS_2}ep:{train_ms/1000:.1f}s | "
      f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | "
      f"top1:{test_acc:.1%} top5:{test_top5:.1%}")
del model, optimizer; torch.cuda.empty_cache()

del images_bench, trainloader, testloader, trainset, testset
torch.cuda.empty_cache()
print()


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 3: HQKANsformer GPT-2 — 2000 iters
# ══════════════════════════════════════════════════════════════════════════════

N_ITERS_3 = 2000
BLOCK_SIZE = 1024
BATCH_GPT = 1
LR_GPT = 3e-4

print("=" * 70)
print(f"Benchmark 3: HQKANsformer GPT-2 ({N_ITERS_3} iters, block_size={BLOCK_SIZE})")
print("=" * 70)


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head, self.n_embd, self.dropout = config.n_head, config.n_embd, config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class HQKANBlock(nn.Module):
    def __init__(self, config, solver="exact", ansatz="real"):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        els = math.ceil(math.log2(config.n_embd))
        self.mlp = nn.ModuleDict(dict(
            c=nn.Sequential(
                nn.Linear(config.n_embd, els),
                QKAN(width=[els, els], reps=1,
                    #  preact_trainable=True,
                    #  postact_weight_trainable=True,
                    #  postact_bias_trainable=True,
                     ba_trainable=True,
                     device=device, solver=solver, ansatz=ansatz,
                     **_real_kwargs(ansatz)),
                nn.Linear(els, config.n_embd)),
            dropout=nn.Dropout(config.dropout)))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c(x))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlpf(self.ln_2(x))


class MLPBlock(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.dropout(self.c_proj(self.gelu(self.c_fc(self.ln_2(x)))))


@triton.jit
def _fused_bias_gelu_kernel(
    x_ptr, bias_ptr, out_ptr,
    N,  # number of columns
    stride_x_row, stride_o_row,
    BLOCK_N: tl.constexpr,
):
    """Fused bias + GELU(tanh approx): out = 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3))), z=x+bias."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask)
    b = tl.load(bias_ptr + cols, mask=mask)
    z = x + b
    # tanh-approximation GELU: 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3)))
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    z3 = z * z * z
    inner = SQRT_2_OVER_PI * (z + 0.044715 * z3)
    # tanh via exp: tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    out = 0.5 * z * (1.0 + tanh_val)
    tl.store(out_ptr + row * stride_o_row + cols, out, mask=mask)


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Launch the fused bias+GELU kernel. x: (*, N), bias: (N,)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    rows, N = x_2d.shape
    out = torch.empty_like(x_2d)
    BLOCK_N = triton.next_power_of_2(N)
    _fused_bias_gelu_kernel[(rows,)](
        x_2d, bias, out, N,
        x_2d.stride(0), out.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return out.reshape(orig_shape)


class TritonMLPBlock(nn.Module):
    """GPT-2 transformer block with Triton-fused bias+GELU MLP."""
    def __init__(self, config, **kwargs):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # Use bias=False on c_fc so we fuse bias into the GELU kernel
        self.c_fc_weight = nn.Parameter(torch.empty(4 * config.n_embd, config.n_embd))
        self.c_fc_bias = nn.Parameter(torch.zeros(4 * config.n_embd)) if config.bias else None
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.c_fc_weight, std=0.02)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = F.linear(self.ln_2(x), self.c_fc_weight)  # no bias — fused below
        h = fused_bias_gelu(h, self.c_fc_bias)
        return x + self.dropout(self.c_proj(h))


class HKANBlock(nn.Module):
    """GPT-2 transformer block with classical KAN (B-spline) MLP."""
    def __init__(self, config, **kwargs):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        els = math.ceil(math.log2(config.n_embd))
        self.mlp_down = nn.Linear(config.n_embd, els)
        self.kan = KAN([els, els], device=device)
        self.mlp_up = nn.Linear(els, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        B, T, C = x.size()
        h = self.mlp_down(self.ln_2(x))
        h = self.kan(h.reshape(B * T, -1)).reshape(B, T, -1)
        return x + self.dropout(self.mlp_up(h))


class GPTConfig:
    def __init__(self, **kw):
        self.block_size, self.vocab_size = BLOCK_SIZE, 50304
        self.n_layer, self.n_head, self.n_embd = 12, 12, 768
        self.dropout, self.bias = 0.0, True
        for k, v in kw.items(): setattr(self, k, v)


class GPTModel(nn.Module):
    def __init__(self, config, block_cls, block_kwargs=None):
        super().__init__()
        self.config = config; bk = block_kwargs or {}
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([block_cls(config, **bk) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias)))
        self.lm_head = nn.Linear(config.vocab_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding: n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h: x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return self.lm_head(x[:, [-1], :]), None


# -- TinyShakespeare --
TS_PATH = "./data/tinyshakespeare_input.txt"
if not os.path.isfile(TS_PATH):
    os.makedirs("./data", exist_ok=True)
    print("  Downloading TinyShakespeare...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", TS_PATH)
with open(TS_PATH, "r") as f: text = f.read()
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
data_all = torch.tensor([stoi[ch] for ch in text], dtype=torch.long, device=device)
vocab_size_char = len(chars)
print(f"  TinyShakespeare: {len(text)} chars, vocab_size={vocab_size_char}")

def get_gpt_batch(batch_size=None):
    bs = batch_size or BATCH_GPT
    ix = torch.randint(len(data_all) - BLOCK_SIZE - 1, (bs,))
    x = torch.stack([data_all[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_all[i+1:i+1+BLOCK_SIZE] for i in ix])
    return x, y

gpt_config = GPTConfig(vocab_size=vocab_size_char)
gpt_x_bench, gpt_y_bench = get_gpt_batch()

ALL_VARIANTS_3 = QKAN_VARIANTS + [("kan", None, None), ("triton_mlp", None, None), ("mlp", None, None)]
gpt_results = {}

for label, solver, ansatz in ALL_VARIANTS_3:
    torch.manual_seed(42); random.seed(42)
    warmup_cuda()

    if label == "kan":
        def make_model(): return GPTModel(gpt_config, HKANBlock).to(device)
    elif label == "triton_mlp":
        def make_model(): return GPTModel(gpt_config, TritonMLPBlock).to(device)
    elif label == "mlp":
        def make_model(): return GPTModel(gpt_config, MLPBlock).to(device)
    else:
        def make_model(s=solver, a=ansatz):
            return GPTModel(gpt_config, HQKANBlock, dict(solver=s, ansatz=a)).to(device)

    print(f"  Building GPT ({label})...")
    model, init_ms = timed_init(make_model)
    n_params = model.get_num_params()
    flops = count_flops(model, gpt_x_bench, gpt_y_bench)
    fwd_ms = bench_gpt_forward(model, gpt_x_bench, gpt_y_bench, n_warmup=3, n_iter=10)

    optimizer = optim.AdamW(model.parameters(), lr=LR_GPT, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_samples = []
    t0 = time.perf_counter()
    for it in range(N_ITERS_3):
        x_b, y_b = get_gpt_batch()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_b, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mem_samples.append(torch.cuda.memory_allocated())
        if it % 500 == 0 or it == N_ITERS_3 - 1:
            print(f"    iter {it:4d}: loss {loss.item():.4f}")
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) * 1000
    peak_mem = torch.cuda.max_memory_allocated()
    avg_mem = sum(mem_samples) / len(mem_samples)

    gpt_results[label] = dict(
        init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
        step_ms=train_ms / N_ITERS_3, train_ms=train_ms,
        final_loss=loss.item(),
        peak_mem_mib=peak_mem / 1024**2, avg_mem_mib=avg_mem / 1024**2,
    )
    print(f"  [{label:12s}] params:{n_params/1e6:.2f}M | flops:{fmt_flops(flops)} | fwd:{fwd_ms:.3f}ms | step:{train_ms/N_ITERS_3:.3f}ms | "
          f"{N_ITERS_3}it:{train_ms/1000:.1f}s | "
          f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | "
          f"loss:{loss.item():.3f}")
    del model, optimizer; torch.cuda.empty_cache()

print()


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 4: HQKANsformer GPT-2 — batch=10, 1000 iters on WebText
# ══════════════════════════════════════════════════════════════════════════════

N_ITERS_4 = 1000
BATCH_GPT_4 = 10
LR_GPT_4 = 1e-3

print("=" * 70)
print(f"Benchmark 4: HQKANsformer GPT-2 batch={BATCH_GPT_4} ({N_ITERS_4} iters, WebText, block_size={BLOCK_SIZE})")
print("=" * 70)

# -- WebText dataset (GPT-2 tokenizer, vocab_size=50304) --
WEBTEXT_DIR = "./datasets/webtext"
WEBTEXT_JSONL = os.path.join(WEBTEXT_DIR, "webtext.train.jsonl")
WEBTEXT_PKL = os.path.join(WEBTEXT_DIR, "webtext.train.pkl")

if not os.path.isfile(WEBTEXT_PKL):
    # Download jsonl if missing
    if not os.path.isfile(WEBTEXT_JSONL):
        os.makedirs(WEBTEXT_DIR, exist_ok=True)
        url = "https://openaipublic.blob.core.windows.net/gpt-2/output-dataset/v1/webtext.train.jsonl"
        print(f"  Downloading WebText train split...")
        urllib.request.urlretrieve(url, WEBTEXT_JSONL)
    # Tokenize and cache
    print("  Tokenizing WebText (one-time)...")
    import json
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    webtext_tokens = []
    with open(WEBTEXT_JSONL, "r") as f:
        for line in f:
            text = json.loads(line)["text"]
            webtext_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    with open(WEBTEXT_PKL, "wb") as f:
        pickle.dump(webtext_tokens, f)
    print(f"  Tokenized {len(webtext_tokens)} tokens -> {WEBTEXT_PKL}")
else:
    print("  Loading WebText tokenized data...")
    with open(WEBTEXT_PKL, "rb") as f:
        webtext_tokens = pickle.load(f)

webtext_data = torch.tensor(webtext_tokens, dtype=torch.long)  # keep on CPU
print(f"  WebText: {len(webtext_tokens)} tokens, vocab_size=50304")
del webtext_tokens

def get_webtext_batch(batch_size, block_size=BLOCK_SIZE):
    ix = torch.randint(len(webtext_data) - block_size - 1, (batch_size,))
    x = torch.stack([webtext_data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([webtext_data[i+1:i+1+block_size] for i in ix]).to(device)
    return x, y

gpt_config_4 = GPTConfig(vocab_size=50304)
gpt_x_bench_4, gpt_y_bench_4 = get_webtext_batch(BATCH_GPT_4)

ALL_VARIANTS_4 = QKAN_VARIANTS + [("kan", None, None), ("triton_mlp", None, None), ("mlp", None, None)]
gpt_b10_results = {}

for label, solver, ansatz in ALL_VARIANTS_4:
    torch.manual_seed(42); random.seed(42)
    warmup_cuda()

    if label == "kan":
        def make_model(): return GPTModel(gpt_config_4, HKANBlock).to(device)
    elif label == "triton_mlp":
        def make_model(): return GPTModel(gpt_config_4, TritonMLPBlock).to(device)
    elif label == "mlp":
        def make_model(): return GPTModel(gpt_config_4, MLPBlock).to(device)
    else:
        def make_model(s=solver, a=ansatz):
            return GPTModel(gpt_config_4, HQKANBlock,
                            dict(solver=s, ansatz=a)).to(device)

    print(f"  Building GPT ({label})...")
    model, init_ms = timed_init(make_model)
    n_params = model.get_num_params()
    flops = count_flops(model, gpt_x_bench_4, gpt_y_bench_4)
    fwd_ms = bench_gpt_forward(model, gpt_x_bench_4, gpt_y_bench_4, n_warmup=3, n_iter=10)

    optimizer = optim.AdamW(model.parameters(), lr=LR_GPT_4, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_samples = []
    t0 = time.perf_counter()
    for it in range(N_ITERS_4):
        x_b, y_b = get_webtext_batch(BATCH_GPT_4)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_b, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mem_samples.append(torch.cuda.memory_allocated())
        if it % 500 == 0 or it == N_ITERS_4 - 1:
            print(f"    iter {it:4d}: loss {loss.item():.4f}")
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) * 1000
    peak_mem = torch.cuda.max_memory_allocated()
    avg_mem = sum(mem_samples) / len(mem_samples)

    gpt_b10_results[label] = dict(
        init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
        step_ms=train_ms / N_ITERS_4, train_ms=train_ms,
        final_loss=loss.item(),
        peak_mem_mib=peak_mem / 1024**2, avg_mem_mib=avg_mem / 1024**2,
    )
    print(f"  [{label:12s}] params:{n_params/1e6:.2f}M | flops:{fmt_flops(flops)} | fwd:{fwd_ms:.3f}ms | step:{train_ms/N_ITERS_4:.3f}ms | "
          f"{N_ITERS_4}it:{train_ms/1000:.1f}s | "
          f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | "
          f"loss:{loss.item():.3f}")
    del model, optimizer; torch.cuda.empty_cache()

del webtext_data
torch.cuda.empty_cache()
print()


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 5: Extreme Synthetic QKAN
# ══════════════════════════════════════════════════════════════════════════════

N_STEPS_5 = 50

print("=" * 70)
print("Benchmark 5: Extreme Synthetic QKAN([100, 100], batch=1000)")
print("=" * 70)

x_synth = torch.randn(1000, 100, device=device)
y_synth = torch.randn(1000, 100, device=device)
loss_fn_5 = nn.MSELoss()
extreme_results = {}

for label, solver, ansatz in QKAN_VARIANTS:
    torch.manual_seed(0)
    warmup_cuda()

    def make_model(s=solver, a=ansatz):
        return QKAN([100, 100], reps=3, device=device, seed=0, solver=s, ansatz=a, **_real_kwargs(a))

    model, init_ms = timed_init(make_model)
    n_params = count_params(model)
    flops = count_flops(model, x_synth)
    fwd_ms = bench_forward(model, x_synth, n_warmup=5, n_iter=20)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    # warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss_fn_5(model(x_synth), y_synth).backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_samples = []
    t0 = time.perf_counter()
    for _ in range(N_STEPS_5):
        optimizer.zero_grad()
        loss = loss_fn_5(model(x_synth), y_synth)
        loss.backward()
        optimizer.step()
        mem_samples.append(torch.cuda.memory_allocated())
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) * 1000
    peak_mem = torch.cuda.max_memory_allocated()
    avg_mem = sum(mem_samples) / len(mem_samples)

    extreme_results[label] = dict(
        init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
        step_ms=train_ms / N_STEPS_5, train_ms=train_ms,
        train_loss=loss.item(),
        peak_mem_mib=peak_mem / 1024**2,
        avg_mem_mib=avg_mem / 1024**2,
    )
    print(f"  [{label:12s}] params:{n_params:6d} | flops:{fmt_flops(flops)} | fwd:{fwd_ms:.3f}ms | "
          f"step:{train_ms/N_STEPS_5:.3f}ms | "
          f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | "
          f"loss:{loss.item():.4f}")
    del model, optimizer
    torch.cuda.empty_cache()

# -- KAN baseline for extreme --
torch.manual_seed(0); warmup_cuda()
model, init_ms = timed_init(lambda: KAN([100, 100], device=device, seed=0))
n_params = count_params(model)
flops = count_flops(model, x_synth)
fwd_ms = bench_forward(model, x_synth, n_warmup=5, n_iter=20)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()
for _ in range(3):
    optimizer.zero_grad(); loss_fn_5(model(x_synth), y_synth).backward(); optimizer.step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mem_samples = []
t0 = time.perf_counter()
for _ in range(N_STEPS_5):
    optimizer.zero_grad()
    loss = loss_fn_5(model(x_synth), y_synth)
    loss.backward(); optimizer.step()
    mem_samples.append(torch.cuda.memory_allocated())
torch.cuda.synchronize()
train_ms = (time.perf_counter() - t0) * 1000
peak_mem = torch.cuda.max_memory_allocated()
avg_mem = sum(mem_samples) / len(mem_samples)
extreme_results["kan"] = dict(
    init_ms=init_ms, n_params=n_params, flops=flops, forward_ms=fwd_ms,
    step_ms=train_ms / N_STEPS_5, train_ms=train_ms, train_loss=loss.item(),
    peak_mem_mib=peak_mem / 1024**2, avg_mem_mib=avg_mem / 1024**2,
)
print(f"  [{'kan':12s}] params:{n_params:6d} | flops:{fmt_flops(flops)} | fwd:{fwd_ms:.3f}ms | "
      f"step:{train_ms/N_STEPS_5:.3f}ms | "
      f"peak:{fmt_mem(peak_mem)}MiB avg:{fmt_mem(avg_mem)}MiB | "
      f"loss:{loss.item():.4f}")
del model, optimizer; torch.cuda.empty_cache()

print()


# ══════════════════════════════════════════════════════════════════════════════
# Write bench_cu.md
# ══════════════════════════════════════════════════════════════════════════════

print("Writing bench_cu.md ...")
gpu_name = torch.cuda.get_device_name(0)

# Speedup helper: >1 means faster than exact_pz
def spd(res, v, key, base="exact_pz"):
    return res[base][key] / res[v][key]

QKAN_LABELS = [l for l, _, _ in QKAN_VARIANTS]

md = f"""# cuTile Solver Benchmark

**GPU**: {gpu_name}
**PyTorch**: {torch.__version__}
**CUDA**: {torch.version.cuda}

---

## Benchmark 1: README Function Fitting

**Model**: `QKAN([1, 1], reps=3)` with trainable pre/post activations
**Data**: 1000 train / 1000 test, 1D, function `sin(20x)/(20x)`
**Training**: Adam lr=1e-3, {N_STEPS_1} steps

| Variant | Ansatz | Params | FLOPs | Init | Forward | Fwd vs exact_pz | Train Step | Step vs exact_pz | {N_STEPS_1} Steps | Peak Mem | Avg Mem | Test Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
for label in QKAN_LABELS + ["kan"]:
    r = readme_results[label]
    ans = label.split("_")[1] if "_" in label else "bspline"
    sf = f"{spd(readme_results, label, 'forward_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    ss = f"{spd(readme_results, label, 'step_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    md += f"| {label} | {ans} | {r['n_params']} | {fmt_flops(r['flops'])} | {r['init_ms']:.1f} ms | {r['forward_ms']:.3f} ms | {sf} | {r['step_ms']:.3f} ms | {ss} | {r['train_ms']:.1f} ms | {r['peak_mem_mib']:.1f} MiB | {r['avg_mem_mib']:.1f} MiB | {r['test_loss']:.4f} |\n"

md += f"""
## Benchmark 2: HQKAN CIFAR-100

**HQKAN-44**: `CNet -> Linear(256, 32) -> QKAN([32, 28]) -> Linear(28, 100)` ([notebook](docs/examples/hqkan_cifar100.ipynb))
**Data**: CIFAR-100, batch size 1000
**Training**: Adam lr=1e-3, {N_EPOCHS_2} epochs

| Variant | Ansatz | Params | FLOPs | Init | Forward | Fwd vs exact_pz | Train Step | Step vs exact_pz | {N_EPOCHS_2}ep Time | Peak Mem | Avg Mem | Test Loss | Top-1 | Top-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
for label in QKAN_LABELS + ["kan"]:
    r = cifar_results[label]
    ans = label.split("_")[1] if "_" in label else "bspline"
    sf = f"{spd(cifar_results, label, 'forward_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    ss = f"{spd(cifar_results, label, 'step_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    md += f"| {label} | {ans} | {r['n_params']} | {fmt_flops(r['flops'])} | {r['init_ms']:.1f} ms | {r['forward_ms']:.3f} ms | {sf} | {r['step_ms']:.3f} ms | {ss} | {r['train_ms']/1000:.1f} s | {r['peak_mem_mib']:.1f} MiB | {r['avg_mem_mib']:.1f} MiB | {r['test_loss']:.3f} | {r['test_acc']:.1%} | {r['test_top5']:.1%} |\n"

md += f"""
## Benchmark 3: HQKANsformer vs MLP GPT-2

**HQKANsformer**: GPT-2 (12L, 12H, 768E) with `Linear(768,10) -> QKAN([10,10], reps=1) -> Linear(10,768)` replacing MLP
**MLP GPT-2**: Standard GPT-2 with `Linear(768,3072) -> GELU -> Linear(3072,768)` MLP
**Triton MLP GPT-2**: Same as MLP but with fused Triton `bias+GELU` kernel (avoids intermediate materialization)
**Data**: TinyShakespeare (char-level), batch size {BATCH_GPT}, block size {BLOCK_SIZE}
**Training**: AdamW lr={LR_GPT}, betas=(0.9,0.95), weight_decay=0.1, grad_clip=1.0, {N_ITERS_3} iters

| Variant | Ansatz | Params | FLOPs | Init | Forward | Fwd vs exact_pz | Train Step | Step vs exact_pz | {N_ITERS_3} Iters | Peak Mem | Avg Mem | Final Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
for label in QKAN_LABELS + ["kan", "triton_mlp", "mlp"]:
    r = gpt_results[label]
    ans = label.split("_")[1] if "_" in label else ("bspline" if label == "kan" else "—")
    sf = f"{spd(gpt_results, label, 'forward_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    ss = f"{spd(gpt_results, label, 'step_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    md += f"| {label} | {ans} | {r['n_params']/1e6:.2f}M | {fmt_flops(r['flops'])} | {r['init_ms']:.1f} ms | {r['forward_ms']:.3f} ms | {sf} | {r['step_ms']:.3f} ms | {ss} | {r['train_ms']/1000:.1f} s | {r['peak_mem_mib']:.1f} MiB | {r['avg_mem_mib']:.1f} MiB | {r['final_loss']:.3f} |\n"

# Benchmark 4 table
md += f"""
## Benchmark 4: HQKANsformer vs MLP GPT-2 on WebText (batch={BATCH_GPT_4})

**HQKANsformer**: GPT-2 (12L, 12H, 768E) with `Linear(768,10) -> QKAN([10,10], reps=1) -> Linear(10,768)` replacing MLP
**MLP GPT-2**: Standard GPT-2 with `Linear(768,3072) -> GELU -> Linear(3072,768)` MLP
**Triton MLP GPT-2**: Same as MLP but with fused Triton `bias+GELU` kernel (avoids intermediate materialization)
**Data**: WebText (GPT-2 tokenizer, vocab_size=50304), batch size {BATCH_GPT_4}, block size {BLOCK_SIZE}
**Training**: AdamW lr={LR_GPT}, betas=(0.9,0.95), weight_decay=0.1, grad_clip=1.0, {N_ITERS_4} iters

| Variant | Ansatz | Params | FLOPs | Init | Forward | Fwd vs exact_pz | Train Step | Step vs exact_pz | {N_ITERS_4} Iters | Peak Mem | Avg Mem | Final Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
for label in QKAN_LABELS + ["kan", "triton_mlp", "mlp"]:
    r = gpt_b10_results[label]
    ans = label.split("_")[1] if "_" in label else ("bspline" if label == "kan" else "—")
    sf = f"{spd(gpt_b10_results, label, 'forward_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    ss = f"{spd(gpt_b10_results, label, 'step_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    md += f"| {label} | {ans} | {r['n_params']/1e6:.2f}M | {fmt_flops(r['flops'])} | {r['init_ms']:.1f} ms | {r['forward_ms']:.3f} ms | {sf} | {r['step_ms']:.3f} ms | {ss} | {r['train_ms']/1000:.1f} s | {r['peak_mem_mib']:.1f} MiB | {r['avg_mem_mib']:.1f} MiB | {r['final_loss']:.3f} |\n"

# Benchmark 5 table
md += f"""
## Benchmark 5: Extreme Synthetic QKAN

**Model**: `QKAN([100, 100], reps=3)`
**Data**: Random 1000x100 input/output
**Training**: Adam lr=1e-3, {N_STEPS_5} steps

| Variant | Ansatz | Params | FLOPs | Init | Forward | Fwd vs exact_pz | Train Step | Step vs exact_pz | {N_STEPS_5} Steps | Peak Mem | Avg Mem | Train Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
for label in QKAN_LABELS + ["kan"]:
    r = extreme_results[label]
    ans = label.split("_")[1] if "_" in label else "bspline"
    sf = f"{spd(extreme_results, label, 'forward_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    ss = f"{spd(extreme_results, label, 'step_ms'):.2f}x" if label != "exact_pz" else "1.00x"
    md += f"| {label} | {ans} | {r['n_params']} | {fmt_flops(r['flops'])} | {r['init_ms']:.1f} ms | {r['forward_ms']:.3f} ms | {sf} | {r['step_ms']:.3f} ms | {ss} | {r['train_ms']:.1f} ms | {r['peak_mem_mib']:.1f} MiB | {r['avg_mem_mib']:.1f} MiB | {r['train_loss']:.4f} |\n"

# Summary
best_q2 = min(QKAN_LABELS, key=lambda v: cifar_results[v]["step_ms"])
best_q3 = min(QKAN_LABELS, key=lambda v: gpt_results[v]["step_ms"])
bq2 = cifar_results[best_q2]
bq3, ml3 = gpt_results[best_q3], gpt_results["mlp"]
tm3 = gpt_results["triton_mlp"]

# Flash vs cuTile comparison
flash_labels = [l for l in QKAN_LABELS if l.startswith("flash_")]
cutile_labels = [l for l in QKAN_LABELS if l.startswith("cutile_")]

md += f"""
## Summary

### CIFAR-100
- **Best QKAN**: `{best_q2}` — {bq2['step_ms']:.3f} ms/step, top-1 {bq2['test_acc']:.1%} / top-5 {bq2['test_top5']:.1%}

### GPT-2 (TinyShakespeare)
- **Best QKAN**: `{best_q3}` — {bq3['step_ms']:.3f} ms/step vs MLP {ml3['step_ms']:.3f} ms/step (QKAN is **{ml3['step_ms']/bq3['step_ms']:.2f}x faster**)
- **Triton MLP**: {tm3['step_ms']:.3f} ms/step (**{ml3['step_ms']/tm3['step_ms']:.2f}x** vs standard MLP)
- **Loss**: QKAN {bq3['final_loss']:.3f} vs Triton MLP {tm3['final_loss']:.3f} vs MLP {ml3['final_loss']:.3f}
- **Parameters**: QKAN {bq3['n_params']/1e6:.2f}M vs MLP {ml3['n_params']/1e6:.2f}M (MLP uses **{ml3['n_params']/bq3['n_params']:.1f}x** more)

### Flash (Triton) vs cuTile Comparison
"""

if cutile_labels:
    for fl, cl in zip(flash_labels, cutile_labels):
        ansatz_name = fl.split("_")[1]
        for bench_name, bench_res in [("B1 Fn Fit", readme_results), ("B2 CIFAR", cifar_results),
                                       ("B3 GPT", gpt_results), ("B4 GPT-WebText", gpt_b10_results),
                                       ("B5 Extreme", extreme_results)]:
            if fl in bench_res and cl in bench_res:
                fr, cr = bench_res[fl], bench_res[cl]
                fwd_ratio = fr["forward_ms"] / cr["forward_ms"] if cr["forward_ms"] > 0 else float("inf")
                step_ratio = fr["step_ms"] / cr["step_ms"] if cr["step_ms"] > 0 else float("inf")
                md += f"- **{bench_name} ({ansatz_name})**: flash fwd={fr['forward_ms']:.3f}ms, cutile fwd={cr['forward_ms']:.3f}ms (**{fwd_ratio:.2f}x**) | flash step={fr['step_ms']:.3f}ms, cutile step={cr['step_ms']:.3f}ms (**{step_ratio:.2f}x**)\n"
else:
    md += "*cuTile was not available on this system (requires CUDA driver >= 13.0).*\n"

md += f"""
## Notes

- **cutile** uses fused cuTile (NVIDIA Tile Language) kernels for both forward and backward passes.
- **cutn_real** fuses X@RY(theta)@Z into a single gate and contracts the entire circuit
  as one tensor network with a cached optimal contraction path (via opt_einsum).
- **triton_mlp** uses a fused Triton kernel for `bias + GELU(tanh approx)`, eliminating one
  intermediate materialization per block compared to the standard MLP.
- **flash** uses fused Triton kernels for both forward and backward passes.
- **exact** is the baseline PyTorch implementation using sequential `einsum` calls.
- **pz** = `pz_encoding` ansatz; **real** = `real` ansatz. cutn supports pz, rpz, and real.
- CIFAR-100 setup follows [hqkan_cifar100.ipynb](docs/examples/hqkan_cifar100.ipynb).
- GPT-2 Benchmark 3 follows [gqkan_gpt.ipynb](docs/examples/gqkan_gpt.ipynb) but uses TinyShakespeare (char-level).
- GPT-2 Benchmark 4 uses WebText (GPT-2 tokenizer) with the same HQKAN architecture as Benchmark 3.
"""

with open("bench_cu.md", "w") as f:
    f.write(md)

print("Done! Results saved to bench_cu.md")
