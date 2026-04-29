# QKAN: Quantum-inspired Kolmogorov-Arnold Network

<div align='center'>
    <a>"Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks"</a>

</div>
<div align='center'>
    <a href='https://scholar.google.com/citations?user=W_I27S8AAAAJ' target='_blank'>Jiun-Cheng Jiang</a><sup>1</sup> 
    <a href='https://scholar.google.com/citations?user=1u3Kvh8AAAAJ' target='_blank'>Morris Yu-Chao Huang</a><sup>2</sup> 
    <a href='https://scholar.google.com/citations?user=LE3ctn0AAAAJ' target='_blank'>Tianlong Chen</a><sup>2</sup> 
    <a href='https://scholar.google.com/citations?user=PMnNYPcAAAAJ' target='_blank'>Hsi-Sheng Goan</a><sup>1</sup> 

</div>
<div align='center'>
    <sup>1</sup>National Taiwan University  <sup>2</sup>UNC Chapel Hill 
</div>

<div align='center'>

[![page](https://img.shields.io/badge/Project%20Page-5745BB?logo=google-chrome&logoColor=white)](https://jim137.github.io/qkan/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.14026-b31b1b.svg)](https://arxiv.org/abs/2509.14026)
[![pypi](https://img.shields.io/pypi/v/qkan)](https://pypi.org/project/qkan/)
![License](https://img.shields.io/github/license/Jim137/qkan)
[![DOI](https://zenodo.org/badge/1006914967.svg)](https://doi.org/10.5281/zenodo.17437425)

</div>

<!-- [![build](https://github.com/Jim137/qkan/actions/workflows/publish.yml/badge.svg)](https://github.com/Jim137/qkan/actions/workflows/publish.yml)
[![lint](https://github.com/Jim137/qkan/actions/workflows/lint.yml/badge.svg)](https://github.com/Jim137/qkan/actions/workflows/lint.yml) -->

This is the official repository for the paper:
**["Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks"](https://arxiv.org/abs/2509.14026)**

📖 Documentation: [https://qkan.jimq.cc/](https://qkan.jimq.cc/)

We provide a PyTorch implementation of QKAN with:

- Pre- and post-activation processing support
- Grouped QVAFs for efficient training
- Plot the nodes and pruning unnecessary nodes
- Layer extension for more complex features
- and more ...

A basic PennyLane version of the quantum circuit is also included for demonstration, but not optimized for performance.

2026-03: Released v0.2.0 with a more efficient quantum circuit implementation—using cuQuantum for the `cutn` solver and Triton for the `flash` solver—which significantly speeds up the activation function.

## Installation

You can install QKAN using pip:

```bash
pip install qkan
```

If you want to install the latest development version, you can use:

```bash
pip install git+https://github.com/Jim137/qkan.git
```

To use the GPU-optimized solvers (including `flash` and `cutn` solver), you can install with the `gpu` extra:

```bash
pip install qkan[gpu]
```

To use the CuTe DSL solver (`solver="cute"`) with pre-built CUDA kernels:

```bash
# Pre-built wheel (recommended — no compilation needed)
pip install qkan[cute] --extra-index-url https://qkan.jimq.cc/whl/

# Or compile locally (auto-downloads CUTLASS headers if needed)
pip install --no-build-isolation qkan[cute]
```

<!-- To install QKAN from source, you can use the following command:

```bash
git clone https://github.com/Jim137/qkan.git && cd qkan
pip install -e .
```

It is recommended to use a virtual environment to avoid conflicts with other packages.

```bash
python -m venv qkan-env
source qkan-env/bin/activate  # On Windows: qkan-env\Scripts\activate
pip install qkan
``` -->

## Quick Start

Here's a minimal working example for function fitting using QKAN:

```python
import torch

from qkan import QKAN, create_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

f = lambda x: torch.sin(20*x)/x/20 # J_0(20x)
dataset = create_dataset(f, n_var=1, ranges=[0,1], device=device, train_num=1000, test_num=1000, seed=0)

qkan = QKAN(
    [1, 1], 
    reps=3, 
    device=device, 
    seed=0,
    preact_trainable=True, 
    postact_weight_trainable=True,
    postact_bias_trainable=True, 
    ba_trainable=True,
    save_act=True, # enable to plot from saved activation
)

optimizer = torch.optim.LBFGS(qkan.parameters(), lr=5e-2)

qkan.train_(
    dataset,
    steps=100,
    optimizer=optimizer,
    reg_metric="edge_forward_dr_n",
)

qkan.plot(from_acts=True, metric=None)
```

You can find more examples in the [examples](https://jim137.github.io/qkan/examples) for different tasks, such as function fitting, classification, and generative modeling.

## Solver Guiding

| Case                                                                      | Device       | Recommended solver    | Why                                                             | Notes                                                                            |
| ------------------------------------------------------------------------- | ------------ | --------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Small models, CPU runs, debugging, or you want a trusted baseline         | CPU (or GPU) | `exact` *(default)*   | Simple + “reference” behavior                                   | First run may include one-time init overhead—do a warmup step before timing.     |
| Most training workloads (medium → large models) / inference               | GPU          | `flash`               | Best overall speed / memory tradeoff, widest compatibility      | Good first choice for practical GPU training.                                    |
| BF16/FP8 maximum throughput with CUTLASS                                  | GPU          | `cute`                | CuTe smem trig caching + __sincosf + FP8 prescaled states      | **1.95x** on GPT-2 HQKANsformer. Requires CUTLASS headers.                      |
| BF16/FP8 mixed-precision training (alternative)                           | GPU          | `cutile`              | cuTile fused kernels with BF16/FP8 + coalesced state layout     | Best with `real` ansatz. Requires CUDA Toolkit 13.1+.                            |
| Extremely large / memory-bound runs (near OOM, very large layers/batches) | GPU (or CPU) | `cutn`                | Best scaling and peak-memory reduction in the extreme benchmark | Use when size/memory dominates. Or CPU case better than `exact`. |

**Ansatz choice (`pz` vs `real`)**
- **Default: `pz`** — most reliable quality across tasks.
- **`real`** can be faster/smaller, but may **hurt accuracy/convergence** on some workloads—only use if you validate it on your task.

See [#8](https://github.com/Jim137/qkan/issues/8) for more discussion on solver choices and tradeoffs.

## Mixed Precision

The `flash`, `cute`, and `cutile` solvers support BF16 and FP8 mixed-precision via the `c_dtype` parameter:

```python
# BF16 (recommended for training)
qkan = QKAN([10, 10], solver="cute", c_dtype=torch.bfloat16, p_dtype=torch.bfloat16, device="cuda")

# FP8 prescaled state checkpoints (max backward throughput)
qkan = QKAN([10, 10], solver="cute", c_dtype=torch.float8_e4m3fn, p_dtype=torch.bfloat16, device="cuda")
```

- `c_dtype` controls the **compute dtype** for quantum simulation kernels (state vectors, trig ops).
- `p_dtype` controls the **parameter storage dtype** (theta, preacts).
- BF16 is the sweet spot: **2.3-2.5x faster training, 45% less peak memory**, with identical convergence.
- FP8 (`torch.float8_e4m3fn`) provides additional memory savings for state checkpoints via prescaled storage (prescale=224, leveraging unitarity-bounded [-1,1] states).
- All ansatzes (`pz`, `rpz`, `real`) are supported.

See [#12](https://github.com/Jim137/qkan/issues/12) for full benchmarks (GPT-2 HQKANsformer, isolated kernel timings, and dtype performance matrix).

## Inference Acceleration

Wrap any QKAN-based model with `compile_inference` for 2–3× faster inference via CUDA graph replay:

```python
import qkan

model = qkan.compile_inference(model).eval()
with torch.no_grad():
    y = model(x)
```

For transformer stacks, wrap each MLP block instead of the full model:

```python
qkan.graph_submodules(transformer, sample, predicate=lambda m: isinstance(m, MyMLP))
```

## Contributing

We are very welcome to all kinds of contributions, including but not limited to bug reports, documentation improvements, and code contributions.

To start contributing, please fork the repository and create a new branch for your feature or bug fix. Then, submit a pull request with a clear description of your changes.

In your environment, you can install the development dependencies with:

```bash
# clone your forked repository and navigate to the project directory
# for example `git clone https://github.com/Jim137/qkan.git && cd qkan`

pip install -e .[dev] # install development dependencies
pip install -e .[doc] # install documentation dependencies
pip install -e .[all] # install all optional dependencies
```

## Citation

```bibtex
@article{jiang2025qkan,
  title={Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks},
  author={Jiang, Jiun-Cheng and Huang, Morris Yu-Chao and Chen, Tianlong and Goan, Hsi-Sheng},
  journal={arXiv preprint arXiv:2509.14026},
  year={2025},
  url={https://arxiv.org/abs/2509.14026}
}
@misc{jiang2025qkan_software,
  title={QKAN: Quantum-inspired Kolmogorov-Arnold Network},
  author={Jiang, Jiun-Cheng},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17437425},
  url={https://doi.org/10.5281/zenodo.17437425}
}
```
