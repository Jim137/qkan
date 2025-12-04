# Quantum-inspired Kolmogorov-Arnold Network (QKAN)

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

## Installation

You can install QKAN using pip:

```bash
pip install qkan
```

If you want to install the latest development version, you can use:

```bash
pip install git+https://github.com/Jim137/qkan.git
```

To install QKAN from source, you can use the following command:

```bash
git clone https://github.com/Jim137/qkan.git && cd qkan
pip install -e .
```

It is recommended to use a virtual environment to avoid conflicts with other packages.

```bash
python -m venv qkan-env
source qkan-env/bin/activate  # On Windows: qkan-env\Scripts\activate
pip install qkan
```

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

## Contributing

We are very welcome to all kinds of contributions, including but not limited to bug reports, documentation improvements, and code contributions.

To start contributing, please fork the repository and create a new branch for your feature or bug fix. Then, submit a pull request with a clear description of your changes.

In your environment, you can install the development dependencies with:

```bash
pip install .[dev] # install development dependencies
pip install .[doc] # install documentation dependencies
pip install .[all] # install all optional dependencies
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
