# Quantum-inspired Kolmogorov-Arnold Network (QKAN)

This is the official repository for the paper:  
**["Quantum Variational Activation Function Empowers Kolmogorov-Arnold Networks"](https://arxiv.org/abs/)**

📖 Documentation: [https://jim137.github.io/qkan/](https://jim137.github.io/qkan/)

We provide a PyTorch implementation of QKAN with:

- Pre- and post-activation processing support
- Grouped QVAFs for efficient training
- Plot the nodes and pruning unnecessary nodes
- Layer extension for more complex architectures
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
    num_qlayers=3, 
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

## Citation

```bibtex
@article{jiang2025qkan,
  title={Quantum variational activation function empowers Kolmogorov-Arnold networks},
  author={Jiang, Jiun-Cheng and Huang, Yu-Chao and Chen, Tianlong and Goan, Hsi-Sheng},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```
