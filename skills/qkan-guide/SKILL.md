---
name: "qkan-guide"
title: "QKAN"
description: "Onboarding guide for the qkan package (Quantum-inspired Kolmogorov-Arnold Networks): installing qkan, training a first QKAN, choosing solver backends, mixed precision, circuit packing, and real QPU deployment. Not for generic PyTorch/CUDA setup."
version: "1.0.0"
author: "Jiun-Cheng Jiang <jcjiang@phys.ntu.edu.tw>"
tags: [qkan, kolmogorov-arnold-networks, quantum-machine-learning, pytorch, onboarding, getting-started]
tools: [Read, Glob, Grep]
license: "Apache-2.0"
compatibility: "Python 3.10+, PyTorch 2.0+"
metadata:
    author: "Jiun-Cheng Jiang <jcjiang@phys.ntu.edu.tw>"
    tags:
        - qkan
        - kolmogorov-arnold-networks
        - quantum-machine-learning
        - pytorch
        - onboarding
        - getting-started
    languages:
        - python
    domain: "quantum-machine-learning"
---

## QKAN Getting Started Guide

You are a QKAN expert assistant. Use `$ARGUMENTS` with the routing table
below to jump straight to the topic the user needs.

## Purpose

Guide users through QKAN (Quantum-inspired Kolmogorov-Arnold Network,
arXiv:2509.14026): installation, training a first model, choosing among the
eight solver backends, mixed-precision and memory tuning, packing circuits
onto calibrated qubit tiles, running inference on real quantum hardware
(IBM via Qiskit, AWS Braket via CUDA-Q), and exploring the example
applications.

## Prerequisites

- Python 3.10+ and PyTorch 2.0+ (the only hard dependencies, plus tqdm and
  matplotlib)
- NVIDIA GPU (optional; the default `exact` solver runs on CPU)
- For GPU-fused solvers: `pip install qkan[gpu]` (Triton, cuTile, cuQuantum,
  CuTe — CUDA 13 family by default, `-cu12` variants available)
- For real QPU inference: `pip install qkan[real-device]` plus provider
  credentials (IBM Quantum or AWS)

## Instructions

- Invoke with `/qkan-guide [argument]`
- If no argument is given, display the full onboarding menu and ask what
  the user wants to explore
- Pass an argument from the routing table below to jump directly to that topic
- Read local QKAN documentation files to answer questions accurately

## References

| Section | Doc file |
| --- | --- |
| Install | `docs/index.rst` (Installation), `README.md` |
| Test Program | `docs/index.rst` (Quick Start), `docs/intro/qkan.ipynb` |
| Solvers | `docs/solver_guide.rst` (Solver Overview, Ansatz Choice) |
| Precision & memory | `docs/solver_guide.rst` (Mixed Precision, Performance Tuning), `docs/optim_guide.rst` |
| Packing | `docs/packing_guide.rst` |
| QPU | `docs/solver_guide.rst` (Real Quantum Device Deployment, Error Mitigation) |
| Fast inference | `docs/graph_guide.rst` |
| Examples | `docs/examples.rst`, `docs/examples/*.ipynb`, `docs/intro.rst`, `docs/applications.rst` |

## Routing by Argument

| Argument | Action |
|---|---|
| `install` | Walk through installation (see Install section) |
| `test-program` | Train a one-neuron QKAN on a Bessel function to verify the install |
| `solvers` | Choose a solver backend (see Solvers section) |
| `precision` | BF16/FP8 mixed precision and memory tuning (see Precision section) |
| `packing` | Pack circuit copies onto calibrated qubit tiles (see Packing section) |
| `qpu` | Run QKAN inference on real quantum hardware (see QPU section) |
| `examples` | Explore the tutorial and application notebooks (see Examples section) |
| _(none)_ | Print the full menu below and ask what they'd like to explore |

---

## Full Menu (no argument)

Present this when invoked with no argument

```text
QKAN Getting Started

QKAN is a PyTorch implementation of Kolmogorov-Arnold Networks whose learnable
activations are single-qubit data re-uploading circuits (DARUAN).
Paper https://arxiv.org/abs/2509.14026 - Docs https://qkan.jimq.cc/

Choose a topic
  /qkan-guide install        Install QKAN (CPU, GPU-fused, or QPU extras)
  /qkan-guide test-program   Train your first QKAN and plot it
  /qkan-guide solvers        Pick the right solver backend
  /qkan-guide precision      BF16/FP8 and memory tuning
  /qkan-guide packing        Calibration-aware circuit packing
  /qkan-guide qpu            Run inference on IBM or AWS Braket hardware
  /qkan-guide examples       Tutorials and applications (MNIST, GPT, HQKAN)
```

---

## Install

Instructions

- Default to `pip install qkan` (pure Python, torch>=2.0, Python 3.10+); the
  default `exact` solver needs no GPU.
- If the user has an NVIDIA GPU and wants speed, add the GPU extras:
  `pip install qkan[gpu]` — pulls Triton (`flash`), cuda-tile (`cutile`),
  cuQuantum (`cutn`), and the CuTe runtime deps, all in the CUDA 13 family.
  On a CUDA 12 stack use the family-pinned extras instead:
  `pip install qkan[flash,cutile,cutn-cu12,cute-cu12]`.
- The CuTe solver ships pre-built wheels per CUDA family:
  `pip install qkan[cute] --extra-index-url https://qkan.jimq.cc/whl/cu13/`
  (or `.../whl/cu12/`). Compiling locally instead requires a CUDA toolkit
  matching torch: `pip install --no-build-isolation qkan[cute]` — CUTLASS
  headers are auto-downloaded when absent (env knobs: `CUTLASS_PATH`,
  `QKAN_CUDA_ARCHS="80;90;120"`, `QKAN_FORCE_BUILD=TRUE`,
  `QKAN_NO_CUTLASS_DOWNLOAD=1`).
- Real-hardware extras: `pip install qkan[real-device]` = qiskit +
  qiskit-ibm-runtime + cuda-quantum.
- Contributors: `git clone https://github.com/Jim137/qkan && pip install -e
  .[dev]`; `make lint` (ruff + mypy) is the CI gate, and `pytest tests/`
  runs the packing/layout suite (qiskit and cudaq cases skip automatically
  when those extras are missing).
- Always validate the install:

  ```bash
  python -c "import qkan; print(qkan.__version__)"
  ```

  then run the Quick Start example from the Test Program section and confirm
  the training loss decreases.

---

## Test Program

The canonical smoke test fits the Bessel-like function J_0(20x) with a
one-input, one-output QKAN (README Quick Start; CPU-safe):

```python
import torch
from qkan import QKAN, create_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

f = lambda x: torch.sin(20 * x) / x / 20  # J_0(20x)
dataset = create_dataset(f, n_var=1, ranges=[0, 1], device=device, seed=0)

model = QKAN(
    [1, 1],                        # layer widths
    reps=3,                        # data re-uploading repetitions
    device=device,
    seed=0,
    preact_trainable=True,
    postact_weight_trainable=True,
    postact_bias_trainable=True,
    ba_trainable=True,
    save_act=True,                 # needed for plotting and pruning
)

optimizer = torch.optim.LBFGS(model.parameters(), lr=5e-2)
model.train_(dataset, steps=100, optimizer=optimizer,
             reg_metric="edge_forward_dr_n")

model.plot(from_acts=True, metric=None)
```

Key concepts to explain

- `QKAN(width, reps=...)` - a PyTorch `nn.Module`; each edge activation is a
  single-qubit data re-uploading circuit (DARUAN) with learnable angles
- `create_dataset(f, n_var, ranges, ...)` - returns a dict with
  `train_input/train_label/test_input/test_label`
- `model.train_(dataset, ...)` - built-in loop; defaults to Adam(lr=5e-4)
  and MSE loss; LBFGS gets an automatic closure; returns
  `{"train_loss": [...], "test_loss": [...], "reg": [...]}`
- `save_act=True` is required for regularization (`lamb>0`), `plot(from_acts=True)`,
  `attribute()`, and pruning (`prune`, `prune_node`, `prune_edge`, `prune_input`)
- Scaling up: `refine(new_reps)` grows the circuit depth of a trained model;
  QKAN-aware optimizers (`qkan.optim.QKANAdamMini`, `QKANBeliefMini`,
  L-BFGS finisher schedules) are covered in `docs/optim_guide.rst`;
  CUDA-graph inference capture (~2-3x at small batch) in `docs/graph_guide.rst`

---

## Solvers

The `solver` kwarg of `QKAN`/`QKANLayer` selects the backend; per-solver
options go in `solver_kwargs={...}`. Details:
`docs/solver_guide.rst` (Solver Overview).

| Solver | Runs on | Description | Use when |
|---|---|---|---|
| `exact` (default) | CPU/GPU | Pure-PyTorch exact statevector | Default, debugging, CPU-only boxes |
| `flash` | GPU | Triton fused forward+backward | Most GPU training (`qkan[gpu]`) |
| `cute` | GPU | Hand-written CUTLASS CuTe DSL kernels | Max throughput incl. BF16/FP8 (`qkan[cute]`) |
| `cutile` | GPU | cuTile (NVIDIA Tile Language) fused kernels | BF16/FP8 alternative; needs CUDA Toolkit 13.1+ |
| `cutn` | GPU/CPU | Tensor-network contraction (cuQuantum / opt-einsum) | Extremely large layers near OOM |
| `qml` | CPU | PennyLane per-edge loop | Demonstration only (slow) |
| `qiskit` | IBM QPU | Qiskit Runtime EstimatorV2 | Real IBM hardware inference |
| `cudaq` | QPU/GPU | NVIDIA CUDA-Q (GPU simulators, AWS Braket QPUs) | Braket hardware or CUDA-Q simulators |

Notes

- `flash`/`cutile`/`cute` are real-valued kernels; complex `c_dtype` is
  mapped to real automatically. Unsupported ansatzes fall back to `exact`.
- `cutn` accepts alias `"tn"`; falls back to `exact` for `reps > 11`.
- A custom callable can be passed as `solver` (requires `theta_size`).
- Ansatz choice (`ansatz=`): `pz` (default, most reliable), `rpz` (fewer
  parameters, trainable pre-activation bias), `real` (fastest, may cost
  accuracy). See `docs/examples/ansatz_comparison.ipynb`.

---

## Precision

`c_dtype` sets the compute dtype, `p_dtype` the parameter dtype
(`docs/solver_guide.rst`, Mixed Precision):

```python
model = QKAN([10, 10], solver="cute", device="cuda",
             c_dtype=torch.bfloat16, p_dtype=torch.bfloat16)
```

- BF16 on `flash`/`cutile`/`cute`: ~2.3-2.5x faster training, ~45% less peak
  memory
- FP8 (`c_dtype=torch.float8_e4m3fn`) is supported for compute;
  `p_dtype=torch.float8_e4m3fn` is NOT supported - keep parameters in
  float32 or bfloat16

Memory/perf tuning (`docs/solver_guide.rst`, Performance Tuning - all
opt-in, default off):

- `checkpoint_reps=True` - recompute rep states in backward instead of
  saving them (~33% saved-tensor memory for one extra forward)
- Fused Triton epilogue - `layer.set_fused_epilogue(True)` per layer or
  `QKAN_FUSED_EPILOGUE=1` process-wide (CUDA-only fast path)
- BF16 optimizer mini-state - `qkan.optim.QKANBeliefMini` (pure PyTorch)
  or `TritonAdaBelief` (~2x optimizer-state memory reduction; see
  `docs/optim_guide.rst`)

---

## Packing

`qkan.solver` ships a provider-neutral packing toolkit
(`docs/packing_guide.rst`): run `k` independent copies of a small circuit
in parallel on one QPU, placed on disjoint calibration-aware tiles.

```python
from qkan import DeviceProfile, pack_circuit

profile = DeviceProfile.from_qiskit(backend)      # or .from_braket(device)
packed = pack_circuit(backend, circuit, k=8)      # qiskit QuantumCircuit
packed = pack_circuit(profile, kernel, k=8)       # or a plain @cudaq.kernel
```

- One entry point for both stacks: a qiskit `QuantumCircuit` packs against
  a backend (SWAP-free transpilation pinned to the tiles); a plain
  `@cudaq.kernel` packs against a `DeviceProfile` (gates extracted from
  the compiled Quake IR and rebuilt at physical indices).
- Two payoff patterns: the variational batch (k parameter sets per job via
  per-copy parameters / `block_args_batch`, re-bound each step with
  `rebind`) and shot reduction (k identical copies pooled with the
  `"mean"` observable - same statistics at shots/k per job).
- Readout: `packed.observable(obs, tile)` maps block-level observables on
  either stack; `packed.expectation(result, pauli, tile)` reads sampled
  counts; `packed.basis_kernel(pauli)` is the hardware-safe X/Y route.
- Selection helpers `tile_disjoint`, `best_subgraph`, and `rank_qubits`
  are usable standalone; `k=1` packing = best-calibrated single placement.
- Hardware caveat: whether physical indices survive to the QPU depends on
  the provider (Braket/Rigetti's compiler may rewire - results stay
  program-indexed and logically correct; verified on hardware).

---

## QPU

The documented workflow (`docs/solver_guide.rst`, Real Quantum Device
Deployment) is train-locally, infer-on-hardware:

1. Train with `solver="exact"` or `"flash"` locally.
2. Transfer weights to a hardware-backed copy with
   `initialize_from_another_model`.
3. Run inference with `solver="qiskit"` or `"cudaq"` and
   **`fast_measure=False`** - real devices measure Born-rule probabilities
   (|alpha|^2 - |beta|^2), not the quantum-inspired shortcut.

### IBM Quantum (Qiskit)

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.least_busy(operational=True, simulator=False)

hw_model = QKAN([1, 2, 1], reps=3, solver="qiskit", fast_measure=False,
                solver_kwargs={
                    "backend": backend,
                    "shots": 1000,
                    "optimization_level": 3,
                    "parallel_qubits": backend.num_qubits,
                    "initial_layout": "auto",
                })
hw_model.initialize_from_another_model(trained_model)
```

- `parallel_qubits` packs N independent single-qubit circuits into one
  N-qubit job (`"auto"` = `backend.num_qubits`) - N-fold fewer jobs.
- `initial_layout` pins physical qubits: `None` (transpiler default),
  `"auto"` (best-calibrated qubits via `qkan.solver.best_qubits`), or an
  explicit list. Optional thresholds in `solver_kwargs` gate the auto
  selection: `max_readout_error`, `qubit_error_threshold`; when fewer than
  N qubits satisfy a threshold, `best_qubits` raises instead of silently
  using noisier qubits.
- Backends without calibration data fall back to the transpiler default
  with a warning; qiskit-ibm-runtime caches `backend.properties()` -
  refresh before long runs.

### AWS Braket (CUDA-Q)

```python
hw_model = QKAN([1, 2, 1], reps=3, solver="cudaq", fast_measure=False,
                solver_kwargs={
                    "target": "braket",
                    "machine": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
                    "shots": 1000,
                    "parallel_qubits": 8,
                    "initial_layout": "auto",
                })
```

- The CUDA-Q solver supports the same `initial_layout` values (`None`,
  `"auto"`, `list[int]`) on its packed path; the layout is applied by
  construction (idle register qubits get zero-angle padding so compilers
  cannot renumber the indices). Simulator targets ignore it with a warning.
- SAFETY: `cudaq.set_target` is process-global. A leaked `braket` target
  plus AWS credentials submits real paid QPU tasks from any later cudaq
  call. Pin a simulator explicitly (`solver_kwargs={"target": "qpp-cpu"}`)
  when you do not intend to spend money, and confirm cost with the user
  before any hardware submission.
- GPU simulation via the same solver: `target="nvidia"` (single GPU) or
  `"nvidia-mqpu"` (multi-GPU).

### Error mitigation

`solver_kwargs["mitigation"]` works on both hardware solvers
(`docs/solver_guide.rst`, Error Mitigation):

| Option | Meaning | Circuit cost |
|---|---|---|
| `{"clip_expvals": True}` | Clamp <Z> to [-1, 1] | 1x (always safe) |
| `{"n_repeats": 3}` | Run 3x and average (shot noise) | 3x |
| `{"zne": {"scale_factors": [1, 3, 5]}}` | Gate-folding ZNE + Richardson (gate noise) | 3x |
| ZNE + n_repeats=3 | Maximum accuracy, inference only | 9x |

IBM-native knobs are passed directly in `solver_kwargs` (not under
`mitigation`): `resilience_level=2` (Qiskit-native ZNE) and
`twirling={"enable_gates": True, "enable_measure": True}`.

Gradients on hardware use the parameter-shift rule (2 evaluations per
scalar parameter), so training on a QPU works out of the box when the
user explicitly asks for it - default to local training otherwise and
budget the extra circuit evaluations before submitting.

---

## Examples

Tutorials (`docs/intro.rst`): `intro/qml.ipynb` (QML 101),
`intro/kan.ipynb` (KAN theory), `intro/qkan.ipynb` (first QKAN + DARUAN).

Applications (`docs/examples.rst`)

| Notebook | Shows |
|---|---|
| `layer_ext.ipynb` | Extending circuit depth (larger `reps` via weight transfer) for fine-grained DARUAN |
| `mnist.ipynb` | MNIST classification - QKAN as an MLP replacement with fewer parameters |
| `hqkan_cifar100.ipynb` | Hybrid QKAN (FCN encoder/decoder around a QKAN core) on CIFAR-100 |
| `ansatz_comparison.ipynb` | pz vs rpz vs real ansatz trade-offs |
| `gqkan_gpt.ipynb` | "Kansformer" - nanoGPT with MLPs replaced by grouped QKAN |
| `qkan2kan.ipynb` | Transferring trained QKAN weights into a classical B-spline KAN |

`docs/applications.rst` lists published and preprint work built on QKAN;
`datasets/*.sh` fetch MNIST/CIFAR/tinyshakespeare/webtext for the
experiments.

---

## Examples of skill invocations

- `/qkan-guide` — print the onboarding menu and ask which topic to explore.
- `/qkan-guide install` — recommend `pip install qkan` (or `qkan[gpu]` /
  `qkan[cute]` / `qkan[real-device]` by need, minding the cu12/cu13
  family), then validate with the version check and Quick Start.
- `/qkan-guide test-program` — walk through the J_0(20x) Quick Start and
  confirm the loss decreases.
- `/qkan-guide solvers` — recommend `flash` for single-GPU training, `cutn`
  near OOM, `exact` on CPU.
- `/qkan-guide packing` — explain `pack_circuit` on either stack, tile
  selection, and the batch/shot-reduction patterns.
- `/qkan-guide qpu` — walk the train-locally/infer-on-QPU workflow with
  `fast_measure=False`, `parallel_qubits`, `initial_layout="auto"`.
- `/qkan-guide examples` — route to the tutorial or application notebook
  matching the user's goal.

---

## Limitations

- Real-hardware solvers (`qiskit`, `cudaq`) default to inference; training
  on a QPU is supported via the parameter-shift rule when the user
  explicitly requests it, at 2 circuit evaluations per scalar parameter
  per step — budget accordingly
- `flash`/`cutile`/`cute` require an NVIDIA GPU; `cutile` needs CUDA
  Toolkit 13.1+; `cute` needs CUTLASS headers unless the pre-built wheel
  index is used; extras come in cu12/cu13 families
- `plot()` is not supported with `is_map=True` or `is_batchnorm=True`
- Packing requires the block's interaction graph to embed in the coupling
  map without routing; whether physical indices survive to hardware is
  provider-dependent
- The test suite (`pytest tests/`) covers the packing/layout toolkit;
  `make lint` is the CI quality gate

## Troubleshooting

- `ImportError: Triton is required for solver=flash` - `pip install triton`
  (or `qkan[gpu]`)
- `CuTe DSL solver requires CUTLASS headers` - set `CUTLASS_PATH` or use the
  pre-built wheel index `https://qkan.jimq.cc/whl/cu13/` (or `/cu12/`)
- CuTe local build skipped with CUDA-mismatch warning - system `nvcc` must
  match `torch.version.cuda`, or install with `--no-build-isolation`
- RuntimeWarning "Regularization is not supported without saving
  activations" - construct the model with `save_act=True` when passing
  `lamb > 0`
- Hardware expectations look biased - verify `fast_measure=False` and
  consider `initial_layout="auto"` plus `mitigation={"clip_expvals": True}`
- `best_qubits: only m qubits satisfy ...; need n` - relax
  `max_readout_error`/`qubit_error_threshold` or pass `strict=False`
