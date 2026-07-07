# Eval guidance for qkan-guide

Developer guidance for generating and refining `evals.json`. This outranks
generated defaults during eval generation and refinement.

## Questions

- How do I install QKAN for GPU training and confirm it works?
- Train and plot a minimal QKAN to verify my setup.
- Which solver should I use for single-GPU training, and which when a layer
  is near OOM?
- How do I run a trained QKAN on real IBM Quantum hardware with the
  best-calibrated qubits?
- How do I run QKAN inference on an AWS Braket QPU, and what must I watch
  out for?
- How do I evaluate the same small circuit at many parameter sets in one QPU job?
- My hardware expectation values are noisy — what error mitigation does QKAN
  provide?
- (negative) Unrelated creative or general-programming requests.
- (negative) Near-miss prompts that mention PyTorch/CUDA installs but are not
  about QKAN, to guard against over-routing.

## Behaviors

- The agent read skills/qkan-guide/SKILL.md before acting.
- The agent recommended the documented install extra for the scenario
  (`qkan`, `qkan[gpu]`, `qkan[cute]`, `qkan[real-device]`).
- The agent recommended the documented solver for the scenario (`exact`,
  `flash`, `cute`/`cutile`, `cutn`, `qiskit`, `cudaq`).
- For batching tasks the agent recommended the packing toolkit
  (`pack_circuit`, variational batch / shot reduction patterns).
- For QPU tasks the agent followed the documented workflow: train locally,
  transfer with `initialize_from_another_model`, infer with
  `fast_measure=False`, pack with `parallel_qubits`, and pin qubits with
  `initial_layout="auto"` (qiskit) — plus the cudaq global-target/cost
  caution for Braket.

## Notes

- qkan-guide is a documentation/onboarding skill with **no executable
  script**, so `expected_script` is `null` for every case and the agent
  should never run a script.
- Ground truth is intentionally derived from SKILL.md content (install
  matrix, solver table, QPU workflow, mitigation table), so cases remain
  answerable in an isolated workspace without staging the repo's docs.
- Negative cases set `expected_skill: null` and `should_trigger: false`.
