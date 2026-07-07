# Evaluation Report

Evaluation of the `qkan-guide` skill before publication.

This benchmark summarizes a subagent-based verification of the skill: fresh-context
agent runs over the evaluation dataset, independent judges grading each transcript
against ground truth, and adversarial fact-checkers auditing every factual claim in
SKILL.md against the repository. The goal is to document whether the skill is
accurate, discoverable, and effective for agents before it is published for broader
workflow use.

## Evaluation Summary

- Skill: `qkan-guide`
- Evaluation date: 2026-07-07
- Method: Claude Code subagent harness (runner + judge per case, plus 2
  adversarial fact-checkers over SKILL.md content)
- Dataset: 10 evaluation tasks (7 positive, 3 negative)
- Rounds: 2 (initial full run; fix round re-running the affected cases)
- Overall verdict: PASS

## Agents Used

- `claude-code` (fresh-context subagents; every runner saw only the skill
  name + description and the repo, mirroring real skill discovery)

## Metrics Used

- Correctness: the agent's answer matches the ground truth derived from
  SKILL.md and the repo docs (graded by an independent judge per case).
- Discoverability: the agent reads the skill when relevant and skips it when
  irrelevant (`read_skill` self-report cross-checked against answer content).
- Grounding: adversarial fact-checkers verify every factual claim in
  SKILL.md against the repository sources (severity: wrong / misleading /
  minor).
- Behavior check: expected behavior steps followed, including safety
  expectations (`fast_measure=False` on hardware, cudaq global-target/cost
  caution).

## Test Tasks

The benchmark dataset contained 10 evaluation tasks:

- Positive tasks: 7 (install, test-program, solvers, packing, qpu-ibm,
  qpu-braket, mitigation) where the skill was expected to activate.
- Negative tasks: 3 (creative writing, generic JavaScript, PyTorch+CUDA
  install near-miss) where no skill was expected.

## Results

| Dimension | Round 1 | Round 2 (after fixes) |
|---|---:|---:|
| Correctness (positive cases) | 7/7 | 7/7 |
| Discoverability (all cases) | 9/10 | 10/10 |
| Grounding (SKILL.md claims) | 7 discrepancies | 0 open |

Round-1 findings and fixes:

- Discoverability: the PyTorch+CUDA near-miss (`neg-003`) over-routed into
  the skill. Fixed by scoping the frontmatter description to the qkan
  package and adding an explicit "Not for generic PyTorch/CUDA setup"
  negative scope; re-run passed and the positive sentinel still routed.
- Grounding (2 misleading, 5 minor, all fixed in SKILL.md): CUTLASS headers
  are auto-downloaded for local CuTe builds (not a prerequisite); BF16
  optimizer mini-state pairs with `QKANBeliefMini`/`TritonAdaBelief` (not
  `QKANAdamMini`); `QKANMuon` is exported but not covered by
  `docs/optim_guide.rst`; the docs' Solver Overview table omits `cute`, so
  the skill no longer cites it as the "full" table; `layer_ext.ipynb`
  extends depth via weight transfer rather than `refine()`; two
  Troubleshooting entries now quote the actual warning/error strings.

## Publication Recommendation

The skill is suitable to proceed toward publication based on this benchmark.
Skill owners should keep this file with the skill and refresh it when the
evaluation dataset, skill behavior, or target agents materially change.
