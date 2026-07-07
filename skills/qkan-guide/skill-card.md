## Description: <br>
Onboarding guide for the qkan package (Quantum-inspired Kolmogorov-Arnold Networks): installing qkan, training a first QKAN, choosing solver backends, mixed precision, circuit packing, and real QPU deployment. Not for generic PyTorch/CUDA setup. <br>

This skill is ready for commercial/non-commercial use under Apache-2.0. <br>

## Owner
Jiun-Cheng Jiang (Jim137/qkan) <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and researchers use this skill to onboard onto QKAN (Quantum-inspired Kolmogorov-Arnold Networks): installing the right extras for their hardware, training and plotting a first model, selecting among the eight solver backends, enabling BF16/FP8 mixed precision, packing circuit copies onto calibration-aware qubit tiles, and deploying inference to IBM Quantum or AWS Braket hardware with calibration-aware qubit layouts and error mitigation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>
Risk: The `cudaq` solver's target selection is process-global; a leaked hardware target plus cloud credentials can submit paid QPU tasks. <br>
Mitigation: The skill instructs pinning a simulator target and confirming cost with the user before any hardware submission. <br>

## Reference(s): <br>
- [QKAN Documentation](https://qkan.jimq.cc/) <br>
- [Paper: Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks](https://arxiv.org/abs/2509.14026) <br>
- [Repository](https://github.com/Jim137/qkan) <br>

## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Code, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` (subagent verification) <br>

## Evaluation Tasks: <br>
Evaluated against 9 evaluation tasks (6 positive skill-activation tasks, 3 negative tasks); see `BENCHMARK.md` for results. <br>

## Evaluation Metrics Used: <br>
- Correctness: the agent's answer matches the ground truth derived from SKILL.md and the repo docs. <br>
- Discoverability: the agent reads the skill when relevant and skips it when irrelevant. <br>
- Behavior check: expected behavior steps followed, including safety expectations (e.g. `fast_measure=False` on hardware, cudaq global-target caution). <br>

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
Users are responsible for validating guidance before running paid quantum-hardware workloads; the skill explicitly requires user confirmation before hardware submissions. Report issues at https://github.com/Jim137/qkan/issues. <br>
