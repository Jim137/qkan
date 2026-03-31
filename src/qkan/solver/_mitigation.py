# Copyright (c) 2026, Jiun-Cheng Jiang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Error mitigation utilities for QKAN real-device solvers.

Provides framework-level, backend-agnostic mitigation techniques:
- Zero-Noise Extrapolation (ZNE) via Richardson extrapolation
- Multi-shot averaging (repeated execution)
- Expectation value clipping

Usage via solver_kwargs:
    solver_kwargs={
        "mitigation": {
            "zne": {"scale_factors": [1, 3, 5]},
            "n_repeats": 3,
            "clip_expvals": True,
        }
    }
"""


def _richardson_extrapolate(scale_factors: list, values: list) -> float:
    """Lagrange interpolation at x=0 for Zero-Noise Extrapolation.

    Given expectation values measured at different noise scale factors,
    extrapolate to the zero-noise limit.

    Args:
        scale_factors: noise amplification factors, e.g. [1, 3, 5]
        values: corresponding expectation values at each scale factor
    Returns:
        Extrapolated zero-noise expectation value
    """
    n = len(scale_factors)
    result = 0.0
    for i in range(n):
        weight = 1.0
        for j in range(n):
            if j != i:
                weight *= -scale_factors[j] / (scale_factors[i] - scale_factors[j])
        result += weight * values[i]
    return result


def _clip_expvals(expvals: list) -> list:
    """Clamp expectation values to [-1, 1] (valid range for <Z>)."""
    return [max(-1.0, min(1.0, v)) for v in expvals]


def _apply_mitigation(
    run_fn,
    mitigation: dict,
) -> list:
    """Apply error mitigation to a circuit execution function.

    Orchestrates ZNE, multi-shot averaging, and clipping.

    Args:
        run_fn: callable(scale_factor) -> list[float] of expectation values
        mitigation: dict with keys "zne", "n_repeats", "clip_expvals"
    Returns:
        list of mitigated expectation values
    """
    zne_config = mitigation.get("zne", None)
    n_repeats = mitigation.get("n_repeats", 1)
    clip = mitigation.get("clip_expvals", False)

    all_repeat_results = []
    for _ in range(n_repeats):
        if zne_config:
            scale_factors = zne_config.get("scale_factors", [1, 3, 5])
            # Run at each noise scale
            scaled_results = [run_fn(sf) for sf in scale_factors]
            # Richardson extrapolate per circuit
            n_circuits = len(scaled_results[0])
            extrapolated = [
                _richardson_extrapolate(
                    scale_factors, [sr[i] for sr in scaled_results]
                )
                for i in range(n_circuits)
            ]
            all_repeat_results.append(extrapolated)
        else:
            all_repeat_results.append(run_fn(1))

    # Average across repeats
    n_circuits = len(all_repeat_results[0])
    if n_repeats > 1:
        expvals = [
            sum(r[i] for r in all_repeat_results) / n_repeats
            for i in range(n_circuits)
        ]
    else:
        expvals = all_repeat_results[0]

    if clip:
        expvals = _clip_expvals(expvals)

    return expvals
