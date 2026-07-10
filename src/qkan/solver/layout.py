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
Device-independent, calibration-aware qubit selection and layout.

This module is torch-free and provider-neutral: calibration data enters
through :class:`DeviceProfile` (with adapters for qiskit backends and AWS
Braket devices) and layouts come out as plain physical-qubit index lists,
so the same selection logic serves the qiskit solver, the CUDA-Q solver,
and standalone users.

Public API:

- :class:`DeviceProfile` — frozen snapshot of per-qubit / per-edge calibration.
- :func:`rank_qubits` — top-``n`` independent qubit ranking (the
  device-independent generalization of ``best_qubits``).
- :func:`score_layout` — score a candidate layout for an interaction graph.
- :func:`best_subgraph` — best-calibrated connected subgraph for one
  multi-qubit circuit (thin complement to qiskit's own VF2 passes).
- :func:`tile_disjoint` — tile a chip with disjoint calibration-aware
  subgraphs to run several independent circuits in parallel.
"""

import functools
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Union

try:
    import rustworkx as rx  # type: ignore

    _RUSTWORKX_AVAILABLE = True
except ImportError:
    _RUSTWORKX_AVAILABLE = False

__all__ = [
    "DeviceProfile",
    "best_subgraph",
    "rank_qubits",
    "score_layout",
    "tile_disjoint",
]

# Pessimistic fallbacks for missing calibration data (a qubit or edge with
# no data must never outrank a measured one). The qubit-level constants
# match best_qubits in the qiskit solver.
_RO_FALLBACK = 0.5
_G1_FALLBACK = 1e-2
_G2_FALLBACK = 0.5
_T2_FALLBACK_US = 50.0
# Finite stand-in for -log(0) when an error rate is reported as >= 1.
_BIG = 1e6

Interaction = Union[int, Sequence[tuple[int, int]]]


@dataclass(frozen=True)
class DeviceProfile:
    """Provider-neutral snapshot of device calibration.

    Physical qubit labels run ``0..num_qubits-1``; devices with
    non-contiguous or 1-indexed labels simply have holes in the maps, and
    unreported labels are marked non-operational by the adapters. Coupling
    ``edges`` and the ``gate_error_2q`` map are keyed by sorted
    ``(low, high)`` pairs; ``t2_us`` is in microseconds. All error values
    are probabilities in [0, 1]; NaN entries are kept and treated as
    missing data at scoring time, and missing dict keys mean the device
    reported nothing for that qubit or edge (common for faulty qubits).
    Qubits mapped to False in ``operational`` are never selected.
    """

    num_qubits: int
    edges: tuple[tuple[int, int], ...] = ()
    readout_error: Mapping[int, float] = field(default_factory=dict)
    gate_error_1q: Mapping[int, float] = field(default_factory=dict)
    gate_error_2q: Mapping[tuple[int, int], float] = field(default_factory=dict)
    t2_us: Optional[Mapping[int, float]] = None
    operational: Optional[Mapping[int, bool]] = None

    def has_calibration(self) -> bool:
        """True when the profile carries any per-qubit calibration data."""
        return bool(self.readout_error or self.gate_error_1q or self.gate_error_2q)

    @functools.cached_property
    def _edge_set(self) -> frozenset:
        # Cached for the layout searches, which test coupling membership
        # once per interaction edge per VF2 candidate.
        return frozenset(self.edges)

    @classmethod
    def from_qiskit(cls, backend, *, refresh: bool = False) -> "DeviceProfile":
        """Build a profile from a qiskit backend.

        Prefers the BackendV2 ``Target`` (structured per-instruction errors);
        falls back to the legacy ``properties()`` API. With ``refresh=True``
        the backend is asked to re-fetch calibration first — via
        ``backend.refresh()`` on BackendV2 (qiskit-ibm-runtime) or
        ``properties(refresh=True)`` on the legacy path — since both APIs
        cache their first snapshot.
        """
        target = getattr(backend, "target", None)
        if target is not None:
            if refresh:
                try:
                    backend.refresh()
                    target = getattr(backend, "target", target)
                except Exception:
                    pass
            return cls._from_qiskit_target(backend, target)
        if hasattr(backend, "properties"):
            return cls._from_qiskit_properties(backend, refresh=refresh)
        raise ValueError(
            "DeviceProfile.from_qiskit: backend exposes neither a Target nor "
            "the legacy properties() calibration API"
        )

    @classmethod
    def _from_qiskit_target(cls, backend, target) -> "DeviceProfile":
        num_qubits = int(target.num_qubits or getattr(backend, "num_qubits", 0))
        readout: dict[int, float] = {}
        g1: dict[int, float] = {}
        g2: dict[tuple[int, int], float] = {}
        edges: set[tuple[int, int]] = set()
        t2_us: dict[int, float] = {}

        def _op_errors(name):
            try:
                return target[name].items()
            except Exception:
                return []

        for qargs, iprops in _op_errors("measure"):
            if qargs and len(qargs) == 1 and iprops is not None:
                err = getattr(iprops, "error", None)
                if err is not None:
                    readout[qargs[0]] = float(err)
        for name in ("sx", "x"):
            for qargs, iprops in _op_errors(name):
                if qargs and len(qargs) == 1 and qargs[0] not in g1:
                    err = getattr(iprops, "error", None) if iprops else None
                    if err is not None:
                        g1[qargs[0]] = float(err)
            if g1:
                break
        for name in getattr(target, "operation_names", []):
            for qargs, iprops in _op_errors(name):
                if qargs is None or len(qargs) != 2:
                    continue
                edge = (min(qargs), max(qargs))
                edges.add(edge)
                err = getattr(iprops, "error", None) if iprops else None
                if err is not None:
                    # A pair may carry several 2q gates/directions; keep the
                    # best one — the transpiler uses the calibrated direction.
                    prev = g2.get(edge)
                    if prev is None or float(err) < prev:
                        g2[edge] = float(err)
        qprops = getattr(target, "qubit_properties", None)
        if qprops:
            for q, qp in enumerate(qprops):
                t2 = getattr(qp, "t2", None) if qp is not None else None
                if t2 is not None:
                    t2_us[q] = float(t2) * 1e6
        return cls(
            num_qubits=num_qubits,
            edges=tuple(sorted(edges)),
            readout_error=readout,
            gate_error_1q=g1,
            gate_error_2q=g2,
            t2_us=t2_us or None,
        )

    @classmethod
    def _from_qiskit_properties(
        cls, backend, *, refresh: bool = False
    ) -> "DeviceProfile":
        try:
            props = (
                backend.properties(refresh=True) if refresh else backend.properties()
            )
        except TypeError:
            props = backend.properties()
        if props is None:
            raise ValueError(
                "DeviceProfile.from_qiskit: backend.properties() returned None "
                "(no calibration available)"
            )
        num_qubits = int(getattr(backend, "num_qubits", 0) or len(props.qubits))
        readout: dict[int, float] = {}
        g1: dict[int, float] = {}
        t2_us: dict[int, float] = {}
        operational: dict[int, bool] = {}
        for q in range(num_qubits):
            try:
                readout[q] = float(props.readout_error(q))
            except Exception:
                pass
            try:
                g1[q] = float(props.gate_error("sx", [q]))
            except Exception:
                pass
            try:
                t2_us[q] = float(props.t2(q)) * 1e6
            except Exception:
                pass
            try:
                operational[q] = bool(props.is_qubit_operational(q))
            except Exception:
                pass
        g2: dict[tuple[int, int], float] = {}
        edges: set[tuple[int, int]] = set()
        config = None
        try:
            config = backend.configuration()
        except Exception:
            pass
        coupling = getattr(config, "coupling_map", None) if config else None
        for pair in coupling or []:
            a, b = int(pair[0]), int(pair[1])
            edge = (min(a, b), max(a, b))
            edges.add(edge)
            for gate in ("ecr", "cz", "cx"):
                try:
                    err = float(props.gate_error(gate, [a, b]))
                except Exception:
                    continue
                prev = g2.get(edge)
                if prev is None or err < prev:
                    g2[edge] = err
                break
        return cls(
            num_qubits=num_qubits,
            edges=tuple(sorted(edges)),
            readout_error=readout,
            gate_error_1q=g1,
            gate_error_2q=g2,
            t2_us=t2_us or None,
            operational=operational or None,
        )

    @classmethod
    def from_braket(cls, device) -> "DeviceProfile":
        """Build a profile from an AWS Braket device (experimental).

        ``device`` is a ``braket.aws.AwsDevice`` (or any object exposing the
        same ``properties`` model). Reads the standardized gate-model QPU
        calibration schema: per-qubit T1/T2, readout fidelity (or IQM's
        readout-error entries), and 1Q (simultaneous) randomized-benchmarking
        fidelity, plus per-pair 2Q gate fidelities. Fidelities are converted
        to errors as ``1 - fidelity``; entries whose type name starts with
        ``READOUT_ERROR`` are already errors and are averaged as-is.
        """
        props = getattr(device, "properties", None)
        std = getattr(props, "standardized", None)
        if std is None:
            raise ValueError(
                "DeviceProfile.from_braket: device exposes no standardized "
                "calibration properties (IonQ and simulators do not publish "
                "per-qubit calibration)"
            )
        paradigm = getattr(props, "paradigm", None)
        one_q = getattr(std, "oneQubitProperties", {}) or {}
        two_q = getattr(std, "twoQubitProperties", {}) or {}

        readout: dict[int, float] = {}
        g1: dict[int, float] = {}
        t2_us: dict[int, float] = {}
        labels: set[int] = set()
        for label, qprops in one_q.items():
            q = int(label)
            labels.add(q)
            t2 = getattr(qprops, "T2", None)
            if t2 is not None and getattr(t2, "value", None) is not None:
                scale = 1e6 if str(getattr(t2, "unit", "S")).upper() == "S" else 1.0
                t2_us[q] = float(t2.value) * scale
            ro_err: list[float] = []
            rb, srb = None, None
            for fid in getattr(qprops, "oneQubitFidelity", []) or []:
                ftype = str(getattr(getattr(fid, "fidelityType", None), "name", ""))
                value = getattr(fid, "fidelity", None)
                if value is None:
                    continue
                if ftype.startswith("READOUT_ERROR"):
                    ro_err.append(float(value))
                elif ftype == "READOUT":
                    readout[q] = 1.0 - float(value)
                elif ftype == "SIMULTANEOUS_RANDOMIZED_BENCHMARKING":
                    srb = 1.0 - float(value)
                elif ftype == "RANDOMIZED_BENCHMARKING":
                    rb = 1.0 - float(value)
            if q not in readout and ro_err:
                readout[q] = sum(ro_err) / len(ro_err)
            # Packed jobs drive all qubits at once: prefer simultaneous RB.
            if srb is not None:
                g1[q] = srb
            elif rb is not None:
                g1[q] = rb

        g2: dict[tuple[int, int], float] = {}
        edges: set[tuple[int, int]] = set()
        for pair_label, pprops in two_q.items():
            try:
                a_s, b_s = str(pair_label).split("-")
                a, b = int(a_s), int(b_s)
            except ValueError:
                continue
            edge = (min(a, b), max(a, b))
            edges.add(edge)
            labels.update((a, b))
            for fid in getattr(pprops, "twoQubitGateFidelity", []) or []:
                value = getattr(fid, "fidelity", None)
                if value is None:
                    continue
                err = 1.0 - float(value)
                prev = g2.get(edge)
                if prev is None or err < prev:
                    g2[edge] = err

        num_qubits = int(getattr(paradigm, "qubitCount", 0) or 0)
        if labels:
            num_qubits = max(num_qubits, max(labels) + 1)
        # Devices with 1-indexed or holey label spaces (IQM, faulty Rigetti
        # qubits) leave phantom indices with no data; never select those.
        reported = {int(label) for label in one_q}
        operational = (
            {q: (q in reported) for q in range(num_qubits)} if reported else None
        )
        return cls(
            num_qubits=num_qubits,
            edges=tuple(sorted(edges)),
            readout_error=readout,
            gate_error_1q=g1,
            gate_error_2q=g2,
            t2_us=t2_us or None,
            operational=operational,
        )


def _neg_log(err: float) -> float:
    """-ln(1 - err), clamped to finite values (fidelity-product cost)."""
    if err <= 0.0:
        return 0.0
    if err >= 1.0:
        return _BIG
    return -math.log1p(-err)


def _qubit_cost(
    profile: DeviceProfile,
    q: int,
    *,
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    filtering: bool = False,
) -> Optional[float]:
    """Additive cost of physical qubit ``q``.

    With ``filtering=True``, returns None for qubits failing a threshold or
    flagged non-operational (NaN fails an active threshold). Without
    filtering, missing/NaN data falls back to pessimistic constants and
    non-operational qubits cost ``_BIG``.
    """
    operational = True
    if profile.operational is not None:
        operational = bool(profile.operational.get(q, True))
    if not operational:
        if filtering:
            return None
        return _BIG
    ro = float(profile.readout_error.get(q, float("nan")))
    if math.isnan(ro):
        if filtering and max_readout_error is not None:
            return None
        ro = _RO_FALLBACK
    if filtering and max_readout_error is not None and ro > max_readout_error:
        return None
    g1 = float(profile.gate_error_1q.get(q, float("nan")))
    if math.isnan(g1):
        if filtering and qubit_error_threshold is not None:
            return None
        g1 = _G1_FALLBACK
    if filtering and qubit_error_threshold is not None and g1 > qubit_error_threshold:
        return None
    t2 = float("nan")
    if profile.t2_us is not None:
        t2 = float(profile.t2_us.get(q, float("nan")))
    if math.isnan(t2):
        t2 = _T2_FALLBACK_US
    return _neg_log(ro) + _neg_log(g1) + 1e-4 / max(t2, 1.0)


def _edge_error(profile: DeviceProfile, a: int, b: int) -> Optional[float]:
    """Calibrated 2q error for the physical pair, or None if not coupled."""
    edge = (min(a, b), max(a, b))
    if edge not in profile._edge_set:
        return None
    err = float(profile.gate_error_2q.get(edge, float("nan")))
    if math.isnan(err):
        err = _G2_FALLBACK
    return err


def _interaction_edges(
    interaction: Interaction,
) -> tuple[list[tuple[int, int]], int, dict[tuple[int, int], int]]:
    """Normalize an interaction spec to (unique edges, n_logical, multiplicity).

    Repeated pairs and both orientations of a pair (multi-layer ansaetze,
    direction flips) collapse to one undirected edge whose multiplicity is
    used as the default 2q gate count in scoring.
    """
    if isinstance(interaction, int):
        if interaction < 1:
            raise ValueError(f"interaction qubit count must be >= 1, got {interaction}")
        return [], interaction, {}
    multiplicity: dict[tuple[int, int], int] = {}
    for a, b in interaction:
        a, b = int(a), int(b)
        if a == b or a < 0 or b < 0:
            raise ValueError(f"invalid interaction edge ({a}, {b})")
        edge = (min(a, b), max(a, b))
        multiplicity[edge] = multiplicity.get(edge, 0) + 1
    if not multiplicity:
        raise ValueError("interaction edge list is empty; pass an int instead")
    edges = sorted(multiplicity)
    n_logical = max(max(a, b) for a, b in edges) + 1
    return edges, n_logical, multiplicity


def _usable_costs(
    profile: DeviceProfile,
    *,
    exclude: Optional[set[int]] = None,
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
) -> list[tuple[float, int]]:
    """Sorted ``(cost, qubit)`` over qubits passing the threshold filters."""
    scored = []
    for q in range(profile.num_qubits):
        if exclude and q in exclude:
            continue
        cost = _qubit_cost(
            profile,
            q,
            max_readout_error=max_readout_error,
            qubit_error_threshold=qubit_error_threshold,
            filtering=True,
        )
        if cost is not None:
            scored.append((cost, q))
    scored.sort()
    return scored


def rank_qubits(
    profile: DeviceProfile,
    n: int,
    *,
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    strict: bool = True,
) -> list[int]:
    """Return the ``n`` best-calibrated qubit indices, best first.

    Device-independent counterpart of the qiskit solver's ``best_qubits``:
    ranks by readout error + 1q gate error (as fidelity-product costs) with
    a small short-T2 penalty. NaN calibration fails an active threshold and
    otherwise falls back to pessimistic constants; non-operational qubits
    are skipped. Returns ``[]`` when the profile carries no calibration and
    no threshold was requested; raises ``ValueError`` when a threshold
    cannot be satisfied (unless ``strict=False``, which returns all usable
    qubits).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"rank_qubits: n must be a positive integer, got {n!r}")
    has_threshold = max_readout_error is not None or qubit_error_threshold is not None
    if not profile.has_calibration():
        if has_threshold:
            raise ValueError(
                "rank_qubits: profile carries no calibration data; cannot "
                "enforce qubit calibration thresholds"
            )
        return []
    if n > profile.num_qubits:
        raise ValueError(
            f"rank_qubits: requested n={n} exceeds num_qubits={profile.num_qubits}"
        )
    scored = _usable_costs(
        profile,
        max_readout_error=max_readout_error,
        qubit_error_threshold=qubit_error_threshold,
    )
    if len(scored) < n:
        if not strict and scored:
            return [q for _cost, q in scored]
        raise ValueError(
            f"rank_qubits: only {len(scored)} qubits satisfy the requested "
            f"thresholds; need {n}"
        )
    return [q for _cost, q in scored[:n]]


def score_layout(
    profile: DeviceProfile,
    interaction: Interaction,
    layout: Sequence[int],
    *,
    gate_counts: Optional[Mapping[Any, int]] = None,
) -> float:
    """Score a candidate layout (lower is better).

    ``layout`` is positional: ``layout[i]`` is the physical qubit hosting
    logical qubit ``i``. The score is a fidelity-product cost: the sum of
    ``count * -ln(1 - error)`` over every mapped 2q edge plus each logical
    qubit's readout/1q-gate cost and a small short-T2 penalty. Logical
    edges whose physical pair is not coupled cost a large finite penalty
    (they would require routing). Repeated interaction edges count as
    their multiplicity. ``gate_counts`` optionally overrides weights: int
    keys are per-logical-qubit 1q+readout multiplicity, sorted-tuple keys
    are per-logical-edge 2q gate counts.
    """
    edges, n_logical, multiplicity = _interaction_edges(interaction)
    if len(layout) < n_logical:
        raise ValueError(
            f"layout has {len(layout)} entries but the interaction uses "
            f"{n_logical} logical qubits"
        )
    if len(set(layout[:n_logical])) != n_logical:
        raise ValueError("layout assigns the same physical qubit twice")
    for i in range(n_logical):
        if not 0 <= int(layout[i]) < profile.num_qubits:
            raise ValueError(
                f"layout entry {layout[i]} is outside the device's "
                f"{profile.num_qubits}-qubit index space"
            )
    return _score_normalized(
        profile, edges, multiplicity, n_logical, layout, gate_counts or {}
    )


def _score_normalized(
    profile: DeviceProfile,
    edges: Sequence[tuple[int, int]],
    multiplicity: Mapping[tuple[int, int], int],
    n_logical: int,
    layout: Sequence[int],
    counts: Mapping[Any, int],
    qubit_costs: Optional[Mapping[int, float]] = None,
) -> float:
    """:func:`score_layout` core over a pre-normalized, pre-validated
    interaction — the per-VF2-candidate hot path, so no re-normalization
    and per-qubit costs can be supplied from a cache."""
    total = 0.0
    for i in range(n_logical):
        phys = int(layout[i])
        cost = (
            qubit_costs[phys] if qubit_costs is not None else _qubit_cost(profile, phys)
        )
        assert cost is not None
        total += float(counts.get(i, 1)) * cost
    for a, b in edges:
        err = _edge_error(profile, int(layout[a]), int(layout[b]))
        edge_cost = _BIG if err is None else _neg_log(err)
        key = (a, b) if (a, b) in counts else (b, a)
        total += float(counts.get(key, multiplicity.get((a, b), 1))) * edge_cost
    return total


def _pruned_coupling_graph(
    profile: DeviceProfile,
    *,
    exclude: Optional[set[int]] = None,
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    edge_error_threshold: Optional[float] = None,
):
    """Coupling graph with thresholded/excluded qubits and edges removed."""
    exclude = exclude or set()
    graph = rx.PyGraph()
    node_of: dict[int, int] = {}
    for q in range(profile.num_qubits):
        if q in exclude:
            continue
        cost = _qubit_cost(
            profile,
            q,
            max_readout_error=max_readout_error,
            qubit_error_threshold=qubit_error_threshold,
            filtering=True,
        )
        if cost is None:
            continue
        node_of[q] = graph.add_node(q)
    for a, b in profile.edges:
        if a not in node_of or b not in node_of:
            continue
        err = float(profile.gate_error_2q.get((a, b), float("nan")))
        # Dead edges (error at the physical bound) can never belong to a good
        # layout, and VF2's enumeration order is not score-aware — scoring
        # alone cannot keep them out of the sampled candidates, so prune them
        # from the search graph unconditionally.
        if not math.isnan(err) and err >= 0.999:
            continue
        if edge_error_threshold is not None:
            if math.isnan(err) or err > edge_error_threshold:
                continue
        graph.add_edge(node_of[a], node_of[b], None)
    return graph


def _best_subgraph_on(
    profile: DeviceProfile,
    graph,
    edges: list[tuple[int, int]],
    n_logical: int,
    multiplicity: Mapping[tuple[int, int], int],
    *,
    call_limit: Optional[int],
    max_trials: int,
) -> Optional[tuple[list[int], float]]:
    """Best layout of the interaction onto the pruned graph, or None."""
    # Isolated logical qubits (no interaction edges) are assigned greedily
    # to the best remaining qubits after the edged part is placed.
    edged_logicals = sorted({q for e in edges for q in e})
    needle = rx.PyGraph()
    needle_node: dict[int, int] = {}
    for lq in edged_logicals:
        needle_node[lq] = needle.add_node(lq)
    for a, b in edges:
        needle.add_edge(needle_node[a], needle_node[b], None)
    if edges and rx.number_connected_components(needle) > 1:
        raise ValueError(
            "interaction graph has multiple connected components with edges; "
            "place each component separately (see tile_disjoint)"
        )

    # Per-qubit costs are reused for the availability ordering and for
    # every candidate's score (they do not vary across VF2 trials).
    qubit_costs: dict[int, float] = {}
    for node in graph.node_indices():
        q = graph[node]
        cost = _qubit_cost(profile, q)
        assert cost is not None  # non-filtering mode always yields a cost
        qubit_costs[q] = cost
    available = sorted(qubit_costs, key=qubit_costs.__getitem__)

    best: Optional[tuple[list[int], float]] = None
    if edges:
        mappings = rx.vf2_mapping(
            graph,
            needle,
            subgraph=True,
            induced=False,
            id_order=False,
            call_limit=call_limit,
        )
        trials = 0
        for mapping in mappings:
            # rustworkx yields {haystack node -> needle node}.
            phys_of_logical: dict[int, int] = {}
            for g_node, n_node in mapping.items():
                phys_of_logical[needle[n_node]] = graph[g_node]
            used = set(phys_of_logical.values())
            layout = [-1] * n_logical
            for lq, phys in phys_of_logical.items():
                layout[lq] = phys
            free = iter(q for q in available if q not in used)
            try:
                for lq in range(n_logical):
                    if layout[lq] < 0:
                        layout[lq] = next(free)
            except StopIteration:
                continue
            score = _score_normalized(
                profile, edges, multiplicity, n_logical, layout, {}, qubit_costs
            )
            if best is None or score < best[1]:
                best = (layout, score)
            trials += 1
            if trials >= max_trials:
                break
    else:
        if len(available) >= n_logical:
            layout = available[:n_logical]
            score = _score_normalized(
                profile, [], {}, n_logical, layout, {}, qubit_costs
            )
            best = (layout, score)
    return best


def best_subgraph(
    profile: DeviceProfile,
    interaction: Interaction,
    *,
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    edge_error_threshold: Optional[float] = None,
    strict: bool = True,
    call_limit: Optional[int] = 5_000_000,
    max_trials: int = 2_500,
) -> list[int]:
    """Best-calibrated placement of one circuit's interaction graph.

    ``interaction`` is either an edge list over logical qubits (2q-gate
    pairs) or a bare int for an edge-free register (which reduces to
    :func:`rank_qubits`). Returns a positional layout (``result[i]`` hosts
    logical qubit ``i``), directly usable as a qiskit ``initial_layout``.

    Note qiskit's own transpiler already performs noise-aware VF2 placement
    at ``optimization_level`` 2-3; this function exists for explicit
    thresholds, custom scoring, level-1 pipelines, and non-qiskit stacks.

    Raises ``ValueError`` when no placement satisfies the thresholds
    (``strict=True``) or returns ``[]`` (``strict=False``). The interaction
    graph must be connected (tile disconnected pieces via
    :func:`tile_disjoint`) and must embed in the coupling map without
    routing.
    """
    edges, n_logical, multiplicity = _interaction_edges(interaction)
    if not edges:
        return rank_qubits(
            profile,
            n_logical,
            max_readout_error=max_readout_error,
            qubit_error_threshold=qubit_error_threshold,
            strict=strict,
        )
    if not _RUSTWORKX_AVAILABLE:
        raise ImportError(
            "best_subgraph requires rustworkx (installed with qiskit): "
            "pip install rustworkx"
        )
    if not profile.has_calibration() and (
        max_readout_error is not None or qubit_error_threshold is not None
    ):
        raise ValueError(
            "best_subgraph: profile carries no calibration data; cannot "
            "enforce calibration thresholds"
        )
    graph = _pruned_coupling_graph(
        profile,
        max_readout_error=max_readout_error,
        qubit_error_threshold=qubit_error_threshold,
        edge_error_threshold=edge_error_threshold,
    )
    best = _best_subgraph_on(
        profile,
        graph,
        edges,
        n_logical,
        multiplicity,
        call_limit=call_limit,
        max_trials=max_trials,
    )
    if best is None:
        if strict:
            raise ValueError(
                "best_subgraph: no placement of the interaction graph "
                "satisfies the coupling map and thresholds"
            )
        return []
    return best[0]


def tile_disjoint(
    profile: DeviceProfile,
    interaction: Interaction,
    k: Union[int, Literal["max"]] = "max",
    *,
    buffer_hops: int = 0,
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    edge_error_threshold: Optional[float] = None,
    tile_score_threshold: Optional[float] = None,
    strict: bool = True,
    call_limit: Optional[int] = 50_000,
    max_trials: int = 500,
    n_logical: Optional[int] = None,
) -> list[list[int]]:
    """Tile the chip with disjoint calibration-aware copies of one circuit.

    Greedy peel: repeatedly place the interaction graph on the best
    remaining subgraph (:func:`best_subgraph` semantics), then remove the
    used qubits plus a ``buffer_hops``-neighborhood shell before placing
    the next copy. Stops at ``k`` tiles, when nothing placeable remains, or
    when the next tile's score exceeds ``tile_score_threshold``.

    Returns tiles ordered best-first; each tile is a positional layout for
    one circuit copy. For a merged qiskit job, concatenate them tile-major:
    ``flat = [q for tile in tiles for q in tile]``.

    ``n_logical`` widens each tile beyond the interaction's edge span:
    logical qubits without interaction edges (single-qubit-gate-only or
    idle wires) are assigned best-effort to the best remaining qubits.

    With integer ``k`` and ``strict=True``, raises ``ValueError`` when
    fewer than ``k`` tiles fit; ``strict=False`` returns what fits.
    """
    if k != "max" and (not isinstance(k, int) or k < 1):
        raise ValueError(f"tile_disjoint: k must be 'max' or a positive int, got {k!r}")
    if buffer_hops < 0:
        raise ValueError(f"tile_disjoint: buffer_hops must be >= 0, got {buffer_hops}")
    edges, n_logical_span, multiplicity = _interaction_edges(interaction)
    if n_logical is None:
        n_logical = n_logical_span
    elif n_logical < n_logical_span:
        raise ValueError(
            f"tile_disjoint: n_logical={n_logical} is smaller than the "
            f"interaction's span of {n_logical_span} logical qubits"
        )
    if edges and not _RUSTWORKX_AVAILABLE:
        raise ImportError(
            "tile_disjoint requires rustworkx (installed with qiskit): "
            "pip install rustworkx"
        )

    # Adjacency over the full (unpruned) coupling map for buffer removal.
    neighbors: dict[int, set[int]] = {}
    for a, b in profile.edges:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    limit = profile.num_qubits if k == "max" else int(k)
    removed: set[int] = set()
    tiles: list[tuple[list[int], float]] = []
    while len(tiles) < limit:
        if edges:
            graph = _pruned_coupling_graph(
                profile,
                exclude=removed,
                max_readout_error=max_readout_error,
                qubit_error_threshold=qubit_error_threshold,
                edge_error_threshold=edge_error_threshold,
            )
            best = _best_subgraph_on(
                profile,
                graph,
                edges,
                n_logical,
                multiplicity,
                call_limit=call_limit,
                max_trials=max_trials,
            )
        else:
            usable = _usable_costs(
                profile,
                exclude=removed,
                max_readout_error=max_readout_error,
                qubit_error_threshold=qubit_error_threshold,
            )
            if len(usable) < n_logical or not profile.has_calibration():
                best = None
            else:
                layout = [q for _c, q in usable[:n_logical]]
                best = (layout, score_layout(profile, n_logical, layout))
        if best is None:
            break
        layout, score = best
        if tile_score_threshold is not None and score > tile_score_threshold:
            break
        tiles.append((layout, score))
        shell = set(layout)
        frontier = set(layout)
        for _hop in range(buffer_hops):
            frontier = {nb for q in frontier for nb in neighbors.get(q, ())} - shell
            shell |= frontier
        removed |= shell

    if isinstance(k, int) and len(tiles) < k and strict:
        raise ValueError(
            f"tile_disjoint: only {len(tiles)} disjoint tiles satisfy the "
            f"thresholds; need {k}"
        )
    tiles.sort(key=lambda t: t[1])
    return [layout for layout, _score in tiles]
