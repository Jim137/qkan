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
Multi-qubit circuit packing.

Run ``k`` independent copies of a small circuit in parallel on one QPU,
placed on disjoint calibration-aware tiles selected by
:func:`~qkan.solver.layout.tile_disjoint`.

:func:`pack_circuit` is the single entry point for both stacks and every
result is one :class:`PackedCircuit`: a qiskit ``QuantumCircuit`` becomes
a swap-free ISA circuit pinned to the tiles with per-tile observable
mapping, and a plain CUDA-Q kernel is rebuilt (gates extracted from the
compiled Quake IR) into one kernel applying each copy at its tile's
physical indices, with per-tile readout from sampled counts.

Everything builds on :class:`~qkan.solver.layout.DeviceProfile` and is
torch-free.
"""

import math
import operator
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

from .layout import DeviceProfile, tile_disjoint

__all__ = [
    "PackedCircuit",
    "interaction_of",
    "kernel_interaction_of",
    "pack_circuit",
]


def interaction_of(circuit) -> list[tuple[int, int]]:
    """Extract a qiskit circuit's 2-qubit interaction edge list.

    Returns one entry per 2-qubit gate (duplicates preserved — they weight
    the tile scoring). Directives (barriers, measurements) are ignored;
    gates on three or more qubits raise ``ValueError``.
    """
    edges: list[tuple[int, int]] = []
    for instruction in circuit.data:
        op = instruction.operation
        if getattr(op, "_directive", False) or op.name in ("measure", "reset"):
            continue
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if len(qubits) == 2:
            edges.append((qubits[0], qubits[1]))
        elif len(qubits) > 2:
            raise ValueError(
                f"interaction_of: gate '{op.name}' acts on {len(qubits)} "
                "qubits; decompose to 1- and 2-qubit gates first"
            )
    return edges


def _edges_of_ops(ops) -> list[tuple[int, int]]:
    """2-qubit edge list of an extracted ``(name, params, qubits)`` gate list."""
    edges: list[tuple[int, int]] = []
    for name, _, qubits in ops:
        if len(qubits) == 2:
            edges.append((qubits[0], qubits[1]))
        elif len(qubits) > 2:
            raise ValueError(
                f"gate '{name}' acts on {len(qubits)} qubits; decompose "
                "to 1- and 2-qubit gates first"
            )
    return edges


def kernel_interaction_of(block, *args) -> list[tuple[int, int]]:
    """Extract a plain CUDA-Q kernel's 2-qubit interaction edge list.

    The CUDA-Q analogue of :func:`interaction_of`: one entry per 2-qubit
    gate, duplicates preserved, measurements ignored, gates on three or
    more qubits raise ``ValueError``. ``args`` bind the kernel's runtime
    arguments; closure captures, loops, and ``list`` indexing resolve to
    literal qubit indices (see :mod:`~qkan.solver._quake`).
    """
    from ._quake import kernel_ops

    ops, _, _ = kernel_ops(block, *args)
    return _edges_of_ops(ops)


def _validate_tile(tile: int, copies: int) -> None:
    """Range-check a tile index against the number of packed copies."""
    if not 0 <= tile < copies:
        raise IndexError(f"tile {tile} out of range for {copies} copies")


@dataclass(frozen=True)
class PackedCircuit:
    """A packed job: ``copies`` circuit copies pinned to disjoint tiles.

    One class serves both stacks — :func:`pack_circuit` fills the half
    that applies, and the placement accessors are shared.

    qiskit (estimator-driven): ``circuit`` is the merged logical circuit
    (copy ``t`` on logical qubits ``t*block_qubits ..``) and ``isa`` its
    transpilation pinned to ``layout``; map observables per tile with
    :meth:`observable` and bind parameter batches with
    :meth:`parameter_values` (each copy carries its own renamed
    parameters, ``parameters[t]`` in the source ``circuit.parameters``
    order) or :meth:`rebind`.

    CUDA-Q (counts-driven): ``kernel`` applies the block at each tile's
    physical indices and (in automatic mode) measures the full register;
    sample it and read per-tile results with :meth:`expectation`,
    X/Y observables through :meth:`basis_kernel` (hardware-safe) or
    :meth:`observable` (simulator ``observe``), and bind argument
    batches with :meth:`rebind`. ``gates`` holds one extracted gate
    list per tile and ``block`` the source kernel.
    """

    tiles: tuple[tuple[int, ...], ...]
    # qiskit half
    circuit: Any = None
    isa: Any = None
    parameters: tuple[tuple[str, ...], ...] = ()
    # CUDA-Q half
    kernel: Any = None
    gates: Optional[tuple] = None
    block: Any = None

    @property
    def layout(self) -> tuple[int, ...]:
        """Tile-major flattening of ``tiles`` (the pinned initial layout)."""
        return tuple(q for tile in self.tiles for q in tile)

    @property
    def width(self) -> int:
        """Register width of the packed CUDA-Q kernel (max index + 1)."""
        return max(self.layout) + 1

    @property
    def block_qubits(self) -> int:
        """Qubits per circuit copy."""
        return len(self.tiles[0])

    @property
    def copies(self) -> int:
        """Number of packed circuit copies."""
        return len(self.tiles)

    def physical_qubits(self, tile: int) -> list[int]:
        """Physical qubit indices hosting circuit copy ``tile``."""
        return list(self.tiles[tile])

    def parameter_values(self, batch) -> dict[str, float]:
        """Bind one batch of block parameters — copy ``t`` gets ``batch[t]``.

        ``batch[t][j]`` binds the ``j``-th parameter (in the source
        circuit's ``circuit.parameters`` order) of copy ``t``. Returns the
        name-to-value mapping for an ``EstimatorV2`` PUB alongside ``isa``,
        so a variational loop packs once and re-binds every step.
        """
        if self.kernel is not None:
            raise TypeError(
                "parameter_values applies to qiskit packs; CUDA-Q packs "
                "bind argument batches with rebind(batch)"
            )
        if not self.parameters:
            raise ValueError("the packed circuit has no parameters")
        if len(batch) != self.copies:
            raise ValueError(
                f"batch has {len(batch)} entries but the pack has {self.copies} copies"
            )
        values: dict[str, float] = {}
        for names, row in zip(self.parameters, batch):
            if len(row) != len(names):
                raise ValueError(
                    f"parameter row has {len(row)} values but each copy "
                    f"takes {len(names)}"
                )
            for name, value in zip(names, row):
                values[name] = float(value)
        return values

    def observable(self, obs, tile: Union[int, str, None] = None):
        """Map a block-level observable onto the packed placement.

        Returns the mapped operator for one ``tile``, the list over all
        copies (``None``), or their average (``"mean"``) — when every
        tile runs the same circuit, the mean pools the tiles into one
        estimate at ``copies``-fold shots.

        qiskit packs take a Pauli string (little-endian) or
        ``SparsePauliOp`` over the block's ``block_qubits`` qubits and
        return the ISA-mapped operator, ready for an ``EstimatorV2`` PUB
        alongside ``isa``. CUDA-Q packs take a Pauli string in index
        order (``pauli[i]`` acts on block qubit ``i``) and return a
        ``cudaq.spin`` operator at the tile's physical indices, for
        ``cudaq.observe(packed.observe_kernel(), ...)``. That is the
        simulator route — hardware targets that compact idle qubits
        reject sparse-index observables; use :meth:`basis_kernel` plus
        :meth:`expectation` there.
        """
        if tile is None:
            return [self.observable(obs, t) for t in range(self.copies)]
        if isinstance(tile, str):
            if tile != "mean":
                raise ValueError(
                    f"tile must be an index, None, or 'mean'; got {tile!r}"
                )
            ops = [self.observable(obs, t) for t in range(self.copies)]
            pooled = ops[0]
            for op in ops[1:]:
                pooled = pooled + op
            return pooled * (1.0 / self.copies)
        _validate_tile(tile, self.copies)
        if self.isa is not None:
            return self._isa_observable(obs, tile)
        return self._spin_observable(obs, tile)

    def _isa_observable(self, obs, tile: int):
        """qiskit leaf of :meth:`observable`: embed and pin to the ISA layout."""
        from qiskit.quantum_info import SparsePauliOp  # type: ignore

        op = obs if isinstance(obs, SparsePauliOp) else SparsePauliOp(obs)
        if op.num_qubits != self.block_qubits:
            raise ValueError(
                f"observable acts on {op.num_qubits} qubits but the packed "
                f"block has {self.block_qubits}"
            )
        base = tile * self.block_qubits
        embedded = op.apply_layout(
            list(range(base, base + self.block_qubits)),
            num_qubits=self.block_qubits * self.copies,
        )
        return embedded.apply_layout(self.isa.layout)

    def _spin_observable(self, obs, tile: int):
        """CUDA-Q leaf of :meth:`observable`: spin operator at physical indices."""
        import cudaq  # type: ignore

        if not isinstance(obs, str):
            raise TypeError(
                f"CUDA-Q packs map Pauli strings; got '{type(obs).__name__}'"
            )
        pauli = self._validated_pauli(obs)
        factors = {"X": cudaq.spin.x, "Y": cudaq.spin.y, "Z": cudaq.spin.z}
        op = None
        for i, letter in enumerate(pauli):
            if letter == "I":
                continue
            term = factors[letter](self.tiles[tile][i])
            op = term if op is None else op * term
        if op is None:
            op = cudaq.spin.i(self.tiles[tile][0])
        return op

    def _validated_pauli(self, pauli: str) -> str:
        pauli = pauli.upper()
        if len(pauli) != self.block_qubits:
            raise ValueError(
                f"'{pauli}' has {len(pauli)} entries but the packed block "
                f"has {self.block_qubits} qubits"
            )
        if any(letter not in "IXYZ" for letter in pauli):
            raise ValueError("Pauli strings may only contain I, X, Y, or Z")
        return pauli

    def _require_kernel(self, method: str) -> None:
        if self.kernel is None:
            raise TypeError(
                f"{method} applies to CUDA-Q packs; qiskit packs read "
                "results through observable/parameter_values"
            )

    def rebind(self, batch) -> "PackedCircuit":
        """Re-pack the same tiles with a new batch of values.

        The per-step call of a variational loop — tile selection is
        reused. CUDA-Q packs re-extract the block per tile (``batch``
        holds one runtime-argument set per tile, as ``block_args_batch``
        in :func:`pack_circuit`; milliseconds per step). qiskit packs
        return a copy with the parameters bound (``batch`` rows as in
        :meth:`parameter_values`; prefer ``parameter_values`` in
        estimator PUBs, which keeps the circuit parametric).
        """
        if self.kernel is not None:
            if self.block is None:
                raise ValueError(
                    "rebind requires automatic packing (a plain block kernel)"
                )
            return _pack_cudaq(
                None, self.block, tiles=self.tiles, block_args_batch=batch
            )
        values = self.parameter_values(batch)
        return PackedCircuit(
            tiles=self.tiles,
            circuit=self.circuit.assign_parameters(
                {p: values[p.name] for p in self.circuit.parameters}
            ),
            isa=self.isa.assign_parameters(
                {p: values[p.name] for p in self.isa.parameters}
            ),
        )

    def basis_kernel(self, bases: str):
        """Packed kernel with per-qubit basis rotations before measurement.

        ``bases`` has one letter per block qubit, position ``i`` acting on
        block qubit ``i``: ``X`` and ``Y`` append that basis change (``H``,
        or ``RZ(-pi/2)`` then ``H``) on the corresponding physical qubit of
        every tile; ``Z`` and ``I`` leave it untouched. Sample the returned
        kernel and read Pauli expectations with :meth:`expectation` — the
        hardware-safe observable path (sparse-index ``observe`` fails on
        targets that compact idle qubits).
        """
        self._require_kernel("basis_kernel")
        if self.gates is None:
            raise ValueError(
                "basis_kernel requires automatic packing (a plain block "
                "kernel); legacy convention blocks apply basis changes "
                "inside the block"
            )
        bases = self._validated_pauli(bases)
        return _recompose(self.gates, self.tiles, self.width, bases=bases)

    def expectation(self, result, pauli: str, tile: Union[int, str, None] = None):
        """Pauli expectation values from a sample of the matching basis.

        ``pauli[i]`` acts on block qubit ``i``; ``result`` must come from
        sampling :meth:`basis_kernel` for that string (the plain
        :attr:`kernel` suffices when it only contains ``Z`` and ``I``).
        The value is the Z-parity of the non-identity positions — for one
        ``tile``, the list over all copies (``None``), or their average
        (``"mean"``): when every tile runs the same circuit, the mean
        pools the tiles into one estimate at ``copies``-fold shots.
        """
        self._require_kernel("expectation")
        pauli = self._validated_pauli(pauli)
        if isinstance(tile, str) and tile != "mean":
            raise ValueError(f"tile must be an index, None, or 'mean'; got {tile!r}")
        positions = [i for i, letter in enumerate(pauli) if letter != "I"]
        if not positions:
            if tile is None:
                return [1.0] * self.copies
            if isinstance(tile, int):
                _validate_tile(tile, self.copies)
            return 1.0
        if tile is None:
            return [self._tile_parity(result, positions, t) for t in range(self.copies)]
        if isinstance(tile, str):
            values = [
                self._tile_parity(result, positions, t) for t in range(self.copies)
            ]
            return sum(values) / len(values)
        return self._tile_parity(result, positions, tile)

    def _tile_parity(self, result, positions, tile: int) -> float:
        """Z-parity of a tile's block positions from full-register counts."""
        _validate_tile(tile, self.copies)
        physical = [self.tiles[tile][p] for p in positions]
        acc = 0
        total = 0
        for bits, count in result.items():
            if len(bits) != self.width:
                raise ValueError(
                    f"bitstrings span {len(bits)} bits but the packed "
                    f"register has {self.width} qubits — the block kernel "
                    "must not measure (explicit mz compacts the sampled "
                    "register; sampling the packed kernel measures the "
                    "full register implicitly)"
                )
            total += count
            parity = sum(bits[p] == "1" for p in physical) % 2
            acc += count if parity == 0 else -count
        return acc / total if total else 0.0

    def observe_kernel(self):
        """Measurement-free packed kernel for ``cudaq.observe``.

        ``observe`` rejects kernels with measurements, so the sampled
        :attr:`kernel` (which measures the full register) cannot be used
        with :meth:`observable`; this rebuild omits the measurement.
        Simulator route only — without the full-register measurement,
        hardware pipelines may compact idle qubits.
        """
        self._require_kernel("observe_kernel")
        if self.gates is None:
            raise ValueError(
                "observe_kernel requires automatic packing (a plain block kernel)"
            )
        return _recompose(self.gates, self.tiles, self.width, measure=False)


def _select_tiles(
    profile: DeviceProfile,
    edges: Sequence[tuple[int, int]],
    m: int,
    k: Union[int, Literal["max"]],
    caller: str,
    **thresholds,
) -> list[tuple[int, ...]]:
    """Shared tile selection: interaction -> tile_disjoint -> validation."""
    interaction: Union[int, Sequence[tuple[int, int]]] = list(edges) if edges else m
    # n_logical widens tiles to the full block: qubits without interaction
    # edges (1q-gate-only or idle wires) get best-effort physical qubits.
    tiles = tile_disjoint(profile, interaction, k=k, n_logical=m, **thresholds)
    if not tiles:
        raise ValueError(f"{caller}: no tiles satisfy the coupling map and thresholds")
    return [tuple(t) for t in tiles]


def pack_circuit(
    backend,
    circuit,
    k: Union[int, Literal["max"]] = "max",
    *,
    tiles: Optional[Sequence[Sequence[int]]] = None,
    profile: Optional[DeviceProfile] = None,
    block_args: Sequence[Any] = (),
    block_args_batch: Optional[Sequence[Sequence[Any]]] = None,
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    edge_error_threshold: Optional[float] = None,
    tile_score_threshold: Optional[float] = None,
    buffer_hops: int = 0,
    strict: bool = True,
    optimization_level: int = 1,
) -> "PackedCircuit":
    """Pack ``k`` copies of a circuit onto disjoint calibrated tiles.

    Overloaded over both supported stacks, with the same selection logic
    (interaction graph -> :func:`~qkan.solver.layout.tile_disjoint` ->
    copies composed at the tiles' physical qubits):

    - qiskit: ``pack_circuit(backend, circuit, k)`` with a
      ``QuantumCircuit`` returns a :class:`PackedCircuit` — the copies are
      transpiled pinned to the tiles. ``profile`` overrides the
      calibration snapshot (default ``DeviceProfile.from_qiskit(backend)``)
      and ``optimization_level`` steers the transpilation. Tiles are
      coupled subgraphs, so the result routes without SWAPs; a routed
      result (e.g. from a profile whose coupling disagrees with
      ``backend``) raises ``RuntimeError``.
    - CUDA-Q: ``pack_circuit(profile, kernel, k)`` with a plain
      ``@cudaq.kernel`` and a :class:`~qkan.solver.layout.DeviceProfile`
      — the kernel's gates are extracted from the compiled Quake IR
      (``block_args`` bind runtime arguments) and rebuilt at the tiles'
      physical indices with a full-register measurement. A block written
      over the legacy ``(q: cudaq.qview, layout: list[int], offset:
      int)`` convention is composed by sub-kernel calls exactly as
      written instead (explicit ``tiles`` required); to thread runtime
      arguments through to sample time, write the whole packed kernel
      by hand (see the packing guide).

    Both stacks require measurement-free inputs: readout belongs to the
    primitive (qiskit) or is appended automatically (CUDA-Q). Selection
    thresholds, ``buffer_hops``, ``k="max"``, and ``strict`` semantics are
    those of :func:`~qkan.solver.layout.tile_disjoint`.

    Batched evaluation packs one job per batch: a parameterized qiskit
    circuit gets per-copy parameters (bind with
    :meth:`PackedCircuit.parameter_values`), and a CUDA-Q kernel takes one
    runtime-argument set per tile via ``block_args_batch`` (re-bind the
    same tiles with :meth:`PackedCircuit.rebind`). Explicit ``tiles``
    (hand-picked or from :func:`~qkan.solver.layout.tile_disjoint`) skip
    the calibration-driven selection on either stack.
    """
    is_qiskit = hasattr(circuit, "find_bit") and hasattr(circuit, "num_qubits")
    # prepare_call filters callables that merely carry a `signature`
    # attribute (e.g. numpy.vectorize) as well as builder PyKernels.
    is_kernel = (
        callable(circuit)
        and hasattr(circuit, "signature")
        and hasattr(circuit, "prepare_call")
    )
    if not (is_qiskit or is_kernel):
        raise TypeError(
            f"pack_circuit: unsupported circuit type "
            f"'{type(circuit).__name__}' — expected a qiskit QuantumCircuit "
            "or a @cudaq.kernel (kernel-builder PyKernels cannot be "
            "introspected; pass the decorated kernel)"
        )
    thresholds: dict[str, Any] = dict(
        max_readout_error=max_readout_error,
        qubit_error_threshold=qubit_error_threshold,
        edge_error_threshold=edge_error_threshold,
        tile_score_threshold=tile_score_threshold,
        buffer_hops=buffer_hops,
        strict=strict,
    )
    if is_qiskit:
        if isinstance(backend, DeviceProfile):
            raise TypeError(
                "pack_circuit: qiskit circuits transpile against a backend, "
                "not a DeviceProfile — pass the backend (profile=... "
                "overrides its calibration)"
            )
        if block_args or block_args_batch is not None:
            raise ValueError(
                "pack_circuit: block_args apply to CUDA-Q kernels only — "
                "parameterized qiskit circuits bind through "
                "PackedCircuit.parameter_values"
            )
        return _pack_qiskit(
            backend,
            circuit,
            k,
            tiles=tiles,
            profile=profile,
            optimization_level=optimization_level,
            **thresholds,
        )
    calibration = backend if isinstance(backend, DeviceProfile) else profile
    if calibration is None and tiles is None:
        raise TypeError(
            "pack_circuit: CUDA-Q kernels pack against calibration — "
            "pass a DeviceProfile as the first argument (or explicit tiles)"
        )
    return _pack_cudaq(
        calibration,
        circuit,
        k,
        tiles=tiles,
        block_args=block_args,
        block_args_batch=block_args_batch,
        **thresholds,
    )


def _pack_qiskit(
    backend,
    circuit,
    k: Union[int, Literal["max"]],
    *,
    tiles: Optional[Sequence[Sequence[int]]],
    profile: Optional[DeviceProfile],
    optimization_level: int,
    **thresholds,
) -> PackedCircuit:
    """qiskit half of :func:`pack_circuit`: compose + transpile pinned."""
    from qiskit import QuantumCircuit  # type: ignore
    from qiskit.circuit import Parameter  # type: ignore
    from qiskit.transpiler.preset_passmanagers import (  # type: ignore
        generate_preset_pass_manager,
    )

    if circuit.num_clbits:
        raise ValueError(
            "pack_circuit: the circuit has classical bits — packing a "
            "measured circuit would map every copy onto the same clbits "
            "(and in-circuit measurements corrupt estimator results); "
            "remove measurements and read results via observable() or the "
            "sampler that runs the packed circuit"
        )
    m = circuit.num_qubits
    if tiles is not None:
        interaction_of(circuit)  # >2-qubit gates fail identically in both modes
        tiles, m_tiles, _flat, _width, _copies = _validated_tiles(tiles)
        if m_tiles != m:
            raise ValueError(
                f"pack_circuit: tiles have {m_tiles} qubits but the block has {m}"
            )
    else:
        if profile is None:
            profile = DeviceProfile.from_qiskit(backend)
        tiles = _select_tiles(
            profile, interaction_of(circuit), m, k, "pack_circuit", **thresholds
        )
    copies = len(tiles)
    merged = QuantumCircuit(m * copies)
    # Each copy gets its own renamed parameters so one packed job can
    # evaluate a batch of parameter values (compose would otherwise share
    # the Parameter objects and bind every copy identically).
    source_params = list(circuit.parameters)
    parameter_names = []
    for t in range(copies):
        block = circuit
        if source_params:
            renamed = {p: Parameter(f"t{t}.{p.name}") for p in source_params}
            block = circuit.assign_parameters(renamed)
            parameter_names.append(tuple(renamed[p].name for p in source_params))
        merged.compose(block, qubits=range(t * m, (t + 1) * m), inplace=True)
    flat = [q for tile in tiles for q in tile]
    isa = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        initial_layout=flat,
    ).run(merged)
    # Routing SWAPs are lowered to basis gates before this point, so detect
    # routing through the layout permutation rather than a literal swap op.
    permutation = isa.layout.routing_permutation() if isa.layout else []
    if permutation != list(range(len(permutation))):
        raise RuntimeError(
            "pack_circuit: transpilation routed the packed circuit — the "
            "tiles do not embed the block's interaction graph in the "
            "backend's coupling map; the calibration profile likely "
            "disagrees with the transpilation backend (stale snapshot or "
            "hand-built edges) — rebuild it with DeviceProfile.from_qiskit"
        )
    return PackedCircuit(
        circuit=merged,
        isa=isa,
        tiles=tuple(tiles),
        parameters=tuple(parameter_names),
    )


def _recompose(
    gates_per_tile,
    tiles,
    width: int,
    bases: Optional[str] = None,
    measure: bool = True,
):
    """Build one kernel applying each tile's gate list at its physical qubits."""
    import cudaq  # type: ignore

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(width)
    for tile, gates in zip(tiles, gates_per_tile):
        for name, params, qubits in gates:
            method = getattr(kernel, name, None)
            if method is None:
                raise ValueError(
                    f"pack_circuit: gate '{name}' has no cudaq kernel-builder equivalent"
                )
            method(*params, *(q[tile[i]] for i in qubits))
        if bases is not None:
            for i, basis in enumerate(bases):
                if basis == "X":
                    kernel.h(q[tile[i]])
                elif basis == "Y":
                    kernel.rz(-math.pi / 2, q[tile[i]])
                    kernel.h(q[tile[i]])
    if measure:
        # The full-register measurement keeps idle qubits alive through
        # hardware pipelines that compact untouched qubits (dqe) and pins
        # the sampled bitstring width to the register width.
        kernel.mz(q)
    return kernel


def _validated_tiles(
    tiles,
) -> tuple[tuple[tuple[int, ...], ...], int, list[int], int, int]:
    """Normalize and validate tiles; return (tiles, size, flat, width, copies)."""
    norm = tuple(tuple(operator.index(q) for q in tile) for tile in tiles)
    if not norm:
        raise ValueError("packing: tiles must be non-empty")
    m = len(norm[0])
    if any(len(t) != m for t in norm):
        raise ValueError("packing: all tiles must have the same size")
    flat = [q for tile in norm for q in tile]
    if any(q < 0 for q in flat):
        raise ValueError("packing: tile indices must be non-negative")
    if len(set(flat)) != len(flat):
        raise ValueError("packing: tiles overlap")
    return norm, m, flat, max(flat) + 1, len(norm)


def _pack_cudaq(
    profile: Optional[DeviceProfile],
    block,
    k: Union[int, Literal["max"]] = "max",
    *,
    tiles: Optional[Sequence[Sequence[int]]] = None,
    block_args: Sequence[Any] = (),
    block_args_batch: Optional[Sequence[Sequence[Any]]] = None,
    **thresholds,
) -> PackedCircuit:
    """CUDA-Q half of :func:`pack_circuit`: extract + recompose pinned.

    Automatic mode (the default): ``block`` is a plain ``@cudaq.kernel``
    that allocates its own qubits — no signature convention. Its gate
    list is extracted from the compiled Quake IR (loops, closure
    captures, and ``list`` indexing resolve to literal gates), tiles are
    selected from ``profile`` calibration via
    :func:`~qkan.solver.layout.tile_disjoint` (or taken from ``tiles``
    when given), and the copies are rebuilt into one kernel that applies
    each copy's gates at its tile's physical indices and measures the
    full register — which also keeps idle qubits alive on hardware
    pipelines that compact untouched qubits. The block must not measure
    (readout is added automatically) or branch on measurement results,
    and must not call sub-kernels.

    Legacy mode: with explicit ``tiles`` and a block written over
    ``(q: cudaq.qview, layout: list[int], offset: int)``, the copies are
    composed by sub-kernel calls exactly as written; runtime-parameterized
    packed kernels can be hand-written this way (see the packing guide).
    """
    import cudaq  # type: ignore

    signature = getattr(block, "signature", None)
    arg_types = [str(t) for t in getattr(signature, "arg_types", None) or []]
    if arg_types and arg_types[0].startswith("!quake.veq"):
        # Legacy (qview, layout, offset) convention block.
        if len(arg_types) != 3 or not (
            arg_types[1].startswith("!cc.stdvec<i") and arg_types[2].startswith("i")
        ):
            raise ValueError(
                "pack_circuit: blocks taking a qview must use the legacy "
                "convention (q: cudaq.qview, layout: list[int], offset: "
                "int); for runtime-parameterized packing write the packed "
                "kernel by hand (see the packing guide)"
            )
        if tiles is None:
            raise ValueError(
                "pack_circuit: automatic tiling requires a plain kernel; "
                "blocks over (qview, layout, offset) need explicit tiles"
            )
        if block_args or block_args_batch is not None:
            raise ValueError("pack_circuit: block_args only apply to plain kernels")
        tiles, m, flat, width, copies = _validated_tiles(tiles)

        @cudaq.kernel
        def packed():
            q = cudaq.qvector(width)
            for t in range(copies):
                block(q, flat, t * m)
            # Full-register measurement: keeps idle qubits alive through
            # hardware pipelines that compact untouched qubits (dqe).
            mz(q)  # noqa: F821

        return PackedCircuit(kernel=packed, tiles=tiles)

    from ._quake import kernel_ops

    if block_args_batch is not None:
        if block_args:
            raise ValueError(
                "pack_circuit: pass block_args (one set for every copy) or "
                "block_args_batch (one set per tile), not both"
            )
        arg_sets = [tuple(args) for args in block_args_batch]
        if not arg_sets:
            raise ValueError("pack_circuit: block_args_batch must be non-empty")
    else:
        arg_sets = [tuple(block_args)]

    extracted = []
    for args in arg_sets:
        ops, m, flags = kernel_ops(block, *args)
        if "measurements" in flags:
            raise ValueError(
                "pack_circuit: the block measures — packing measures the full "
                "register automatically; remove mz/reset from the block"
            )
        if "conditionals" in flags:
            raise ValueError(
                "pack_circuit: mid-circuit conditionals are not supported by "
                "automatic packing"
            )
        extracted.append((ops, m, _edges_of_ops(ops)))
    ops0, m, edges0 = extracted[0]
    coupling = {tuple(sorted(edge)) for edge in edges0}
    for ops_t, m_t, edges_t in extracted[1:]:
        if m_t != m:
            raise ValueError(
                "pack_circuit: block_args_batch entries change the block's "
                f"qubit count ({m_t} vs {m})"
            )
        if {tuple(sorted(edge)) for edge in edges_t} != coupling:
            raise ValueError(
                "pack_circuit: block_args_batch entries change the block's "
                "interaction graph — every tile must host the same coupling"
            )

    if block_args_batch is not None:
        if k == "max":
            k = len(arg_sets)
        elif k != len(arg_sets):
            raise ValueError(
                f"pack_circuit: k={k} but block_args_batch has {len(arg_sets)} entries"
            )

    if tiles is None:
        assert profile is not None  # pack_circuit guarantees calibration or tiles
        tiles = _select_tiles(profile, edges0, m, k, "pack_circuit", **thresholds)
    tiles, m_tiles, flat, width, copies = _validated_tiles(tiles)
    if m_tiles != m:
        raise ValueError(
            f"pack_circuit: tiles have {m_tiles} qubits but the block has {m}"
        )
    normalized = [
        tuple((name, tuple(params), tuple(qubits)) for name, params, qubits in ops)
        for ops, _, _ in extracted
    ]
    if block_args_batch is not None:
        if copies != len(normalized):
            raise ValueError(
                f"pack_circuit: {copies} tiles but block_args_batch has "
                f"{len(normalized)} entries"
            )
        gates = tuple(normalized)
    else:
        gates = (normalized[0],) * copies
    return PackedCircuit(
        kernel=_recompose(gates, tiles, width),
        tiles=tiles,
        gates=gates,
        block=block,
    )
