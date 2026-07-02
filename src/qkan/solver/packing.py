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
:func:`~qkan.solver.layout.tile_disjoint`:

- :func:`pack_circuit` packs a qiskit circuit into one swap-free ISA
  circuit pinned to the tiles, with per-tile observable mapping.
- :func:`pack_kernel` packs a plain CUDA-Q kernel into one kernel that
  applies each copy at its tile's physical indices (gates are extracted
  from the compiled Quake IR and rebuilt with the kernel builder), with
  per-tile Z-parity readout from sampled counts and basis-change kernels
  for X/Y observables.

Both build on :class:`~qkan.solver.layout.DeviceProfile` and are
torch-free.
"""

import math
import operator
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

from .layout import DeviceProfile, tile_disjoint

__all__ = [
    "PackedCircuit",
    "PackedKernel",
    "interaction_of",
    "kernel_interaction_of",
    "pack_circuit",
    "pack_kernel",
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
                f"kernel_interaction_of: gate '{name}' acts on {len(qubits)} "
                "qubits; decompose to 1- and 2-qubit gates first"
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


@dataclass(frozen=True)
class PackedCircuit:
    """A packed qiskit job: ``copies`` circuit copies pinned to disjoint tiles.

    ``circuit`` is the merged logical circuit (copy ``t`` on logical qubits
    ``t*block_qubits .. (t+1)*block_qubits-1``), ``isa`` its transpilation
    pinned to ``layout`` (the tile-major flattening of ``tiles``).
    """

    circuit: Any
    isa: Any
    tiles: tuple[tuple[int, ...], ...]
    layout: tuple[int, ...]
    block_qubits: int
    copies: int

    def physical_qubits(self, tile: int) -> list[int]:
        """Physical qubit indices hosting circuit copy ``tile``."""
        return list(self.tiles[tile])

    def observable(self, obs, tile: Optional[int] = None):
        """Map a block-level observable onto the packed ISA circuit.

        ``obs`` is a Pauli string or ``SparsePauliOp`` over the block's
        ``block_qubits`` qubits. Returns the ISA-mapped operator for one
        ``tile``, or the list over all copies when ``tile`` is None —
        ready to pass to an ``EstimatorV2`` PUB alongside ``isa``.
        """
        from qiskit.quantum_info import SparsePauliOp  # type: ignore

        if tile is None:
            return [self.observable(obs, t) for t in range(self.copies)]
        if not 0 <= tile < self.copies:
            raise IndexError(f"tile {tile} out of range for {self.copies} copies")
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
    if len(tiles[0]) < m:
        raise RuntimeError(
            f"{caller}: tiles span {len(tiles[0])} qubits but the block has "
            f"{m} (this is a bug; please report)"
        )
    return [tuple(t[:m]) for t in tiles]


def pack_circuit(
    backend,
    circuit,
    k: Union[int, Literal["max"]] = "max",
    *,
    profile: Optional[DeviceProfile] = None,
    block_args: Sequence[Any] = (),
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    edge_error_threshold: Optional[float] = None,
    tile_score_threshold: Optional[float] = None,
    buffer_hops: int = 0,
    strict: bool = True,
    optimization_level: int = 1,
) -> "Union[PackedCircuit, PackedKernel]":
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
      returns a :class:`PackedKernel` — the kernel's gates are extracted
      from the compiled Quake IR (``block_args`` bind runtime arguments)
      and rebuilt at the tiles' physical indices with a full-register
      measurement. Explicit tiles or convention blocks go through
      :func:`pack_kernel`.

    Both stacks require measurement-free inputs: readout belongs to the
    primitive (qiskit) or is appended automatically (CUDA-Q). Selection
    thresholds, ``buffer_hops``, ``k="max"``, and ``strict`` semantics are
    those of :func:`~qkan.solver.layout.tile_disjoint`.
    """
    if hasattr(circuit, "find_bit") and hasattr(circuit, "num_qubits"):
        if block_args:
            raise ValueError("pack_circuit: block_args apply to CUDA-Q kernels only")
        return _pack_qiskit(
            backend,
            circuit,
            k,
            profile=profile,
            max_readout_error=max_readout_error,
            qubit_error_threshold=qubit_error_threshold,
            edge_error_threshold=edge_error_threshold,
            tile_score_threshold=tile_score_threshold,
            buffer_hops=buffer_hops,
            strict=strict,
            optimization_level=optimization_level,
        )
    if callable(circuit) and hasattr(circuit, "signature"):
        selected = backend if isinstance(backend, DeviceProfile) else profile
        if selected is None:
            raise TypeError(
                "pack_circuit: CUDA-Q kernels pack against calibration — "
                "pass a DeviceProfile as the first argument"
            )
        return pack_kernel(
            circuit,
            profile=selected,
            k=k,
            block_args=block_args,
            max_readout_error=max_readout_error,
            qubit_error_threshold=qubit_error_threshold,
            edge_error_threshold=edge_error_threshold,
            tile_score_threshold=tile_score_threshold,
            buffer_hops=buffer_hops,
            strict=strict,
        )
    raise TypeError(
        f"pack_circuit: unsupported circuit type "
        f"'{type(circuit).__name__}' — expected a qiskit QuantumCircuit "
        "or a @cudaq.kernel"
    )


def _pack_qiskit(
    backend,
    circuit,
    k: Union[int, Literal["max"]],
    *,
    profile: Optional[DeviceProfile],
    optimization_level: int,
    **thresholds,
) -> PackedCircuit:
    """qiskit half of :func:`pack_circuit`: compose + transpile pinned."""
    from qiskit import QuantumCircuit  # type: ignore
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
    if profile is None:
        profile = DeviceProfile.from_qiskit(backend)
    m = circuit.num_qubits
    tiles = _select_tiles(
        profile, interaction_of(circuit), m, k, "pack_circuit", **thresholds
    )
    copies = len(tiles)
    merged = QuantumCircuit(m * copies)
    for t in range(copies):
        merged.compose(circuit, qubits=range(t * m, (t + 1) * m), inplace=True)
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
        layout=tuple(flat),
        block_qubits=m,
        copies=copies,
    )


def _z_parity(counts, positions, width: Optional[int] = None) -> float:
    """Z-parity expectation over register positions from bitstring counts."""
    acc = 0
    total = 0
    for bits, count in counts.items():
        if width is not None and len(bits) != width:
            raise ValueError(
                f"bitstrings span {len(bits)} bits but the packed register "
                f"has {width} qubits — the block kernel must not measure "
                "(explicit mz compacts the sampled register; sampling the "
                "packed kernel measures the full register implicitly)"
            )
        total += count
        parity = sum(bits[p] == "1" for p in positions) % 2
        acc += count if parity == 0 else -count
    if total == 0:
        return 0.0
    return acc / total


@dataclass(frozen=True)
class PackedKernel:
    """A packed CUDA-Q kernel: ``copies`` block copies at disjoint tiles.

    ``kernel`` allocates ``width`` qubits, applies the block at each
    tile's physical indices, and (in automatic mode) measures the full
    register. Sample it with ``cudaq.sample`` and read per-tile results
    with :meth:`z_parity`; X/Y observables go through :meth:`basis_kernel`
    (hardware-safe) or :meth:`spin_op` (simulator ``observe``). ``gates``
    holds the extracted block gate list in automatic mode.
    """

    kernel: Any
    tiles: tuple[tuple[int, ...], ...]
    flat: tuple[int, ...]
    width: int
    block_qubits: int
    copies: int
    gates: Optional[tuple] = None

    def physical_qubits(self, tile: int) -> list[int]:
        """Physical qubit indices hosting block copy ``tile``."""
        return list(self.tiles[tile])

    def basis_kernel(self, bases: str):
        """Packed kernel with per-qubit basis rotations before measurement.

        ``bases`` has one letter per block qubit, position ``i`` acting on
        block qubit ``i``: ``X`` and ``Y`` append that basis change (``H``,
        or ``RZ(-pi/2)`` then ``H``) on the corresponding physical qubit of
        every tile; ``Z`` and ``I`` leave it untouched. Sample the returned
        kernel and read Pauli expectations with :meth:`z_parity` — the
        hardware-safe observable path (sparse-index ``observe`` fails on
        targets that compact idle qubits).
        """
        if self.gates is None:
            raise ValueError(
                "basis_kernel requires automatic packing (a plain block "
                "kernel); legacy convention blocks apply basis changes "
                "inside the block"
            )
        bases = bases.upper()
        if len(bases) != self.block_qubits:
            raise ValueError(
                f"bases has {len(bases)} entries but the packed block has "
                f"{self.block_qubits} qubits"
            )
        if any(b not in "IXYZ" for b in bases):
            raise ValueError("bases may only contain I, X, Y, or Z")
        return _recompose(self.gates, self.tiles, self.width, bases=bases)

    def observe_kernel(self):
        """Measurement-free packed kernel for ``cudaq.observe``.

        ``observe`` rejects kernels with measurements, so the sampled
        :attr:`kernel` (which measures the full register) cannot be used
        with :meth:`spin_op`; this rebuild omits the measurement.
        Simulator route only — without the full-register measurement,
        hardware pipelines may compact idle qubits.
        """
        if self.gates is None:
            raise ValueError(
                "observe_kernel requires automatic packing (a plain block kernel)"
            )
        return _recompose(self.gates, self.tiles, self.width, measure=False)

    def spin_op(self, pauli: str, tile: Optional[int] = None):
        """CUDA-Q spin operator for a block-level Pauli string at a tile.

        ``pauli`` has one letter per block qubit, position ``i`` acting on
        block qubit ``i`` (index order — unlike qiskit's little-endian
        strings). Returns the operator at ``tile``'s physical indices, or
        the list over all copies when ``tile`` is None, for
        ``cudaq.observe(packed.observe_kernel(), ...)``. Simulator route:
        hardware targets that compact idle qubits reject sparse-index
        observables — use :meth:`basis_kernel` plus :meth:`z_parity` there.
        """
        import cudaq  # type: ignore

        if tile is None:
            return [self.spin_op(pauli, t) for t in range(self.copies)]
        if not 0 <= tile < self.copies:
            raise IndexError(f"tile {tile} out of range for {self.copies} copies")
        pauli = pauli.upper()
        if len(pauli) != self.block_qubits:
            raise ValueError(
                f"pauli has {len(pauli)} entries but the packed block has "
                f"{self.block_qubits} qubits"
            )
        factors = {"X": cudaq.spin.x, "Y": cudaq.spin.y, "Z": cudaq.spin.z}
        op = None
        for i, letter in enumerate(pauli):
            if letter == "I":
                continue
            if letter not in factors:
                raise ValueError("pauli may only contain I, X, Y, or Z")
            term = factors[letter](self.tiles[tile][i])
            op = term if op is None else op * term
        if op is None:
            op = cudaq.spin.i(self.tiles[tile][0])
        return op

    def z_parity(self, result, positions: Sequence[int], tile: Optional[int] = None):
        """Z-parity of the block-local ``positions`` from a sample result.

        ``positions`` index qubits within the block (0..block_qubits-1).
        Returns the correlator for one ``tile``, or the list over all
        copies when ``tile`` is None. A single position yields that
        qubit's ``<Z>`` marginal; several yield ``<Z...Z>``.
        """
        if tile is None:
            return [self.z_parity(result, positions, t) for t in range(self.copies)]
        if not 0 <= tile < self.copies:
            raise IndexError(f"tile {tile} out of range for {self.copies} copies")
        physical = [self.tiles[tile][p] for p in positions]
        return _z_parity(result, physical, width=self.width)


# Gate names whose kernel-builder method is spelled differently.
_BUILDER_GATES = {"ccx": "ctx"}


def _recompose(
    gates,
    tiles,
    width: int,
    bases: Optional[str] = None,
    measure: bool = True,
):
    """Build one kernel applying ``gates`` at every tile's physical qubits."""
    import cudaq  # type: ignore

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(width)
    for tile in tiles:
        for name, params, qubits in gates:
            method = getattr(kernel, _BUILDER_GATES.get(name, name), None)
            if method is None:
                raise ValueError(
                    f"pack_kernel: gate '{name}' has no cudaq kernel-builder equivalent"
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
        raise ValueError("pack_kernel: tiles must be non-empty")
    m = len(norm[0])
    if any(len(t) != m for t in norm):
        raise ValueError("pack_kernel: all tiles must have the same size")
    flat = [q for tile in norm for q in tile]
    if any(q < 0 for q in flat):
        raise ValueError("pack_kernel: tile indices must be non-negative")
    if len(set(flat)) != len(flat):
        raise ValueError("pack_kernel: tiles overlap")
    return norm, m, flat, max(flat) + 1, len(norm)


def pack_kernel(
    block,
    tiles: Optional[Sequence[Sequence[int]]] = None,
    *,
    profile: Optional[DeviceProfile] = None,
    k: Union[int, Literal["max"]] = "max",
    block_args: Sequence[Any] = (),
    max_readout_error: Optional[float] = None,
    qubit_error_threshold: Optional[float] = None,
    edge_error_threshold: Optional[float] = None,
    tile_score_threshold: Optional[float] = None,
    buffer_hops: int = 0,
    strict: bool = True,
) -> PackedKernel:
    """Pack ``k`` copies of a CUDA-Q kernel onto disjoint calibrated tiles.

    Automatic mode (the default): ``block`` is a plain ``@cudaq.kernel``
    that allocates its own qubits — no signature convention. Its gate
    list is extracted from the compiled Quake IR (``block_args`` bind any
    runtime arguments; loops, closure captures, and ``list`` indexing
    resolve to literal gates), tiles are selected from ``profile``
    calibration via :func:`~qkan.solver.layout.tile_disjoint` (or taken
    from ``tiles`` when given), and the copies are rebuilt into one
    kernel that applies each copy's gates at its tile's physical indices
    and measures the full register — which also keeps idle qubits alive
    on hardware pipelines that compact untouched qubits. The block must
    not measure (readout is added automatically) or branch on
    measurement results, and must not call sub-kernels.

    Legacy mode: with explicit ``tiles`` and a block written over
    ``(q: cudaq.qview, layout: list[int], offset: int)``, the copies are
    composed by sub-kernel calls exactly as written; runtime-parameterized
    packed kernels can be hand-written this way (see the packing guide).

    Whether the physical indices survive to hardware depends on the
    CUDA-Q target (see the packing guide); on simulators the placement
    is exact.
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
                "pack_kernel: blocks taking a qview must use the legacy "
                "convention (q: cudaq.qview, layout: list[int], offset: "
                "int); for runtime-parameterized packing write the packed "
                "kernel by hand (see the packing guide)"
            )
        if tiles is None:
            raise ValueError(
                "pack_kernel: automatic tiling requires a plain kernel; "
                "blocks over (qview, layout, offset) need explicit tiles"
            )
        if block_args:
            raise ValueError("pack_kernel: block_args only apply to plain kernels")
        tiles, m, flat, width, copies = _validated_tiles(tiles)

        @cudaq.kernel
        def packed():
            q = cudaq.qvector(width)
            for t in range(copies):
                block(q, flat, t * m)
            # Full-register measurement: keeps idle qubits alive through
            # hardware pipelines that compact untouched qubits (dqe).
            mz(q)  # noqa: F821

        return PackedKernel(
            kernel=packed,
            tiles=tiles,
            flat=tuple(flat),
            width=width,
            block_qubits=m,
            copies=copies,
        )

    if not hasattr(block, "prepare_call"):
        raise TypeError(
            f"pack_kernel: expected a @cudaq.kernel, got "
            f"'{type(block).__name__}' — kernel-builder PyKernels cannot "
            "be introspected; pass the decorated kernel"
        )

    from ._quake import kernel_ops

    ops, m, flags = kernel_ops(block, *block_args)
    if "measurements" in flags:
        raise ValueError(
            "pack_kernel: the block measures — packing measures the full "
            "register automatically; remove mz/reset from the block"
        )
    if "conditionals" in flags:
        raise ValueError(
            "pack_kernel: mid-circuit conditionals are not supported by "
            "automatic packing"
        )
    if tiles is None:
        if profile is None:
            raise ValueError(
                "pack_kernel: automatic tiling needs calibration — pass "
                "profile=DeviceProfile.from_...(...) or explicit tiles"
            )
        tiles = _select_tiles(
            profile,
            _edges_of_ops(ops),
            m,
            k,
            "pack_kernel",
            max_readout_error=max_readout_error,
            qubit_error_threshold=qubit_error_threshold,
            edge_error_threshold=edge_error_threshold,
            tile_score_threshold=tile_score_threshold,
            buffer_hops=buffer_hops,
            strict=strict,
        )
    else:
        _edges_of_ops(ops)  # >2-qubit gates fail identically in both modes
    tiles, m_tiles, flat, width, copies = _validated_tiles(tiles)
    if m_tiles != m:
        raise ValueError(
            f"pack_kernel: tiles have {m_tiles} qubits but the block has {m}"
        )
    gates = tuple((name, tuple(params), tuple(qubits)) for name, params, qubits in ops)
    return PackedKernel(
        kernel=_recompose(gates, tiles, width),
        tiles=tiles,
        flat=tuple(flat),
        width=width,
        block_qubits=m,
        copies=copies,
        gates=gates,
    )
