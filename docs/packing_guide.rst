.. packing_guide:

Circuit Packing and Calibration-Aware Layouts
=============================================

``qkan.solver`` ships a provider-neutral circuit-packing toolkit: calibration-aware selection of qubits, connected subgraphs, and disjoint tiles (``qkan.solver.layout``), plus a packer that turns ``k`` independent copies of a small circuit into one pinned, result-mappable job.
Packing serves two repeated-execution patterns.
The first is the variational batch: one parameterized circuit evaluated at ``k`` parameter sets per job, so a training loop advances ``k`` evaluations per round trip instead of one.
The second is shot reduction for a single circuit: only one circuit needs to run, but packing it ``k`` times collects statistics in parallel, so each job needs only ``shots/k`` shots — 8 copies at 256 shots match 2048 shots of a single placement at one eighth of the QPU time, at the cost of tying the estimate to ``k`` tiles' average noise instead of one tile's.
:func:`~qkan.solver.packing.pack_circuit` is overloaded over both supported stacks with the same selection logic: a qiskit ``QuantumCircuit`` packs against a backend, a plain ``@cudaq.kernel`` packs against a :class:`~qkan.solver.layout.DeviceProfile`; :func:`~qkan.solver.packing.pack_kernel` covers explicit tiles and hand-written composition.
Everything is torch-free and usable independently of QKAN models.

Device profiles
---------------

Calibration enters through :class:`~qkan.solver.layout.DeviceProfile`, a frozen snapshot of per-qubit and per-edge calibration.

.. code-block:: python

   from qkan import DeviceProfile

   profile = DeviceProfile.from_qiskit(backend)            # BackendV2 Target or legacy properties()
   profile = DeviceProfile.from_braket(aws_device)         # Braket standardized schema (experimental)
   profile = DeviceProfile(num_qubits=..., edges=..., ...) # or supply your own data

NaN calibration entries are treated as missing data, unreported qubit labels are marked non-operational, and dead coupling edges (2-qubit error at the physical bound) are excluded from all searches.
Profiles are snapshots: pass ``refresh=True`` (qiskit) or rebuild the profile before long runs to pick up recalibration.

Selection API
-------------

- ``rank_qubits(profile, n, ...)``: top-``n`` independent qubit ranking with threshold (``max_readout_error``, ``qubit_error_threshold``) and ``strict`` semantics.
- ``best_subgraph(profile, interaction, ...)``: best-calibrated connected placement for one multi-qubit circuit, where ``interaction`` is its 2-qubit-gate edge list.
  qiskit's transpiler already does this well at ``optimization_level`` 2-3; use this for explicit thresholds, level-1 pipelines, custom objectives, or non-qiskit stacks.
- ``tile_disjoint(profile, interaction, k, ...)``: tile the chip with ``k`` (or ``"max"``) *disjoint* calibration-aware copies of one circuit — for running independent circuits in parallel on one QPU, which no transpiler currently offers.
- ``score_layout(profile, interaction, layout)``: the fidelity-product cost used by both searches, usable for custom comparisons.

Layouts are positional (``layout[i]`` hosts logical qubit ``i``) and tiles are returned best-first.

Selection is stack-neutral, and CUDA-Q kernels reach it through the same :func:`~qkan.solver.packing.pack_circuit` entry point: ``k=1`` reduces packing to single-circuit placement (``best_subgraph`` semantics).

.. code-block:: python

   import cudaq
   from qkan import pack_circuit

   @cudaq.kernel
   def ghz20():
       q = cudaq.qvector(20)
       h(q[0])
       for i in range(19):
           x.ctrl(q[i], q[i + 1])

   placed = pack_circuit(profile, ghz20, k=1)   # one copy on the best-calibrated path
   result = cudaq.sample(placed.kernel, shots_count=1000)
   zz = [placed.z_parity(result, [i, i + 1], 0) for i in range(19)]

The QKAN CUDA-Q solver applies the same selection to its own packed single-qubit jobs via ``initial_layout="auto"`` (see the solver guide).

Packing with qiskit
-------------------

:func:`~qkan.solver.packing.pack_circuit` is the packing API: it derives the circuit's interaction graph, selects disjoint tiles from live calibration, composes the copies into one circuit, and transpiles it pinned to the tiles (SWAP-free by construction; a routed result, e.g. from a calibration profile that disagrees with the backend, raises ``RuntimeError``).
Packing pays off when the same circuit runs many times, in one of two ways.

The first is the variational batch: pack a parameterized circuit once, and every training step evaluates ``k`` parameter sets in one hardware job — each copy carries its own renamed parameters, bound per step with ``parameter_values``.

.. code-block:: python

   from qkan import pack_circuit
   from qiskit import QuantumCircuit
   from qiskit.circuit import Parameter
   from qiskit_ibm_runtime import EstimatorV2

   theta = Parameter("theta")
   vqc = QuantumCircuit(2)
   vqc.ry(theta, 0)
   vqc.cx(0, 1)

   packed = pack_circuit(backend, vqc, k=8, max_readout_error=0.05)
   obs = packed.observable("ZI")                # any Pauli/SparsePauliOp, mapped per tile
   estimator = EstimatorV2(mode=backend)

   for step in range(steps):
       batch = optimizer.ask(8)                 # 8 parameter sets, one per tile
       values = packed.parameter_values([[t] for t in batch])
       evs = estimator.run([(packed.isa, obs, values)]).result()[0].data.evs
       optimizer.tell(batch, evs)               # evs[t] belongs to batch[t]

The second is shot reduction for a single circuit: pack ``k`` identical copies, run each job at ``shots/k``, and pool the tiles with the averaged observable — same statistics, a ``k``-fold shorter shot budget per job.

.. code-block:: python

   packed = pack_circuit(backend, bell, k=8)
   pooled = packed.observable("ZZ", "mean")     # (1/8) * sum of the per-tile operators
   value = estimator.run([(packed.isa, pooled)]).result()[0].data.evs

``packed.observable(obs, tile)`` maps a block-level Pauli observable onto the packed ISA circuit — one ``tile``, the list over all copies (``None``), or their pooled average (``"mean"``); ``packed.tiles`` / ``packed.physical_qubits(t)`` expose the placement, and the selection thresholds, ``buffer_hops``, ``k="max"``, and ``strict`` semantics are those of :func:`~qkan.solver.layout.tile_disjoint`.
For custom flows the underlying pieces (``tile_disjoint`` plus a concatenated tile-major ``initial_layout``) remain available.

Measured results (Bell pairs, ``|Phi+>`` fidelity per pair):

- ``FakeSherbrooke`` noise model, 8 pairs: mean fidelity **0.969** (worst 0.961) for tiles vs 0.894 (worst 0.775) for qiskit level-3 placement of the merged circuit and 0.816 (worst 0.231) for the trivial layout.
- ``ibm_berlin`` (real hardware, live calibration, default mitigation): worst pair **0.982** for tiles vs 0.951 for level 3, with means at the mitigated ceiling.
- ``ibm_berlin`` GHZ-20 line via ``best_subgraph``: a clean 20-qubit path with min neighbor ``<ZZ>`` **0.970**; before dead-edge pruning the search crossed a calibrated dead edge and that stabilizer collapsed to 0.742 — one bad edge poisons a 20-qubit entangled state, which is exactly what calibration-aware selection prevents.

Packing with CUDA-Q
-------------------

The same :func:`~qkan.solver.packing.pack_circuit` call packs a plain ``@cudaq.kernel`` — no signature convention and no manual tiles.
The kernel's gate list is extracted from its compiled Quake IR (loops, closure captures, and ``list`` indexing resolve to literal gates), tiles come from the same calibration-aware selection, and the copies are rebuilt into one kernel that applies each copy's gates at its tile's physical indices and measures the full register.
The same two repeated-use patterns apply.

The variational batch binds one runtime-argument set per tile (``block_args_batch``); each step re-binds the same tiles with ``rebind`` — only extraction and the kernel rebuild run, in milliseconds.

.. code-block:: python

   import cudaq
   from qkan import DeviceProfile, pack_circuit

   @cudaq.kernel
   def rotate(theta: float):
       q = cudaq.qvector(2)
       ry(theta, q[0])
       x.ctrl(q[0], q[1])

   packed = pack_circuit(profile, rotate, block_args_batch=[(0.3,), (0.9,), (1.7,)])

   for step in range(steps):
       result = cudaq.sample(packed.kernel, shots_count=2048)
       z0 = packed.expectation(result, "ZI")   # one value per tile
       packed = packed.rebind(optimizer.ask(3))

Shot reduction for a single circuit packs identical copies (``block_args`` or none) and pools the tiles with ``"mean"``.

.. code-block:: python

   @cudaq.kernel
   def bell():
       q = cudaq.qvector(2)
       h(q[0])
       x.ctrl(q[0], q[1])

   packed = pack_circuit(profile, bell, k=8, max_readout_error=0.05)
   result = cudaq.sample(packed.kernel, shots_count=256)      # 8 x 256 = 2048 effective
   zz = packed.expectation(result, "ZZ", "mean")

   xx_result = cudaq.sample(packed.basis_kernel("XX"), shots_count=256)
   xx = packed.expectation(xx_result, "XX", "mean")

``packed.expectation(result, pauli, tile)`` is the observable mapping: it reads ``<pauli>`` from a sample of the matching basis (the plain ``kernel`` for Z/I strings, ``basis_kernel(pauli)`` otherwise) — one ``tile``, the list over copies (``None``), or the pooled ``"mean"``.
``packed.basis_kernel(bases)`` rebuilds the packed kernel with per-qubit basis changes before the measurement, so X/Y observables read as Z-parities — the hardware-safe observable path; ``packed.z_parity`` remains the positional primitive underneath.
``packed.spin_op(pauli, tile)`` builds ``cudaq.spin`` operators at a tile's physical indices for ``cudaq.observe`` on ``packed.observe_kernel()``; hardware targets that compact idle qubits reject sparse-index observables, so treat that as the simulator route.
This path is verified numerically on the ``qpp-cpu`` simulator (exact Bell/GHZ correlations at sparse physical tiles; parameterized blocks match ``cos(theta)`` per tile) and at the wire level (the OpenQASM submitted through the ``quantum_machines`` target carries the gates on the exact tile indices over the full register).
Whether a hardware target honors the physical indices matches the single-qubit case: Braket's vendor compiler may rewire (verified on Rigetti hardware for single-qubit packing and for a packed 20-qubit job, whose compiled Quil ran on a different chip region than requested) and ``quantum_machines`` mapping is decided by the Qoperator server.
The native ``iqm`` / ``oqc`` / ``anyon`` pipelines compact untouched qubits, and the packed kernel's full-register measurement is the verified mitigation — gates stay at the exact physical indices through the emulated ``iqm`` codegen.

For explicit control, :func:`~qkan.solver.packing.pack_kernel` accepts hand-picked ``tiles`` (from :func:`~qkan.solver.layout.tile_disjoint` or :func:`~qkan.solver.layout.best_subgraph`) with a plain kernel, or a legacy block written over ``(q: cudaq.qview, layout: list[int], offset: int)`` composed by sub-kernel calls.
Automatic mode bakes argument values at pack time (per tile via ``block_args_batch``, refreshed with ``rebind``); to thread runtime arguments through to sample time unchanged, write the packed kernel by hand following the legacy pattern (a parameterized block kernel plus per-tile argument offsets).

Caveats
-------

- The interaction graph must be connected and embed in the coupling map without routing; pre-transpile and extract the routed interaction graph for circuits that need SWAPs.
- Inputs are measurement-free: ``pack_circuit`` rejects circuits with classical bits (copies would share clbits and in-circuit measurements corrupt estimator values) and kernels that measure (the packed kernel measures the full register itself); ``z_parity`` rejects results whose bitstrings are narrower than the packed register.
- CUDA-Q introspection covers flat kernels: sub-kernel calls, mid-circuit conditionals, and qubit indices that stay dynamic after constant folding are rejected with clear errors, and parameterized kernels are extracted per bound argument set (angles are baked at pack time; ``block_args_batch`` extracts once per tile and ``rebind`` once per step, all entries preserving the block's qubit count and interaction graph).
- The Quake extraction uses internal-but-stable ``cudaq.mlir`` entry points (the same ones cudaq's kernel builder uses) and is pinned per cudaq release by the packing test suite (verified on cudaq 0.15).
- Crosstalk between neighboring tiles is not modeled by public calibration data; ``buffer_hops=1`` leaves a one-qubit shell between tiles at the cost of fewer tiles (52 to 31 Bell tiles on ``FakeSherbrooke``).
- Calibration drifts: rebuild profiles before long runs.
