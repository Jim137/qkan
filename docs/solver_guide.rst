.. solver_guide:

Solver Guide
============

QKAN supports multiple solver backends for computing quantum variational activation functions.
Choose the right solver based on your hardware, model size, and deployment target.

Solver Overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 12 12 40 36

   * - Solver
     - Device
     - Use Case
     - Install
   * - ``exact``
     - CPU / GPU
     - Default. Pure PyTorch reference implementation. Best for debugging and small models.
     - Included with ``qkan``
   * - ``flash``
     - GPU
     - Triton fused kernels. Best speed/memory tradeoff for most GPU training.
     - Included with ``qkan[gpu]``
   * - ``cutile``
     - GPU
     - cuTile (NVIDIA Tile Language) fused kernels. BF16/FP8 mixed-precision with coalesced state layout.
     - ``pip install cuda-tile``
   * - ``cutn``
     - GPU / CPU
     - cuQuantum tensor network contraction. Best for extremely large layers near OOM.
     - GPU: ``pip install cuquantum``, CPU: ``pip install opt-einsum``
   * - ``qml``
     - CPU
     - PennyLane quantum circuits. For demonstration, not optimized.
     - ``pip install pennylane``
   * - ``qiskit``
     - IBM QPU
     - IBM Quantum backends via Qiskit Runtime. For real quantum device inference.
     - ``pip install qiskit qiskit-ibm-runtime``
   * - ``cudaq``
     - QPU / GPU
     - NVIDIA CUDA-Q. Supports AWS Braket QPUs, GPU simulators, and CPU simulators.
     - See `CUDA-Q installation <https://nvidia.github.io/cuda-quantum/>`_


Ansatz Choice
-------------

- **``pz`` (default)**: Most reliable quality across tasks. Uses RZ-RY-RZ rotation layers with data re-uploading.
- **``rpz``**: Reduced pz encoding with trainable preactivation. Fewer parameters per layer.
- **``real``**: Real-valued ansatz (no complex arithmetic). Can be faster but may hurt accuracy on some tasks.

**Recommendation**: Start with ``pz``. Only switch to ``real`` if you validate it on your task.


Mixed Precision
---------------

The ``flash`` and ``cutile`` solvers support BF16 and FP8 compute via the ``c_dtype`` parameter:

.. code-block:: python

   qkan = QKAN([10, 10], solver="flash", c_dtype=torch.bfloat16, device="cuda")

- ``c_dtype``: Compute dtype for quantum simulation kernels (state vectors, trig ops).
- ``p_dtype``: Parameter storage dtype (theta, preacts). Keep at ``float32``.

Performance (from `#12 <https://github.com/Jim137/qkan/issues/12>`_):

- **BF16**: 2.3--2.5x faster training, 45% less peak memory, identical convergence.
- **FP8** (``torch.float8_e4m3fn``): Additional memory savings for state checkpoints via prescaled storage.
- All ansatzes (``pz``, ``rpz``, ``real``) are supported.

.. note::
   ``p_dtype=torch.float8_e4m3fn`` is **not supported** — PyTorch has no FP8 arithmetic kernels.
   Use FP8 only for ``c_dtype``.


Performance Tuning
------------------

Three opt-in features trade compute or accuracy for memory. All default to off,
so behavior matches earlier releases unless you enable them.

.. list-table::
   :header-rows: 1
   :widths: 28 20 26 26

   * - Feature
     - Scope
     - Saves
     - Costs
   * - ``checkpoint_reps``
     - exact solver
     - ~1/reps rep-state memory
     - +1 forward pass
   * - ``fused_epilogue``
     - flash / cute / cutile epilogue
     - 2 intermediate buffers, ~3 launches
     - CUDA-only fast path
   * - bf16 optimizer state
     - ``QKANBeliefMini`` / ``TritonAdaBelief``
     - 50% of optimizer state
     - bf16 storage for ``m``, ``s``

Activation Checkpointing the Rep Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``checkpoint_reps`` flag on ``QKANLayer`` / ``QKAN`` wraps each iteration of
the inner rep loop with ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``.
Intermediate state tensors are recomputed in the backward pass instead of being
stashed for autograd, removing ~1/reps of the saved-for-backward memory.

.. code-block:: python

   qkan = QKAN(
       [10, 10],
       reps=8,
       solver="exact",
       checkpoint_reps=True,   # rep states recomputed on backward
       device="cuda",
   )

When to use:

- Training the ``exact`` solver near OOM (large ``batch × reps × group``).
- ``reps`` is high enough that the extra forward is cheaper than swapping.

Tradeoff: ~33% saved-tensor memory reduction at the cost of one extra forward
pass per step. Numerics are bit-identical to the unflagged path. Non-``exact``
solvers ignore this kwarg; they manage rep memory in fused kernels.

Fused Triton Epilogue
~~~~~~~~~~~~~~~~~~~~~

``QKANLayer`` ends every forward with
``(postacts + postact_bias) * postact_weights`` summed across the rep dim, plus
``base_weight @ base_input``. By default this is about five eager ops. Enabling
the fused epilogue collapses it into a single Triton kernel
(``qkan_epilogue_forward`` / ``qkan_epilogue_backward``), skipping two
intermediate buffers and ~3 kernel launches per layer.

Enable per layer:

.. code-block:: python

   qkan = QKAN([10, 10], solver="flash", device="cuda")
   for layer in qkan.qkan_layers:
       layer.set_fused_epilogue(True)

Or enable it process-wide before model construction:

.. code-block:: bash

   QKAN_FUSED_EPILOGUE=1 python train.py

When to use:

- CUDA training/inference where the post-activation epilogue is a measurable
  fraction of step time (deep stacks, small ``out_dim``, or after the quantum
  kernel itself has been fused).

Tradeoff: CUDA-only fast path; CPU tensors fall back to the eager chain
transparently. Numerics match eager within bf16/f32 rounding; gradients are
fused into one backward kernel and the matmul-shaped grads stay on cuBLAS.

BF16 Optimizer Mini-State
~~~~~~~~~~~~~~~~~~~~~~~~~

The QKAN-aware optimizers ``QKANBeliefMini`` (pure-PyTorch) and
``TritonAdaBelief`` (fused Triton) accept a ``state_dtype`` kwarg. Setting it
to ``torch.bfloat16`` stores the first/second moment buffers (``m`` and the
block-reduced ``s``) in bf16, halving optimizer-state memory. Compute remains
fp32 — the Triton kernel upcasts on load; the eager path uses torch's
bf16-aware add/mul.

.. code-block:: python

   from qkan.optim import TritonAdaBelief

   opt = TritonAdaBelief(
       qkan.parameters(),
       lr=1e-3,
       state_dtype=torch.bfloat16,   # halves optimizer memory
   )

When to use:

- Optimizer state dominates GPU memory (millions of params, small batch).
- You are already comfortable with bf16 training.

Tradeoff: ~2x reduction in optimizer-state memory. Bit-exact convergence is
not guaranteed against fp32 state, but tracks closely in practice. If your
params are themselves bf16 and you pass ``state_dtype=None``, ``s`` accumulates
squared residuals in bf16 and may underflow on long runs — pass
``state_dtype=torch.float32`` explicitly to be safe. See :doc:`optim_guide` for
the full optimizer API.


Real Quantum Device Deployment
------------------------------

QKAN can run inference on real quantum hardware. The workflow:

1. **Train locally** with ``exact`` or ``flash`` solver on CPU/GPU.
2. **Transfer weights** to a device-backed model using ``initialize_from_another_model``.
3. **Run inference** on the QPU.

IBM Quantum (Qiskit)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from qiskit_ibm_runtime import QiskitRuntimeService

   service = QiskitRuntimeService(channel="ibm_quantum_platform")
   backend = service.least_busy(operational=True, simulator=False)

   ibm_model = QKAN(
       [1, 2, 1], solver="qiskit", fast_measure=False,
       solver_kwargs={
           "backend": backend,
           "shots": 1000,
           "optimization_level": 3,
           "parallel_qubits": backend.num_qubits,
       },
   )
   ibm_model.initialize_from_another_model(trained_model)

AWS Braket via CUDA-Q
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   qpu_model = QKAN(
       [1, 2, 1], solver="cudaq", fast_measure=False,
       solver_kwargs={
           "target": "braket",
           "machine": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
           "shots": 1000,
           "parallel_qubits": 20,
       },
   )
   qpu_model.initialize_from_another_model(trained_model)

Key parameters:

- ``fast_measure=False``: Required for real devices. Uses ``|alpha|^2 - |beta|^2`` (Born rule) instead of the quantum-inspired ``|alpha| - |beta|`` shortcut.
- ``parallel_qubits``: Packs N independent single-qubit circuits onto N qubits of one multi-qubit job, reducing QPU submissions by ~Nx.
- ``shots``: Number of measurement samples per circuit. More shots = less statistical noise.
- ``initial_layout``: Controls which physical qubits receive the packed circuit (see below).
  Defaults to ``None`` (let the transpiler choose).
  Applies only when executing through a ``backend``; a user-supplied ``estimator`` receives circuits without qkan-side transpilation.
- ``qubit_error_threshold``: Optional single-qubit ``sx`` gate-error threshold for ``initial_layout="auto"``.
  For example, ``0.001`` means ``sx`` error <= 0.1%, equivalent to single-qubit fidelity >= 99.9%.
- ``max_readout_error``: Optional readout-error threshold for ``initial_layout="auto"``.


Qubit-calibration layout
~~~~~~~~~~~~~~~~~~~~~~~~

Modern IBM devices have substantial per-qubit calibration variance — on a snapshot of ``FakeSherbrooke``, readout error across the physical qubits ``0..19`` that a level-1 transpile assigns by default spans 1.1% to 25.7% (a 23x spread), and the chip-wide spread is larger still.
When the transpiler maps a packed N-qubit circuit onto physical qubits ``0..N-1``, several poorly-calibrated qubits end up in the mix, and their readout and gate errors directly bias every expectation value in the packed job.

Fix: pin the packed circuit to the best-calibrated qubits via ``initial_layout`` in ``solver_kwargs``.

.. code-block:: python

   from qkan.solver import best_qubits

   layout = best_qubits(backend, 20)

   model = QKAN(
       [1, 2, 1], solver="qiskit", fast_measure=False,
       solver_kwargs={
           "backend": backend,
           "shots": 1024,
           "parallel_qubits": 20,
           "initial_layout": layout,
       },
   )

Alternatively, pass ``"auto"`` to let ``qiskit_solver`` compute the layout internally from the current backend calibration:

.. code-block:: python

   solver_kwargs={
       "backend": backend,
       "shots": 1024,
       "parallel_qubits": 20,
       "initial_layout": "auto",
   }

To exclude qubits above a single-qubit gate-error threshold, add ``qubit_error_threshold``.
For example, ``0.001`` keeps qubits whose IBM-native ``sx`` gate error is at or below 0.1%:

.. code-block:: python

   solver_kwargs={
       "backend": backend,
       "shots": 1024,
       "parallel_qubits": 20,
       "initial_layout": "auto",
       "qubit_error_threshold": 0.001,
   }

``best_qubits(backend, n)`` scores each physical qubit by

.. math::

   \mathrm{score}(q) = \mathrm{readout\_error}(q) +
                        \mathrm{sx\_err}(q) +
                        10^{-4} / \max(T_2(q)\,[\mu s],\, 1)

and returns the ``n`` lowest-scoring qubit indices.
Readout error dominates the sum; sx error breaks ties; short :math:`T_2` is penalised only slightly because QKAN's shallow single-qubit circuits aren't T2-sensitive.
Qubits whose calibration reports NaN (faulty qubits) are treated as missing data.

Use ``max_readout_error`` when readout filtering is also desired.
If a threshold is supplied and fewer than ``n`` qubits satisfy it, ``best_qubits`` raises ``ValueError`` instead of silently using noisier qubits.
With ``parallel_qubits="auto"`` and ``initial_layout="auto"``, QKAN instead packs onto all qubits that satisfy the threshold, so the packing width follows the current calibration quality.

**Empirical impact.** Smoke test on ``FakeSherbrooke`` with a trained single-sample forecast, ``parallel_qubits=20``, ``shots=1024``:

+-----------------------------+--------------------------+
| Layout                      | rel MSE vs noiseless ref |
+=============================+==========================+
| ``parallel_qubits=1``       | 0.134%                   |
| (serial baseline, qubit 0)  |                          |
+-----------------------------+--------------------------+
| naive ``0..19``             | 5.218%                   |
+-----------------------------+--------------------------+
| ``best_qubits(backend, 20)``| 0.127%                   |
+-----------------------------+--------------------------+

The smart layout at ``parallel_qubits=20`` fully recovers the ``parallel_qubits=1`` fidelity (a ~40x improvement over the naive layout) at identical QPU cost.

**Caveats and scope.**

- Real-backend calibration drifts over time.
  ``initial_layout="auto"`` re-resolves the layout on every forward call, but ``qiskit_ibm_runtime`` caches ``backend.properties()`` after the first fetch — call ``backend.properties(refresh=True)`` (or re-create the backend) before long runs to pick up fresh calibration.
  An explicit layout computed once via ``best_qubits`` is frozen for the lifetime of the model.
- The helper assumes independent single-qubit circuits (the QKAN ``parallel_qubits`` packing pattern).
  If you add 2-qubit gates, you also need connectivity-aware routing.
- ``best_qubits`` returns ``[]`` when ``backend.properties()`` is unavailable (e.g. ``AerSimulator`` or ``GenericBackendV2``, which lack the legacy calibration API); the solver treats an empty layout as ``None`` and warns before falling back to the transpiler default.
- A layout longer than a packed chunk is truncated to the chunk's width (keeping the best-ranked qubits), so ragged final chunks are handled; a layout *shorter* than the packed width raises ``ValueError``.
- ``initial_layout`` applies only when executing through a ``backend``; a user-supplied ``estimator`` receives circuits without qkan-side transpilation (a warning is emitted).


Error Mitigation
----------------

Real quantum hardware introduces gate errors, readout noise, and shot noise. QKAN provides
framework-level error mitigation via the ``mitigation`` key in ``solver_kwargs``:

.. code-block:: python

   solver_kwargs={
       "backend": backend,
       "shots": 1000,
       "parallel_qubits": 127,
       "mitigation": {
           "zne": {"scale_factors": [1, 3, 5]},  # Zero-Noise Extrapolation
           "n_repeats": 3,                        # Multi-shot averaging
           "clip_expvals": True,                  # Clamp <Z> to [-1, 1]
       },
   }

**Zero-Noise Extrapolation (ZNE)**: Runs circuits at amplified noise levels (via gate folding)
and Richardson-extrapolates to the zero-noise limit. For Qiskit, you can alternatively use
``resilience_level=2`` for Qiskit-native ZNE.

**Multi-shot averaging** (``n_repeats``): Runs the entire batch N times and averages results.
Reduces variance from shot-to-shot fluctuations.

**Expectation clipping** (``clip_expvals``): Clamps values to [-1, 1] after mitigation.
Prevents catastrophic outliers from ZNE extrapolation.

.. list-table:: Mitigation Cost
   :header-rows: 1

   * - Technique
     - Circuit multiplier
     - When to use
   * - ``clip_expvals`` only
     - 1x
     - Always (zero cost)
   * - ``n_repeats=3``
     - 3x
     - When shot noise dominates
   * - ZNE ``[1, 3, 5]``
     - 3x
     - When gate noise dominates
   * - ZNE + ``n_repeats=3``
     - 9x
     - Maximum accuracy, inference only

IBM-specific options (passed directly, not under ``mitigation``):

.. code-block:: python

   solver_kwargs={
       "resilience_level": 2,  # Qiskit-native ZNE
       "twirling": {"enable_gates": True, "enable_measure": True},
   }
