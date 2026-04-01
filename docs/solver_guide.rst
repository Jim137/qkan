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
