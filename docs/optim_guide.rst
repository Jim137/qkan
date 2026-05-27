.. optim_guide:

Optimizer Guide
===============

.. note::

   **Pre-release.** The ``qkan.optim`` module is shipping in v0.2.3 as a
   preview. APIs, defaults, and the choice of which variants are kept may
   change before the next stable release. Pin ``qkan==0.2.3.*`` if you
   depend on the current surface.

QKAN ships a small family of optimizers tuned for its parameter layout.
The main parameters (``theta``, ``preacts_*``, ``base_weight``,
``postact_*``) are indexed by output / input edges ``(O, I)`` plus a
small trailing fan (reps ``R``, ``K``). That structure makes
Adam-mini-style block-shared second moments and single-kernel fused
updates useful.

This guide covers ``qkan.optim``: a vanilla AdaBelief, a fused
Triton AdaBelief, three block-aware variants (``QKANAdamMini``,
``QKANSpectralMini``, ``QKANBeliefMini``), an L-BFGS finisher schedule,
and a checkpoint-portability helper.


Optimizer Overview
------------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 24 40

   * - Optimizer
     - State vs AdamW
     - Speed (RTX 5090, GPT-2-small stack)
     - Use case
   * - ``AdaBelief``
     - same (2N)
     - 4.79 ms/step (eager)
     - Drop-in Adam replacement. Better preconditioner for QKAN's noisy quantum-circuit gradients.
   * - ``TritonAdaBelief``
     - same (2N), or 0.5x with bf16
     - ~2.6 ms/step (~2.0 ms bf16)
     - Same algorithm as ``AdaBelief``, single fused Triton kernel. ~7x fewer launches per step.
   * - ``QKANAdamMini``
     - ~0.5x (N + #blocks)
     - n/a
     - Block-shared 2nd moment, per ``(o, i, r)`` for theta. ~30% less optimizer state.
   * - ``QKANBeliefMini``
     - ~0.5x; ~0.25x with bf16 state
     - n/a
     - AdaBelief + Adam-mini block partitioning. State between Adam and full AdaBelief in convergence.
   * - ``QKANSpectralMini``
     - same (2N), state on rank-1 sub-manifold
     - n/a
     - Eigenvalue-aware Adam-mini using a Sherman-Morrison rank-1 GN preconditioner. No eigendecomp.
   * - ``LBFGSFinisher``
     - depends on early opt
     - n/a
     - Adam (or any optimizer) for the warmup phase, then L-BFGS line search. Polishes the minimum 2-10x on KAN-style fits.

Benchmark setup for the Triton numbers: a 50257x768 embedding, 96
weights of shape 768x768, and 48 biases of length 768
(GPT-2-small-shaped). All optimizers run the same per-parameter
algorithm; the speedup comes from collapsing ~7 elementwise kernels
into one Triton launch.


AdaBelief
---------

Drop-in Adam replacement from `arXiv:2010.07468
<https://arxiv.org/abs/2010.07468>`_. AdaBelief replaces Adam's second
moment ``v = EMA(g^2)`` with ``s = EMA((g - m)^2)`` — the variance of
the gradient around its EMA. The update is otherwise identical to Adam.

For QKAN this is materially better than vanilla Adam: the
data-reuploading angle injects stochasticity through the input ``x``, so
raw gradient magnitude is a noisy preconditioner. The variance form
down-weights noisy directions and amplifies consistent ones.

.. code-block:: python

   from qkan.optim import AdaBelief

   opt = AdaBelief(model.parameters(), lr=1e-2, weight_decay=0.0)

   for step in range(num_steps):
       opt.zero_grad()
       loss = loss_fn(model(x), y)
       loss.backward()
       opt.step()

Memory and per-step compute match Adam. For QKAN, the default
``lr=1e-2`` is much larger than Adam's typical ``1e-3`` — sweep on your
task.


TritonAdaBelief
---------------

``TritonAdaBelief`` uses the same algorithm as
:class:`~qkan.optim.AdaBelief`, but collapses the full per-parameter
step (lerp + sub + mul + addcmul + sqrt + add + addcdiv) into one
Triton kernel. PyTorch ships ``AdamW(fused=True)`` but has no equivalent
``_fused_adabelief_``; this fills the gap.

Speed on the GPT-2-small-shaped parameter stack (RTX 5090):

.. list-table::
   :header-rows: 1

   * - Optimizer
     - ms / step
     - Notes
   * - ``AdamW(fused=True)``
     - 2.43
     - PyTorch's fused Adam
   * - ``AdaBelief`` (eager)
     - 4.79
     - Reference
   * - ``TritonAdaBelief``
     - ~2.6
     - -46% vs eager AdaBelief
   * - ``TritonAdaBelief`` bf16 state
     - ~2.0
     - -58% vs eager, -50% optimizer memory

.. code-block:: python

   import torch
   from qkan.optim import TritonAdaBelief

   opt = TritonAdaBelief(
       model.parameters(),
       lr=1e-2,
       state_dtype=torch.bfloat16,   # halve optimizer memory
   )

The kernel computes in fp32 via implicit upcasts on load, even when
state is bf16. CPU tensors and non-Triton builds fall back to eager
AdaBelief — the same algorithm, just per-op.


QKANAdamMini
------------

Adam-mini (`arXiv:2406.16793 <https://arxiv.org/abs/2406.16793>`_) with
QKAN-aware block partitioning. The first moment ``m`` stays
per-parameter; the second moment ``v`` collapses to one scalar per
block.

Block partition (empirically tuned for QKAN, one level finer than the
strict Hessian per-output rule):

* ``theta`` natural ``(O, I, R+1, K)`` → block per ``(o, i, r)``,
  collapse only the trailing ``K`` axis. ``v`` shape ``(O, I, R+1)``.
* ``preacts_*`` natural ``(O, I, R)`` → block per ``(o, i)``, collapse
  ``R``. ``v`` shape ``(O, I)``.
* ``(O, I)`` params (``base_weight``, ``postact_*``) → one block per
  tensor.
* Non-QKAN parameters: per-row for 2-D weight matrices (matches the
  Adam-mini paper's MLP rule), per-tensor for 1-D / LayerNorm / bias.

Memory is ``N + #blocks`` floats instead of Adam's ``2N``. The natural
shape comes from the ``_qkan_natural_shape`` attribute that ``QKANLayer``
stores on its parameters, so the partition is independent of the
storage ``p_dim``.

.. code-block:: python

   from qkan.optim import QKANAdamMini

   # Pass named_parameters — names enable per-edge blocking for theta / preacts.
   opt = QKANAdamMini(model.named_parameters(), lr=1e-3)

Passing bare ``model.parameters()`` (without names) falls back to
per-tensor blocking. This is still correct, but it gives no memory
savings.


QKANBeliefMini
--------------

``QKANBeliefMini`` is the AdaBelief variant of ``QKANAdamMini``: it
uses the same block partition rule, but applies it to AdaBelief's
variance ``s`` instead of Adam's ``v``. It sits between Adam and full
AdaBelief in convergence quality while keeping ~30% smaller optimizer
state.

.. code-block:: python

   import torch
   from qkan.optim import QKANBeliefMini

   opt = QKANBeliefMini(
       model.named_parameters(),
       lr=1e-2,
       state_dtype=torch.bfloat16,   # half-size m and s
   )

The ``state_dtype`` knob is shared with ``TritonAdaBelief`` — see the
bf16 caveat below.


QKANSpectralMini
----------------

``QKANSpectralMini`` is a sibling of ``QKANAdamMini`` that exploits the
rank-1-plus-diagonal structure of the QKAN per-block Hessian
(empirically a Gauss-Newton outer product plus a small diagonal — see
``analyze_qkan_hessian.py``).

For each block ``b``, it stores a single direction vector ``c_b`` (EMA
of the gradient inside the block) and preconditions with the
Sherman-Morrison inverse of ``c_b c_b^T + lambda I``:

.. code-block:: text

   upd_b = (1 / lambda) * (g_b - (<c_b, g_b> / (lambda + |c_b|^2)) * c_b)

No eigendecomposition — the rank-1 inverse is closed-form. ``eps``
(default ``1e-3``) plays the role of ``lambda`` and acts as a damping
floor on the curvature estimate.

.. code-block:: python

   from qkan.optim import QKANSpectralMini

   opt = QKANSpectralMini(
       model.named_parameters(),
       lr=1e-3,
       eps=1e-3,   # Sherman-Morrison damping
   )

   # Inspect the block partition before training:
   for name, shape, block_ndim, n_blocks in opt.describe_layout():
       print(f"{name}: shape={shape}  block_ndim={block_ndim}  n_blocks={n_blocks}")

State cost is ``m`` (per-parameter, like Adam) plus ``c`` (also
per-parameter — ``c`` lives at parameter resolution and is reshaped into
blocks at step time). Memory is comparable to Adam, but the
"second-moment" information lives on a rank-1 sub-manifold per block.


bf16 ``state_dtype``
--------------------

Both :class:`~qkan.optim.TritonAdaBelief` and
:class:`~qkan.optim.QKANBeliefMini` accept a ``state_dtype`` argument:

.. code-block:: python

   opt = TritonAdaBelief(
       model.parameters(),
       lr=1e-2,
       state_dtype=torch.bfloat16,   # -50% optimizer memory
   )

``None`` (default) inherits the parameter dtype. Passing
``torch.bfloat16`` halves optimizer memory at near-zero quality cost
for fp32 parameters. The Triton kernel recomputes in fp32 via implicit
upcasts on load, and torch's add/mul handle bf16 EMAs correctly enough
for these small accumulators.

.. warning::

   **bf16 params + ``state_dtype=None``**. If the model parameters
   themselves are bf16 and you leave ``state_dtype=None``, the variance
   ``s`` accumulates squared residuals in bf16 and may underflow on
   long runs. Pass ``state_dtype=torch.float32`` explicitly when
   training bf16 weights.


L-BFGS Finisher
---------------

The original KAN paper (`arXiv:2404.19756
<https://arxiv.org/abs/2404.19756>`_) and ``pykan`` use a two-phase
schedule for symbolic-regression fits: a first-order optimizer for the
bulk of training to find a good basin, then L-BFGS to polish the
minimum. The BFGS curvature approximation typically reduces final loss
by 2-10x on KAN-style tasks.

:class:`~qkan.optim.LBFGSFinisher` wraps any early optimizer for the
first ``pct_early`` fraction of total steps, then auto-switches to
``torch.optim.LBFGS`` with strong-Wolfe line search. The closure
interface is uniform across the swap point — L-BFGS *requires* a
closure that re-evaluates the loss, and the early phase uses the same
closure.

.. code-block:: python

   from qkan.optim import adam_then_lbfgs

   opt = adam_then_lbfgs(
       model,
       total_steps=2000,
       lr_adam=1e-2,
       pct_adam=0.7,             # Adam for 1400 steps, L-BFGS for 600
       use_adam_mini=True,       # use QKANAdamMini in the early phase
   )

   def closure():
       opt.zero_grad()
       loss = loss_fn(model(x), y)
       loss.backward()
       return loss

   for _ in range(2000):
       loss = opt.step(closure)
       if opt.using_lbfgs:
           ...   # post-switch logging, e.g. early stop on tolerance_grad

For manual control, build the wrapper directly:

.. code-block:: python

   from qkan.optim import LBFGSFinisher, QKANAdamMini

   early = QKANAdamMini(model.named_parameters(), lr=1e-2)
   opt = LBFGSFinisher(
       early=early,
       params=model.parameters(),
       total_steps=2000,
       pct_early=0.7,
       lbfgs_kwargs=dict(max_iter=20, history_size=100,
                         tolerance_grad=1e-7,
                         line_search_fn="strong_wolfe"),
   )


Cross-``p_dim`` Checkpoint Portability
--------------------------------------

QKAN's ``p_dim`` knob changes the storage rank of ``theta``,
``preacts_*``, and ``(O, I)`` parameters (for example, 4-D natural vs
2-D collapsed). Model ``state_dict`` entries are reshaped on load via
per-module hooks, but optimizer state (``exp_avg``, ``exp_avg_sq``,
``momentum_buffer``, ...) stays in the parameter shape first seen by the
optimizer.

Call :func:`~qkan.optim.reshape_optimizer_state` immediately after
``optimizer.load_state_dict(...)`` if the loaded model's ``p_dim``
differs from the checkpoint's:

.. code-block:: python

   from qkan.optim import reshape_optimizer_state

   opt.load_state_dict(checkpoint["optimizer"])
   n_reshaped = reshape_optimizer_state(opt)
   print(f"reshaped {n_reshaped} optimizer tensors to current p_dim")

State tensors whose ``numel`` matches the new parameter are reshaped in
place. Tensors with different element counts are left untouched, and the
optimizer will raise on the next step — the right behavior for a genuine
model/checkpoint mismatch.


When to use which
-----------------

- **Default for QKAN GPU training**: ``TritonAdaBelief`` with
  ``state_dtype=torch.bfloat16``. Cheapest per-step on small / medium
  parameter stacks, with a variance-form denominator suited to noisy
  QKAN gradients.
- **CPU or no-Triton build**: ``AdaBelief`` — same algorithm,
  eager backend.
- **Memory-bound** (large QKAN, optimizer state matters):
  ``QKANAdamMini`` or ``QKANBeliefMini`` (with bf16 state for an extra
  halving).
- **Curvature-aware experimentation**: ``QKANSpectralMini``. The rank-1
  GN preconditioner is the cheapest second-order signal you can layer
  onto an Adam-mini block partition.
- **Final loss matters more than wall-clock**: ``adam_then_lbfgs`` —
  the L-BFGS polish at the end is where ``pykan``-style fits get their
  symbolic-regression accuracy.
