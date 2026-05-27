.. graph_guide:

Inference and Training Guide
============================

QKAN is launch-bound at small batch sizes: a single QKAN forward issues
roughly 8 GPU kernels, each preceded by ~2 us of Python / CUDA driver
overhead. The quantum-activation kernels are short, so for typical
GPT-style HQKAN blocks the host launch cost dominates the GPU work —
both at inference time and during training. ``qkan.inference`` provides
three graph-capture helpers that collapse those launches:

- **``torch.compile``-style shape-cached wrapper** —
  ``compile_inference`` / ``CompiledInference``. Drop-in; lazily
  captures one CUDA graph per input shape.
- **Direct single-shape CUDA-graph capture** — ``make_graphed_inference``
  for eval, ``make_graphed_train_step`` for a full forward + backward
  + zero_grad train step.
- **Selective submodule capture** — ``graph_submodules`` for models
  where only a few submodules (e.g. every MLP block) are launch-bound.

On launch-bound workloads each typically yields a 2--3x speedup.

This guide covers :ref:`inference <graph-inference>`,
:ref:`training <graph-training>`, and shared
:ref:`caveats <graph-caveats>`.


Which helper should I use?
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Helper
     - Use when
     - Avoid when
   * - ``compile_inference``
     - Drop-in: you want graphs, but the eval batch shape can vary
       across calls. Lazily captures one graph per shape (up to
       ``max_shapes``) and falls back to eager for training /
       grad-tracking forwards.
     - You need a single ultra-low-overhead callable and you already
       know the shape (use the lower-level ``make_graphed_inference``
       instead).
   * - ``make_graphed_inference``
     - You know the inference shape ahead of time and want the minimum
       per-call dispatch overhead — no cache lookup, just replay.
     - Shapes change between calls (raises ``ValueError``).
   * - ``make_graphed_train_step``
     - Training is launch-bound and forward + backward + grad-zero fit
       one graph. Captures everything except ``optimizer.step()`` for
       one fixed shape.
     - The optimizer or loss path has host-side scalar work that must
       remain eager, or you need shape variability.
   * - ``graph_submodules``
     - Full-model capture fails (Python control flow, ``.item()`` calls,
       SDPA RNG counters, etc.) but the launch-bound cost is
       concentrated in a few submodules (e.g. every MLP block).
     - The entire model is graph-safe — wrap the whole thing instead.

Quick decision tree:

::

    Is the full model graph-safe?
    |-- No  --> graph_submodules(model, sample, predicate=...)
    |-- Yes
        |
        Is this training (forward + backward + optimizer)?
        |-- Yes --> make_graphed_train_step(model, x, y, loss_fn, opt)
        |-- No
            |
            Does the eval shape vary across calls?
            |-- Yes --> compile_inference(model)
            |-- No  --> make_graphed_inference(model, x)


.. _graph-inference:

Inference
---------

``compile_inference`` (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The drop-in API. It wraps any module so eval / no-grad forwards capture
and replay per shape; train / grad forwards fall back to eager.

.. code-block:: python

   import torch
   from qkan import QKAN, compile_inference

   model = QKAN(width=[2, 8, 2], reps=2, solver="flash", device="cuda").eval()
   model = compile_inference(model)         # drop-in wrapper

   x = torch.randn(64, 2, device="cuda")
   with torch.no_grad():
       y = model(x)                          # first call: captures
       y = model(x)                          # subsequent calls: replays

The cache key is ``(shape, dtype, device)``. On a miss the wrapper
captures a new graph (up to ``max_shapes`` entries, default 8). When
the cache is saturated, further unseen shapes fall through to eager
rather than evict an existing graph mid-replay.

``compile_inference`` is a thin shortcut for
``CompiledInference(module, ...)``. The class form is equivalent:

.. code-block:: python

   from qkan import CompiledInference
   model = CompiledInference(model, max_shapes=16, warmup=3)

After modifying parameters in place (e.g. loading a new checkpoint),
call ``model.clear_cache()`` to drop stale graphs.
``model.train()`` clears the cache automatically.


``make_graphed_inference`` (low-level, single shape)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use when the inference shape is known and fixed, and you want the
absolute minimum per-call overhead — no shape cache lookup, just
``copy_`` + replay.

.. code-block:: python

   import torch
   from qkan import QKAN, make_graphed_inference

   model = QKAN(width=[2, 8, 2], reps=2, solver="flash", device="cuda")
   x = torch.randn(64, 2, device="cuda")

   graphed = make_graphed_inference(model, x)
   y = graphed(x)                            # replays graph

The returned callable accepts inputs with the same shape / dtype /
device as ``sample_input``. Mismatches raise ``ValueError``. The output
tensor is **reused across calls** — clone it before the next replay if
you need to keep the value:

.. code-block:: python

   y1 = graphed(x1).clone()                  # keep this result
   y2 = graphed(x2)                          # overwrites the previous static buffer


``graph_submodules`` (selective capture)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When full-model capture fails (for example, because an SDPA backend
increments an RNG counter even at ``dropout_p=0``, or because Python
``if`` branches depend on tensor values), wrap only the launch-bound
submodules. QKAN's small-launch cost typically lives in the MLP blocks,
so graphing those captures most of the gain without making the whole
model graph-safe.

.. code-block:: python

   import torch
   from qkan import graph_submodules
   from my_model import GPT2, HQKANMLP

   gpt2 = GPT2(...).cuda().eval()
   sample_idx = torch.randint(0, vocab_size, (1, 128), device="cuda")

   graph_submodules(
       gpt2,
       sample_idx,
       predicate=lambda m: isinstance(m, HQKANMLP),
   )

   with torch.no_grad():
       logits = gpt2(sample_idx)             # MLP blocks now graphed; rest is eager

Each matching submodule is replaced in place with a
``CompiledInference`` wrapper that shares parameters with the original.
``named_parameters()`` is unchanged. The helper runs one initial eval
forward, so the first live call does not pay the capture cost.


.. _graph-training:

Training
--------

QKAN training is launch-bound for the same reason inference is — the
quantum-activation kernels are short, and a single train step issues
forward + backward + zero_grad worth of small kernels. Capturing the
whole train step as one CUDA graph cuts the same host overhead.

``make_graphed_train_step``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Captures **forward + loss + backward + zero_grad** into a single CUDA
graph for one fixed-shape train step. ``optimizer.step()`` is **not**
captured — optimizers like Adam contain host-side scalar work that
breaks capture, so the caller invokes ``step()`` after each replay. This
also gives you a stable place to plug in QKAN-aware optimizers like
``TritonAdaBelief`` from :doc:`optim_guide`.

.. code-block:: python

   import torch
   from torch import nn
   from qkan import QKAN, TritonAdaBelief, make_graphed_train_step

   model = QKAN(width=[2, 8, 2], reps=2, solver="flash", device="cuda")
   opt = TritonAdaBelief(model.parameters(), lr=1e-3)
   loss_fn = nn.MSELoss()

   # Sample tensors must match the shape / dtype of real batches.
   x = torch.randn(64, 2, device="cuda")
   y = torch.randn(64, 2, device="cuda")

   train_step = make_graphed_train_step(model, x, y, loss_fn, opt)

   for xb, yb in data_loader:                # xb, yb must match shape / dtype
       loss = train_step(xb, yb)             # replays graph (zero_grad + fwd + bwd)
       opt.step()                            # not in graph — call yourself


Side effects on ``model`` and ``optimizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Every trainable ``Parameter`` is re-allocated onto the capture stream
  so the autograd accumulator binds to the right stream. References
  held by the caller (e.g. ``my_layer.theta``) are updated, but
  references taken **before** calling ``make_graphed_train_step`` go
  stale — re-fetch ``model.parameters()`` if you cached anything.
- ``optimizer.param_groups`` is rewired to the new ``Parameter``
  objects. ``optimizer.state`` is re-keyed best-effort, but state
  tensors are **not migrated** across streams. Run this on a
  freshly-built optimizer (no momentum yet) for cleanest behavior.

Inside the captured graph,
``optimizer.zero_grad(set_to_none=False)`` zeroes ``.grad`` in place
to preserve graph-internal pointers. **Do not** call
``optimizer.zero_grad(set_to_none=True)`` between replays — swapping
``.grad`` for a new tensor outside the graph invalidates the capture.

The returned loss is a view into a static buffer; clone it if you
need to retain it past the next replay.


Mixing in eager steps
~~~~~~~~~~~~~~~~~~~~~

If you need an occasional eager step (eval pass, validation step, LR
scheduler tick that mutates parameter groups, etc.), run it between
graph replays as long as you do not reassign ``.grad`` or reshape any
captured tensor. Eval-mode forwards through the model also work — the
train-step graph only references the train pathway.

For sporadic shape changes (e.g. last batch has a different size),
fall back to an eager step for that batch; capturing per shape isn't
supported by ``make_graphed_train_step`` today.


.. _graph-caveats:

Caveats
-------

**CUDA graphs require fixed shapes.**
   A captured graph hard-codes input shape, dtype, and device. Calling
   a graphed callable with a different shape raises ``ValueError``
   for the low-level helpers; ``compile_inference`` captures a new
   graph (up to ``max_shapes``) and falls back to eager once the
   cache is saturated. If your eval batch size varies a lot, either
   widen ``max_shapes`` or pad inputs to a fixed shape.

**Output tensors are reused buffers.**
   ``make_graphed_inference`` returns a callable whose output is a
   view into a static buffer overwritten on every replay. Clone the
   result if you need to keep it past the next call.
   The same applies to the loss returned by ``make_graphed_train_step``.
   ``CompiledInference`` inherits this behavior for cache hits.

**Parameter mutation invalidates graphs.**
   The captured graph references specific parameter storage. Editing
   parameters in place (e.g. loading a checkpoint) requires
   ``CompiledInference.clear_cache()``.
   ``train()`` / ``eval()`` transitions clear the cache automatically.

**Non-graph-safe ops break full-model capture.**
   Python control flow on tensor values, ``.item()``, ``.cpu()``, CPU
   syncs, and SDPA backends that touch RNG state all break capture.
   If ``compile_inference`` raises during capture, narrow the scope
   with ``graph_submodules`` to wrap only the graph-safe portions.

**``compile_inference`` falls back to eager in several cases.**
   The wrapper takes the eager path whenever any of the following
   hold: training mode, grad enabled, multiple positional arguments,
   any keyword arguments, non-tensor input, non-CUDA input, or
   ``x.requires_grad``. This keeps the wrapper safe to drop on top of
   any model, but you will see no speedup unless you wrap the forward in
   ``torch.no_grad()`` and call the eval-mode model with a single CUDA
   tensor.

**Warmup cost.**
   Capture is preceded by ``warmup`` (default 3) forward passes on a
   side stream to stabilize cuBLAS / cuDNN algorithm selection. For
   ``make_graphed_train_step``, warmup includes a full backward so
   ``.grad`` buffers are allocated on the capture stream. Expect the
   first call to be noticeably slower; amortize by capturing once
   before timing.

**``compile_inference`` vs. ``torch.compile``.**
   ``compile_inference`` is a CUDA-graph wrapper, not a tracing JIT.
   It does not recompile graphs across shapes — it captures a new one
   per shape. If you want kernel-level fusion or shape-polymorphic
   compilation, use ``torch.compile`` directly. Combine it with CUDA
   graphs by passing ``mode="reduce-overhead"``, or compose
   ``compile_inference`` on top of a ``torch.compile``\d module.
