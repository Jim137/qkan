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
CUDA-Q kernel introspection via the Quake MLIR representation.

Extracts a ``(gate, params, qubits)`` list from a plain ``@cudaq.kernel``
so :func:`~qkan.solver.packing.pack_circuit` can derive the interaction
graph and rebuild the kernel at tile physical indices with no signature
convention. The pipeline resolves closure captures and runtime arguments
(``prepare_call`` + ``synthesize``), unrolls loops, and constant-folds
``list[int]`` indexing down to literal ``extract_ref`` indices; the
resulting Quake text is regex-parsed. Gate identities are preserved
(controls become a ``c`` prefix, adjoints a ``dg`` suffix).

The pass-manager entry points live under ``cudaq.mlir`` — internal but
stable APIs that cudaq's own kernel builder uses; the Quake text format
is pinned per cudaq release by the packing test suite.
"""

import re

_PIPELINE = (
    "builtin.module(unrolling-pipeline,func.func(canonicalize,cse,"
    "constant-propagation,canonicalize,cse))"
)

_ALLOCA = re.compile(r"(%\w+) = quake\.alloca !quake\.veq<(\d+)>")
_CONST_I = re.compile(r"(%\w+) = arith\.constant (-?\d+) : i64")
_CONST_F = re.compile(r"(%\w+) = arith\.constant (-?[\d.eE+-]+) : f64")
_EXTRACT = re.compile(
    r"(%\w+) = quake\.extract_ref (%\w+)\[(%?\w+)\]"
    r" : \(!quake\.veq<\d+>(?:, i64)?\) -> !quake\.ref"
)
_GATE = re.compile(
    r"^\s*quake\.(\w+)(<adj>)?\s*(\([^)]*\))?\s*(\[[^\]]*\])?\s*([%\w, ]+) :"
)
# Any quake op line, assigned or not; group 1 is the op mnemonic. Ops that
# are neither skippable nor parseable as gates must fail loud — matching
# type annotations like '!quake.veq<2>' instead of the mnemonic previously
# let unsupported ops (exp_pauli, custom unitaries) vanish silently.
_OP = re.compile(r"^\s*(?:[%\w:, ]+=\s*)?quake\.(\w+)")
_ENTRY = re.compile(
    r"func\.func @__nvqpp__mlirgen__\S+\([^)]*\)[^{]*\{(.*?)\n  \}", re.S
)
# Non-gate quake ops that legitimately appear in a block's body.
_SKIP = {
    "mz",
    "mx",
    "my",
    "reset",
    "alloca",
    "dealloc",
    "extract_ref",
    "subveq",
    "discriminate",
}


def kernel_ops(kernel, *args):
    """Extract a plain CUDA-Q kernel's gate list.

    Returns ``(ops, num_qubits, flags)`` where ``ops`` is a list of
    ``(name, params, qubits)`` with literal physical parameters and
    register indices, and ``flags`` is a subset of ``{"measurements",
    "conditionals"}``. ``args`` bind the kernel's runtime arguments
    (required whenever it takes any); closure-captured values are
    resolved automatically. Raises ``ValueError`` for kernels this
    route cannot represent faithfully (sub-kernel calls, dynamic qubit
    indices, no or multiple registers).
    """
    import cudaq  # type: ignore
    from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime  # type: ignore
    from cudaq.mlir.passmanager import PassManager  # type: ignore

    # prepare_call resolves closure-captured values into explicit launch
    # args; synthesize bakes them (and any user args) as constants.
    processed, _ = kernel.prepare_call(*args)
    baked = cudaq.synthesize(kernel, *processed) if processed else kernel
    module = cudaq_runtime.cloneModule(baked.qkeModule)
    manager = PassManager.parse(_PIPELINE, context=module.context)
    cudaq_runtime.runPassManager(manager, module)
    entry = _ENTRY.search(str(module))
    if not entry:
        raise ValueError("kernel_ops: no entry-point function in the Quake module")
    body = entry.group(1)
    if re.search(r"\bcall @|cc\.call_callable", body):
        raise ValueError(
            "kernel_ops: the kernel calls sub-kernels, which cannot be "
            "inlined at the Quake level in cudaq 0.15 — flatten the kernel"
        )

    flags = set()
    if re.search(r"quake\.(mz|mx|my|reset)\b", body):
        flags.add("measurements")
    if "cc.if" in body or "quake.discriminate" in body:
        flags.add("conditionals")

    consts_i = {m.group(1): int(m.group(2)) for m in _CONST_I.finditer(body)}
    consts_f = {m.group(1): float(m.group(2)) for m in _CONST_F.finditer(body)}
    registers = {m.group(1): int(m.group(2)) for m in _ALLOCA.finditer(body)}
    if len(registers) != 1:
        raise ValueError(
            f"kernel_ops: expected the kernel to allocate exactly one "
            f"qvector, found {len(registers)}"
        )
    ((register, num_qubits),) = registers.items()
    refs = {}
    for m in _EXTRACT.finditer(body):
        name, source, index = m.group(1), m.group(2), m.group(3)
        if source != register:
            raise ValueError(f"kernel_ops: extract_ref from unknown register {source}")
        if index.startswith("%"):
            if index not in consts_i:
                raise ValueError(
                    f"kernel_ops: qubit index {index} is not resolvable at compile time"
                )
            refs[name] = consts_i[index]
        else:
            refs[name] = int(index)

    def resolve_qubits(group, line):
        resolved = []
        for token in group.split(","):
            token = token.strip()
            if not token:
                continue
            if token in refs:
                resolved.append(refs[token])
            elif token == register:
                resolved.append(("veq", num_qubits))
            elif token in consts_f or token in consts_i:
                continue
            else:
                raise ValueError(
                    f"kernel_ops: unresolved qubit token {token} in {line.strip()}"
                )
        return resolved

    ops = []
    for line in body.splitlines():
        mnemonic = _OP.match(line)
        if not mnemonic:
            continue
        if mnemonic.group(1) in _SKIP:
            continue
        gate = _GATE.match(line)
        if not gate:
            raise ValueError(
                f"kernel_ops: unsupported quake op "
                f"'{mnemonic.group(1)}': {line.strip()}"
            )
        name, adjoint, params_s, controls_s, targets_s = gate.groups()
        params = []
        if params_s:
            for token in params_s.strip("()").split(","):
                token = token.strip()
                params.append(consts_f[token] if token in consts_f else float(token))
        controls = resolve_qubits((controls_s or "").strip("[]"), line)
        qubits = controls + resolve_qubits(targets_s or "", line)
        if adjoint:
            name += "dg"
        name = "c" * len(controls) + name
        if any(isinstance(q, tuple) for q in qubits):
            # Broadcast form (gate applied to the whole register).
            if len(qubits) == 1:
                ops.extend((name, params, [i]) for i in range(num_qubits))
                continue
            raise ValueError(
                f"kernel_ops: broadcast in multi-qubit gate: {line.strip()}"
            )
        ops.append((name, params, qubits))
    return ops, num_qubits, flags
