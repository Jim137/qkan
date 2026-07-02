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


"""Regression tests for the CUDA-Q half of qkan.solver.packing."""

import pytest

cudaq = pytest.importorskip("cudaq")

from qkan.solver.layout import DeviceProfile  # noqa: E402
from qkan.solver.packing import (  # noqa: E402
    kernel_interaction_of,
    pack_circuit,
    pack_kernel,
)

SHOTS = 4096


@pytest.fixture(autouse=True)
def _local_simulator():
    # Never let a leaked remote target receive direct kernel executions.
    cudaq.set_target("qpp-cpu")
    assert not cudaq.get_target().is_remote()


@pytest.fixture(scope="module")
def profile():
    ring = [(i, (i + 1) % 8) for i in range(8)]
    return DeviceProfile(
        num_qubits=8,
        edges=tuple(tuple(sorted(e)) for e in ring),
        readout_error={i: 0.01 for i in range(8)},
        gate_error_1q={i: 0.001 for i in range(8)},
        gate_error_2q={tuple(sorted(e)): 0.005 for e in ring},
    )


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])  # noqa: F821
    x.ctrl(q[0], q[1])  # noqa: F821


def test_auto_pack_dispatch_and_correlations(profile):
    packed = pack_circuit(profile, bell, k=3)
    assert packed.copies == 3
    assert packed.gates is not None
    flat = [q for tile in packed.tiles for q in tile]
    assert len(set(flat)) == len(flat)
    result = cudaq.sample(packed.kernel, shots_count=SHOTS)
    assert packed.z_parity(result, [0, 1]) == [1.0, 1.0, 1.0]
    # Sampled bitstrings span the full register (explicit mz(q)).
    assert all(len(bits) == packed.width for bits in result)


def test_basis_kernel_bell_stabilizers(profile):
    packed = pack_circuit(profile, bell, k=2)
    xx = packed.z_parity(
        cudaq.sample(packed.basis_kernel("XX"), shots_count=SHOTS), [0, 1]
    )
    yy = packed.z_parity(
        cudaq.sample(packed.basis_kernel("YY"), shots_count=SHOTS), [0, 1]
    )
    assert xx == [1.0, 1.0]
    assert yy == [-1.0, -1.0]
    with pytest.raises(ValueError, match="2 qubits"):
        packed.basis_kernel("XXX")
    with pytest.raises(ValueError, match="I, X, Y"):
        packed.basis_kernel("XQ")


def test_spin_op_observe(profile):
    packed = pack_circuit(profile, bell, k=2)
    bare = packed.observe_kernel()
    for pauli, want in (("XX", 1.0), ("YY", -1.0), ("ZZ", 1.0)):
        for t in range(packed.copies):
            ev = cudaq.observe(bare, packed.spin_op(pauli, t)).expectation()
            assert abs(ev - want) < 1e-9
    with pytest.raises(IndexError, match="out of range"):
        packed.spin_op("ZZ", 5)


def test_parameterized_block_args(profile):
    import math

    @cudaq.kernel
    def rotate(theta: float, n: int):
        q = cudaq.qvector(2)
        for _ in range(n):
            ry(theta, q[0])  # noqa: F821
        x.ctrl(q[0], q[1])  # noqa: F821

    packed = pack_circuit(profile, rotate, k=2, block_args=(0.4, 3))
    result = cudaq.sample(packed.kernel, shots_count=100000)
    for value in packed.z_parity(result, [0]):
        assert abs(value - math.cos(1.2)) < 0.03
    assert packed.z_parity(result, [0, 1]) == [1.0, 1.0]


def test_kernel_interaction_of_matches_qiskit():
    qiskit = pytest.importorskip("qiskit")

    @cudaq.kernel
    def line4():
        q = cudaq.qvector(4)
        h(q[0])  # noqa: F821
        for i in range(3):
            x.ctrl(q[i], q[i + 1])  # noqa: F821
        x.ctrl(q[0], q[1])  # noqa: F821

    from qkan.solver.packing import interaction_of

    circuit = qiskit.QuantumCircuit(4)
    circuit.h(0)
    for i in range(3):
        circuit.cx(i, i + 1)
    circuit.cx(0, 1)
    assert kernel_interaction_of(line4) == interaction_of(circuit)


def test_legacy_convention_block():
    @cudaq.kernel
    def block(q: cudaq.qview, layout: list[int], off: int):
        h(q[layout[off]])  # noqa: F821
        x.ctrl(q[layout[off]], q[layout[off + 1]])  # noqa: F821

    packed = pack_kernel(block, [(0, 1), (3, 4)])
    assert packed.gates is None
    result = cudaq.sample(packed.kernel, shots_count=SHOTS)
    assert packed.z_parity(result, [0, 1]) == [1.0, 1.0]
    with pytest.raises(ValueError, match="automatic"):
        packed.basis_kernel("XX")
    with pytest.raises(ValueError, match="block_args"):
        pack_kernel(block, [(0, 1)], block_args=(1,))
    with pytest.raises(ValueError, match="explicit tiles"):
        pack_kernel(block)


def test_plain_kernel_manual_tiles():
    packed = pack_kernel(bell, [(5, 6), (2, 1)])
    assert packed.gates is not None
    result = cudaq.sample(packed.kernel, shots_count=SHOTS)
    assert packed.z_parity(result, [0, 1]) == [1.0, 1.0]


def test_pack_validation(profile):
    @cudaq.kernel
    def measured():
        q = cudaq.qvector(2)
        h(q[0])  # noqa: F821
        mz(q[0])  # noqa: F821

    with pytest.raises(ValueError, match="measures"):
        pack_circuit(profile, measured, k=2)

    @cudaq.kernel
    def toffoli():
        q = cudaq.qvector(3)
        x.ctrl(q[0], q[1], q[2])  # noqa: F821

    with pytest.raises(ValueError, match="3 qubits"):
        pack_circuit(profile, toffoli, k=1)

    @cudaq.kernel
    def sub(q: cudaq.qview):
        h(q[0])  # noqa: F821

    @cudaq.kernel
    def composed():
        q = cudaq.qvector(2)
        sub(q)

    with pytest.raises(ValueError, match="sub-kernel"):
        pack_circuit(profile, composed, k=1)

    with pytest.raises(TypeError, match="DeviceProfile"):
        pack_circuit(None, bell, k=2)
    with pytest.raises(TypeError, match="unsupported"):
        pack_circuit(profile, 42, k=2)
    with pytest.raises(ValueError, match="overlap"):
        pack_kernel(bell, [(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="same size"):
        pack_kernel(bell, [(0, 1), (2,)])
    with pytest.raises(ValueError, match="non-empty"):
        pack_kernel(bell, [])
    with pytest.raises(ValueError, match="block has 2"):
        pack_kernel(bell, [(0, 1, 2)])


def test_z_parity_rejects_compacted_bitstrings():
    # A result whose bitstrings are narrower than the packed register
    # (e.g. from a hand-written block with explicit mz) must be rejected.
    packed = pack_kernel(bell, [(0, 1), (3, 4)])
    with pytest.raises(ValueError, match="must not measure"):
        packed.z_parity({"01": 10}, [0], 0)


def test_uncoupled_qubit_gets_best_effort_entry(profile):
    @cudaq.kernel
    def with_1q_tail():
        q = cudaq.qvector(3)
        h(q[0])  # noqa: F821
        x.ctrl(q[0], q[1])  # noqa: F821
        x(q[2])  # noqa: F821

    packed = pack_circuit(profile, with_1q_tail, k=2)
    assert all(len(tile) == 3 for tile in packed.tiles)
    result = cudaq.sample(packed.kernel, shots_count=SHOTS)
    assert packed.z_parity(result, [0, 1]) == [1.0, 1.0]
    assert packed.z_parity(result, [2]) == [-1.0, -1.0]


def test_unsupported_quake_ops_fail_loud(profile):
    # Ops the extractor cannot rebuild must raise, never vanish silently.
    @cudaq.kernel
    def with_exp_pauli():
        q = cudaq.qvector(2)
        h(q[0])  # noqa: F821
        exp_pauli(0.25, q, "XY")  # noqa: F821

    with pytest.raises(ValueError, match="unsupported quake op"):
        pack_circuit(profile, with_exp_pauli, k=1)

    numpy = pytest.importorskip("numpy")
    cudaq.register_operation("qkan_test_gate", numpy.array([0, 1, 1, 0]))

    @cudaq.kernel
    def with_custom():
        q = cudaq.qvector(2)
        h(q[0])  # noqa: F821
        qkan_test_gate(q[1])  # noqa: F821

    with pytest.raises(ValueError, match="unsupported quake op"):
        pack_circuit(profile, with_custom, k=1)


def test_non_convention_qview_block_rejected_at_pack_time():
    @cudaq.kernel
    def qview_only(q: cudaq.qview):
        h(q[0])  # noqa: F821

    with pytest.raises(ValueError, match="legacy convention"):
        pack_kernel(qview_only, [(0, 1)])

    @cudaq.kernel
    def qview_param(q: cudaq.qview, theta: float):
        ry(theta, q[0])  # noqa: F821

    with pytest.raises(ValueError, match="legacy convention"):
        pack_kernel(qview_param, [(0, 1)])


def test_tile_entry_normalization():
    numpy = pytest.importorskip("numpy")
    tiles = [
        (numpy.int64(0), numpy.int64(1)),
        (numpy.int64(3), numpy.int64(4)),
    ]
    packed = pack_kernel(bell, tiles)
    assert packed.tiles == ((0, 1), (3, 4))
    result = cudaq.sample(packed.kernel, shots_count=SHOTS)
    assert packed.z_parity(result, [0, 1]) == [1.0, 1.0]
    with pytest.raises(ValueError, match="non-negative"):
        pack_kernel(bell, [(-1, 0)])
    with pytest.raises(TypeError):
        pack_kernel(bell, [(0.5, 1.5)])


def test_builder_kernel_rejected():
    builder = cudaq.make_kernel()
    q = builder.qalloc(2)
    builder.h(q[0])
    with pytest.raises(TypeError, match="kernel-builder"):
        pack_kernel(builder, [(0, 1)])
