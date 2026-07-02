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


"""Regression tests for the qiskit half of qkan.solver.packing."""

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402

from qkan.solver.layout import DeviceProfile  # noqa: E402
from qkan.solver.packing import interaction_of, pack_circuit  # noqa: E402


def line_backend(n=8, seed=1234):
    coupling = [[i, i + 1] for i in range(n - 1)]
    return GenericBackendV2(num_qubits=n, coupling_map=coupling, seed=seed)


def bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_interaction_of():
    qc = bell()
    qc.cx(0, 1)
    assert interaction_of(qc) == [(0, 1), (0, 1)]  # duplicates weight scoring
    qc3 = QuantumCircuit(3)
    qc3.ccx(0, 1, 2)
    with pytest.raises(ValueError, match="3 qubits"):
        interaction_of(qc3)


def test_pack_circuit_pins_disjoint_tiles():
    backend = line_backend()
    packed = pack_circuit(backend, bell(), k=2)
    assert packed.copies == 2
    flat = [q for tile in packed.tiles for q in tile]
    assert len(set(flat)) == len(flat)
    perm = packed.isa.layout.routing_permutation()
    assert perm == list(range(len(perm)))
    assert packed.isa.num_qubits == backend.num_qubits


def test_pack_circuit_detects_routing_from_mismatched_profile():
    # (0, 2) and (1, 3) are not coupled on a line backend, so honoring
    # these tiles requires routing — which must raise, not pass silently
    # (routing SWAPs are lowered to CX before count_ops, so a literal
    # "swap" check can never fire).
    backend = line_backend(n=4)
    profile = DeviceProfile(
        num_qubits=4,
        edges=((0, 2), (1, 3)),
        gate_error_2q={(0, 2): 0.01, (1, 3): 0.01},
    )
    with pytest.raises(RuntimeError, match="routed"):
        pack_circuit(backend, bell(), k=2, profile=profile)


def test_pack_circuit_rejects_measured_circuits():
    # compose() would map every copy's measurements onto the same clbits
    # and in-circuit measurements corrupt estimator expectation values.
    qc = bell()
    qc.measure_all()
    with pytest.raises(ValueError, match="classical bits"):
        pack_circuit(line_backend(), qc, k=2)


def test_pack_circuit_uncoupled_qubits_get_best_effort_entries():
    # Qubits with no 2-qubit coupling to the rest (1q-gate-only or idle)
    # are placed best-effort — the tiles still span the full block.
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    packed = pack_circuit(line_backend(), qc, k=2)
    assert all(len(tile) == 3 for tile in packed.tiles)
    flat = [q for tile in packed.tiles for q in tile]
    assert len(set(flat)) == len(flat)

    idle = QuantumCircuit(3)
    idle.h(0)
    idle.cx(0, 1)
    packed = pack_circuit(line_backend(), idle, k=2)
    assert all(len(tile) == 3 for tile in packed.tiles)


def test_observable_validation():
    packed = pack_circuit(line_backend(), bell(), k=2)
    with pytest.raises(IndexError, match="out of range"):
        packed.observable("ZZ", 2)
    with pytest.raises(IndexError, match="out of range"):
        packed.observable("ZZ", -1)
    with pytest.raises(ValueError, match="block has 2"):
        packed.observable("ZZZ", 0)
    obs = packed.observable("ZZ")
    assert len(obs) == 2
    assert all(o.num_qubits == packed.isa.num_qubits for o in obs)


def test_observable_expectation_roundtrip():
    aer_primitives = pytest.importorskip("qiskit_aer.primitives")

    packed = pack_circuit(line_backend(), bell(), k=2)
    obs = [o for basis in ("XX", "ZZ") for o in packed.observable(basis)]
    noiseless = aer_primitives.EstimatorV2()
    values = noiseless.run([(packed.isa, obs)], precision=0).result()[0].data.evs
    assert all(abs(v - 1.0) < 1e-6 for v in values)


def test_parameter_batch_binds_per_copy():
    import math

    aer_primitives = pytest.importorskip("qiskit_aer.primitives")
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    vqc = QuantumCircuit(2)
    vqc.ry(theta, 0)
    vqc.cx(0, 1)
    packed = pack_circuit(line_backend(), vqc, k=2)
    assert len(packed.parameters) == 2

    batch = [0.4, 1.3]
    values = packed.parameter_values([[t] for t in batch])
    obs = packed.observable("ZI")  # <Z> of block qubit 1 (little-endian)
    estimator = aer_primitives.EstimatorV2()
    evs = estimator.run([(packed.isa, obs, values)], precision=0).result()[0].data.evs
    for got, t in zip(evs, batch):
        assert abs(got - math.cos(t)) < 1e-6

    with pytest.raises(ValueError, match="2 copies"):
        packed.parameter_values([[0.1]])
    with pytest.raises(ValueError, match="takes 1"):
        packed.parameter_values([[0.1, 0.2], [0.3, 0.4]])


def test_pooled_mean_observable():
    aer_primitives = pytest.importorskip("qiskit_aer.primitives")

    packed = pack_circuit(line_backend(), bell(), k=2)
    pooled = packed.observable("ZZ", "mean")
    estimator = aer_primitives.EstimatorV2()
    value = estimator.run([(packed.isa, pooled)], precision=0).result()[0].data.evs
    assert abs(float(value) - 1.0) < 1e-9
    with pytest.raises(ValueError, match="no parameters"):
        packed.parameter_values([[1.0], [1.0]])
