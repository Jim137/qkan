"""
Synchrous processing of quantum circuits with PyTorch.

Features:
    - Single-qubit quantum circuits (Faster than other libraries)
    - Two-qubit quantum circuits

Code author: Jiun-Cheng Jiang (Jim137@GitHub)
Contact: [jcjiang@phys.ntu.edu.tw](mailto:jcjiang@phys.ntu.edu.tw)
"""

import math

import torch

# Constants
INV_SQRT2 = math.sqrt(2.0) / 2.0  # 1/sqrt(2) for Hadamard gate


class TorchGates:
    @staticmethod
    def identity_gate(shape) -> torch.Tensor:
        """
        shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """
        return torch.stack(
            [
                torch.stack([torch.ones(*shape), torch.zeros(*shape)]),
                torch.stack([torch.zeros(*shape), torch.ones(*shape)]),
            ],
        )

    @staticmethod
    def i_gate(shape):
        """
        alias for identity_gate

        shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """
        pass

    i_gate = identity_gate  # noqa: F811

    @staticmethod
    def rx_gate(theta: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        """
        theta: torch.Tensor, shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """
        cos = torch.cos(theta / 2).to(dtype)
        jsin = 1j * torch.sin(-theta / 2)
        return torch.stack(
            [
                torch.stack([cos, jsin]),
                torch.stack([jsin, cos]),
            ],
        )

    @staticmethod
    def ry_gate(theta: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        """
        theta: torch.Tensor, shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack(
            [
                torch.stack([cos, -sin]),
                torch.stack([sin, cos]),
            ],
        ).to(dtype)

    @staticmethod
    def rz_gate(theta: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        """
        theta: torch.Tensor, shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """
        exp = torch.exp(-0.5j * theta)
        zero = torch.zeros_like(theta)
        return torch.stack(
            [
                torch.stack([exp, zero]),
                torch.stack([zero, torch.conj(exp)]),
            ],
        ).to(dtype)

    @staticmethod
    def h_gate(shape, device, dtype=torch.complex64) -> torch.Tensor:
        """
        shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """
        # Optimize: use pre-computed constant instead of computing 1/sqrt(2) at runtime
        inv_sqrt2 = torch.full(shape, INV_SQRT2, device=device, dtype=dtype)
        return torch.stack(
            [
                torch.stack([inv_sqrt2, inv_sqrt2]),
                torch.stack([inv_sqrt2, -inv_sqrt2]),
            ],
        )

    @staticmethod
    def s_gate(shape) -> torch.Tensor:
        """
        shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """
        return torch.stack(
            [
                torch.stack([torch.ones(*shape), torch.zeros(*shape)]),
                torch.stack([torch.zeros(*shape), 1j * torch.ones(*shape)]),
            ],
        )

    @staticmethod
    def acrx_gate(theta: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        """
        Complex extension of RX(acos(theta)) gate.
        *Note: Physically unrealizable.*

        theta: torch.Tensor, shape: (out_dim, in_dim)

        return: torch.Tensor, shape: (2, 2, out_dim, in_dim)
        """

        sq = torch.square(theta).flatten()
        diag = torch.mul(
            torch.sqrt(torch.abs(1 - sq)), torch.where(sq < 1, 1j, 1)
        ).reshape(theta.shape)
        return torch.stack(
            [
                torch.stack([theta, diag]),
                torch.stack([diag, theta]),
            ],
        ).to(dtype)

    @staticmethod
    def tensor_product(gate, another_gate, dtype=None):
        """
        Compute tensor product of two gates.

        Arguments
        ---------
            :gate: torch.Tensor, shape: (2, 2, out_dim, in_dim)
            :another_gate: torch.Tensor, shape: (2, 2, out_dim, in_dim)
            :dtype: torch dtype, optional. If None, uses the dtype of the input gate.
                    Both gates should have the same dtype.

        return: torch.Tensor, shape: (4, 4, out_dim, in_dim)
        """
        if dtype is None:
            dtype = gate.dtype
            # Validate that both gates have the same dtype
            if gate.dtype != another_gate.dtype:
                raise ValueError(
                    f"Gate dtypes must match: got {gate.dtype} and {another_gate.dtype}"
                )
        shape = gate.shape[2:]
        gate = gate.view(2, 2, -1)
        another_gate = another_gate.view(2, 2, -1)
        out = torch.empty(
            4,
            4,
            gate.shape[2],
            dtype=dtype,
            device=gate.device,
        )
        for i in range(out.shape[2]):
            out[:, :, i] = torch.kron(gate[:, :, i], another_gate[:, :, i])
        return out.view(4, 4, *shape)

    @staticmethod
    def cx_gate(shape, control: int, device, dtype=torch.complex64) -> torch.Tensor:
        """
        2-qubits CX (CNOT) gate.

        shape: (out_dim, in_dim)
        control: int

        return: torch.Tensor, shape: (4, 4, out_dim, in_dim)
        """
        assert control in (0, 1), "Control qubit must be 0 or 1."

        gate = torch.zeros(4, 4, *shape, dtype=dtype, device=device)
        gate[0, 0] = 1.0
        gate[1, 1] = 1.0
        gate[2, 3] = 1.0
        gate[3, 2] = 1.0
        if control == 1:
            gate = gate.transpose(0, 1)
        return gate

    @staticmethod
    def cz_gate(shape, device, dtype=torch.complex64) -> torch.Tensor:
        """
        2-qubits CZ gate.

        shape: (out_dim, in_dim)
        control: int

        return: torch.Tensor, shape: (4, 4, out_dim, in_dim)
        """

        gate = torch.zeros(4, 4, *shape, dtype=dtype, device=device)
        gate[0, 0] = 1.0
        gate[1, 1] = 1.0
        gate[2, 2] = 1.0
        gate[3, 3] = -1.0
        return gate


class StateVector:
    """
    1-qubit state vector.

    StateVector.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 2)
    """

    state: torch.Tensor

    def __init__(
        self,
        batch_size: int,
        out_dim: int,
        in_dim: int,
        device="cpu",
        dtype=torch.complex64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.state = torch.zeros(
            batch_size, out_dim, in_dim, 2, dtype=dtype, device=self.device
        )
        self.state[:, :, :, 0] = 1.0
        self.dtype = dtype

    def measure_z(self, fast_measure: bool = True) -> torch.Tensor:
        """
        Measure the state vector in the Z basis.

        Arguments
        ---------
            :fast_measure: bool, default: True. If True, for state |ψ⟩ = α|0⟩ + β|1⟩, return |α| - |β|;
                           if False, return |α|^2 - |β|^2.
                           Which is quantum-inspired method and faster when it is True.
        return: torch.Tensor, shape: (batch_size, out_dim, in_dim)
        """
        return (
            self.state[:, :, :, 0].abs() - self.state[:, :, :, 1].abs()
            if fast_measure
            else torch.square(self.state[:, :, :, 0].abs())
            - torch.square(self.state[:, :, :, 1].abs())
        )

    def measure_x(self, fast_measure: bool = True) -> torch.Tensor:
        """
        Measure the state vector in the X basis.

        Arguments
        ---------
            :fast_measure: bool, default: True. If True, for state |ψ⟩ = α|0⟩ + β|1⟩, return |α| - |β|;
                           if False, return |α|^2 - |β|^2.
                           Which is quantum-inspired method and faster when it is True.
        return: torch.Tensor, shape: (batch_size, out_dim, in_dim)
        """
        tmp_state = StateVector(self.batch_size, self.out_dim, self.in_dim, self.device)
        tmp_state.state.copy_(self.state)
        tmp_state.h()
        return tmp_state.measure_z(fast_measure)

    def measure_y(self, fast_measure: bool = True) -> torch.Tensor:
        """
        Measure the state vector in the Y basis.

        Arguments
        ---------
            :fast_measure: bool, default: True. If True, for state |ψ⟩ = α|0⟩ + β|1⟩, return |α| - |β|;
                           if False, return |α|^2 - |β|^2.
                           Which is quantum-inspired method and faster when it is True.
        return: torch.Tensor, shape: (batch_size, out_dim, in_dim)
        """
        tmp_state = StateVector(self.batch_size, self.out_dim, self.in_dim, self.device)
        tmp_state.state.copy_(self.state)
        tmp_state.s(is_dagger=True)
        tmp_state.h()
        return tmp_state.measure_z(fast_measure)

    def s(self, is_dagger: bool = False):
        """
        Apply Phase gate (or S gate) to the state vector.

        Arguments
        ---------
            :is_dagger: bool, default: False
        """
        gate = TorchGates.s_gate(self.state.shape[1:3]).to(self.device)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mnoi,boin->boim", gate, self.state)

    def h(self, is_dagger: bool = False):
        """
        Apply Hadamard gate to the state vector.

        Arguments
        ---------
            :is_dagger: bool, default: False
        """
        gate = TorchGates.h_gate(self.state.shape[1:3], self.device, dtype=self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mnoi,boin->boim", gate, self.state)

    def x(self):
        """
        Apply Pauli-X gate to the state vector.
        """
        self.state = torch.index_select(
            self.state, dim=-1, index=torch.tensor([1, 0], device=self.device)
        )

    def z(self):
        """
        Apply Pauli-Z gate to the state vector.
        """
        self.state[:, :, :, 1] = -self.state[:, :, :, 1]

    def rx(self, theta: torch.Tensor, is_dagger: bool = False):
        """
        Apply Rotation-X gate to the state vector.

        Arguments
        ---------
            :theta: torch.Tensor, shape: (out_dim, in_dim)
            :is_dagger: bool, default: False
        """
        gate = TorchGates.rx_gate(theta, dtype=self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mnoi,boin->boim", gate, self.state)

    def ry(self, theta: torch.Tensor, is_dagger: bool = False):
        """
        Apply Rotation-Y gate to the state vector.

        Arguments
        ---------
            :theta: torch.Tensor, shape: (out_dim, in_dim)
            :is_dagger: bool, default: False
        """
        gate = TorchGates.ry_gate(theta, dtype=self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mnoi,boin->boim", gate, self.state)

    def rz(self, theta: torch.Tensor, is_dagger: bool = False):
        """
        Apply Rotation-Z gate to the state vector.

        Arguments
        ---------
            :theta: torch.Tensor, shape: (out_dim, in_dim)
            :is_dagger: bool, default: False
        """
        gate = TorchGates.rz_gate(theta, dtype=self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mnoi,boin->boim", gate, self.state)


class DQStateVector:
    """
    2-qubit state vector.

    DQStateVector.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 4)
    """

    state: torch.Tensor

    def __init__(
        self,
        batch_size: int,
        out_dim: int,
        in_dim: int,
        device="cpu",
        dtype=torch.complex64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.dtype = dtype
        self.state = torch.zeros(
            batch_size, out_dim, in_dim, 4, dtype=dtype, device=self.device
        )
        self.state[:, :, :, 0] = 1.0

    def measure_z(self, target: int = 0) -> torch.Tensor:
        """
        Measure the state vector in the Z basis.

        return: torch.Tensor, shape: (batch_size, out_dim, in_dim)
        """
        if target == 0:
            return (
                +self.state[:, :, :, 0].abs()
                - self.state[:, :, :, 1].abs()
                + self.state[:, :, :, 2].abs()
                - self.state[:, :, :, 3].abs()
            )
        else:
            return (
                +self.state[:, :, :, 0].abs()
                + self.state[:, :, :, 1].abs()
                - self.state[:, :, :, 2].abs()
                - self.state[:, :, :, 3].abs()
            )

    def cx(self, control: int):
        """
        Apply CX (CNOT) gate to the state vector.

        Arguments
        ---------
            :control: int
        """
        cx_gate = TorchGates.cx_gate(
            self.state.shape[1:3], control, self.device, dtype=self.dtype
        )
        self.state = torch.einsum("mnoi,boin->boim", cx_gate, self.state)

    def cz(self):
        """
        Apply CZ gate to the state vector.
        """
        cz_gate = TorchGates.cz_gate(
            self.state.shape[1:3], self.device, dtype=self.dtype
        )
        self.state = torch.einsum("mnoi,boin->boim", cz_gate, self.state)

    def apply_gate(self, gate: torch.Tensor, target: int = 0):
        """
        Apply a gate to the state vector.

        Arguments
        ---------
            :gate: torch.Tensor, shape: (4, 4, out_dim, in_dim)
        """
        if target == 0:
            gate = TorchGates.tensor_product(
                gate, TorchGates.identity_gate(self.state.shape[1:3])
            )
        else:
            gate = TorchGates.tensor_product(
                TorchGates.identity_gate(self.state.shape[1:3]), gate
            )
        self.state = torch.einsum("mnoi,boin->boim", gate, self.state)

    def apply_2gates(self, gate1: torch.Tensor, gate2: torch.Tensor):
        """
        Apply two gates to the state vector.

        Arguments
        ---------
            :gate1: torch.Tensor, shape: (4, 4, out_dim, in_dim)
            :gate2: torch.Tensor, shape: (4, 4, out_dim, in_dim)
        """
        gate = TorchGates.tensor_product(gate1, gate2)
        self.state = torch.einsum("mnoi,boin->boim", gate, self.state)

    def hh(self):
        """
        Apply Hadamard gate to the state vector.

        Arguments
        ---------
            :is_dagger: bool, default: False
        """
        h_gate = TorchGates.h_gate(self.state.shape[1:3], self.device, dtype=self.dtype)
        self.apply_2gates(h_gate, h_gate)
