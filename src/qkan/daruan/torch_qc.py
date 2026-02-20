# Copyright (c) 2025, Jiun-Cheng Jiang. All rights reserved.
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
Synchronous processing of quantum circuits with PyTorch.

NOTE: We will migrate all quantum circuit solvers to this module.

Features:
    - Single-qubit quantum circuits
"""

import math

import torch

INV_SQRT2 = math.sqrt(2.0) / 2.0


class TorchGates:
    @staticmethod
    def identity_gate(dim) -> torch.Tensor:
        """
        return: (2, 2, dim)
        """
        ones = torch.ones(dim)
        zeros = torch.zeros(dim)
        return torch.stack(
            [
                torch.stack([ones, zeros]),
                torch.stack([zeros, ones]),
            ]
        )

    i_gate = identity_gate

    @staticmethod
    def rx_gate(theta: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        """
        theta: (dim,)
        return: (2, 2, dim)
        """
        cos = torch.cos(theta / 2).to(dtype)
        jsin = 1j * torch.sin(-theta / 2)
        return torch.stack(
            [
                torch.stack([cos, jsin]),
                torch.stack([jsin, cos]),
            ]
        )

    @staticmethod
    def ry_gate(theta: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack(
            [
                torch.stack([cos, -sin]),
                torch.stack([sin, cos]),
            ]
        ).to(dtype)

    @staticmethod
    def rz_gate(theta: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        exp = torch.exp(-0.5j * theta)
        zero = torch.zeros_like(theta)
        return torch.stack(
            [
                torch.stack([exp, zero]),
                torch.stack([zero, torch.conj(exp)]),
            ]
        ).to(dtype)

    @staticmethod
    def h_gate(dim, device, dtype=torch.complex64) -> torch.Tensor:
        inv_sqrt2 = torch.full((dim,), INV_SQRT2, device=device, dtype=dtype)
        return torch.stack(
            [
                torch.stack([inv_sqrt2, inv_sqrt2]),
                torch.stack([inv_sqrt2, -inv_sqrt2]),
            ]
        )

    @staticmethod
    def s_gate(dim) -> torch.Tensor:
        ones = torch.ones(dim)
        zeros = torch.zeros(dim)
        return torch.stack(
            [
                torch.stack([ones, zeros]),
                torch.stack([zeros, 1j * ones]),
            ]
        )

    @staticmethod
    def tensor_product(g1, g2):
        """
        g1, g2: (2, 2, dim)
        return: (4, 4, dim)
        """
        dim = g1.shape[-1]
        out = torch.empty(4, 4, dim, dtype=g1.dtype, device=g1.device)
        for i in range(dim):
            out[:, :, i] = torch.kron(g1[:, :, i], g2[:, :, i])
        return out

    @staticmethod
    def cx_gate(dim, control: int, device, dtype=torch.complex64):
        gate = torch.zeros(4, 4, dim, dtype=dtype, device=device)
        gate[0, 0] = 1
        gate[1, 1] = 1
        gate[2, 3] = 1
        gate[3, 2] = 1
        if control == 1:
            gate = gate.transpose(0, 1)
        return gate

    @staticmethod
    def cz_gate(dim, device, dtype=torch.complex64):
        gate = torch.zeros(4, 4, dim, dtype=dtype, device=device)
        gate[0, 0] = 1
        gate[1, 1] = 1
        gate[2, 2] = 1
        gate[3, 3] = -1
        return gate


class StateVector:
    """
    1-qubit state vector.

    state: (batch_size, dim, 2)
    """

    def __init__(
        self,
        batch_size: int,
        dim: int,
        device="cpu",
        dtype=torch.complex64,
    ):
        self.batch_size = batch_size
        self.dim = dim
        self.device = device
        self.dtype = dtype

        self.state = torch.zeros(
            batch_size, dim, 2, dtype=dtype, device=device
        )
        self.state[..., 0] = 1.0

    def measure_z(self, fast_measure=True):
        if fast_measure:
            return self.state[..., 0].abs() - self.state[..., 1].abs()
        return (
            self.state[..., 0].abs().square()
            - self.state[..., 1].abs().square()
        )

    def measure_x(self, fast_measure=True):
        tmp = StateVector(self.batch_size, self.dim, self.device, self.dtype)
        tmp.state.copy_(self.state)
        tmp.h()
        return tmp.measure_z(fast_measure)

    def measure_y(self, fast_measure=True):
        tmp = StateVector(self.batch_size, self.dim, self.device, self.dtype)
        tmp.state.copy_(self.state)
        tmp.s(is_dagger=True)
        tmp.h()
        return tmp.measure_z(fast_measure)

    def s(self, is_dagger=False):
        gate = TorchGates.s_gate(self.dim).to(self.device)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mni,bin->bim", gate, self.state)

    def h(self, is_dagger=False):
        gate = TorchGates.h_gate(self.dim, self.device, self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mni,bin->bim", gate, self.state)

    def x(self):
        self.state = self.state[..., [1, 0]]

    def z(self):
        self.state[..., 1] *= -1

    def rx(self, theta, is_dagger=False):
        gate = TorchGates.rx_gate(theta, self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mni,bin->bim", gate, self.state)

    def ry(self, theta, is_dagger=False):
        gate = TorchGates.ry_gate(theta, self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mni,bin->bim", gate, self.state)

    def rz(self, theta, is_dagger=False):
        gate = TorchGates.rz_gate(theta, self.dtype)
        if is_dagger:
            gate = torch.conj_physical(gate).transpose(0, 1)
        self.state = torch.einsum("mni,bin->bim", gate, self.state)
