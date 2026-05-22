# Copyright (c) 2024, Jiun-Cheng Jiang. All rights reserved.
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


import torch
from torch.utils.checkpoint import checkpoint

from ..torch_qc import StateVector, TorchGates
from ._base import QKANSolver, register


def torch_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Single-qubit data reuploading circuit.

    Args
    ----
        x : torch.Tensor
            shape: (batch_size, in_dim)
        theta : torch.Tensor
            shape: (\\*group, reps, 2)
        preacts_weight : torch.Tensor
            shape: (\\*group, reps)
        preacts_bias : torch.Tensor
            shape: (\\*group, reps)
        reps : int
        ansatz : str
            options: ["pz_encoding", "px_encoding"], default: "pz_encoding"
        n_group : int
            number of neurons in a group, default: in_dim of x

    Returns
    -------
        torch.Tensor
            shape: (batch_size, out_dim, in_dim)
    """
    batch, in_dim = x.shape
    device = x.device
    ansatz = kwargs.get("ansatz", "pz_encoding")
    # group = kwargs.get("group", in_dim)
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", in_dim)
    dtype = kwargs.get("dtype", torch.complex64)
    # Opt-in activation checkpointing: recompute per-rep state in backward
    # instead of storing it. Trades ~1 extra forward pass for reps× less
    # rep-state memory. Only meaningful when grad is required.
    checkpoint_reps = kwargs.get("checkpoint_reps", False) and torch.is_grad_enabled()

    if len(theta.shape) != 4:
        theta = theta.unsqueeze(0)
    if theta.shape[1] != in_dim:
        repeat_out = out_dim
        repeat_in = in_dim // theta.shape[1] + 1
        theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]
    # rpz_encoding always needs encoded_x (with bias), even when preacts_trainable=False
    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    if _needs_encoded_x:
        if len(preacts_weight.shape) != 3:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]
        encoded_x = torch.einsum("oir,bi->boir", preacts_weight, x).add(preacts_bias)
        # encoded_x shape: (batch_size, out_dim, in_dim, reps)

    def _pz_encoding(theta: torch.Tensor):
        """
        Args
        ----
            theta : torch.Tensor
                shape: (\\*group, reps, 2)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 2)
        psi.h()
        if not preacts_trainable:
            rug = TorchGates.rz_gate(x, dtype=dtype)

        def _step(state, th0, th1, data_gate):
            psi.state = state
            psi.rz(th0)
            psi.ry(th1)
            if not preacts_trainable:
                psi.state = torch.einsum("mnbi,boin->boim", data_gate, psi.state)
            else:
                psi.state = torch.einsum("mnboi,boin->boim", data_gate, psi.state)
            return psi.state

        for l in range(reps):
            th0 = theta[:, :, l, 0]
            th1 = theta[:, :, l, 1]
            if preacts_trainable:
                data_gate = TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype)
            else:
                data_gate = rug
            if checkpoint_reps:
                psi.state = checkpoint(
                    _step, psi.state, th0, th1, data_gate, use_reentrant=False
                )
            else:
                psi.state = _step(psi.state, th0, th1, data_gate)

        psi.rz(theta[:, :, reps, 0])
        psi.ry(theta[:, :, reps, 1])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _rpz_encoding(theta: torch.Tensor):
        """
        Args
        ----
            theta : torch.Tensor
                shape: (\\*group, reps, 2)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )
        psi.h()

        def _step(state, th0, data_gate):
            psi.state = state
            psi.ry(th0)
            psi.state = torch.einsum("mnboi,boin->boim", data_gate, psi.state)
            return psi.state

        for l in range(reps):
            th0 = theta[:, :, l, 0]
            data_gate = TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype)
            if checkpoint_reps:
                psi.state = checkpoint(
                    _step, psi.state, th0, data_gate, use_reentrant=False
                )
            else:
                psi.state = _step(psi.state, th0, data_gate)
        psi.ry(theta[:, :, reps, 0])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _px_encoding(theta: torch.Tensor):
        """
        Args
        ----
            theta: torch.Tensor
                shape: (\\*group, reps, 1)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size * g, out_dim, n_group, 2)
        psi.h()

        def _step(state, th0, data_gate):
            psi.state = state
            psi.rz(th0)
            psi.state = torch.einsum("mnboi,boin->boim", data_gate, psi.state)
            return psi.state

        for l in range(reps):
            th0 = theta[:, :, l, 0]
            data_gate = TorchGates.rx_gate(
                torch.acos(encoded_x[:, :, :, l]), dtype=dtype
            )
            if checkpoint_reps:
                psi.state = checkpoint(
                    _step, psi.state, th0, data_gate, use_reentrant=False
                )
            else:
                psi.state = _step(psi.state, th0, data_gate)
            """
            # complex extension implementation
            psi.state = torch.einsum(
                "mnboi,boin->boim",
                TorchGates.acrx_gate(
                    torch.einsum("oi,bi->boi", preacts_weight[:, :, l], x)
                ),
                psi.state,
            )
            """
        psi.rz(theta[:, :, reps, 0])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _real(theta: torch.Tensor):
        """
        Args
        ----
            theta: torch.Tensor
                shape: (\\*group, reps, 1)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 2)
        psi.h()
        if not preacts_trainable:
            rug = TorchGates.ry_gate(x, dtype=dtype)

        def _step(state, th0, data_gate):
            psi.state = state
            psi.x()
            psi.ry(th0)
            psi.z()
            if not preacts_trainable:
                psi.state = torch.einsum("mnbi,boin->boim", data_gate, psi.state)
            else:
                psi.state = torch.einsum("mnboi,boin->boim", data_gate, psi.state)
            return psi.state

        for l in range(reps):
            th0 = theta[:, :, l, 0]
            if preacts_trainable:
                data_gate = TorchGates.ry_gate(encoded_x[:, :, :, l], dtype=dtype)
            else:
                data_gate = rug
            if checkpoint_reps:
                psi.state = checkpoint(
                    _step, psi.state, th0, data_gate, use_reentrant=False
                )
            else:
                psi.state = _step(psi.state, th0, data_gate)
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _mix(theta: torch.Tensor):
        """
        Args
        ----
            theta: torch.Tensor
                shape: (\\*group, reps, 2)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 2)
        psi.h()
        if not preacts_trainable:
            rug_y = TorchGates.ry_gate(x, dtype=dtype)

        def _step(state, th0, th1, data_gate):
            psi.state = state
            psi.rz(th0)
            psi.rx(th1)
            if not preacts_trainable:
                psi.state = torch.einsum("mnbi,boin->boim", data_gate, psi.state)
            else:
                psi.state = torch.einsum("mnboi,boin->boim", data_gate, psi.state)
            return psi.state

        for l in range(reps):
            th0 = theta[:, :, l, 0]
            th1 = theta[:, :, l, 1]
            if preacts_trainable:
                data_gate = TorchGates.ry_gate(encoded_x[:, :, :, l], dtype=dtype)
            else:
                data_gate = rug_y
            if checkpoint_reps:
                psi.state = checkpoint(
                    _step, psi.state, th0, th1, data_gate, use_reentrant=False
                )
            else:
                psi.state = _step(psi.state, th0, th1, data_gate)
        psi.rz(theta[:, :, reps, 0])
        psi.rx(theta[:, :, reps, 1])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    if ansatz == "pz_encoding" or ansatz == "pz":
        circuit = _pz_encoding
    elif ansatz == "rpz_encoding" or ansatz == "rpz":
        circuit = _rpz_encoding
    elif ansatz == "px_encoding" or ansatz == "px":
        circuit = _px_encoding
    elif ansatz == "real":
        circuit = _real
    elif ansatz == "mix":
        circuit = _mix
    elif callable(ansatz):
        circuit = ansatz
    else:
        raise NotImplementedError()
    x = circuit(theta)  # shape: (batch_size, out_dim, in_dim)
    return x


class ExactTorchSolver(QKANSolver):
    """Pure-PyTorch reference solver (registered as ``"exact"``)."""

    name = "exact"

    def __call__(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        preacts_weight: torch.Tensor,
        preacts_bias: torch.Tensor,
        reps: int,
        **kwargs,
    ) -> torch.Tensor:
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )


register(ExactTorchSolver())
