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

import random
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn

from .torch_qc import StateVector, TorchGates


def torch_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    x: torch.Tensor, shape: (batch_size, dim)
    theta: torch.Tensor, shape: (dim, reps, 2)
    reps: int
    device: str, default: "cpu"
    no_sum: bool, default: False
    ansatz: str, options: ["pz_encoding", "px_encoding"], default: "pz_encoding"
    """
    device = kwargs.get("device", "cpu")
    ansatz = kwargs.get("ansatz", "pz_encoding")

    encoded_x = [
        torch.einsum("i,bi->bi", preacts_weight[:, l], x).add(preacts_bias[:, l])
        for l in range(reps)
    ]

    def pz_encoding(encoded_x: list[torch.Tensor], theta: torch.Tensor):
        """
        x: torch.Tensor, shape: (batch_size, dim)
        theta: torch.Tensor, shape: (dim, reps, 2)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            device=device,
        )  # psi.state: torch.Tensor, shape: (batch_size, dim, 2)
        psi.h()
        for l in range(reps):
            psi.rz(theta[:, l, 0])
            psi.ry(theta[:, l, 1])
            psi.state = torch.einsum(
                "mnbi,bin->bim",
                TorchGates.rz_gate(encoded_x[l]),
                psi.state,
            )

        psi.rz(theta[:, reps, 0])
        psi.ry(theta[:, reps, 1])
        return psi.measure_z()  # shape: (batch_size, dim)

    if ansatz == "pz_encoding":
        circuit = pz_encoding
    else:
        raise NotImplementedError()
    return circuit(encoded_x, theta)  # shape: (batch_size, dim)


class DARUAN(nn.Module):
    def __init__(
        self,
        dim: int,
        reps: int,
        device="cpu",
        solver: Literal["qml", "exact"] = "exact",
        ansatz="pz_encoding",
        preact_trainable: bool = False,
        postact_weight_trainable: bool = False,
        postact_bias_trainable: bool = False,
        seed=0,
    ):
        super(DARUAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.dim = dim
        self.reps = reps
        self.device = device
        self.solver: Literal["qml", "exact"] = solver
        self.ansatz = ansatz

        if ansatz == "pz_encoding":
            self.theta = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(dim, reps + 1, 2, device=device))
            )
        elif ansatz == "px_encoding":
            self.theta = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(dim, reps + 1, 1, device=device))
            )
        else:
            raise NotImplementedError()

        self.preacts_trainable = preact_trainable
        self.preacts_weight = nn.Parameter(
            torch.ones(dim, reps, device=device),
            requires_grad=preact_trainable,
        )
        self.preacts_bias = nn.Parameter(
            torch.zeros(dim, reps, device=device),
            requires_grad=preact_trainable,
        )

        self.postact_weight_trainable = postact_weight_trainable
        self.postact_weights = nn.Parameter(
            torch.ones(dim, device=device),
            requires_grad=postact_weight_trainable,
        )
        self.postact_bias_trainable = postact_bias_trainable
        self.postact_bias = nn.Parameter(
            torch.zeros(dim, device=device),
            requires_grad=postact_bias_trainable,
        )

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.dim, "Invalid input dimension"

        x = x.to(self.device)
        if self.solver == "qml":
            raise NotImplementedError
        elif self.solver == "exact":
            postacts = torch_exact_solver(
                x,
                self.theta,
                self.preacts_weight,
                self.preacts_bias,
                self.reps,
                device=self.device,
                ansatz=self.ansatz,
            )
        else:
            assert False, "Invalid solver"
        x = torch.einsum("bi,i->bi", postacts, self.postact_weights).add(
            self.postact_bias
        )
        return x
