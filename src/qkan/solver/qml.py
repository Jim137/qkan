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


import numpy as np
import torch


def qml_solver(x: torch.Tensor, theta: torch.Tensor, reps: int, **kwargs):
    """
    Single-qubit data reuploading circuit using PennyLane.

    Args
    ----
        x : torch.Tensor
            shape: (batch_size, in_dim)
        theta : torch.Tensor
            shape: (reps, 2)
        reps : int
        qml_device : str
            default: "default.qubit"
    """
    import pennylane as qml  # type: ignore

    qml_device: str = kwargs.get("qml_device", "default.qubit")
    dev = qml.device(qml_device, wires=1)

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor, theta: torch.Tensor):
        """
        Args
        ----
            x : torch.Tensor
                shape: (batch_size, in_dim)
            theta : torch.Tensor
                shape: (reps, 2)
        """
        qml.RY(np.pi / 2, wires=0)
        for l in range(reps):
            qml.RZ(theta[l, 0], wires=0)
            qml.RY(theta[l, 1], wires=0)
            qml.RZ(x, wires=0)
        qml.RZ(theta[reps, 0], wires=0)
        qml.RY(theta[reps, 1], wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit(x, theta)
