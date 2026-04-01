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
Fourier-based Kolmogorov-Arnold Network (FourierKAN) for baseline comparison.

Replaces B-spline basis functions with Fourier basis (sin/cos) as the learnable
activation functions on edges.
"""

import math
from typing import TYPE_CHECKING, Union

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .qkan import QKAN


class FourierKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        scale_base: float = 1.0,
        scale_fourier: float = 1.0,
        enable_standalone_scale_fourier: bool = True,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(FourierKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Precompute the frequency indices: 1, 2, ..., grid_size
        freqs = torch.arange(1, grid_size + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

        # Scale input from grid_range to [-pi, pi]
        self.grid_range = grid_range
        self.register_buffer(
            "input_scale",
            torch.tensor(2.0 * math.pi / (grid_range[1] - grid_range[0])),
        )
        self.register_buffer(
            "input_shift",
            torch.tensor(-grid_range[0] * 2.0 * math.pi / (grid_range[1] - grid_range[0]) - math.pi),
        )

        # Fourier coefficients: (out_features, in_features, 2 * grid_size)
        # First grid_size entries are cosine coefficients, last grid_size are sine
        self.fourier_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, 2 * grid_size)
        )

        # Base weight (linear residual connection through activation)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        if enable_standalone_scale_fourier:
            self.fourier_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_base = scale_base
        self.scale_fourier = scale_fourier
        self.enable_standalone_scale_fourier = enable_standalone_scale_fourier
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            # Initialize Fourier coefficients with small random values
            std = self.scale_fourier / math.sqrt(self.in_features * self.grid_size)
            self.fourier_weight.data.normal_(0, std)
            if self.enable_standalone_scale_fourier:
                torch.nn.init.kaiming_uniform_(
                    self.fourier_scaler, a=math.sqrt(5) * self.scale_fourier
                )

    def fourier_basis(self, x: torch.Tensor):
        """
        Compute the Fourier basis for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Fourier basis tensor of shape (batch_size, in_features, 2 * grid_size).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Map input to [-pi, pi]
        x_scaled = x * self.input_scale + self.input_shift  # (batch, in_features)

        # (batch, in_features, grid_size)
        phase = x_scaled.unsqueeze(-1) * self.freqs

        # Concatenate cos and sin: (batch, in_features, 2 * grid_size)
        bases = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return bases.contiguous()

    @property
    def scaled_fourier_weight(self):
        return self.fourier_weight * (
            self.fourier_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_fourier
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        fourier_output = F.linear(
            self.fourier_basis(x).view(x.size(0), -1),
            self.scaled_fourier_weight.view(self.out_features, -1),
        )
        output = base_output + fourier_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    def fit_from_qkan(
        self, x0: torch.Tensor, y: torch.Tensor, max_iter: int = 200, tol: float = 1e-5
    ):
        """
        Fit FourierKAN layer from QKAN with early stopping.

        Args
        ----
            x0: torch.Tensor
                Input tensor
            y: torch.Tensor
                Target tensor
            max_iter: int
                Maximum number of iterations, default: 200
            tol: float
                Tolerance for early stopping, default: 1e-5
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        criterion = torch.nn.MSELoss()
        prev_loss = float("inf")
        for _ in range(max_iter):
            optimizer.zero_grad()
            x = self.forward(x0)
            loss = criterion(x, y)
            loss.backward()
            optimizer.step()
            if abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        L1 regularization on the Fourier coefficients, analogous to the spline
        weight regularization in KANLinear.
        """
        l1_fake = self.fourier_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class FourierKANModuleList(torch.nn.ModuleList):
    def __init__(self):
        super(FourierKANModuleList, self).__init__()

    def __getitem__(self, idx) -> Union[FourierKANLinear, "FourierKANModuleList"]:  # type: ignore
        return super(FourierKANModuleList, self).__getitem__(idx)


class FourierKAN(torch.nn.Module):
    """
    Fourier KAN (Kolmogorov-Arnold Network) model.

    Uses Fourier basis (sin/cos) as the learnable activation functions on edges,
    serving as a baseline for comparison with B-spline KAN and QKAN.
    """

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        scale_base=1.0,
        scale_fourier=1.0,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
        device="cpu",
        seed=0,
        **kwargs,
    ):
        super(FourierKAN, self).__init__()

        torch.manual_seed(seed)

        self.grid_size = grid_size

        self.layers = FourierKANModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                FourierKANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    scale_base=scale_base,
                    scale_fourier=scale_fourier,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def initialize_from_qkan(self, qkan: "QKAN", x0: torch.Tensor, sampling: int = 100):
        """
        Initialize FourierKAN from a QKAN.

        Args
        ----
            qkan: QKAN
            x0: torch.Tensor (batch, in_dim)
            sampling: int
        """
        assert len(self.layers) == len(qkan.layers), "Mismatched architecture"
        if qkan.is_map:
            raise RuntimeError("Cannot initialize from a QKAN with a map layer")

        for i, fourier_layer, qkan_layer in zip(
            range(len(self.layers)), self.layers, qkan.layers
        ):
            if i == 0:
                x = x0
            else:
                ymin = torch.min(out.cpu().detach(), dim=0).values  # noqa: F821
                ymax = torch.max(out.cpu().detach(), dim=0).values  # noqa: F821
                x = torch.stack(
                    [
                        torch.linspace(
                            ymin[j],
                            ymax[j],
                            steps=sampling,
                            device=x0.device,
                        )
                        for j in range(qkan_layer.in_dim)
                    ]
                ).permute(1, 0)  # x.shape = (sampling, in_dim)
            with torch.no_grad():
                out: torch.Tensor = qkan_layer.forward(
                    x
                )  # out.shape = (batch, out_dim)
            fourier_layer.fit_from_qkan(x, out)
