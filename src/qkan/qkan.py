"""
Quantum-inspired Kolmogorov Arnold Networks (QKANs) implementation in PyTorch.
Paper: Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks: https://arxiv.org/abs/2509.14026

Supported solvers:
    - PennyLane
    - Exact solver implemented in PyTorch (faster)
    - Custom solvers api

Code author: Jiun-Cheng Jiang (Jim137@GitHub)
Contact: [jcjiang@phys.ntu.edu.tw](mailto:jcjiang@phys.ntu.edu.tw)
"""

import os
import random
import warnings
from copy import deepcopy
from glob import glob
from typing import Callable, Literal, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # type: ignore

from .info import get_dist_info, print0, print_version
from .solver import qml_solver, torch_exact_solver


class QKANLayer(nn.Module):
    """
    QKANLayer Class

    Attributes
    ----------
        in_dim : int
            Input dimension
        out_dim : int
            Output dimension
        reps : int
            Repetitions of quantum layers
        group : int
            Group of neurons
        device :
            Device to use
        solver : Union[Literal["qml", "exact"], Callable]
            Solver to use
        ansatz : Union[str, Callable]
            Ansatz to use, "pz_encoding", "px_encoding", "rpz_encoding" or custom
        qml_device : str
            PennyLane device to use
        theta : nn.Parameter
            Learnable parameter of quantum circuit
        base_weight : nn.Parameter
            Learnable parameter of base activation
        preact_trainable : bool
            Whether preact weights are trainable
        preacts_weight : nn.Parameter
            Learnable parameter of preact weights
        preacts_bias : nn.Parameter
            Learnable parameter of preact bias
        postact_weight_trainable : bool
            Whether postact weights are trainable
        postact_weights : nn.Parameter
            Learnable parameter of postact weights
        postact_bias_trainable : bool
            Whether postact bias are trainable
        postact_bias : nn.Parameter
            Learnable parameter of postact bias
        mask : nn.Parameter
            Mask for pruning
        is_batchnorm : bool
            Whether to use batch normalization
        fast_measure : bool
            Enable to use fast measurement in exact solver. Which would be quantum-inspired method.
            When False, the exact solver simulates the exact measurement process of quantum circuit.
        c_dtype : torch.dtype
            Compute dtype for quantum simulation
        p_dtype : torch.dtype
            Parameter dtype for quantum simulation
        _x0 : Optional[torch.Tensor]
            Leave for ResQKANLayer
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        reps: int = 3,
        group: Union[int, tuple] = -1,
        device="cpu",
        solver: Union[Literal["qml", "exact"], Callable] = "exact",
        qml_device="default.qubit",
        ansatz: Union[str, Callable] = "pz_encoding",
        theta_size: Optional[list[int]] = None,
        preact_trainable: bool = False,
        preact_init: bool = False,
        postact_weight_trainable: bool = False,
        postact_bias_trainable: bool = False,
        base_activation=torch.nn.SiLU(),
        ba_trainable: bool = True,
        is_batchnorm: bool = False,
        fast_measure: bool = True,
        c_dtype=torch.complex64,
        p_dtype=torch.float32,
        seed=0,
    ):
        super(QKANLayer, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if isinstance(group, int):
            if group == -1:
                group = (out_dim, in_dim)
            else:
                group = tuple([group])

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reps = reps
        self.group = group
        self.device = device
        self.solver: Union[Literal["qml", "exact"], Callable] = solver
        self.qml_device = qml_device
        self.ansatz = ansatz
        self.theta_size = theta_size
        self.base_activation = base_activation
        self.ba_trainable = ba_trainable
        self.is_batchnorm = is_batchnorm
        self.fast_measure = fast_measure
        self.seed = seed
        self.c_dtype = c_dtype
        self.p_dtype = p_dtype

        if callable(solver) or callable(ansatz):
            if not theta_size:
                raise ValueError("theta_size is required for custom ansatz")
            self.theta = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(*theta_size, device=device, dtype=p_dtype)
                )
            )
        elif ansatz == "pz_encoding" or ansatz == "pz":
            self.theta = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(*group, reps + 1, 2, device=device, dtype=p_dtype)
                )
            )
        elif ansatz == "rpz_encoding" or ansatz == "rpz":
            if not preact_trainable:
                warnings.warn(
                    "Reduced pz encoding requires preact_trainable=True, set automatically."
                )
                preact_trainable = True
            self.theta = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(*group, reps + 1, 1, device=device, dtype=p_dtype)
                )
            )
        elif ansatz == "px_encoding" or ansatz == "px":
            self.theta = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(*group, reps + 1, 1, device=device, dtype=p_dtype)
                )
            )
        else:
            raise NotImplementedError()

        if ba_trainable:
            self.base_weight = torch.nn.Parameter(
                0.5 * torch.ones(out_dim, in_dim, device=device, dtype=p_dtype),
                requires_grad=ba_trainable,
            )
        else:
            self.base_weight = torch.nn.Parameter(
                torch.zeros(out_dim, in_dim, device=device, dtype=p_dtype),
                requires_grad=ba_trainable,
            )

        self.preact_trainable = preact_trainable
        if not preact_init:
            self.preacts_weight = nn.Parameter(
                torch.ones(*group, reps, device=device, dtype=p_dtype),
                requires_grad=preact_trainable,
            )
            self.preacts_bias = nn.Parameter(
                torch.zeros(*group, reps, device=device, dtype=p_dtype),
                requires_grad=preact_trainable,
            )
        else:
            self.preacts_weight = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(*group, reps, device=device, dtype=p_dtype)
                ),
                requires_grad=preact_trainable,
            )
            self.preacts_bias = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(*group, reps, device=device, dtype=p_dtype)
                ),
                requires_grad=preact_trainable,
            )
        self.preact_init = preact_init

        self.postact_weight_trainable = postact_weight_trainable
        self.postact_weights = nn.Parameter(
            torch.ones(out_dim, in_dim, device=device, dtype=p_dtype),
            requires_grad=postact_weight_trainable,
        )
        self.postact_bias_trainable = postact_bias_trainable
        self.postact_bias = nn.Parameter(
            torch.zeros(out_dim, in_dim, device=device, dtype=p_dtype),
            requires_grad=postact_bias_trainable,
        )
        self.mask = nn.Parameter(
            torch.ones(out_dim, in_dim, device=device, dtype=p_dtype),
            requires_grad=False,
        )
        if is_batchnorm:
            self.bn = nn.BatchNorm1d(in_dim, device=device, dtype=p_dtype)
        self._x0: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        """
        Move the layer to the specified device.

        Args
        ----
            device : str | torch.device
                Device to move the layer to, default: "cpu"
        """
        device = None
        for arg in args:
            if isinstance(arg, str) or isinstance(arg, torch.device):
                device = arg
                break
            elif isinstance(arg, torch.dtype):
                self.p_dtype = arg
        if "device" in kwargs:
            device = kwargs["device"]
        if device:
            self.device = device
            for param in self.parameters():
                param.data = param.to(device)
        return super(QKANLayer, self).to(*args, **kwargs)

    @property
    def param_size(self):
        if hasattr(self, "_param_size"):
            return self._param_size
        count = 0
        for param in self.parameters():
            if param.requires_grad:
                count += param.numel()
        self._param_size = count
        return self._param_size

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x: torch.Tensor):
        self._x0 = None

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.in_dim, "Invalid input dimension"

        batch = x.shape[0]

        if self.is_batchnorm:
            x = self.bn(x)
        base_output = torch.einsum(
            "oi,bi->boi", self.base_weight, self.base_activation(x)
        )
        if self.solver == "qml":
            postacts = torch.zeros(
                batch, self.out_dim, self.in_dim, dtype=self.p_dtype
            ).to(self.device)
            for j in range(self.out_dim):
                for i in range(self.in_dim):
                    postacts[:, j, i] = qml_solver(
                        x=x[:, i],
                        theta=self.theta[i, j],
                        reps=self.reps,
                        device=self.device,
                        qml_device=self.qml_device,
                    ).to(self.p_dtype)
        elif self.solver == "exact":
            postacts = torch_exact_solver(
                x,
                self.theta,
                self.preacts_weight,
                self.preacts_bias,
                self.reps,
                device=self.device,
                ansatz=self.ansatz,
                group=self.group,
                preacts_trainable=self.preact_trainable,
                fast_measure=self.fast_measure,
                out_dim=self.out_dim,
                dtype=self.c_dtype,
            ).to(self.p_dtype)
        elif callable(self.solver):
            postacts = self.solver(
                x,
                self.theta,
                self.preacts_weight,
                self.preacts_bias,
                self.reps,
                device=self.device,
                ansatz=self.ansatz,
            )
        else:
            raise NotImplementedError()
        if postacts.shape[1] != self.out_dim:
            postacts = postacts.expand(-1, self.out_dim, -1)
        x = torch.sum(
            (
                (postacts + self.postact_bias) * self.postact_weights[None, :, :]
                + base_output
            )
            * self.mask[None, :, :],
            dim=2,
        )
        return x

    def reset_parameters(self):
        self.theta.data.copy_(torch.zeros(self.theta.shape, dtype=self.p_dtype))

    @torch.no_grad()
    def forward_no_sum(self, x: torch.Tensor):
        assert x.shape[1] == self.in_dim, "Invalid input dimension"

        base_output = torch.einsum(
            "oi,bi->boi", self.base_weight, self.base_activation(x)
        )

        if self.solver == "qml":
            postacts = torch.cat(
                [
                    torch.stack(
                        [
                            qml_solver(
                                x=x[:, i],
                                theta=self.theta[i, j],
                                reps=self.reps,
                                device=self.device,
                                qml_device=self.qml_device,
                            )
                            for i in range(self.in_dim)
                        ],
                    )
                    .unsqueeze(-1)
                    .permute(1, 2, 0)
                    for j in range(self.out_dim)
                ],
                dim=1,
            ).to(self.p_dtype)
        elif self.solver == "exact":
            postacts = torch_exact_solver(
                x,
                self.theta,
                self.preacts_weight,
                self.preacts_bias,
                self.reps,
                device=self.device,
                ansatz=self.ansatz,
                group=self.group,
                preacts_trainable=self.preact_trainable,
                fast_measure=self.fast_measure,
                dtype=self.c_dtype,
            ).to(self.p_dtype)
        else:
            raise NotImplementedError()
        x_new = (
            (postacts + self.postact_bias) * self.postact_weights[None, :, :]
            + base_output
        ) * self.mask[None, :, :]
        return x_new

    def get_subset(self, in_id, out_id):
        """
        Get a smaller QKANLayer from a larger QKANLayer (used for pruning).

        Args
        ----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns
        -------
            QKANLayer
                New QKANLayer with selected neurons
        """
        spb = QKANLayer(
            in_dim=len(in_id),
            out_dim=len(out_id),
            reps=self.reps,
            device=self.device,
            solver=self.solver,
            qml_device=self.qml_device,
            ansatz=self.ansatz,
            preact_trainable=self.preact_trainable,
            postact_weight_trainable=self.postact_weight_trainable,
            postact_bias_trainable=self.postact_bias_trainable,
            base_activation=self.base_activation,
            ba_trainable=self.ba_trainable,
            seed=self.seed,
        )
        spb.theta.data = self.theta[out_id][:, in_id]
        spb.base_weight.data = self.base_weight[out_id][:, in_id]
        spb.preacts_weight.data = self.preacts_weight[out_id][:, in_id]
        spb.preacts_bias.data = self.preacts_bias[out_id][:, in_id]
        spb.postact_weights.data = self.postact_weights[out_id][:, in_id]
        spb.postact_bias.data = self.postact_bias[out_id][:, in_id]
        spb.mask.data = self.mask[out_id][:, in_id]
        return spb


class QKANModuleList(nn.ModuleList):
    def __init__(self):
        super(QKANModuleList, self).__init__()

    # make type hint for getitem method
    def __getitem__(self, idx) -> Union[QKANLayer, nn.Linear, "QKANModuleList"]:
        return super(QKANModuleList, self).__getitem__(idx)


class QKAN(nn.Module):
    """
    Quantum-inspired Kolmogorov Arnold Network (QKAN) Class

    A quantum-inspired neural network that uses DatA Re-Uploading ActivatioN (DARUAN)
    as its learnable variation activation function.

    References:
        Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks: https://arxiv.org/abs/2509.14026

    Attributes
    ----------
        width : list[int]
            List of width of each layer
        reps : int
            Repetitions of quantum layers
        group : int
            Group of neurons
        device : Literal["cpu", "cuda"]
            Device to use
        solver : Literal["qml", "exact"]
            Solver to use
        qml_device : str
            PennyLane device to use
        layers : QKANModuleList
            List of layers
        is_map : bool
            Whether to use map layer
        is_batchnorm : bool
            Whether to use batch normalization
        reps : int
            Repetitions of quantum layers
        norm_out : int
            Normalize output
        postact_weight_trainable : bool
            Whether postact weights are trainable
        postact_bias_trainable : bool
            Whether postact bias are trainable
        preact_trainable : bool
            Whether preact weights are trainable
        base_activation : torch.nn.Module or lambda function
            Base activation function
        ba_trainable : bool
            Whether base activation weights are trainable
        fast_measure : bool
            Enable to use fast measurement in exact solver. Which would be quantum-inspired method.
            When False, the exact solver simulates the exact measurement process of quantum circuit.
        save_act : bool
            Whether to save activations
        seed : int
            Random seed
    """

    def __init__(
        self,
        width: list[int],
        reps: int = 3,
        group: int = -1,
        is_map: bool = False,
        is_batchnorm: bool = False,
        hidden: int = 0,
        device="cpu",
        solver: Union[Literal["qml", "exact"], Callable] = "exact",
        qml_device: str = "default.qubit",
        ansatz: Union[str, Callable] = "pz_encoding",
        theta_size: Optional[list[int]] = None,
        norm_out: int = 0,
        preact_trainable: bool = False,
        preact_init: bool = False,
        postact_weight_trainable: bool = False,
        postact_bias_trainable: bool = False,
        base_activation=nn.SiLU(),
        ba_trainable: bool = False,
        fast_measure: bool = True,
        save_act: bool = False,
        c_dtype=torch.complex64,
        p_dtype=torch.float32,
        seed=0,
        **kwargs,
    ):
        """
        Initialize QKAN model

        Args
        ----
            width : list[int]
                List of width of each layer
            reps : int
                Repetitions of quantum layers, default: 3
            group : int
                Group of neurons, default: -1
            is_map : bool
                Whether to use map layer, default: False
            is_batchnorm: bool
                Whether to add a batchnorm layer before QKANLayer, default: False
            hidden : int
                Number of hidden units in map layer, default: 0
            device :
                Device to use, default: "cpu"
            solver : Union[Literal["qml", "exact"], Callable]
                Solver to use, default: "exact"
            ansatz : Union[str, Callable]
                Ansatz to use, "pz_encoding" ("pz"), "px_encoding" ("px"), "rpz_encoding" ("rpz", reduced pz encoding) or custom
            qml_device : str
                PennyLane device to use, default: "default.qubit"
            ansatz : str | Callable
                Ansatz to use, default: "pz_encoding"
            norm_out : int
                Normalize output, default: 0
            postact_weight_trainable : bool
                Whether postact weights are trainable, default: False
            postact_bias_trainable : bool
                Whether postact bias are trainable, default: False
            base_activation : torch.nn.Module | lambda function
                Base activation function, default: torch.nn.SiLU()
            ba_trainable : bool
                Whether base activation weights are trainable, default: False
            save_act : bool
                Whether to save activations, default: False
            fast_measure : bool
                Enable to use fast measurement in exact solver. Which would be quantum-inspired method.
                When False, the exact solver simulates the exact measurement process of quantum circuit.
            p_dtype : torch.dtype
                Parameter dtype for quantum simulation, default: torch.float32
            c_dtype : torch.dtype
                Compute dtype for quantum simulation, default: torch.complex64
            seed : int
                Random seed, default: 0
        """
        super(QKAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.depth = len(width) - (2 if is_map else 1)
        self.width = width
        self.reps = reps
        self.group = group
        self.device = device
        self.solver: Union[Literal["qml", "exact"], Callable] = solver
        self.ansatz = ansatz
        self.qml_device = qml_device
        self.norm_out = norm_out
        self.postact_weight_trainable = postact_weight_trainable
        self.postact_bias_trainable = postact_bias_trainable
        self.preact_trainable = preact_trainable
        self.preact_init = preact_init
        self.theta_size = theta_size
        self.base_activation = base_activation
        self.ba_trainable = ba_trainable
        self.fast_measure = fast_measure
        self.save_act = save_act
        self.c_dtype = c_dtype
        self.p_dtype = p_dtype
        self.seed = seed

        self.layers = QKANModuleList()
        for l in range(self.depth):
            self.layers.append(
                QKANLayer(
                    in_dim=width[l],
                    out_dim=width[l + 1],
                    reps=reps,
                    group=group,
                    device=self.device,
                    solver=self.solver,
                    qml_device=self.qml_device,
                    ansatz=self.ansatz,
                    theta_size=theta_size,
                    preact_trainable=preact_trainable,
                    preact_init=preact_init,
                    postact_weight_trainable=postact_weight_trainable,
                    postact_bias_trainable=postact_bias_trainable,
                    base_activation=base_activation,
                    ba_trainable=ba_trainable,
                    is_batchnorm=is_batchnorm,
                    fast_measure=fast_measure,
                    c_dtype=c_dtype,
                    p_dtype=p_dtype,
                    seed=seed,
                )
            )

        self.is_batchnorm = is_batchnorm
        self.is_map = is_map
        self.hidden = hidden
        if is_map:
            self.layers.append(
                nn.Linear(width[-2], hidden, device=self.device, dtype=self.p_dtype)
            )
            self.layers.append(nn.SiLU())
            self.layers.append(
                nn.Linear(hidden, width[-1], device=self.device, dtype=self.p_dtype)
            )
        self.input_id: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        """
        Move the model to the specified device.

        Args
        ----
            device : str | torch.device
                Device to move the model to, default: "cpu"
        """
        device = None
        for arg in args:
            if isinstance(arg, str) or isinstance(arg, torch.device):
                device = arg
                break
            elif isinstance(arg, torch.dtype):
                self.p_dtype = arg
        if "device" in kwargs:
            device = kwargs["device"]
        if device:
            self.device = device
        for layer in self.layers:
            layer.to(*args, **kwargs)
        return super(QKAN, self).to(*args, **kwargs)

    @property
    def param_size(self):
        if hasattr(self, "_param_size"):
            return self._param_size
        count = 0
        for layer in self.layers:
            if not isinstance(layer, QKANLayer):
                count += sum(p.numel() for p in layer.parameters())
                continue
            count += layer.param_size
        self._param_size = count
        return self._param_size

    def forward(self, x: torch.Tensor):
        shape_size = len(x.shape)

        if shape_size == 3:
            B, C, T = x.shape
        elif shape_size == 2:
            B, T = x.shape
        else:
            raise NotImplementedError()

        x = x.view(-1, T)
        if self.input_id is not None:
            x = x[:, self.input_id.long()]

        if self.save_act:
            self.cache_data = x
            self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
            self.subnode_actscale = []
            self.dr_preacts = []
            self.dr_postacts = []
            self.acts_scale = []
            self.acts_scale_dr = []
            self.edge_actscale = []
            self.acts.append(x.detach())

        for layer in self.layers:
            if self.save_act and isinstance(layer, QKANLayer):
                self.subnode_actscale.append(torch.std(x, dim=0).detach())
                preacts = x[:, None, :].expand(B, layer.out_dim, layer.in_dim)
                postacts = layer.forward_no_sum(x)  # shape: (batch, out_dim, in_dim)

            x = layer(x)

            if self.save_act and isinstance(layer, QKANLayer):
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_dr = torch.std(
                    postacts, dim=0
                )  # for training, only penalize the dr part
                output_range = torch.std(
                    postacts, dim=0
                )  # leave for symbolic (Not implemented yet)
                # save edge_scale
                self.edge_actscale.append(output_range)
                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_dr.append(output_range_dr / input_range)
                self.dr_preacts.append(preacts.detach())
                self.dr_postacts.append(postacts.detach())
                self.acts.append(x.detach())

        if self.norm_out:
            x = F.normalize(x, p=self.norm_out, dim=1)

        U = x.shape[1]

        if shape_size == 3:
            x = x.view(B, C, U)
        elif shape_size == 2:
            assert x.shape == (B, U)
        return x

    def initialize_from_another_model(self, another_model: "QKAN"):
        """
        Initialize from another model.
        Used for layer extension to refine the model.

        Args
        ----
            another_model : QKAN
                Another model to initialize from
        """
        assert all(x == y for x, y in zip(self.width, another_model.width)), (
            "Cannot initialize from another model with different width"
        )
        count = -2
        for l, layer in enumerate(self.layers):
            if isinstance(layer, QKANLayer):
                layer.reset_parameters()
                for i in range(another_model.layers[l].reps):
                    layer.theta.data[:, :, i, :].copy_(
                        another_model.layers[l].theta.data[:, :, i, :]
                    )
                    layer.preacts_weight.data[:, :, i].copy_(
                        another_model.layers[l].preacts_weight.data[:, :, i]
                    )
                    layer.preacts_bias.data[:, :, i].copy_(
                        another_model.layers[l].preacts_bias.data[:, :, i]
                    )
                layer.theta.data[:, :, another_model.layers[l].reps, :].copy_(
                    another_model.layers[l].theta.data[
                        :, :, another_model.layers[l].reps, :
                    ]
                )
                layer.postact_weights.data.copy_(
                    another_model.layers[l].postact_weights.data
                )
                layer.postact_bias.data.copy_(another_model.layers[l].postact_bias.data)
                layer.base_weight.data.copy_(another_model.layers[l].base_weight.data)
            if isinstance(layer, nn.Linear):
                layer.weight.data.copy_(another_model.layers[count - 1].weight.data)
                layer.bias.data.copy_(another_model.layers[count - 1].bias.data)
                count += 2
        return self

    def _reg(
        self,
        reg_metric: str,
        lamb_l1: float,
        lamb_entropy: float,
        lamb_coef: float,
        lamb_coefdiff: float,
    ):
        """
        Get regularization.

        Adapted from "pykan".

        Args
        ----
            reg_metric : the regularization metric
                'edge_forward_dr_n', 'edge_forward_dr_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient penalty strength
            lamb_coefdiff : float
                coefficient smoothness strength

        Returns
        -------
            torch.Tensor
        """
        if reg_metric == "edge_forward_dr_n":
            acts_scale = self.acts_scale_dr
        elif reg_metric == "edge_forward_sum":
            acts_scale = self.acts_scale
        elif reg_metric == "edge_forward_dr_u":
            acts_scale = self.edge_actscale
        elif reg_metric == "edge_backward":
            acts_scale = self.edge_scores
        elif reg_metric == "node_backward":
            acts_scale = self.node_attribute_scores
        else:
            raise RuntimeError(f"reg_metric = {reg_metric} not recognized!")

        reg_: torch.Tensor = torch.tensor(0.0, device=self.device)
        for i in range(len(acts_scale)):
            vec = acts_scale[i]

            l1 = torch.sum(vec)
            # Optimize: reuse sum computations for normalization
            sum_row = torch.sum(vec, dim=1, keepdim=True) + 1
            sum_col = torch.sum(vec, dim=0, keepdim=True) + 1
            p_row = vec / sum_row
            p_col = vec / sum_col
            entropy_row = -torch.mean(
                torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1)
            )
            entropy_col = -torch.mean(
                torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0)
            )
            reg_ += lamb_l1 * l1 + lamb_entropy * (
                entropy_row + entropy_col
            )  # both l1 and entropy

        # regularize coefficient to encourage activation to be zero
        for layer in self.layers:
            if not isinstance(layer, QKANLayer):
                continue
            coeff_l1 = torch.sum(torch.mean(torch.abs(layer.postact_weights), dim=1))
            coeff_diff_l1 = torch.sum(
                torch.mean(torch.abs(torch.diff(layer.postact_weights)), dim=1)
            )
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    def get_reg(
        self,
        reg_metric: str,
        lamb_l1: float,
        lamb_entropy: float,
        lamb_coef: float,
        lamb_coefdiff: float,
    ):
        """
        Get regularization from the model.

        Adapted from "pykan".

        Args
        ----
            reg_metric : str
                Regularization metric.
                'edge_forward_dr_n', 'edge_forward_dr_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            lamb_l1 : float
                L1 Regularization parameter
            lamb_entropy : float
                Entropy Regularization parameter
            lamb_coef : float
                Coefficient Regularization parameter
            lamb_coefdiff : float
                Coefficient Smoothness Regularization parameter

        Returns
        -------
            torch.Tensor
        """
        return self._reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)

    def attribute(self, l=None, i=None, out_score=None, plot=True):
        """
        Get attribution scores

        Adapted from "pykan".

        Args
        ----
            l : None | int
                layer index
            i : None | int
                neuron index
            out_score : None | torch.Tensor
                specify output scores
            plot : bool
                when plot = True, display the bar show

        Returns
        -------
            torch.Tensor
                attribution scores
        """
        if not self.save_act:
            warnings.warn(
                "Activations are not saved, cannot get attribution scores",
                RuntimeWarning,
            )
            return None

        if l is not None:
            self.attribute()
            out_score = self.node_scores[l]

        node_scores = []
        subnode_scores = []
        edge_scores = []

        l_query = l
        if l is None:
            l_end = self.depth
        else:
            l_end = l

        # back propagate from the queried layer
        out_dim = self.width[l_end]
        if out_score is None:
            node_score = torch.eye(out_dim).requires_grad_(True)
        else:
            node_score = torch.diag(out_score).requires_grad_(True)
        node_scores.append(node_score)

        # Pre-compute inverse subnode_actscale for efficiency
        inv_subnode_actscale = [1 / (scale + 1e-4) for scale in self.subnode_actscale]

        for l in range(l_end, 0, -1):
            subnode_score = node_score[:, : self.width[l]]

            subnode_scores.append(subnode_score)
            # subnode to edge
            edge_score = torch.einsum(
                "oi,ko,i->koi",
                self.edge_actscale[l - 1],
                subnode_score.to(self.device),
                inv_subnode_actscale[l - 1],
            )
            edge_scores.append(edge_score)

            # edge to node
            node_score = torch.sum(edge_score, dim=1)
            node_scores.append(node_score)

        self.node_scores_all = list(reversed(node_scores))
        self.edge_scores_all = list(reversed(edge_scores))
        self.subnode_scores_all = list(reversed(subnode_scores))

        self.node_scores = [torch.mean(l, dim=0) for l in self.node_scores_all]
        self.edge_scores = [torch.mean(l, dim=0) for l in self.edge_scores_all]
        self.subnode_scores = [torch.mean(l, dim=0) for l in self.subnode_scores_all]

        # return: (out_dim, in_dim)
        if l_query is not None:
            if i is None:
                return self.node_scores_all[0]
            else:
                # plot
                if plot:
                    in_dim = self.width[0]
                    plt.figure(figsize=(1 * in_dim, 3))
                    plt.bar(
                        range(in_dim), self.node_scores_all[0][i].cpu().detach().numpy()
                    )
                    plt.xticks(range(in_dim))

                return self.node_scores_all[0][i]

    def node_attribute(self):
        """
        Get node attribution scores.

        Adapted from "pykan".
        """
        self.node_attribute_scores = []
        for l in range(1, self.depth + 1):
            node_attr = self.attribute(l)
            self.node_attribute_scores.append(node_attr)

    def train_(
        self,
        dataset,
        optimizer=None,
        closure=None,
        scheduler=None,
        steps: int = 10,
        log: int = 1,
        loss_fn=None,
        batch=-1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        reg_metric="edge_forward_dr_n",
        verbose=True,
    ):
        """
        Train the model

        Args
        ----
            dataset : dict
                Dictionary containing train_input, train_label, test_input, test_label
            optimizer : torch.optim.Optimizer | None
                Optimizer to use, default: None
            closure : Callable | None
                Closure function for optimizer, default: None
            scheduler : torch.optim.lr_scheduler | None
                Scheduler to use, default: None
            steps : int
                Number of steps, default: 10
            log : int
                Logging frequency, default: 1
            loss_fn : torch.nn.Module | Callable |None
                Loss function to use, default: None
            batch : int
                batch size, if -1 then full., default: -1
            lamb : float
                L1 Regularization parameter. If 0, no regularization.
            lamb_l1 : float
                L1 Regularization parameter
            lamb_entropy : float
                Entropy Regularization parameter
            lamb_coef : float
                Coefficient Regularization parameter
            lamb_coefdiff : float
                Coefficient Smoothness Regularization parameter
            reg_metric : str
                Regularization metric.
                'edge_forward_dr_n', 'edge_forward_dr_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            verbose : bool
                Verbose mode, default: True

        Returns
        -------
            dict
                Dictionary containing train_loss and test_loss
        """
        if verbose:
            print_version()

        if lamb > 0.0 and not self.save_act:
            lamb = 0.0
            warnings.warn(
                "Regularization is not supported without saving activations",
                RuntimeWarning,
            )

        pbar = tqdm(range(steps), ncols=100)

        if loss_fn is None:
            loss_fn = loss_fn_eval = torch.nn.MSELoss()
        else:
            loss_fn = loss_fn_eval = loss_fn

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        else:
            optimizer = optimizer

        results: dict = {}
        results["train_loss"] = []
        results["test_loss"] = []
        results["reg"] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size = dataset["train_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        def _closure():
            nonlocal train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset["train_input"][train_id].to(self.device))
            train_loss = loss_fn(pred, dataset["train_label"][train_id].to(self.device))
            if self.save_act:
                if reg_metric == "edge_backward":
                    self.attribute()
                if reg_metric == "node_backward":
                    self.node_attribute()
                reg_ = self.get_reg(
                    reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                )
            else:
                reg_ = torch.tensor(0.0, device=self.device)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if closure is None and isinstance(optimizer, torch.optim.LBFGS):
            closure = _closure

        for _ in pbar:
            self.train()
            train_id = np.random.choice(
                dataset["train_input"].shape[0], batch_size, replace=False
            )
            test_id = np.random.choice(
                dataset["test_input"].shape[0], batch_size_test, replace=False
            )

            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                pred = self.forward(dataset["train_input"][train_id].to(self.device))
                train_loss = loss_fn(
                    pred, dataset["train_label"][train_id].to(self.device)
                )
                if self.save_act:
                    if reg_metric == "edge_backward":
                        self.attribute()
                    if reg_metric == "node_backward":
                        self.node_attribute()
                    reg_ = self.get_reg(
                        reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                    )
                else:
                    reg_ = torch.tensor(0.0, device=self.device)
                loss = train_loss + lamb * reg_
                loss.backward()
                optimizer.step(closure)

            self.eval()
            test_loss = loss_fn_eval(
                self.forward(dataset["test_input"][test_id].to(self.device)),
                dataset["test_label"][test_id].to(self.device),
            )
            if scheduler is not None:
                scheduler.step(test_loss)

            if _ % log == 0:
                pbar.set_postfix(
                    {
                        "train loss": train_loss.cpu().detach().numpy(),
                        "test loss": test_loss.cpu().detach().numpy(),
                    }
                )

            results["train_loss"].append(train_loss.cpu().detach().numpy())
            results["test_loss"].append(test_loss.cpu().detach().numpy())
            results["reg"].append(reg_.cpu().detach().numpy())

        return results

    def plot(
        self,
        x0=None,
        sampling=1000,
        from_acts=False,
        scale=0.5,
        beta=3,
        metric="forward_n",
        mask=False,
        in_vars=None,
        out_vars=None,
        title=None,
    ):
        """
        Plot the model.

        Adapted from "pykan".

        Arguments
        ---------
            x0 : torch.Tensor | None
                Input tensor to plot, if None, plot from saved activations
            sampling : int
                Sampling frequency
            from_acts : bool
                Plot from saved activations
            scale : float
                Scale of the plot
            beta : float
                Beta value
            metric : str
                Metric to use. 'forward_n', 'forward_u', 'backward'
            in_vars : list[int] | None
                Input variables to plot
            out_vars : list[int] | None
                Output variables to plot
            title : str | None
                Title of the plot
        """
        if self.is_map:
            warnings.warn("Not supported for map layer", RuntimeWarning)
            return None
        if self.is_batchnorm:
            warnings.warn("Not supported for batchnorm layer", RuntimeWarning)
            return None
        if x0 is None and not from_acts:
            warnings.warn(
                "x0 is not provided, try plot from saved activations.", RuntimeWarning
            )
            from_acts = True
        if from_acts and not self.acts:
            warnings.warn(
                "Activations are not saved, cannot plot from activations",
                RuntimeWarning,
            )
            return None
        if mask and not hasattr(self, "mask"):
            warnings.warn(
                "Make sure to run model.prune_node() first to compute mask. Continue without mask.",
                RuntimeWarning,
            )
            mask = False

        if not os.path.exists("./figures"):
            os.makedirs("./figures")

        if metric == "backward":
            self.attribute()

        save_act = self.save_act
        self.save_act = False
        self.eval()
        for idx, qkan_layer in enumerate(self.layers):
            assert isinstance(qkan_layer, QKANLayer)

            if idx == 0:
                x = x0
            else:
                ymin = torch.min(ynew.cpu().detach(), dim=0).values  # noqa: F821
                ymax = torch.max(ynew.cpu().detach(), dim=0).values  # noqa: F821
                x = torch.stack(
                    [
                        torch.linspace(
                            ymin[i],
                            ymax[i],
                            steps=sampling,
                            device=self.device,
                        )
                        for i in range(qkan_layer.in_dim)
                    ]
                ).permute(1, 0)  # x.shape = (sampling, in_dim)
            if from_acts:
                x = self.acts[idx]

            y = qkan_layer.forward_no_sum(x).transpose(
                1, 2
            )  # y.shape = (sampling, in_dim, out_dim)
            for i in range(self.width[idx]):
                for j in range(self.width[idx + 1]):
                    fig, ax = plt.subplots(figsize=(2, 2))
                    plt.xticks([])
                    plt.yticks([])
                    plt.gca().patch.set_edgecolor("black")
                    plt.gca().patch.set_linewidth(1.5)
                    plt.scatter(
                        x[:, i].detach().cpu().numpy(),
                        y[:, i, j].detach().cpu().numpy(),
                        color="black",
                        s=40,
                    )
                    plt.gca().spines[:].set_color("black")
                    plt.savefig(
                        f"./figures/dr_{idx}_{i}_{j}.png", bbox_inches="tight", dpi=400
                    )
                    plt.close()
            with torch.no_grad():
                ynew = qkan_layer.forward(x)  # noqa: F841

        def score2alpha(score):
            return np.tanh(beta * score)

        alpha = []
        try:
            if save_act and metric is not None:
                if metric == "forward_n":
                    scores = self.acts_scale
                elif metric == "forward_u":
                    scores = self.edge_actscale
                elif metric == "backward":
                    scores = self.edge_scores
                else:
                    raise RuntimeError(f"metric = '{metric}' cannot be recognized")

                alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]
        except RuntimeError:
            warnings.warn(f"metric = '{metric}' cannot be recognized", RuntimeWarning)
        finally:
            if not alpha:
                alpha = [
                    torch.ones(layer.out_dim, layer.in_dim).detach().numpy()
                    for layer in self.layers
                ]

        # draw skeleton
        width = np.array(self.width)
        A = 1
        y0 = 0.4

        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width), 5)

        # max_neuron = np.max(width)
        max_num_weights = np.max(width[:-1] * width[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 3)

        fig, ax = plt.subplots(
            figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * y0)
        )
        # plot scatters and lines
        for l in range(neuron_depth):
            n = width[l]
            # spacing = A / n
            for i in range(n):
                plt.scatter(
                    1 / (2 * n) + i / n,
                    l * y0,
                    s=min_spacing**2 * 10000 * scale**2,
                    color="black",
                )

                if l < neuron_depth - 1:
                    # plot connections
                    n_next = width[l + 1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j
                        if mask:
                            plt.plot(
                                [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N],
                                [l * y0, (l + 1 / 2) * y0 - y1],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i]
                                * self.mask[l][i].item()
                                * self.mask[l + 1][j].item(),
                            )
                            plt.plot(
                                [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                [(l + 1 / 2) * y0 + y1, (l + 1) * y0],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i]
                                * self.mask[l][i].item()
                                * self.mask[l + 1][j].item(),
                            )
                        else:
                            plt.plot(
                                [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N],
                                [l * y0, (l + 1 / 2) * y0 - y1],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i],
                            )
                            plt.plot(
                                [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                [(l + 1 / 2) * y0 + y1, (l + 1) * y0],
                                color="black",
                                lw=2 * scale,
                                alpha=alpha[l][j][i],
                            )

            plt.xlim(0, 1)
            plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

        plt.axis("off")

        # plot splines
        for l in range(neuron_depth - 1):
            n = width[l]
            for i in range(n):
                n_next = width[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f"./figures/dr_{l}_{i}_{j}.png")
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
                    up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    if mask:
                        newax.imshow(
                            im,
                            alpha=alpha[l][j][i]
                            * self.mask[l][i].item()
                            * self.mask[l + 1][j].item(),
                        )
                    else:
                        newax.imshow(im, alpha=alpha[l][j][i])
                    newax.axis("off")

        if in_vars is not None:
            n = self.width[0]
            for i in range(n):
                plt.gcf().get_axes()[0].text(
                    1 / (2 * (n)) + i / (n),
                    -0.1,
                    in_vars[i],
                    fontsize=40 * scale,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if out_vars is not None:
            n = self.width[-1]
            for i in range(n):
                plt.gcf().get_axes()[0].text(
                    1 / (2 * (n)) + i / (n),
                    y0 * (len(self.width) - 1) + 0.1,
                    out_vars[i],
                    fontsize=40 * scale,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if title is not None:
            plt.gcf().get_axes()[0].text(
                0.5,
                y0 * (len(self.width) - 1) + 0.2,
                title,
                fontsize=40 * scale,
                horizontalalignment="center",
                verticalalignment="center",
            )
        self.save_act = save_act

    def prune_node(
        self,
        threshold: float = 1e-2,
        mode: str = "auto",
        active_neurons_id: Optional[list] = None,
    ):
        """
        Pruning nodes.

        Adapted from "pykan".

        Args
        ----
            threshold : float
                if the attribution score of a neuron is below the threshold, it is considered dead and will be removed
            mode : str
                "auto" or "manual". with "auto", nodes are automatically pruned using threshold.
                With "manual", active_neurons_id should be passed in.

        Returns
        -------
            QKAN
                pruned network
        """
        if not hasattr(self, "acts"):
            warnings.warn("No activations, cannot prune nodes", RuntimeWarning)
            return None
        if mode == "manual" and active_neurons_id is None:
            warnings.warn(
                "active_neurons_id is not provided. Continue with auto mode.",
                RuntimeWarning,
            )
            mode = "auto"

        mask = [
            torch.ones(
                self.width[0],
            )
        ]
        active_neurons = [list(range(self.width[0]))]
        for i in range(len(self.acts_scale) - 1):
            if mode == "auto":
                in_important = torch.max(self.acts_scale[i], dim=1)[0] > threshold
                out_important = torch.max(self.acts_scale[i + 1], dim=0)[0] > threshold
                overall_important = in_important * out_important
            elif mode == "manual":
                assert active_neurons_id is not None
                overall_important = torch.zeros(self.width[i + 1], dtype=torch.bool)
                overall_important[active_neurons_id[i + 1]] = True
            mask.append(overall_important.float())
            active_neurons.append(
                torch.where(overall_important == True)[0].tolist()  # noqa: E712
            )
        active_neurons.append(list(range(self.width[-1])))
        mask.append(
            torch.ones(
                self.width[-1],
            )
        )

        self.mask = mask  # for plot

        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width[l + 1]):
                if i not in active_neurons[l + 1]:
                    self.remove_node(l + 1, i)

        model2 = QKAN(
            deepcopy(self.width),
            reps=self.reps,
            is_map=self.is_map,
            is_batchnorm=self.is_batchnorm,
            hidden=self.hidden,
            device=self.device,
            solver=self.solver,
            qml_device=self.qml_device,
            ansatz=self.ansatz,
            norm_out=self.norm_out,
            preact_trainable=self.preact_trainable,
            postact_weight_trainable=self.postact_weight_trainable,
            postact_bias_trainable=self.postact_bias_trainable,
            base_activation=self.base_activation,
            ba_trainable=self.ba_trainable,
            save_act=self.save_act,
            seed=self.seed,
        )
        model2.load_state_dict(self.state_dict())
        for i, layer in enumerate(model2.layers):
            if not isinstance(layer, QKANLayer):
                continue
            model2.layers[i] = layer.get_subset(
                active_neurons[i], active_neurons[i + 1]
            )
            model2.width[i] = len(active_neurons[i])
        model2.cache_data = self.cache_data

        return model2

    def prune_edge(self, threshold: float = 3e-2):
        """
        Pruning edges.

        Adapted from "pykan".

        Args:
            threshold: float
                if the attribution score of an edge is below the threshold, it is considered dead and will be set to zero.
        """
        if not hasattr(self, "acts"):
            warnings.warn("No activations, cannot prune edges", RuntimeWarning)
            return None

        for i in range(len(self.width) - 1):
            old_mask = self.layers[i].mask.data
            self.layers[i].mask.data = (
                (self.edge_scores[i] > threshold) * old_mask
            ).float()

    def prune(self, node_th: float = 1e-2, edge_th: float = 3e-2):
        """
        Prune (both nodes and edges).

        Adapted from "pykan".

        Args
        ----
            node_th : float
                if the attribution score of a node is below node_th, it is considered dead and will be set to zero.
            edge_th : float
                if the attribution score of an edge is below node_th, it is considered dead and will be set to zero.

        Returns
        -------
            QKAN
                pruned network
        """
        if not hasattr(self, "acts"):
            warnings.warn("No activations, cannot prune.", RuntimeWarning)
            return None

        self = self.prune_node(node_th)
        self.forward(self.cache_data)
        self.attribute()
        self.prune_edge(edge_th)
        return self

    def prune_input(
        self, threshold: float = 1e-2, active_inputs: Optional[list] = None
    ):
        """
        Prune inputs.

        Adapted from "pykan".

        Args
        ----
            threshold : float
                if the attribution score of the input feature is below threshold, it is considered irrelevant.
            active_inputs : list | None
                if a list is passed, the manual mode will disregard attribution score and prune as instructed.

        Returns
        -------
            QKAN
                pruned network
        """
        if active_inputs is None:
            self.attribute()
            input_score = self.node_scores[0]
            input_mask = input_score > threshold
            print0("keep:", input_mask.tolist())
            input_id = torch.where(input_mask == True)[0]  # noqa: E712

        else:
            input_id = torch.tensor(active_inputs, dtype=torch.long).to(self.device)

        model2 = QKAN(
            deepcopy(self.width),
            reps=self.reps,
            is_map=self.is_map,
            is_batchnorm=self.is_batchnorm,
            hidden=self.hidden,
            device=self.device,
            solver=self.solver,
            qml_device=self.qml_device,
            ansatz=self.ansatz,
            norm_out=self.norm_out,
            preact_trainable=self.preact_trainable,
            postact_weight_trainable=self.postact_weight_trainable,
            postact_bias_trainable=self.postact_bias_trainable,
            base_activation=self.base_activation,
            ba_trainable=self.ba_trainable,
            save_act=self.save_act,
            seed=self.seed,
        )
        model2.load_state_dict(self.state_dict())

        model2.layers[0] = model2.layers[0].get_subset(
            input_id, torch.arange(self.width[1])
        )

        model2.cache_data = self.cache_data

        model2.width[0] = len(input_id)
        model2.input_id = input_id

        return model2

    def remove_edge(self, layer_idx, in_idx, out_idx):
        """
        Remove activtion phi(layer_idx, in_idx, out_idx) (set its mask to zero)

        Args
        ----
            layer_idx : int
                Layer index
            in_idx : int
                Input node index
            out_idx : int
                Output node index
        """
        if not isinstance(self.layers[layer_idx], QKAN):
            return
        self.layers[layer_idx].mask[out_idx, in_idx] = 0.0

    def remove_node(self, layer_idx, in_idx, mode="all"):
        """
        remove neuron (layer_idx, in_idx) (set the masks of all incoming and outgoing activation functions to zero)

        Args
        ----
            layer_idx : int
                Layer index
            in_idx : int
                Input node index
            mode : str
                Mode to remove. "all" or "up" or "down", default: "all"
        """
        if mode == "down":
            if not isinstance(self.layers[layer_idx - 1], QKAN):
                return
            self.layers[layer_idx - 1].mask[in_idx, :] = 0.0
        elif mode == "up":
            if not isinstance(self.layers[layer_idx], QKAN):
                return
            self.layers[layer_idx].mask[:, in_idx] = 0.0
        else:
            self.remove_node(layer_idx, in_idx, mode="up")
            self.remove_node(layer_idx, in_idx, mode="down")

    @staticmethod
    def clear_ckpts(folder="./model_ckpt"):
        """
        Clear all checkpoints.

        Args
        ----
            folder : str
                Folder containing checkpoints, default: "./model_ckpt"
        """
        is_ddp, ddp_rank, _, _ = get_dist_info()
        if not is_ddp or ddp_rank == 0:
            if os.path.exists(folder):
                files = glob(folder + "/*")
                for f in files:
                    os.remove(f)
            else:
                os.makedirs(folder)

    def save_ckpt(self, name, folder="./model_ckpt"):
        """
        Save the current model as checkpoint.

        Args
        ----
            name : str
                Name of the checkpoint
            folder : str
                Folder to save the checkpoint, default: "./model_ckpt"
        """
        is_ddp, ddp_rank, _, _ = get_dist_info()
        if not is_ddp or ddp_rank == 0:
            if not os.path.exists(folder):
                os.makedirs(folder)

            torch.save(self.state_dict(), folder + "/" + name)
            print0("save this model to", folder + "/" + name)

    def load_ckpt(self, name, folder="./model_ckpt"):
        """
        Load a checkpoint to the current model.

        Args
        ----
            name : str
                Name of the checkpoint
            folder : str
                Folder containing the checkpoint, default: "./model_ckpt"
        """
        self.load_state_dict(torch.load(folder + "/" + name))
