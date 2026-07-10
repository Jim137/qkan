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

__version__ = "0.2.3dev"

from .daruan import DARUAN
from .feynman import dataset_range, get_feynman_dataset
from .fourier_kan import FourierKAN
from .inference import (
    CompiledInference,
    compile_inference,
    graph_submodules,
    make_graphed_inference,
    make_graphed_train_step,
)
from .info import print0, print_banner, print_version
from .kan import KAN
from .optim import (
    AdaBelief,
    LBFGSFinisher,
    QKANAdamMini,
    QKANBeliefMini,
    QKANMuon,
    QKANSpectralMini,
    TritonAdaBelief,
    adam_then_lbfgs,
    reshape_optimizer_state,
)
from .qkan import QKAN, QKANLayer
from .solver import (
    DeviceProfile,
    PackedCircuit,
    pack_circuit,
    tile_disjoint,
)
from .torch_qc import StateVector, TorchGates
from .utils import SYMBOLIC_LIB, create_dataset

__author__ = "Jiun-Cheng Jiang"
__email__ = "jcjiang@phys.ntu.edu.tw"

__all__ = [
    "DARUAN",
    "FourierKAN",
    "KAN",
    "QKAN",
    "AdaBelief",
    "QKANAdamMini",
    "QKANBeliefMini",
    "QKANMuon",
    "QKANLayer",
    "QKANSpectralMini",
    "StateVector",
    "SYMBOLIC_LIB",
    "TorchGates",
    "CompiledInference",
    "DeviceProfile",
    "PackedCircuit",
    "LBFGSFinisher",
    "TritonAdaBelief",
    "adam_then_lbfgs",
    "compile_inference",
    "create_dataset",
    "dataset_range",
    "get_feynman_dataset",
    "tile_disjoint",
    "graph_submodules",
    "pack_circuit",
    "make_graphed_inference",
    "make_graphed_train_step",
    "print0",
    "print_banner",
    "print_version",
    "reshape_optimizer_state",
]
