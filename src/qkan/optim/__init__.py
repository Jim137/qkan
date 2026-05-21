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

"""QKAN-aware optimizers and training-loop helpers."""

from .adabelief import AdaBelief, QKANBeliefMini
from .adamini import QKANAdamMini
from .lbfgs import LBFGSFinisher, adam_then_lbfgs
from .spectral_mini import QKANSpectralMini
from .triton_adabelief import TritonAdaBelief
from .utils import reshape_optimizer_state

__all__ = [
    "AdaBelief",
    "QKANAdamMini",
    "QKANBeliefMini",
    "QKANSpectralMini",
    "LBFGSFinisher",
    "TritonAdaBelief",
    "adam_then_lbfgs",
    "reshape_optimizer_state",
]
