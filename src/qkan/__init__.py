__version__ = "0.1.4"

from .feynman import dataset_range, get_feynman_dataset
from .info import print0, print_banner, print_version
from .kan import KAN
from .qkan import QKAN, QKANLayer
from .torch_qc import StateVector, TorchGates
from .utils import SYMBOLIC_LIB, create_dataset

__author__ = "Jiun-Cheng Jiang"
__email__ = "jcjiang@phys.ntu.edu.tw"

__all__ = [
    "KAN",
    "QKAN",
    "QKANLayer",
    "StateVector",
    "SYMBOLIC_LIB",
    "TorchGates",
    "create_dataset",
    "dataset_range",
    "get_feynman_dataset",
    "print0",
    "print_banner",
    "print_version",
]
