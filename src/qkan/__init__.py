__version__ = "0.0.1"

from .feynman import dataset_range, get_feynman_dataset
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
]
