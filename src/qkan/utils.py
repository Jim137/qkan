# MIT License
#
# Copyright (c) 2024 Ziming Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Create dataset for regression task.

Adapted from [KindXiaoming/pykan@GitHub 91a2f63](https://github.com/KindXiaoming/pykan/tree/91a2f633be2d435b081ef0ef52a7205c7e7bea9e)
"""

import numpy as np
import sympy
import torch

# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)

# singularity protection functions
f_inv = lambda x, y_th: (
    (x_th := 1 / y_th),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x) * (torch.abs(x) >= x_th),
)
f_inv2 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 2)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**2) * (torch.abs(x) >= x_th),
)
f_inv3 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 3)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**3) * (torch.abs(x) >= x_th),
)
f_inv4 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 4)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**4) * (torch.abs(x) >= x_th),
)
f_inv5 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 5)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**5) * (torch.abs(x) >= x_th),
)
f_sqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    x_th / y_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.sqrt(torch.abs(x)) * torch.sign(x))
    * (torch.abs(x) >= x_th),
)
f_power1d5 = lambda x, y_th: torch.abs(x) ** 1.5
f_invsqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_log = lambda x, y_th: (
    (x_th := torch.e ** (-y_th)),
    -y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_tan = lambda x, y_th: (
    (clip := x % torch.pi),
    (delta := torch.pi / 2 - torch.arctan(y_th)),
    -y_th / delta * (clip - torch.pi / 2) * (torch.abs(clip - torch.pi / 2) < delta)
    + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi / 2) >= delta),
)
f_arctanh = lambda x, y_th: (
    (delta := 1 - torch.tanh(y_th) + 1e-4),
    y_th * torch.sign(x) * (torch.abs(x) > 1 - delta)
    + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta),
)
f_arcsin = lambda x, y_th: (
    (),
    torch.pi / 2 * torch.sign(x) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1),
)
f_arccos = lambda x, y_th: (
    (),
    torch.pi / 2 * (1 - torch.sign(x)) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1),
)
f_exp = lambda x, y_th: (
    (x_th := torch.log(y_th)),
    y_th * (x > x_th) + torch.exp(x) * (x <= x_th),
)

SYMBOLIC_LIB = {
    "x": (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
    "x^2": (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
    "x^3": (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
    "x^4": (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
    "x^5": (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
    "1/x": (lambda x: 1 / x, lambda x: 1 / x, 2, f_inv),
    "1/x^2": (lambda x: 1 / x**2, lambda x: 1 / x**2, 2, f_inv2),
    "1/x^3": (lambda x: 1 / x**3, lambda x: 1 / x**3, 3, f_inv3),
    "1/x^4": (lambda x: 1 / x**4, lambda x: 1 / x**4, 4, f_inv4),
    "1/x^5": (lambda x: 1 / x**5, lambda x: 1 / x**5, 5, f_inv5),
    "sqrt": (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    "x^0.5": (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    "x^1.5": (
        lambda x: torch.sqrt(x) ** 3,
        lambda x: sympy.sqrt(x) ** 3,
        4,
        f_power1d5,
    ),
    "1/sqrt(x)": (
        lambda x: 1 / torch.sqrt(x),
        lambda x: 1 / sympy.sqrt(x),
        2,
        f_invsqrt,
    ),
    "1/x^0.5": (lambda x: 1 / torch.sqrt(x), lambda x: 1 / sympy.sqrt(x), 2, f_invsqrt),
    "exp": (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
    "log": (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
    "abs": (
        lambda x: torch.abs(x),
        lambda x: sympy.Abs(x),
        3,
        lambda x, y_th: ((), torch.abs(x)),
    ),
    "sin": (
        lambda x: torch.sin(x),
        lambda x: sympy.sin(x),
        2,
        lambda x, y_th: ((), torch.sin(x)),
    ),
    "cos": (
        lambda x: torch.cos(x),
        lambda x: sympy.cos(x),
        2,
        lambda x, y_th: ((), torch.cos(x)),
    ),
    "tan": (lambda x: torch.tan(x), lambda x: sympy.tan(x), 3, f_tan),
    "tanh": (
        lambda x: torch.tanh(x),
        lambda x: sympy.tanh(x),
        3,
        lambda x, y_th: ((), torch.tanh(x)),
    ),
    "sgn": (
        lambda x: torch.sign(x),
        lambda x: sympy.sign(x),
        3,
        lambda x, y_th: ((), torch.sign(x)),
    ),
    "arcsin": (lambda x: torch.arcsin(x), lambda x: sympy.asin(x), 4, f_arcsin),
    "arccos": (lambda x: torch.arccos(x), lambda x: sympy.acos(x), 4, f_arccos),
    "arctan": (
        lambda x: torch.arctan(x),
        lambda x: sympy.atan(x),
        4,
        lambda x, y_th: ((), torch.arctan(x)),
    ),
    "arctanh": (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
    "0": (lambda x: x * 0, lambda x: x * 0, 0, lambda x, y_th: ((), x * 0)),
    "gaussian": (
        lambda x: torch.exp(-(x**2)),
        lambda x: sympy.exp(-(x**2)),
        3,
        lambda x, y_th: ((), torch.exp(-(x**2))),
    ),
}


def create_dataset(
    f,
    n_var=2,
    f_mode="col",
    ranges=[-1, 1],
    train_num=1000,
    test_num=1000,
    normalize_input=False,
    normalize_label=False,
    device="cpu",
    seed=0,
):
    """
    Create dataset

    Args:
        f: function
            the symbolic formula used to create the synthetic dataset
        ranges: list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num: int
            the number of training samples. Default: 1000.
        test_num: int
            the number of test samples. Default: 1000.
        normalize_input: bool
            If True, apply normalization to inputs. Default: False.
        normalize_label: bool
            If True, apply normalization to labels. Default: False.
        device: str
            device. Default: 'cpu'.
        seed: int
            random seed. Default: 0.

    Returns:
        dataset: dict
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
            dataset['test_input'], dataset['test_label']
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = (
            torch.rand(
                train_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )
        test_input[:, i] = (
            torch.rand(
                test_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )

    if f_mode == "col":
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == "row":
        train_label = f(train_input.T)
        test_label = f(test_input.T)
    else:
        print(f"f_mode {f_mode} not recognized")

    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)

    def normalize(data, mean, std):
        return (data - mean) / std

    if normalize_input:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)

    if normalize_label:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset["train_input"] = train_input.to(device)
    dataset["test_input"] = test_input.to(device)

    dataset["train_label"] = train_label.to(device)
    dataset["test_label"] = test_label.to(device)

    return dataset
