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
Get Feynman dataset from the given name.
Return the symbol, expression, function, and ranges of the dataset.

Adapted from [KindXiaoming/pykan@GitHub 11984f4](https://github.com/KindXiaoming/pykan/tree/11984f49216fc254c1dd40b5bee0a069ee63114a)
"""

from typing import Union as TyUnion

import torch
from sympy import *  # noqa: F403

dataset_range = range(120)


def get_feynman_dataset(name: TyUnion[str, int]):
    """
    Get Feynman dataset from the given name.

    Args:
        name (str | int): The name of the dataset.

    Returns:
        tuple: The symbol, expression, function, and ranges of the dataset.
    """
    tpi = torch.tensor(torch.pi)

    ranges: list[list[TyUnion[int, float, torch.Tensor]]]
    match name:
        case "test" | 0:
            symbol = x, y = symbols("x, y")
            expr = (x + y) * sin(exp(2 * y))
            f = lambda x: (x[:, [0]] + x[:, [1]]) * torch.sin(torch.exp(2 * x[:, [1]]))
            ranges = [[-1, 1]]

        case "I.6.20a" | 1:
            symbol = theta = symbols("theta")
            symbol = [symbol]
            expr = exp(-(theta**2) / 2) / sqrt(2 * pi)
            f = lambda x: torch.exp(-(x[:, [0]] ** 2) / 2) / torch.sqrt(2 * tpi)
            ranges = [[-3, 3]]

        case "I.6.20" | 2:
            symbol = theta, sigma = symbols("theta sigma")
            expr = exp(-(theta**2) / (2 * sigma**2)) / sqrt(2 * pi * sigma**2)
            f = lambda x: torch.exp(
                -(x[:, [0]] ** 2) / (2 * x[:, [1]] ** 2)
            ) / torch.sqrt(2 * tpi * x[:, [1]] ** 2)
            ranges = [[-1, 1], [0.5, 2]]

        case "I.6.20b" | 3:
            symbol = theta, theta1, sigma = symbols("theta theta1 sigma")
            expr = exp(-((theta - theta1) ** 2) / (2 * sigma**2)) / sqrt(
                2 * pi * sigma**2
            )
            f = lambda x: torch.exp(
                -((x[:, [0]] - x[:, [1]]) ** 2) / (2 * x[:, [2]] ** 2)
            ) / torch.sqrt(2 * tpi * x[:, [2]] ** 2)
            ranges = [[-1.5, 1.5], [-1.5, 1.5], [0.5, 2]]

        case "I.8.4" | 4:
            symbol = x1, x2, y1, y2 = symbols("x1 x2 y1 y2")
            expr = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            f = lambda x: torch.sqrt(
                (x[:, [1]] - x[:, [0]]) ** 2 + (x[:, [3]] - x[:, [2]]) ** 2
            )
            ranges = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]

        case "I.9.18" | 5:
            symbol = G, m1, m2, x1, x2, y1, y2, z1, z2 = symbols(
                "G m1 m2 x1 x2 y1 y2 z1 z2"
            )
            expr = G * m1 * m2 / ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                * x[:, [2]]
                / (
                    (x[:, [3]] - x[:, [4]]) ** 2
                    + (x[:, [5]] - x[:, [6]]) ** 2
                    + (x[:, [7]] - x[:, [8]]) ** 2
                )
            )
            ranges = [
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, -0.5],
                [0.5, 1],
                [-1, -0.5],
                [0.5, 1],
                [-1, -0.5],
                [0.5, 1],
            ]

        case "I.10.7" | 6:
            symbol = m0, v, c = symbols("m0 v c")
            expr = m0 / sqrt(1 - v**2 / c**2)
            f = lambda x: x[:, [0]] / torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [2]] ** 2)
            ranges = [[0, 1], [0, 1], [1, 2]]

        case "I.11.19" | 7:
            symbol = x1, y1, x2, y2, x3, y3 = symbols("x1 y1 x2 y2 x3 y3")
            expr = x1 * y1 + x2 * y2 + x3 * y3
            f = (
                lambda x: x[:, [0]] * x[:, [1]]
                + x[:, [2]] * x[:, [3]]
                + x[:, [4]] * x[:, [5]]
            )
            ranges = [[-1, 1]]

        case "I.12.1" | 8:
            symbol = mu, Nn = symbols("mu N_n")
            expr = mu * Nn
            f = lambda x: x[:, [0]] * x[:, [1]]
            ranges = [[-1, 1]]

        case "I.12.2" | 9:
            symbol = q1, q2, eps, r = symbols("q1 q2 epsilon r")
            expr = q1 * q2 / (4 * pi * eps * r**2)
            f = lambda x: x[:, [0]] * x[:, [1]] / (4 * tpi * x[:, [2]] * x[:, [3]] ** 2)
            ranges = [[-1, 1], [-1, 1], [0.5, 2], [0.5, 2]]

        case "I.12.4" | 10:
            symbol = q1, eps, r = symbols("q1 epsilon r")
            expr = q1 / (4 * pi * eps * r**2)
            f = lambda x: x[:, [0]] / (4 * tpi * x[:, [1]] * x[:, [2]] ** 2)
            ranges = [[-1, 1], [0.5, 2], [0.5, 2]]

        case "I.12.5" | 11:
            symbol = q2, Ef = symbols("q2, E_f")
            expr = q2 * Ef
            f = lambda x: x[:, [0]] * x[:, [1]]
            ranges = [[-1, 1]]

        case "I.12.11" | 12:
            symbol = q, Ef, B, v, theta = symbols("q E_f B v theta")
            expr = q * (Ef + B * v * sin(theta))
            f = lambda x: x[:, [0]] * (
                x[:, [1]] + x[:, [2]] * x[:, [3]] * torch.sin(x[:, [4]])
            )
            ranges = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [0, 2 * tpi]]

        case "I.13.4" | 13:
            symbol = m, v, u, w = symbols("m u v w")
            expr = 1 / 2 * m * (v**2 + u**2 + w**2)
            f = (
                lambda x: 1
                / 2
                * x[:, [0]]
                * (x[:, [1]] ** 2 + x[:, [2]] ** 2 + x[:, [3]] ** 2)
            )
            ranges = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]

        case "I.13.12" | 14:
            symbol = G, m1, m2, r1, r2 = symbols("G m1 m2 r1 r2")
            expr = G * m1 * m2 * (1 / r2 - 1 / r1)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                * x[:, [2]]
                * (1 / x[:, [4]] - 1 / x[:, [3]])
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "I.14.3" | 15:
            symbol = m, g, z = symbols("m g z")
            expr = m * g * z
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]]
            ranges = [[0, 1], [0, 1], [-1, 1]]

        case "I.14.4" | 16:
            symbol = ks, x = symbols("k_s x")
            expr = 1 / 2 * ks * x**2
            f = lambda x: 1 / 2 * x[:, [0]] * x[:, [1]] ** 2
            ranges = [[0, 1], [-1, 1]]

        case "I.15.3x" | 17:
            symbol = x, u, t, c = symbols("x u t c")
            expr = (x - u * t) / sqrt(1 - u**2 / c**2)
            f = lambda x: (x[:, [0]] - x[:, [1]] * x[:, [2]]) / torch.sqrt(
                1 - x[:, [1]] ** 2 / x[:, [3]] ** 2
            )
            ranges = [[-1, 1], [-1, 1], [-1, 1], [1, 2]]

        case "I.15.3t" | 18:
            symbol = t, u, x, c = symbols("t u x c")
            expr = (t - u * x / c**2) / sqrt(1 - u**2 / c**2)
            f = lambda x: (
                x[:, [0]] - x[:, [1]] * x[:, [2]] / x[:, [3]] ** 2
            ) / torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [3]] ** 2)
            ranges = [[-1, 1], [-1, 1], [-1, 1], [1, 2]]

        case "I.15.10" | 19:
            symbol = m0, v, c = symbols("m0 v c")
            expr = m0 * v / sqrt(1 - v**2 / c**2)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                / torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [2]] ** 2)
            )
            ranges = [[-1, 1], [-0.9, 0.9], [1.1, 2]]

        case "I.16.6" | 20:
            symbol = u, v, c = symbols("u v c")
            expr = (u + v) / (1 + u * v / c**2)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                / (1 + x[:, [0]] * x[:, [1]] / x[:, [2]] ** 2)
            )
            ranges = [[-0.8, 0.8], [-0.8, 0.8], [1, 2]]

        case "I.18.4" | 21:
            symbol = m1, r1, m2, r2 = symbols("m1 r1 m2 r2")
            expr = (m1 * r1 + m2 * r2) / (m1 + m2)
            f = lambda x: (x[:, [0]] * x[:, [1]] + x[:, [2]] * x[:, [3]]) / (
                x[:, [0]] + x[:, [2]]
            )
            ranges = [[0.5, 1], [-1, 1], [0.5, 1], [-1, 1]]

        case "I.18.4" | 22:
            symbol = r, F, theta = symbols("r F theta")
            expr = r * F * sin(theta)
            f = lambda x: x[:, [0]] * x[:, [1]] * torch.sin(x[:, [2]])
            ranges = [[-1, 1], [-1, 1], [0, 2 * tpi]]

        case "I.18.16" | 23:
            symbol = m, r, v, theta = symbols("m r v theta")
            expr = m * r * v * sin(theta)
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] * torch.sin(x[:, [3]])
            ranges = [[-1, 1], [-1, 1], [-1, 1], [0, 2 * tpi]]

        case "I.24.6" | 24:
            symbol = m, omega, omega0, x = symbols("m omega omega_0 x")
            expr = 1 / 4 * m * (omega**2 + omega0**2) * x**2
            f = (
                lambda x: 1
                / 4
                * x[:, [0]]
                * (x[:, [1]] ** 2 + x[:, [2]] ** 2)
                * x[:, [3]] ** 2
            )
            ranges = [[0, 1], [-1, 1], [-1, 1], [-1, 1]]

        case "I.25.13" | 25:
            symbol = q, C = symbols("q C")
            expr = q / C
            f = lambda x: x[:, [0]] / x[:, [1]]
            ranges = [[-1, 1], [0.5, 2]]

        case "I.26.2" | 26:
            symbol = n, theta2 = symbols("n theta2")
            expr = asin(n * sin(theta2))
            f = lambda x: torch.arcsin(x[:, [0]] * torch.sin(x[:, [1]]))
            ranges = [[0, 0.99], [0, 2 * tpi]]

        case "I.27.6" | 27:
            symbol = d1, d2, n = symbols("d1 d2 n")
            expr = 1 / (1 / d1 + n / d2)
            f = lambda x: 1 / (1 / x[:, [0]] + x[:, [2]] / x[:, [1]])
            ranges = [[0.5, 2], [1, 2], [0.5, 2]]

        case "I.29.4" | 28:
            symbol = omega, c = symbols("omega c")
            expr = omega / c
            f = lambda x: x[:, [0]] / x[:, [1]]
            ranges = [[0, 1], [0.5, 2]]

        case "I.29.16" | 29:
            symbol = x1, x2, theta1, theta2 = symbols("x1 x2 theta1 theta2")
            expr = sqrt(x1**2 + x2**2 - 2 * x1 * x2 * cos(theta1 - theta2))
            f = lambda x: torch.sqrt(
                x[:, [0]] ** 2
                + x[:, [1]] ** 2
                - 2 * x[:, [0]] * x[:, [1]] * torch.cos(x[:, [2]] - x[:, [3]])
            )
            ranges = [[-1, 1], [-1, 1], [0, 2 * tpi], [0, 2 * tpi]]

        case "I.30.3" | 30:
            symbol = I0, n, theta = symbols("I_0 n theta")
            expr = I0 * sin(n * theta / 2) ** 2 / sin(theta / 2) ** 2
            f = (
                lambda x: x[:, [0]]
                * torch.sin(x[:, [1]] * x[:, [2]] / 2) ** 2
                / torch.sin(x[:, [2]] / 2) ** 2
            )
            ranges = [[0, 1], [0, 4], [0.4 * tpi, 1.6 * tpi]]

        case "I.30.5" | 31:
            symbol = lamb, n, d = symbols("lambda n d")
            expr = asin(lamb / (n * d))
            f = lambda x: torch.arcsin(x[:, [0]] / (x[:, [1]] * x[:, [2]]))
            ranges = [[-1, 1], [1, 1.5], [1, 1.5]]

        case "I.32.5" | 32:
            symbol = q, a, eps, c = symbols("q a epsilon c")
            expr = q**2 * a**2 / (eps * c**3)
            f = lambda x: x[:, [0]] ** 2 * x[:, [1]] ** 2 / (x[:, [2]] * x[:, [3]] ** 3)
            ranges = [[-1, 1], [-1, 1], [0.5, 2], [0.5, 2]]

        case "I.32.17" | 33:
            symbol = eps, c, Ef, r, omega, omega0 = symbols(
                "epsilon c E_f r omega omega_0"
            )
            expr = nsimplify(
                (1 / 2 * eps * c * Ef**2)
                * (8 * pi * r**2 / 3)
                * (omega**4 / (omega**2 - omega0**2) ** 2)
            )
            f = (
                lambda x: (1 / 2 * x[:, [0]] * x[:, [1]] * x[:, [2]] ** 2)
                * (8 * tpi * x[:, [3]] ** 2 / 3)
                * (x[:, [4]] ** 4 / (x[:, [4]] ** 2 - x[:, [5]] ** 2) ** 2)
            )
            ranges = [[0, 1], [0, 1], [-1, 1], [0, 1], [0, 1], [1, 2]]

        case "I.34.8" | 34:
            symbol = q, V, B, p = symbols("q V B p")
            expr = q * V * B / p
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] / x[:, [3]]
            ranges = [[-1, 1], [-1, 1], [-1, 1], [0.5, 2]]

        case "I.34.10" | 35:
            symbol = omega0, v, c = symbols("omega_0 v c")
            expr = omega0 / (1 - v / c)
            f = lambda x: x[:, [0]] / (1 - x[:, [1]] / x[:, [2]])
            ranges = [[0, 1], [0, 0.9], [1.1, 2]]

        case "I.34.14" | 36:
            symbol = omega0, v, c = symbols("omega_0 v c")
            expr = omega0 * (1 + v / c) / sqrt(1 - v**2 / c**2)
            f = (
                lambda x: x[:, [0]]
                * (1 + x[:, [1]] / x[:, [2]])
                / torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [2]] ** 2)
            )
            ranges = [[0, 1], [-0.9, 0.9], [1.1, 2]]

        case "I.34.27" | 37:
            symbol = hbar, omega = symbols("hbar omega")
            expr = hbar * omega
            f = lambda x: x[:, [0]] * x[:, [1]]
            ranges = [[-1, 1], [-1, 1]]

        case "I.37.4" | 38:
            symbol = I1, I2, delta = symbols("I_1 I_2 delta")
            expr = I1 + I2 + 2 * sqrt(I1 * I2) * cos(delta)
            f = (
                lambda x: x[:, [0]]
                + x[:, [1]]
                + 2 * torch.sqrt(x[:, [0]] * x[:, [1]]) * torch.cos(x[:, [2]])
            )
            ranges = [[0.1, 1], [0.1, 1], [0, 2 * tpi]]

        case "I.38.12" | 39:
            symbol = eps, hbar, m, q = symbols("epsilon hbar m q")
            expr = 4 * pi * eps * hbar**2 / (m * q**2)
            f = (
                lambda x: 4
                * tpi
                * x[:, [0]]
                * x[:, [1]] ** 2
                / (x[:, [2]] * x[:, [3]] ** 2)
            )
            ranges = [[0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "I.39.10" | 40:
            symbol = pF, V = symbols("p_F V")
            expr = 3 / 2 * pF * V
            f = lambda x: 3 / 2 * x[:, [0]] * x[:, [1]]
            ranges = [[0, 1], [0, 1]]

        case "I.39.11" | 41:
            symbol = gamma, pF, V = symbols("gamma p_F V")
            expr = pF * V / (gamma - 1)
            f = lambda x: 1 / (x[:, [0]] - 1) * x[:, [1]] * x[:, [2]]
            ranges = [[1.5, 3], [0, 1], [0, 1]]

        case "I.39.22" | 42:
            symbol = n, kb, T, V = symbols("n k_b T V")
            expr = n * kb * T / V
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] / x[:, [3]]
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "I.40.1" | 43:
            symbol = n0, m, g, x, kb, T = symbols("n_0 m g x k_b T")
            expr = n0 * exp(-m * g * x / (kb * T))
            f = lambda x: x[:, [0]] * torch.exp(
                -x[:, [1]] * x[:, [2]] * x[:, [3]] / (x[:, [4]] * x[:, [5]])
            )
            ranges = [[0, 1], [-1, 1], [-1, 1], [-1, 1], [1, 2], [1, 2]]

        case "I.41.16" | 44:
            symbol = hbar, omega, c, kb, T = symbols("hbar omega c k_b T")
            expr = hbar * omega**3 / (pi**2 * c**2 * (exp(hbar * omega / (kb * T)) - 1))
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]] ** 3
                / (
                    tpi**2
                    * x[:, [2]] ** 2
                    * (torch.exp(x[:, [0]] * x[:, [1]] / (x[:, [3]] * x[:, [4]])) - 1)
                )
            )
            ranges = [[0.5, 1], [0.5, 1], [0.5, 2], [0.5, 2], [0.5, 2]]

        case "I.43.16" | 45:
            symbol = mu, q, Ve, d = symbols("mu q V_e d")
            expr = mu * q * Ve / d
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] / x[:, [3]]
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "I.43.31" | 46:
            symbol = mu, kb, T = symbols("mu k_b T")
            expr = mu * kb * T
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]]
            ranges = [[0, 1], [0, 1], [0, 1]]

        case "I.43.43" | 47:
            symbol = gamma, kb, v, A = symbols("gamma k_b v A")
            expr = kb * v / A / (gamma - 1)
            f = lambda x: 1 / (x[:, [0]] - 1) * x[:, [1]] * x[:, [2]] / x[:, [3]]
            ranges = [[1.5, 3], [0, 1], [0, 1], [0.5, 2]]

        case "I.44.4" | 48:
            symbol = n, kb, T, V1, V2 = symbols("n k_b T V_1 V_2")
            expr = n * kb * T * log(V2 / V1)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                * x[:, [2]]
                * torch.log(x[:, [4]] / x[:, [3]])
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "I.47.23" | 49:
            symbol = gamma, p, rho = symbols("gamma p rho")
            expr = sqrt(gamma * p / rho)
            f = lambda x: torch.sqrt(x[:, [0]] * x[:, [1]] / x[:, [2]])
            ranges = [[0.1, 1], [0.1, 1], [0.5, 2]]

        case "I.48.20" | 50:
            symbol = m, v, c = symbols("m v c")
            expr = m * c**2 / sqrt(1 - v**2 / c**2)
            f = (
                lambda x: x[:, [0]]
                * x[:, [2]] ** 2
                / torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [2]] ** 2)
            )
            ranges = [[0, 1], [-0.9, 0.9], [1.1, 2]]

        case "I.50.26" | 51:
            symbol = x1, alpha, omega, t = symbols("x_1 alpha omega t")
            expr = x1 * (cos(omega * t) + alpha * cos(omega * t) ** 2)
            f = lambda x: x[:, [0]] * (
                torch.cos(x[:, [2]] * x[:, [3]])
                + x[:, [1]] * torch.cos(x[:, [2]] * x[:, [3]]) ** 2
            )
            ranges = [[0, 1], [0, 1], [0, 2 * tpi], [0, 1]]

        case "II.2.42" | 52:
            symbol = kappa, T1, T2, A, d = symbols("kappa T_1 T_2 A d")
            expr = kappa * (T2 - T1) * A / d
            f = lambda x: x[:, [0]] * (x[:, [2]] - x[:, [1]]) * x[:, [3]] / x[:, [4]]
            ranges = [[0, 1], [0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "II.3.24" | 53:
            symbol = P, r = symbols("P r")
            expr = P / (4 * pi * r**2)
            f = lambda x: x[:, [0]] / (4 * tpi * x[:, [1]] ** 2)
            ranges = [[0, 1], [0.5, 2]]

        case "II.4.23" | 54:
            symbol = q, eps, r = symbols("q epsilon r")
            expr = q / (4 * pi * eps * r)
            f = lambda x: x[:, [0]] / (4 * tpi * x[:, [1]] * x[:, [2]])
            ranges = [[0, 1], [0.5, 2], [0.5, 2]]

        case "II.6.11" | 55:
            symbol = eps, pd, theta, r = symbols("epsilon p_d theta r")
            expr = 1 / (4 * pi * eps) * pd * cos(theta) / r**2
            f = (
                lambda x: 1
                / (4 * tpi * x[:, [0]])
                * x[:, [1]]
                * torch.cos(x[:, [2]])
                / x[:, [3]] ** 2
            )
            ranges = [[0.5, 2], [0, 1], [0, 2 * tpi], [0.5, 2]]

        case "II.6.15a" | 56:
            symbol = eps, pd, z, x, y, r = symbols("epsilon p_d z x y r")
            expr = 3 / (4 * pi * eps) * pd * z / r**5 * sqrt(x**2 + y**2)
            f = (
                lambda x: 3
                / (4 * tpi * x[:, [0]])
                * x[:, [1]]
                * x[:, [2]]
                / x[:, [5]] ** 5
                * torch.sqrt(x[:, [3]] ** 2 + x[:, [4]] ** 2)
            )
            ranges = [[0.5, 2], [0, 1], [0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "II.6.15b" | 57:
            symbol = eps, pd, r, theta = symbols("epsilon p_d r theta")
            expr = 3 / (4 * pi * eps) * pd / r**3 * cos(theta) * sin(theta)
            f = (
                lambda x: 3
                / (4 * tpi * x[:, [0]])
                * x[:, [1]]
                / x[:, [2]] ** 3
                * torch.cos(x[:, [3]])
                * torch.sin(x[:, [3]])
            )
            ranges = [[0.5, 2], [0, 1], [0.5, 2], [0, 2 * tpi]]

        case "II.8.7" | 58:
            symbol = q, eps, d = symbols("q epsilon d")
            expr = 3 / 5 * q**2 / (4 * pi * eps * d)
            f = lambda x: 3 / 5 * x[:, [0]] ** 2 / (4 * tpi * x[:, [1]] * x[:, [2]])
            ranges = [[0, 1], [0.5, 2], [0.5, 2]]

        case "II.8.31" | 59:
            symbol = eps, Ef = symbols("epsilon E_f")
            expr = 1 / 2 * eps * Ef**2
            f = lambda x: 1 / 2 * x[:, [0]] * x[:, [1]] ** 2
            ranges = [[0, 1], [0, 1]]

        case "I.10.9" | 60:
            symbol = sigma, eps, chi = symbols("sigma epsilon chi")
            expr = sigma / eps / (1 + chi)
            f = lambda x: x[:, [0]] / x[:, [1]] / (1 + x[:, [2]])
            ranges = [[0, 1], [0.5, 2], [0, 1]]

        case "II.11.3" | 61:
            symbol = q, Ef, m, omega0, omega = symbols("q E_f m omega_o omega")
            expr = q * Ef / (m * (omega0**2 - omega**2))
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                / (x[:, [2]] * (x[:, [3]] ** 2 - x[:, [4]] ** 2))
            )
            ranges = [[0, 1], [0, 1], [0.5, 2], [1.5, 3], [0, 1]]

        case "II.11.17" | 62:
            symbol = n0, pd, Ef, theta, kb, T = symbols("n_0 p_d E_f theta k_b T")
            expr = n0 * (1 + pd * Ef * cos(theta) / (kb * T))
            f = lambda x: x[:, [0]] * (
                1
                + x[:, [1]] * x[:, [2]] * torch.cos(x[:, [3]]) / (x[:, [4]] * x[:, [5]])
            )
            ranges = [[0, 1], [-1, 1], [-1, 1], [0, 2 * tpi], [0.5, 2], [0.5, 2]]

        case "II.11.20" | 63:
            symbol = n, pd, Ef, kb, T = symbols("n p_d E_f k_b T")
            expr = n * pd**2 * Ef / (3 * kb * T)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]] ** 2
                * x[:, [2]]
                / (3 * x[:, [3]] * x[:, [4]])
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "II.11.27" | 64:
            symbol = n, alpha, eps, Ef = symbols("n alpha epsilon E_f")
            expr = n * alpha / (1 - n * alpha / 3) * eps * Ef
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                / (1 - x[:, [0]] * x[:, [1]] / 3)
                * x[:, [2]]
                * x[:, [3]]
            )
            ranges = [[0, 1], [0, 2], [0, 1], [0, 1]]

        case "II.11.28" | 65:
            symbol = n, alpha = symbols("n alpha")
            expr = 1 + n * alpha / (1 - n * alpha / 3)
            f = lambda x: 1 + x[:, [0]] * x[:, [1]] / (1 - x[:, [0]] * x[:, [1]] / 3)
            ranges = [[0, 1], [0, 2]]

        case "II.13.17" | 66:
            symbol = eps, c, l, r = symbols("epsilon c l r")
            expr = 1 / (4 * pi * eps * c**2) * (2 * l / r)
            f = (
                lambda x: 1
                / (4 * tpi * x[:, [0]] * x[:, [1]] ** 2)
                * (2 * x[:, [2]] / x[:, [3]])
            )
            ranges = [[0.5, 2], [0.5, 2], [0, 1], [0.5, 2]]

        case "II.13.23" | 67:
            symbol = rho, v, c = symbols("rho v c")
            expr = rho / sqrt(1 - v**2 / c**2)
            f = lambda x: x[:, [0]] / torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [2]] ** 2)
            ranges = [[0, 1], [0, 1], [1, 2]]

        case "II.13.34" | 68:
            symbol = rho, v, c = symbols("rho v c")
            expr = rho * v / sqrt(1 - v**2 / c**2)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                / torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [2]] ** 2)
            )
            ranges = [[0, 1], [0, 1], [1, 2]]

        case "II.15.4" | 69:
            symbol = muM, B, theta = symbols("mu_M B theta")
            expr = -muM * B * cos(theta)
            f = lambda x: -x[:, [0]] * x[:, [1]] * torch.cos(x[:, [2]])
            ranges = [[0, 1], [0, 1], [0, 2 * tpi]]

        case "II.15.5" | 70:
            symbol = pd, Ef, theta = symbols("p_d E_f theta")
            expr = -pd * Ef * cos(theta)
            f = lambda x: -x[:, [0]] * x[:, [1]] * torch.cos(x[:, [2]])
            ranges = [[0, 1], [0, 1], [0, 2 * tpi]]

        case "II.21.32" | 71:
            symbol = q, eps, r, v, c = symbols("q epsilon r v c")
            expr = q / (4 * pi * eps * r * (1 - v / c))
            f = lambda x: x[:, [0]] / (
                4 * tpi * x[:, [1]] * x[:, [2]] * (1 - x[:, [3]] / x[:, [4]])
            )
            ranges = [[0, 1], [0.5, 2], [0.5, 2], [0, 1], [1, 2]]

        case "II.24.17" | 72:
            symbol = omega, c, d = symbols("omega c d")
            expr = sqrt(omega**2 / c**2 - pi**2 / d**2)
            f = lambda x: torch.sqrt(
                x[:, [0]] ** 2 / x[:, [1]] ** 2 - tpi**2 / x[:, [2]] ** 2
            )
            ranges = [[1, 1.5], [0.75, 1], [1 * tpi, 1.5 * tpi]]

        case "II.27.16" | 73:
            symbol = eps, c, Ef = symbols("epsilon c E_f")
            expr = eps * c * Ef**2
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] ** 2
            ranges = [[0, 1], [0, 1], [-1, 1]]

        case "II.27.18" | 74:
            symbol = eps, Ef = symbols("epsilon E_f")
            expr = eps * Ef**2
            f = lambda x: x[:, [0]] * x[:, [1]] ** 2
            ranges = [[0, 1], [-1, 1]]

        case "II.34.2a" | 75:
            symbol = q, v, r = symbols("q v r")
            expr = q * v / (2 * pi * r)
            f = lambda x: x[:, [0]] * x[:, [1]] / (2 * tpi * x[:, [2]])
            ranges = [[0, 1], [0, 1], [0.5, 2]]

        case "II.34.2" | 76:
            symbol = q, v, r = symbols("q v r")
            expr = q * v * r / 2
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] / 2
            ranges = [[0, 1], [0, 1], [0, 1]]

        case "II.34.11" | 77:
            symbol = g, q, B, m = symbols("g q B m")
            expr = g * q * B / (2 * m)
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] / (2 * x[:, [3]])
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "II.34.29a" | 78:
            symbol = q, h, m = symbols("q h m")
            expr = q * h / (4 * pi * m)
            f = lambda x: x[:, [0]] * x[:, [1]] / (4 * tpi * x[:, [2]])
            ranges = [[0, 1], [0, 1], [0.5, 2]]

        case "II.34.29b" | 79:
            symbol = g, mu, B, J, hbar = symbols("g mu B J hbar")
            expr = g * mu * B * J / hbar
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] * x[:, [3]] / x[:, [4]]
            ranges = [[0, 1], [0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "II.35.18" | 80:
            symbol = n0, mu, B, kb, T = symbols("n0 mu B k_b T")
            expr = n0 / (exp(mu * B / (kb * T)) + exp(-mu * B / (kb * T)))
            f = lambda x: x[:, [0]] / (
                torch.exp(x[:, [1]] * x[:, [2]] / (x[:, [3]] * x[:, [4]]))
                + torch.exp(-x[:, [1]] * x[:, [2]] / (x[:, [3]] * x[:, [4]]))
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "II.35.21" | 81:
            symbol = n, mu, B, kb, T = symbols("n mu B k_b T")
            expr = n * mu * tanh(mu * B / (kb * T))
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                * torch.tanh(x[:, [1]] * x[:, [2]] / (x[:, [3]] * x[:, [4]]))
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "II.36.38" | 82:
            symbol = mu, B, kb, T, alpha, M, eps, c = symbols(
                "mu B k_b T alpha M epsilon c"
            )
            expr = mu * B / (kb * T) + mu * alpha * M / (eps * c**2 * kb * T)
            f = lambda x: x[:, [0]] * x[:, [1]] / (x[:, [2]] * x[:, [3]]) + x[
                :, [0]
            ] * x[:, [4]] * x[:, [5]] / (
                x[:, [6]] * x[:, [7]] ** 2 * x[:, [2]] * x[:, [3]]
            )
            ranges = [
                [0, 1],
                [0, 1],
                [0.5, 2],
                [0.5, 2],
                [0, 1],
                [0, 1],
                [0.5, 2],
                [0.5, 2],
            ]

        case "II.37.1" | 83:
            symbol = mu, chi, B = symbols("mu chi B")
            expr = mu * (1 + chi) * B
            f = lambda x: x[:, [0]] * (1 + x[:, [1]]) * x[:, [2]]
            ranges = [[0, 1], [0, 1], [0, 1]]

        case "II.38.3" | 84:
            symbol = Y, A, x, d = symbols("Y A x d")
            expr = Y * A * x / d
            f = lambda x: x[:, [0]] * x[:, [1]] * x[:, [2]] / x[:, [3]]
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "II.38.14" | 85:
            symbol = Y, sigma = symbols("Y sigma")
            expr = Y / (2 * (1 + sigma))
            f = lambda x: x[:, [0]] / (2 * (1 + x[:, [1]]))
            ranges = [[0, 1], [0, 1]]

        case "III.4.32" | 86:
            symbol = hbar, omega, kb, T = symbols("hbar omega k_b T")
            expr = 1 / (exp(hbar * omega / (kb * T)) - 1)
            f = lambda x: 1 / (
                torch.exp(x[:, [0]] * x[:, [1]] / (x[:, [2]] * x[:, [3]])) - 1
            )
            ranges = [[0.5, 1], [0.5, 1], [0.5, 2], [0.5, 2]]

        case "III.4.33" | 87:
            symbol = hbar, omega, kb, T = symbols("hbar omega k_b T")
            expr = hbar * omega / (exp(hbar * omega / (kb * T)) - 1)
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                / (torch.exp(x[:, [0]] * x[:, [1]] / (x[:, [2]] * x[:, [3]])) - 1)
            )
            ranges = [[0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "III.7.38" | 88:
            symbol = mu, B, hbar = symbols("mu B hbar")
            expr = 2 * mu * B / hbar
            f = lambda x: 2 * x[:, [0]] * x[:, [1]] / x[:, [2]]
            ranges = [[0, 1], [0, 1], [0.5, 2]]

        case "III.8.54" | 89:
            symbol = E, t, hbar = symbols("E t hbar")
            expr = sin(E * t / hbar) ** 2
            f = lambda x: torch.sin(x[:, [0]] * x[:, [1]] / x[:, [2]]) ** 2
            ranges = [[0, 2 * tpi], [0, 1], [0.5, 2]]

        case "III.9.52" | 90:
            symbol = pd, Ef, t, hbar, omega, omega0 = symbols(
                "p_d E_f t hbar omega omega_0"
            )
            expr = (
                pd
                * Ef
                * t
                / hbar
                * sin((omega - omega0) * t / 2) ** 2
                / ((omega - omega0) * t / 2) ** 2
            )
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                * x[:, [2]]
                / x[:, [3]]
                * torch.sin((x[:, [4]] - x[:, [5]]) * x[:, [2]] / 2) ** 2
                / ((x[:, [4]] - x[:, [5]]) * x[:, [2]] / 2) ** 2
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0, tpi], [0, tpi]]

        case "III.10.19" | 91:
            symbol = mu, Bx, By, Bz = symbols("mu B_x B_y B_z")
            expr = mu * sqrt(Bx**2 + By**2 + Bz**2)
            f = lambda x: x[:, [0]] * torch.sqrt(
                x[:, [1]] ** 2 + x[:, [2]] ** 2 + x[:, [3]] ** 2
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0, 1]]

        case "III.12.43" | 92:
            symbol = n, hbar = symbols("n hbar")
            expr = n * hbar
            f = lambda x: x[:, [0]] * x[:, [1]]
            ranges = [[0, 1], [0, 1]]

        case "III.13.18" | 93:
            symbol = E, d, k, hbar = symbols("E d k hbar")
            expr = 2 * E * d**2 * k / hbar
            f = lambda x: 2 * x[:, [0]] * x[:, [1]] ** 2 * x[:, [2]] / x[:, [3]]
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "III.14.14" | 94:
            symbol = I0, q, Ve, kb, T = symbols("I_0 q V_e k_b T")
            expr = I0 * (exp(q * Ve / (kb * T)) - 1)
            f = lambda x: x[:, [0]] * (
                torch.exp(x[:, [1]] * x[:, [2]] / (x[:, [3]] * x[:, [4]])) - 1
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0.5, 2]]

        case "III.15.12" | 95:
            symbol = U, k, d = symbols("U k d")
            expr = 2 * U * (1 - cos(k * d))
            f = lambda x: 2 * x[:, [0]] * (1 - torch.cos(x[:, [1]] * x[:, [2]]))
            ranges = [[0, 1], [0, 2 * tpi], [0, 1]]

        case "III.15.14" | 96:
            symbol = hbar, E, d = symbols("hbar E d")
            expr = hbar**2 / (2 * E * d**2)
            f = lambda x: x[:, [0]] ** 2 / (2 * x[:, [1]] * x[:, [2]] ** 2)
            ranges = [[0, 1], [0.5, 2], [0.5, 2]]

        case "III.15.27" | 97:
            symbol = alpha, n, d = symbols("alpha n d")
            expr = 2 * pi * alpha / (n * d)
            f = lambda x: 2 * tpi * x[:, [0]] / (x[:, [1]] * x[:, [2]])
            ranges = [[0, 1], [0.5, 2], [0.5, 2]]

        case "III.17.37" | 98:
            symbol = beta, alpha, theta = symbols("beta alpha theta")
            expr = beta * (1 + alpha * cos(theta))
            f = lambda x: x[:, [0]] * (1 + x[:, [1]] * torch.cos(x[:, [2]]))
            ranges = [[0, 1], [0, 1], [0, 2 * tpi]]

        case "III.19.51" | 99:
            symbol = m, q, eps, hbar, n = symbols("m q epsilon hbar n")
            expr = -m * q**4 / (2 * (4 * pi * eps) ** 2 * hbar**2) * 1 / n**2
            f = (
                lambda x: -x[:, [0]]
                * x[:, [1]] ** 4
                / (2 * (4 * tpi * x[:, [2]]) ** 2 * x[:, [3]] ** 2)
                * 1
                / x[:, [4]] ** 2
            )
            ranges = [[0, 1], [0, 1], [0.5, 2], [0.5, 2], [0.5, 2]]

        case "III.21.20" | 100:
            symbol = rho, q, A, m = symbols("rho q A m")
            expr = -rho * q * A / m
            f = lambda x: -x[:, [0]] * x[:, [1]] * x[:, [2]] / x[:, [3]]
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "Rutherforld scattering" | 101:
            symbol = Z1, Z2, alpha, hbar, c, E, theta = symbols(
                "Z_1 Z_2 alpha hbar c E theta"
            )
            expr = (Z1 * Z2 * alpha * hbar * c / (4 * E * sin(theta / 2) ** 2)) ** 2
            f = (
                lambda x: (
                    x[:, [0]]
                    * x[:, [1]]
                    * x[:, [2]]
                    * x[:, [3]]
                    * x[:, [4]]
                    / (4 * x[:, [5]] * torch.sin(x[:, [6]] / 2) ** 2)
                )
                ** 2
            )
            ranges = [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0.5, 2],
                [0.1 * tpi, 0.9 * tpi],
            ]

        case "Friedman equation" | 102:
            symbol = G, rho, kf, c, af = symbols("G rho k_f c a_f")
            expr = sqrt(8 * pi * G / 3 * rho - kf * c**2 / af**2)
            f = lambda x: torch.sqrt(
                8 * tpi * x[:, [0]] / 3 * x[:, [1]]
                - x[:, [2]] * x[:, [3]] ** 2 / x[:, [4]] ** 2
            )
            ranges = [[1, 2], [1, 2], [0, 1], [0, 1], [1, 2]]

        case "Compton scattering" | 103:
            symbol = E, m, c, theta = symbols("E m c theta")
            expr = E / (1 + E / (m * c**2) * (1 - cos(theta)))
            f = lambda x: x[:, [0]] / (
                1
                + x[:, [0]] / (x[:, [1]] * x[:, [2]] ** 2) * (1 - torch.cos(x[:, [3]]))
            )
            ranges = [[0, 1], [0.5, 2], [0.5, 2], [0, 2 * tpi]]

        case "Radiated gravitational wave power" | 104:
            symbol = G, c, m1, m2, r = symbols("G c m_1 m_2 r")
            expr = -32 / 5 * G**4 / c**5 * (m1 * m2) ** 2 * (m1 + m2) / r**5
            f = (
                lambda x: -32
                / 5
                * x[:, [0]] ** 4
                / x[:, [1]] ** 5
                * (x[:, [2]] * x[:, [3]]) ** 2
                * (x[:, [2]] + x[:, [3]])
                / x[:, [4]] ** 5
            )
            ranges = [[0, 1], [0.5, 2], [0, 1], [0, 1], [0.5, 2]]

        case "Relativistic aberration" | 105:
            symbol = theta2, v, c = symbols("theta_2 v c")
            expr = acos((cos(theta2) - v / c) / (1 - v / c * cos(theta2)))
            f = lambda x: torch.arccos(
                (torch.cos(x[:, [0]]) - x[:, [1]] / x[:, [2]])
                / (1 - x[:, [1]] / x[:, [2]] * torch.cos(x[:, [0]]))
            )
            ranges = [[0, tpi], [0, 1], [1, 2]]

        case "N-slit diffraction" | 106:
            symbol = I0, alpha, delta, N = symbols("I_0 alpha delta N")
            expr = (
                I0
                * (sin(alpha / 2) / (alpha / 2) * sin(N * delta / 2) / sin(delta / 2))
                ** 2
            )
            f = (
                lambda x: x[:, [0]]
                * (
                    torch.sin(x[:, [1]] / 2)
                    / (x[:, [1]] / 2)
                    * torch.sin(x[:, [3]] * x[:, [2]] / 2)
                    / torch.sin(x[:, [2]] / 2)
                )
                ** 2
            )
            ranges = [[0, 1], [0.1 * tpi, 0.9 * tpi], [0.1 * tpi, 0.9 * tpi], [0.5, 1]]

        case "Goldstein 3.16" | 107:
            symbol = m, E, U, L, r = symbols("m E U L r")
            expr = sqrt(2 / m * (E - U - L**2 / (2 * m * r**2)))
            f = lambda x: torch.sqrt(
                2
                / x[:, [0]]
                * (
                    x[:, [1]]
                    - x[:, [2]]
                    - x[:, [3]] ** 2 / (2 * x[:, [0]] * x[:, [4]] ** 2)
                )
            )
            ranges = [[1, 2], [2, 3], [0, 1], [0, 1], [1, 2]]

        case "Goldstein 3.55" | 108:
            symbol = m, kG, L, E, theta1, theta2 = symbols("m k_G L E theta_1 theta_2")
            expr = (
                m
                * kG
                / L**2
                * (1 + sqrt(1 + 2 * E * L**2 / (m * kG**2)) * cos(theta1 - theta2))
            )
            f = (
                lambda x: x[:, [0]]
                * x[:, [1]]
                / x[:, [2]] ** 2
                * (
                    1
                    + torch.sqrt(
                        1
                        + 2 * x[:, [3]] * x[:, [2]] ** 2 / (x[:, [0]] * x[:, [1]] ** 2)
                    )
                    * torch.cos(x[:, [4]] - x[:, [5]])
                )
            )
            ranges = [[0.5, 2], [0.5, 2], [0.5, 2], [0, 1], [0, 2 * tpi], [0, 2 * tpi]]

        case "Goldstein 3.64 (ellipse)" | 109:
            symbol = d, alpha, theta1, theta2 = symbols("d alpha theta_1 theta_2")
            expr = d * (1 - alpha**2) / (1 + alpha * cos(theta2 - theta1))
            f = (
                lambda x: x[:, [0]]
                * (1 - x[:, [1]] ** 2)
                / (1 + x[:, [1]] * torch.cos(x[:, [2]] - x[:, [3]]))
            )
            ranges = [[0, 1], [0, 0.9], [0, 2 * tpi], [0, 2 * tpi]]

        case "Goldstein 3.74 (Kepler)" | 110:
            symbol = d, G, m1, m2 = symbols("d G m_1 m_2")
            expr = 2 * pi * d ** (3 / 2) / sqrt(G * (m1 + m2))
            f = (
                lambda x: 2
                * tpi
                * x[:, [0]] ** (3 / 2)
                / torch.sqrt(x[:, [1]] * (x[:, [2]] + x[:, [3]]))
            )
            ranges = [[0, 1], [0.5, 2], [0.5, 2], [0.5, 2]]

        case "Goldstein 3.99" | 111:
            symbol = eps, E, L, m, Z1, Z2, q = symbols("epsilon E L m Z_1 Z_2 q")
            expr = sqrt(1 + 2 * eps**2 * E * L**2 / (m * (Z1 * Z2 * q**2) ** 2))
            f = lambda x: torch.sqrt(
                1
                + 2
                * x[:, [0]] ** 2
                * x[:, [1]]
                * x[:, [2]] ** 2
                / (x[:, [3]] * (x[:, [4]] * x[:, [5]] * x[:, [6]] ** 2) ** 2)
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0.5, 2], [0.5, 2], [0.5, 2], [0.5, 2]]

        case "Goldstein 8.56" | 112:
            symbol = p, q, A, c, m, Ve = symbols("p q A c m V_e")
            expr = sqrt((p - q * A) ** 2 * c**2 + m**2 * c**4) + q * Ve
            f = (
                lambda x: torch.sqrt(
                    (x[:, [0]] - x[:, [1]] * x[:, [2]]) ** 2 * x[:, [3]] ** 2
                    + x[:, [4]] ** 2 * x[:, [3]] ** 4
                )
                + x[:, [1]] * x[:, [5]]
            )
            ranges = [[0, 1]]

        case "Goldstein 12.80" | 113:
            symbol = m, p, omega, x, alpha, y = symbols("m p omega x alpha y")
            expr = 1 / (2 * m) * (p**2 + m**2 * omega**2 * x**2 * (1 + alpha * y / x))
            f = (
                lambda x: 1
                / (2 * x[:, [0]])
                * (
                    x[:, [1]] ** 2
                    + x[:, [0]] ** 2
                    * x[:, [2]] ** 2
                    * x[:, [3]] ** 2
                    * (1 + x[:, [4]] * x[:, [3]] / x[:, [5]])
                )
            )
            ranges = [[0.5, 2], [0, 1], [0, 1], [0, 1], [0, 1], [0.5, 2]]

        case "Jackson 2.11" | 114:
            symbol = q, eps, y, Ve, d = symbols("q epsilon y V_e d")
            expr = (
                q
                / (4 * pi * eps * y**2)
                * (4 * pi * eps * Ve * d - q * d * y**3 / (y**2 - d**2) ** 2)
            )
            f = (
                lambda x: x[:, [0]]
                / (4 * tpi * x[:, [1]] * x[:, [2]] ** 2)
                * (
                    4 * tpi * x[:, [1]] * x[:, [3]] * x[:, [4]]
                    - x[:, [0]]
                    * x[:, [4]]
                    * x[:, [2]] ** 3
                    / (x[:, [2]] ** 2 - x[:, [4]] ** 2) ** 2
                )
            )
            ranges = [[0, 1], [0.5, 2], [1, 2], [0, 1], [0, 1]]

        case "Jackson 3.45" | 115:
            symbol = q, r, d, alpha = symbols("q r d alpha")
            expr = q / sqrt(r**2 + d**2 - 2 * d * r * cos(alpha))
            f = lambda x: x[:, [0]] / torch.sqrt(
                x[:, [1]] ** 2
                + x[:, [2]] ** 2
                - 2 * x[:, [1]] * x[:, [2]] * torch.cos(x[:, [3]])
            )
            ranges = [[0, 1], [0, 1], [0, 1], [0, 2 * tpi]]

        case "Jackson 4.60" | 116:
            symbol = Ef, theta, alpha, d, r = symbols("E_f theta alpha d r")
            expr = Ef * cos(theta) * ((alpha - 1) / (alpha + 2) * d**3 / r**2 - r)
            f = (
                lambda x: x[:, [0]]
                * torch.cos(x[:, [1]])
                * (
                    (x[:, [2]] - 1) / (x[:, [2]] + 2) * x[:, [3]] ** 3 / x[:, [4]] ** 2
                    - x[:, [4]]
                )
            )
            ranges = [[0, 1], [0, 2 * tpi], [0, 2], [0, 1], [0.5, 2]]

        case "Jackson 11.38 (Doppler)" | 117:
            symbol = omega, v, c, theta = symbols("omega v c theta")
            expr = sqrt(1 - v**2 / c**2) / (1 + v / c * cos(theta)) * omega
            f = (
                lambda x: torch.sqrt(1 - x[:, [1]] ** 2 / x[:, [2]] ** 2)
                / (1 + x[:, [1]] / x[:, [2]] * torch.cos(x[:, [3]]))
                * x[:, [0]]
            )
            ranges = [[0, 1], [0, 1], [1, 2], [0, 2 * tpi]]

        case "Weinberg 15.2.1" | 118:
            symbol = G, c, kf, af, H = symbols("G c k_f a_f H")
            expr = 3 / (8 * pi * G) * (c**2 * kf / af**2 + H**2)
            f = (
                lambda x: 3
                / (8 * tpi * x[:, [0]])
                * (x[:, [1]] ** 2 * x[:, [2]] / x[:, [3]] ** 2 + x[:, [4]] ** 2)
            )
            ranges = [[0.5, 2], [0, 1], [0, 1], [0.5, 2], [0, 1]]

        case "Weinberg 15.2.2" | 119:
            symbol = G, c, kf, af, H, alpha = symbols("G c k_f a_f H alpha")
            expr = (
                -1 / (8 * pi * G) * (c**4 * kf / af**2 + c**2 * H**2 * (1 - 2 * alpha))
            )
            f = (
                lambda x: -1
                / (8 * tpi * x[:, [0]])
                * (
                    x[:, [1]] ** 4 * x[:, [2]] / x[:, [3]] ** 2
                    + x[:, [1]] ** 2 * x[:, [4]] ** 2 * (1 - 2 * x[:, [5]])
                )
            )
            ranges = [[0.5, 2], [0, 1], [0, 1], [0.5, 2], [0, 1], [0, 1]]

        case "Schwarz 13.132 (Klein-Nishina)" | 120:
            symbol = alpha, hbar, m, c, omega0, omega, theta = symbols(
                "alpha hbar m c omega_0 omega theta"
            )
            expr = (
                pi
                * alpha**2
                * hbar**2
                / m**2
                / c**2
                * (omega0 / omega) ** 2
                * (omega0 / omega + omega / omega0 - sin(theta) ** 2)
            )
            f = (
                lambda x: tpi
                * x[:, [0]] ** 2
                * x[:, [1]] ** 2
                / x[:, [2]] ** 2
                / x[:, [3]] ** 2
                * (x[:, [4]] / x[:, [5]]) ** 2
                * (
                    x[:, [4]] / x[:, [5]]
                    + x[:, [5]] / x[:, [4]]
                    - torch.sin(x[:, [6]]) ** 2
                )
            )
            ranges = [
                [0, 1],
                [0, 1],
                [0.5, 2],
                [0.5, 2],
                [0.5, 2],
                [0.5, 2],
                [0, 2 * tpi],
            ]

    return symbol, expr, f, ranges
