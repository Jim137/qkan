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
Parameter-shift gradients for the QKAN real-device solvers.

A device returns expectation values and no gradients. Every parametrised gate
of the QKAN circuits is a rotation ``exp(-i angle P / 2)``, so the derivative
of an expectation ``f`` in one gate's angle is
``(f(angle + pi/2) - f(angle - pi/2)) / 2``.
"""

import itertools
import math

import torch

from .flydsl.flydsl_ops import _first_order_only

_NO_SECOND_DERIVATIVES = (
    "the qiskit and cudaq solvers have no second derivatives, so their "
    "gradients cannot be differentiated; use solver='exact' for higher-order "
    "gradients"
)


def encoding_angles(x, preacts_weight, preacts_bias, config):
    """The encoding angle ``w * x + b`` of every rep, shape (batch_size,
    out_dim, in_dim, reps), or None for circuits that encode ``x`` itself."""
    if not (config["preacts_trainable"] or config["ansatz"] in ("rpz_encoding", "rpz")):
        return None
    in_dim = x.shape[1]
    if len(preacts_weight.shape) != 3:
        preacts_weight = preacts_weight.unsqueeze(0)
        preacts_bias = preacts_bias.unsqueeze(0)
    if preacts_weight.shape[1] != in_dim:
        repeat_out = config["out_dim"]
        repeat_in = in_dim // preacts_weight.shape[1] + 1
        preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]
        preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]
    return torch.einsum("oir,bi->boir", preacts_weight, x).add(preacts_bias)


class ParamShift(torch.autograd.Function):
    """The expectations of ``evaluate`` with parameter-shift gradients.

    ``evaluate(x, theta, preacts_weight, preacts_bias, reps, config)`` runs the
    circuits and returns their (batch_size, out_dim, in_dim) expectations.
    """

    @staticmethod
    def forward(ctx, evaluate, x, theta, preacts_weight, preacts_bias, reps, config):
        ctx.save_for_backward(x, theta, preacts_weight, preacts_bias)
        ctx.evaluate, ctx.reps, ctx.config = evaluate, reps, config
        return evaluate(x, theta, preacts_weight, preacts_bias, reps, config)

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_weight, preacts_bias = ctx.saved_tensors
        need_x, need_theta, need_weight, need_bias = ctx.needs_input_grad[1:5]
        reps, config = ctx.reps, ctx.config

        def run(x=x, theta=theta, preacts_bias=preacts_bias):
            return ctx.evaluate(x, theta, preacts_weight, preacts_bias, reps, config)

        def shift_derivative(name, tensor, index):
            """d f / d angle of every circuit, for the gates ``tensor[index]`` feeds."""
            plus, minus = tensor.clone(), tensor.clone()
            plus[index] += math.pi / 2
            minus[index] -= math.pi / 2
            return (run(**{name: plus}) - run(**{name: minus})) / 2

        grad_x = grad_theta = grad_weight = grad_bias = None
        if need_theta:
            grad_theta = torch.zeros_like(theta)
            for index in itertools.product(*map(range, theta.shape)):
                derivative = shift_derivative("theta", theta, index)
                grad_theta[index] = (grad_output * derivative).sum()

        with torch.inference_mode(False), torch.enable_grad():
            inputs = [
                t.detach().requires_grad_() for t in (x, preacts_weight, preacts_bias)
            ]
            angles = encoding_angles(*inputs, config)
        if angles is None:
            if need_x:
                # x is the angle of all reps encoding gates of a circuit, so f
                # has the frequencies 0..reps in x: the rule with 2 * reps
                # equidistant shifts (Wierichs et al., Quantum 6, 677).
                derivative = torch.zeros_like(grad_output)
                for m in range(2 * reps):
                    shift = (2 * m + 1) * math.pi / (2 * reps)
                    weight = (-1) ** m / (4 * reps * math.sin(shift / 2) ** 2)
                    derivative += weight * run(x=x + shift)
                grad_x = (grad_output * derivative).sum(dim=1)
        elif need_x or need_weight or need_bias:
            # The encoding gate of rep l has the angle w_l * x + b_l: shifting
            # b_l shifts that gate alone, and the chain rule gives the rest.
            derivative = torch.stack(
                [
                    shift_derivative("preacts_bias", preacts_bias, (..., l))
                    for l in range(reps)
                ],
                dim=-1,
            )
            grad_x, grad_weight, grad_bias = torch.autograd.grad(
                angles,
                inputs,
                grad_output.unsqueeze(-1) * derivative,
                create_graph=torch.is_grad_enabled(),
            )

        grads = [grad_x, grad_theta, grad_weight, grad_bias]
        if torch.is_grad_enabled():  # a backward with create_graph=True
            # The gradients are linear in grad_output and stay differentiable
            # in it. Differentiating them in the circuits' inputs would need
            # second derivatives, so that path raises.
            (raises,) = _first_order_only(
                (grad_output.new_zeros(()),),
                (x, theta, preacts_weight, preacts_bias),
                _NO_SECOND_DERIVATIVES,
            )
            grads = [None if g is None else g + raises.to(g.dtype) for g in grads]
        return None, *grads, None, None
