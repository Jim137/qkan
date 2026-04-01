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


def _cast_grads_to_dtype(grad_x, grad_theta, grad_pw, grad_pb, dtype):
    """Cast f32 gradients back to parameter dtype (used after bf16 backward)."""
    grad_x = grad_x.to(dtype)
    grad_theta = grad_theta.to(dtype)
    if grad_pw is not None:
        grad_pw = grad_pw.to(dtype)
    if grad_pb is not None:
        grad_pb = grad_pb.to(dtype)
    return grad_x, grad_theta, grad_pw, grad_pb
