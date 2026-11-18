# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable design-region operations."""

import numpy as np
import torch
from autograd.extend import defvjp, primitive
from ceviche.tensor import as_tensor

from ceviche_challenges import defs


@primitive
def insert_design_variable(
    design_var: defs.Array, destination: defs.Array, coords: tuple[int, int, int, int]
) -> defs.Array:
    """Insert 2D design variable a into a larger 2D ndarray at coordinates.

    NOTE: This function only connects gradients flowing through `design_var` and
    not `destination` or `coords`.

    Args:
      design_var: A NumPy array or Torch tensor specifying the design variable.
      destination: An array specifying the destination for
        `design_var` to be inserted.
      coords: A `Tuple` specifying the coordinates to insert design_var in the
        format `(x_min, y_min, x_max, y_max)`.

    Returns:
      An array with the same type as `design_var` and shape as `destination`, with
        `design_var` inserted at the location specified by `coords`.
    """
    (x_min, y_min, x_max, y_max) = coords
    if design_var.shape[0] > destination.shape[0] or design_var.shape[1] > destination.shape[1]:
        raise ValueError(
            f"The `design_var` array with shape {design_var.shape} does not fit into the "
            f"`destination` array with shape {destination.shape}"
        )
    if not np.all([coord > 0 for coord in coords]):
        raise ValueError("All values in `coord` must be positive")
    if x_min >= x_max:
        raise ValueError("The min x value must be less than the max x value")
    if y_min >= y_max:
        raise ValueError("The min y value must be less than the max y value")
    if x_max >= destination.shape[0] or y_max >= destination.shape[1]:
        raise ValueError("Box defined by `coords` extends outside of `destination` array")
    if isinstance(design_var, torch.Tensor):
        destination_ = (
            as_tensor(destination, device=design_var.device, dtype=design_var.dtype)
            .detach()
            .clone()
        )
    else:
        destination_ = np.copy(destination)
    destination_[x_min:x_max, y_min:y_max] = design_var
    return destination_


def vjp_maker(ans, design_var, destination, coords):
    del ans, design_var, destination
    return lambda x: x[coords[0] : coords[2], coords[1] : coords[3]]


defvjp(insert_design_variable, vjp_maker, None, None)
