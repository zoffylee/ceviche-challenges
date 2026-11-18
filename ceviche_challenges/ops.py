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
"""Helper functions for operations on VectorField types."""

import numpy as np
import torch

from ceviche_challenges import defs


def _conjugate(value: defs.FieldComponent) -> defs.FieldComponent:
    return value.conj() if isinstance(value, torch.Tensor) else np.conj(value)


def cross(a: defs.VectorField, b: defs.VectorField) -> defs.VectorField:
    """Compute the cross product between two VectorFields."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def overlap(
    a: defs.VectorField,
    b: defs.VectorField,
    normal: defs.Direction,
) -> complex | torch.Tensor:
    """Numerically compute the overlap integral of two VectorFields.

    Args:
      a: `VectorField` specifying the first field.
      b: `VectorField` specifying the second field.
      normal: `Direction` specifying the direction normal to the plane (or slice)
        where the overlap is computed.

    Returns:
      Result of the overlap integral.
    """
    ax, ay, az = a
    ac = (_conjugate(ax), _conjugate(ay), _conjugate(az))
    integrand = cross(ac, b)[normal.index]
    return integrand.sum() if isinstance(integrand, torch.Tensor) else np.sum(integrand)
