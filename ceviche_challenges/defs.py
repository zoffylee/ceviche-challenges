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
"""Definitions and types for the ceviche scattering parameters."""

import enum

import numpy as np
import torch

Array = np.ndarray | torch.Tensor
Geometry = np.ndarray
Slice = tuple[np.ndarray, np.ndarray]
Field = Array
FieldComponent = Field | float | complex
VectorField = tuple[FieldComponent, FieldComponent, FieldComponent]


@enum.unique
class Direction(enum.Enum):
    """Defines a signed direction."""

    Y_NEG = -2
    X_NEG = -1
    X_POS = 1
    Y_POS = 2

    @property
    def sign(self):
        """The sign (+/-) of the `Direction`."""
        if self is self.Y_NEG or self is self.X_NEG:
            return -1
        else:
            return +1

    @property
    def is_along_x(self):
        """Is the `Direction` along the x-axis."""
        return self is self.X_POS or self is self.X_NEG

    @property
    def is_along_y(self):
        """Is the `Direction` along the y-axis."""
        return not self.is_along_x

    @property
    def index(self):
        """Integer to index the component of a `VectorField` corresponding to the `Direction`."""
        return np.abs(self.value) - 1
