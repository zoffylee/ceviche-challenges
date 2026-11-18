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
"""Tests for ceviche_challenges.primitives."""

import numpy as np
import torch
from absl.testing import absltest

from ceviche_challenges import primitives


class PrimitivesTest(absltest.TestCase):
    def test_grad_insert_design_variable(self):
        """Test the gradient of the `insert_design_variable` primitive."""

        destination = torch.ones((10, 10), dtype=torch.float64, requires_grad=True)
        design_var = torch.linspace(0.5, 1.5, 25, dtype=torch.float64).reshape(5, 5)
        design_var.requires_grad_()
        coords = (3, 4, 8, 9)

        density = primitives.insert_design_variable(design_var.square() - 1, destination, coords)
        density.sum().backward()
        torch.testing.assert_close(design_var.grad, 2 * design_var.detach())
        self.assertIsNone(destination.grad)

    def test_insert_design_variable_invalid(self):
        """Test that a ValueError is thrown for invalid coords or design vars."""
        with self.assertRaises(ValueError):
            destination = np.ones((10, 10), dtype=np.float64)
            design_var = np.ones((5, 5), dtype=np.float64)
            # Negative coordinates
            coords = (-1, 4, 4, 9)
            primitives.insert_design_variable(design_var, destination, coords)
        with self.assertRaises(ValueError):
            destination = np.ones((10, 10), dtype=np.float64)
            design_var = np.ones((5, 5), dtype=np.float64)
            # Incorrect ordering of coordinates
            coords = (8, 4, 3, 9)
            primitives.insert_design_variable(design_var, destination, coords)
        with self.assertRaises(ValueError):
            destination = np.ones((10, 10), dtype=np.float64)
            design_var = np.ones((5, 5), dtype=np.float64)
            # Extending outside of the destination
            coords = (8, 4, 13, 9)
            primitives.insert_design_variable(design_var, destination, coords)
        with self.assertRaises(ValueError):
            destination = np.ones((10, 10), dtype=np.float64)
            design_var = np.ones((15, 5), dtype=np.float64)
            # Too large of a design variable
            coords = (0, 4, 14, 9)
            primitives.insert_design_variable(design_var, destination, coords)


if __name__ == "__main__":
    absltest.main()
