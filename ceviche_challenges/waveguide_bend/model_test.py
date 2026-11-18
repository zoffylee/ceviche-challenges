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
"""Tests for ceviche_challenges.waveguide_bend.model."""

from typing import Any

import autograd
import autograd.numpy as _autograd_numpy
import numpy as np
import torch
from absl.testing import absltest, parameterized

from ceviche_challenges import units as u
from ceviche_challenges.waveguide_bend import model as _model
from ceviche_challenges.waveguide_bend import prefabs

# Autograd installs the NumPy API dynamically.
npa: Any = _autograd_numpy

_ATOL_SCATTERING_PARAMETERS = 1e-2


class ModelScatteringTest(parameterized.TestCase):
    def test_torch_and_legacy_gradients(self):
        model = _model.WaveguideBendModel(
            prefabs.waveguide_bend_sim_params(), prefabs.waveguide_bend_1umx1um_spec()
        )
        design = np.full(model.design_variable_shape, 0.5)
        tensor_design = torch.tensor(design, requires_grad=True)
        output = model.ports[1]

        def objective(x):
            s_params, fields = model.simulate(x, wavelengths_nm=[1500.0, 1550.0])
            xp = torch if isinstance(x, torch.Tensor) else npa
            return (
                xp.real(s_params[:, 0, 1]).mean()
                + 1e-5 * xp.imag(fields[:, 0, output.x, output.y]).mean()
            )

        loss = objective(tensor_design)
        loss.backward()
        legacy_loss, legacy_grad = autograd.value_and_grad(objective)(design)
        self.assertAlmostEqual(loss.item(), legacy_loss)
        gradient = tensor_design.grad
        assert gradient is not None
        np.testing.assert_allclose(gradient.numpy(), legacy_grad, atol=1e-10)

        step = 1e-5
        expected = (objective(design + step) - objective(design - step)) / (2 * step)
        self.assertGreater(gradient.norm().item(), 0)
        np.testing.assert_allclose(gradient.sum().item(), expected, rtol=1e-5, atol=1e-7)
        with torch.no_grad():
            self.assertFalse(objective(tensor_design).requires_grad)

    @parameterized.named_parameters(
        {
            "testcase_name": "1umx1um_resolution_50nm",
            "resolution": 50 * u.nm,
            "spec_prefab_fn": prefabs.waveguide_bend_1umx1um_spec,
        },
        {
            "testcase_name": "2umx2um_resolution_50nm",
            "resolution": 50 * u.nm,
            "spec_prefab_fn": prefabs.waveguide_bend_2umx2um_spec,
        },
        {
            "testcase_name": "3umx3um_resolution_50nm",
            "resolution": 50 * u.nm,
            "spec_prefab_fn": prefabs.waveguide_bend_3umx3um_spec,
        },
        {
            "testcase_name": "1umx1um_resolution_25nm",
            "resolution": 25 * u.nm,
            "spec_prefab_fn": prefabs.waveguide_bend_1umx1um_spec,
        },
        {
            "testcase_name": "2umx2um_resolution_25nm",
            "resolution": 25 * u.nm,
            "spec_prefab_fn": prefabs.waveguide_bend_2umx2um_spec,
        },
        {
            "testcase_name": "3umx3um_resolution_25nm",
            "resolution": 25 * u.nm,
            "spec_prefab_fn": prefabs.waveguide_bend_3umx3um_spec,
        },
    )
    def test_scattering_matrix_symmetry(
        self,
        resolution=50 * u.nm,
        spec_prefab_fn=prefabs.waveguide_bend_1umx1um_spec,
    ):
        """Test that the scattering matrix is symmetric (reciprocal)."""
        params = prefabs.waveguide_bend_sim_params(resolution=resolution)
        spec = spec_prefab_fn()
        model = _model.WaveguideBendModel(params, spec)

        design_var = np.ones(model.design_variable_shape)
        s_params, _ = model.simulate(
            design_var,
            excite_port_idxs=(0, 1),
        )
        s = np.abs(s_params.squeeze())
        # Check overall scattering matrix symmetry
        np.testing.assert_allclose(
            s,
            s.T,
            atol=_ATOL_SCATTERING_PARAMETERS,
        )


if __name__ == "__main__":
    absltest.main()
