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
"""The base model for planar devices in ceviche with a single design region."""

import abc
import concurrent.futures
import itertools
from collections.abc import Sequence
from typing import cast, overload

import ceviche
import numpy as np
import torch
from autograd.extend import Box, defvjp, primitive
from autograd.tracer import getval
from ceviche.tensor import as_tensor, to_numpy

from ceviche_challenges import defs, modes, primitives
from ceviche_challenges import units as u
from ceviche_challenges.scattering import calculate_amplitudes


def _wavelengths_nm_to_omegas(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Convert array of wavelengths to array of angular frequencies for ceviche."""
    return 2 * np.pi * u.c.to_value("nm/s") / wavelengths_nm


@primitive
def _numpy_with_torch_vjp(design, tensor_design, outputs):
    return tuple(to_numpy(output) for output in outputs)


def _torch_vjp(_answer, design, tensor_design, outputs):
    def vjp(cotangents):
        # HIPS and Torch use conjugate conventions for complex cotangents.
        gradients = tuple(
            as_tensor(value, dtype=output.dtype).conj()
            for value, output in zip(cotangents, outputs)
        )
        (gradient,) = torch.autograd.grad(outputs, tensor_design, gradients, retain_graph=True)
        return to_numpy(gradient.conj()).astype(design.dtype, copy=False)

    return vjp


defvjp(_numpy_with_torch_vjp, _torch_vjp)


class Model(abc.ABC):
    """The base class for planar devices in ceviche with a single design region.

    This class uses a background material density distribution to define features
    such as input and output waveguides, and the design variable is provided
    upon simulating the model. It is up to child classes to define the simulation
    routine.

    No up- or down-sampling of the design variable is performed while inlaying the
    design variable. Instead, the design variable shape must match that of the
    design variable size specified by the model. Often the size of the design
    variable will be identical to that of the design region, however this isn't a
    requirement, e.g. if a model wishes to impose some symmetry. A rough
    illustration of the inlaying for a planar WDM device is shown below:

                                  ___________________
                                  |                 |
                                  |            #####|
                 background       |#####            |
                   density        |            #####|
                                  |_________________|
                                  ___________________
                                  |                 |
                                  |     *******     |
                   design         |     *******     |
                   region         |     *******     |
                                  |_________________|
                                       _________
                   design              |8877733|
                   variable            |8177733|
                                       |1177733|
                                  ___________________
                                  |                 |
                                  |     8877733#####|
                   simulated      |#####8177733     |
                   density        |     1177733#####|
                                  |_________________|

    """

    # Axes for model outputs
    SPARAMS_FREQ_AXIS = -3  # pylint: disable=invalid-name
    SPARAMS_INPUT_PORT_AXIS = -2  # pylint: disable=invalid-name
    SPARAMS_OUTPUT_PORT_AXIS = -1  # pylint: disable=invalid-name
    FIELDS_FREQ_AXIS = -4  # pylint: disable=invalid-name
    FIELDS_INPUT_PORT_AXIS = -3  # pylint: disable=invalid-name
    FIELDS_OUTPUT_PORT_AXIS = -2  # pylint: disable=invalid-name

    def __init__(self):
        """Initializes a new model."""

    @overload
    def simulate(
        self,
        design_variable: torch.Tensor,
        excite_port_idxs: Sequence[int] = (0,),
        wavelengths_nm: np.ndarray | Sequence[float] | None = None,
        max_parallelizm: int | None = None,
        *,
        solver_options: ceviche.SolverOptions | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def simulate(
        self,
        design_variable: np.ndarray,
        excite_port_idxs: Sequence[int] = (0,),
        wavelengths_nm: np.ndarray | Sequence[float] | None = None,
        max_parallelizm: int | None = None,
        *,
        solver_options: ceviche.SolverOptions | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def simulate(
        self,
        design_variable: defs.Array,
        excite_port_idxs: Sequence[int] = (0,),
        wavelengths_nm: np.ndarray | Sequence[float] | None = None,
        max_parallelizm: int | None = None,
        *,
        solver_options: ceviche.SolverOptions | None = None,
    ) -> tuple[defs.Array, defs.Array]:
        """Simulate the device described by the model on a design variable.

        Args:
          design_variable: A NumPy array or Torch tensor specifying the design.
            Torch tensors select the device and precision for the simulation.
          excite_port_idxs: A sorted sequence of ints specifying the ports to
            excite. These values are indexed starting from 0 for port 1. The default
            behavior is to excite only port 1.
          wavelengths_nm: an (optional) ndarray describing the free-space
            wavelengths for which we will perform simulations.  The ndarray must be
            rank-1. If None, the resulting simulation will be as if the user had
            passed `self.output_wavelengths`--a rank-1 array.
          max_parallelizm: Maximum number of CPU threads executing simulations.
            By default, CPU runs one task per excitation port and wavelength.
            CUDA simulations run sequentially on the caller's stream.
          solver_options: Optional sparse-solver settings passed to Ceviche.

        Returns:
          s_params: Complex scattering parameters with axes for wavelength,
            excitation port, and output port.
          fields: Complex steady-state Ez fields with axes for wavelength,
            excitation port, x, and y.
          Both outputs have the same array type as the design variable.
        """
        if len(excite_port_idxs) == 0:
            raise ValueError("At least one excitation port is required.")
        if np.max(excite_port_idxs) > len(self.ports) - 1:
            raise ValueError(
                f"Invalid port index, {np.max(excite_port_idxs)}, which exceeds the number of "
                f"ports in the device, {len(self.ports)}."
            )
        if np.min(excite_port_idxs) < 0:
            raise ValueError(
                f"Invalid port index, {np.min(excite_port_idxs)}, which below the minimum port "
                "index of 0."
            )
        if len(np.unique(excite_port_idxs)) != len(excite_port_idxs):
            raise ValueError("Duplicate port index specified in `excite_port_idxs`.")
        if not np.all(np.sort(excite_port_idxs) == np.asarray(excite_port_idxs)):
            raise ValueError("Ports specified in `excite_port_idxs` are not sorted.")

        wavelengths_nm = np.asarray(
            self.output_wavelengths if wavelengths_nm is None else wavelengths_nm
        )
        if wavelengths_nm.ndim != 1 or wavelengths_nm.size == 0:
            raise ValueError("`wavelengths_nm` arg must be a non-empty rank-1 array.")
        omegas = _wavelengths_nm_to_omegas(wavelengths_nm)

        if isinstance(design_variable, Box):
            with torch.enable_grad():
                design = as_tensor(getval(design_variable)).requires_grad_()
                outputs = self._simulate_tensor(
                    design, excite_port_idxs, omegas, max_parallelizm, solver_options
                )
            return _numpy_with_torch_vjp(design_variable, design, outputs)

        s_params, fields = self._simulate_tensor(
            design_variable, excite_port_idxs, omegas, max_parallelizm, solver_options
        )
        if isinstance(design_variable, torch.Tensor):
            return s_params, fields
        return to_numpy(s_params), to_numpy(fields)

    def _simulate_tensor(
        self, design_variable, excite_port_idxs, omegas, max_parallelizm, solver_options
    ):
        pml_width = self.pml_width
        dl = self.dl
        epsilon_r = as_tensor(self.epsilon_r(design_variable))
        # Port sources and monitor modes depend only on the fixed background.
        epsilon_r_bg = self.epsilon_r_bg()
        ports = self.ports
        grad_enabled = torch.is_grad_enabled()
        inference_mode = torch.is_inference_mode_enabled()

        def _simulate(task):
            omega, excite_port_idx = task
            # Torch's grad and inference modes are local to each worker thread.
            with torch.inference_mode(inference_mode), torch.set_grad_enabled(grad_enabled):
                sim = ceviche.fdfd_ez(
                    omega, dl, epsilon_r, [pml_width, pml_width], solver_options=solver_options
                )
                source = ports[excite_port_idx].source_fdfd(omega, dl, epsilon_r_bg)
                hx, hy, ez = sim.solve(source)
                incoming, outgoing = zip(
                    *(
                        calculate_amplitudes(omega, dl, port, ez, hy, hx, epsilon_r_bg)
                        for port in ports
                    )
                )
                return torch.stack(outgoing) / incoming[excite_port_idx], ez

        tasks = list(itertools.product(omegas, excite_port_idxs))
        num_workers = max_parallelizm or len(tasks)
        if num_workers < 1:
            raise ValueError("`max_parallelizm` must be positive.")
        if epsilon_r.is_cuda or num_workers == 1:
            results = list(map(_simulate, tasks))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_simulate, tasks))

        sparams, efields = zip(*results)
        leading_shape = (len(omegas), len(excite_port_idxs))
        return (
            torch.stack(sparams).reshape(*leading_shape, len(ports)),
            torch.stack(efields).reshape(*leading_shape, *epsilon_r.shape),
        )

    def density(self, design_variable: defs.Array) -> defs.Array:
        """The combined (design + background) density distribution of the model."""
        if design_variable.shape != self.design_variable_shape:
            raise ValueError(
                f"Invalid design variable shape. Got {tuple(design_variable.shape)} "
                f"but expected {self.design_variable_shape}"
            )
        return primitives.insert_design_variable(
            self.transform_design_variable(design_variable),
            self.density_bg,
            self.design_region_coords,
        )

    def epsilon_r(self, design_variable: defs.Array) -> defs.Array:
        """The combined permittivity distribution of the model."""
        return self._epsilon_r(self.density(design_variable))

    def epsilon_r_bg(self) -> np.ndarray:
        """The background permittivity distribution of the model."""
        return cast(np.ndarray, self._epsilon_r(self.density_bg))

    def _epsilon_r(self, density: defs.Array) -> defs.Array:
        """Helper function for mapping density values to permittivity values."""
        return (
            self.cladding_permittivity
            + (self.slab_permittivity - self.cladding_permittivity) * density
        )

    def transform_design_variable(self, design_variable: defs.Array) -> defs.Array:
        """Transformation of design variable before inlaying to the design region.

        By default this transformation is the identity operation and
        `self.design_variable_shape == self.design_region_shape`. However, models
        may override this operation to perform transformations on the design
        variable such as resampling or mirroring.

        Args:
          design_variable: An array of shape `self.design_variable_shape`
            that corresponds to the design variable.

        Returns:
          An array of shape `self.design_region_shape` that can be inlaid to
            the design region.
        """
        return design_variable

    @property
    def design_region(self) -> np.ndarray:
        """A boolean mask for the design region."""
        x0, y0, x1, y1 = self.design_region_coords
        mask = np.zeros(self.shape, dtype=bool)
        mask[x0:x1, y0:y1] = True
        return mask

    @property
    def design_variable_shape(self) -> tuple[int, int]:
        """Shape of the design variable, in grid units."""
        return self.design_region_shape

    @property
    def design_region_shape(self) -> tuple[int, int]:
        """Shape of the design region, in grid units."""
        (x_min, y_min, x_max, y_max) = self.design_region_coords
        return (x_max - x_min, y_max - y_min)

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        """Shape of the simulation domain, in grid units."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def design_region_coords(self) -> tuple[int, int, int, int]:
        """The coordinates of the design region as (x_min, y_min, x_max, y_max)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def slab_permittivity(self) -> float:
        """The slab permittivity of the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cladding_permittivity(self) -> float:
        """The cladding permittivity of the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def density_bg(self) -> np.ndarray:
        """The background density distribution of the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dl(self) -> float:
        """The grid resolution of the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pml_width(self) -> int:
        """The width of the PML region, in grid units."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ports(self) -> Sequence[modes.Port]:
        """A list of the device ports."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_wavelengths(self) -> Sequence[float] | np.ndarray:
        """A list of the wavelengths, in nm, to output fields and s-parameters."""
        raise NotImplementedError
