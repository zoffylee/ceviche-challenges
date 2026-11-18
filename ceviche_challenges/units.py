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
"""Definitions of compatible units, and tools for using unitted quantities."""

from collections.abc import Sequence as Sequence_

import numpy as np
import unyt
import unyt.exceptions

# Pervasive type alias for a quantity with units.
Quantity = unyt.unyt_quantity  # pylint: disable=invalid-name

# Type alias for an array of quantities sharing the same unit. (subclasses
# np.ndarray)
Array = unyt.unyt_array  # pylint: disable=invalid-name

# Type alias for an arbitrary sequence of quantities (e.g. Tuple or List of Q,
# or an `Array`)
Sequence = Sequence_[Quantity] | Array  # pylint: disable=invalid-name

# ------------------------------------------------------------------------------
# Define units.  PyType can't reason about the dynamics in unyt, so we disable a
# check:
# pytype: disable=module-attr

# Length units.
nm = unyt.nm
um = unyt.um

# Vacuum speed of light.
c = unyt.c

# pytype: enable=module-attr

# Quantization tolerance as a fraction of resolution.
_RESOLUTION_TOLERANCE = 0.0001


class ResolutionError(Exception):
    """Exception signaling that a quantity could not be resolved."""


def resolve(v: np.ndarray, resolution: np.ndarray) -> int:
    """Resolves `v` to an integer number of grid units at `resolution`.

    Note that `v` is resolved to an integer multiple (`q`) of `resolution` if it
    is within `_RESOLUTION_TOLERANCE * resolution` of `q * resolution`; otherwise
    the exception `ResolutionError` is raised.

    Args:
      v: a quantity with units.
      resolution: a grid resolution in the same *dimension* (if not units) as `v`.

    Returns:
      the integer number of grid units (with scale `resolution`) that equates to
      `v`, within a numerical tolerance.

    Raises:
      ResolutionError: if `v` is not an integral multiple of `resolution`.
      ValueError: if `v` and `resolution` have incompatible unit dimensions.
    """
    resolved = _resolve_or_none(v, resolution)
    if resolved is None:
        raise ResolutionError(
            f"Cannot neatly resolve quantity ({v!r}) at given resolution ({resolution!r})."
        )
    else:
        return resolved


def _check_compatible(v: np.ndarray, resolution: np.ndarray):
    """Checks compatibility for `Quantity` and dimensions."""
    if not (isinstance(v, Quantity) and isinstance(resolution, Quantity)):
        # Preserve the public exception type for invalid quantities.
        raise ValueError("Arguments to `resolve` must be of type `Quantity`.")  # noqa: TRY004
    if v.units.dimensions != resolution.units.dimensions:
        raise ValueError("Arguments to `resolve` must have compatible unit dimensions.")


def _resolve_or_none(
    v: np.ndarray,
    resolution: np.ndarray,
    tolerance: float = _RESOLUTION_TOLERANCE,
) -> int | None:
    """Resolves `v` on the grid with given `resolution`, or returns `None`.

    If `v` is not within `tolerance` of an integer multiple (`q`) of `resolution`,
    this function returns `None`. Otherwise, it returns `q`.

    Args:
      v: a unitted-quantity.
      resolution: a grid resolution in the same *dimension* (if not units) as `v`.
      tolerance: a float giving the tolerance of the resolution operation.

    Returns:
      the integer multiple `q` resulting from the resolution operation, or `None`.
    """
    _check_compatible(v, resolution)
    count = v / resolution
    count_rounded = np.round(count).astype(int)
    if np.max(np.abs(count - count_rounded)) < tolerance:
        return int(count_rounded)
    else:
        return None
