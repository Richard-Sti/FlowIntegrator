# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Spherical aperture utilities."""

import numpy as np


def _sphere_offsets(cell, radius):
    reach = int(np.ceil(radius / cell))
    grid = np.arange(-reach, reach + 1)
    dx, dy, dz = np.meshgrid(grid, grid, grid, indexing="ij")
    offsets = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
    distances_sq = np.sum((offsets * cell) ** 2, axis=1)
    distances = np.sqrt(distances_sq)
    return offsets[distances <= radius]


def _points_to_indices(points, cell, resolution):
    scaled = np.asarray(points) / cell - 0.5
    return np.rint(scaled).astype(np.int64) % resolution


class SphericalIntegrator:
    """
    Spherical aperture integrator for fixed resolution and box size.

    Parameters
    ----------
    box_size : float
        Physical size of the domain (same units as query coordinates).
    resolution : int
        Grid resolution (assumes a cubic layout with periodic wrapping).
    """

    def __init__(self, box_size, resolution):
        self.box_size = float(box_size)
        self.resolution = int(resolution)
        self.cell = self.box_size / self.resolution
        self._cache = {}

    def _get_indices(self, radius):
        cached_radius = self._cache.get("radius")
        if cached_radius is None or not np.isclose(cached_radius, radius):
            self._cache.clear()
            offsets = _sphere_offsets(self.cell, radius)
            if offsets.size == 0:
                raise ValueError("radius is smaller than half a grid cell")
            self._cache["radius"] = radius
            self._cache["offsets"] = offsets
        return self._cache["offsets"]

    def integrated_density(self, density, points, radius):
        """
        Integrated density for a single cube.

        Parameters
        ----------
        density : array_like
            Density cube with shape
            ``(resolution, resolution, resolution)``.
        points : array_like
            Query position (3,).
        radius : float
            Sphere radius in physical units.

        Returns
        -------
        float
            Enclosed mass in Msun / h.
        """
        density = np.asarray(density)
        shape = (self.resolution,) * 3
        if density.shape != shape:
            raise ValueError(f"density must have shape {shape}")

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 1 or points.shape != (3,):
            raise ValueError(
                "points must have shape (3,) for a single query point"
                )
        points = points[None, :]  # Reshape to (1, 3)

        offsets = self._get_indices(radius)
        centers = _points_to_indices(points, self.cell, self.resolution)

        idx = (centers[:, None, :] + offsets[None, :, :]) % self.resolution
        linear = np.ravel_multi_index(
            (idx[..., 0], idx[..., 1], idx[..., 2]),
            dims=(self.resolution,) * 3,
        )

        values = density.ravel()[linear]

        # The density is in h^2 Msun / kpc^3, and the cell volume is
        # in (Mpc/h)^3.
        enclosed_mass = np.sum(values) * self.cell**3
        return enclosed_mass * 1e9
