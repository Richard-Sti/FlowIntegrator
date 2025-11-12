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
from tqdm import tqdm


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

    def integrated_density_single(self, density, points, radii):
        """
        Integrated density for a single cube for multiple radii.

        Parameters
        ----------
        density : array_like
            Density cube with shape
            ``(resolution, resolution, resolution)``.
        points : array_like
            Query position (3,).
        radii : float or array_like
            Single sphere radius or array of sphere radii in physical units.

        Returns
        -------
        float or np.ndarray
            Enclosed mass in Msun / h. Returns a float if a single radius
            is provided, otherwise returns an array.
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

        # Check if radii is a scalar or an array
        is_scalar_radius = np.isscalar(radii)
        radii = np.atleast_1d(radii).astype(np.float64)

        if radii.ndim != 1:
            raise ValueError("radii must be a scalar or a 1D array")

        enclosed_masses = []
        for r in radii:
            offsets = self._get_indices(r)
            centers = _points_to_indices(points, self.cell, self.resolution)

            idx = (centers[:, None, :] + offsets[None, :, :]) % self.resolution
            linear = np.ravel_multi_index(
                (idx[..., 0], idx[..., 1], idx[..., 2]),
                dims=(self.resolution,) * 3,
            )

            values = density.ravel()[linear]

            # in (Mpc/h)^3. The conversion factor to Msun / h is 1e9.
            enclosed_mass = np.sum(values) * self.cell**3
            enclosed_masses.append(enclosed_mass * 1e9)

        result = np.array(enclosed_masses)
        if is_scalar_radius:
            return result[0]
        return result

    def integrated_density_random_points(self, density, radii, num_points,
                                         seed=None):
        """
        Computes the integrated density for multiple random points in the box.

        Parameters
        ----------
        density : array_like
            Density cube with shape
            ``(resolution, resolution, resolution)``.
        radii : float or array_like
            Single sphere radius or array of sphere radii in physical units.
        num_points : int
            Number of random points to generate.
        seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        np.ndarray
            A 2D array of enclosed masses in Msun / h, with shape
            (num_points, len(radii)).
        """
        if seed is not None:
            np.random.seed(seed)

        random_points = np.random.uniform(
            0, self.box_size, size=(num_points, 3))

        # Ensure radii is an array for consistent length checking
        radii_array = np.atleast_1d(radii)

        # Pre-allocate the results array
        results = np.empty((num_points, len(radii_array)), dtype=np.float64)

        for i, point in enumerate(tqdm(random_points,
                                       desc="Processing points")):
            enclosed_masses = self.integrated_density_single(
                density, point, radii
            )
            results[i] = enclosed_masses

        return results
