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
"""Data loader interfaces and implementations."""

from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np


class FieldLoader(ABC):
    """Abstract base class for simulation data loaders."""

    def __init__(self, boxsize, coordinate_system, resolution):
        self._boxsize = float(boxsize)
        self._coordinate_system = str(coordinate_system)
        self._resolution = int(resolution)

    @property
    def boxsize(self):
        """Return the simulation box size (units: Mpc / h)."""
        return self._boxsize

    @property
    def coordinate_system(self):
        """Return the coordinate system of the simulation."""
        return self._coordinate_system

    @property
    def resolution(self):
        """Return the grid resolution of the simulation."""
        return self._resolution

    @abstractmethod
    def load_density_field(self):
        """Return the density field (units: h^2 Msun / kpc^3)."""

    @abstractmethod
    def load_velocity_field(self):
        """Return the velocity field as a NumPy array (units: km / s)."""


class ManticoreLoader(FieldLoader):
    """
    Loader for Manticore simulation snapshots.

    Parameters
    ----------
    base_folder : str or Path
        Directory containing the HDF5 snapshots.
    simulation_number : int
        Snapshot identifier used in ``mcmc_<id>.hdf5``.
    boxsize : float, optional
        Length of the simulation box in Mpc / h. Default: 681.0.
    """

    def __init__(self, base_folder, simulation_number, boxsize=681.0):
        self.base_folder = Path(base_folder)
        self.simulation_number = int(simulation_number)
        self._density_cache = None  # Initialize cache
        with h5py.File(self._file_path(), "r") as handle:
            resolution = handle["density"].shape[0]
        super().__init__(
            boxsize, coordinate_system="icrs", resolution=resolution
        )

    def _file_path(self):
        name = f"mcmc_{self.simulation_number}.hdf5"
        return self.base_folder / name

    def _load_original_density(self):
        if self._density_cache is None:
            with h5py.File(self._file_path(), "r") as handle:
                self._density_cache = handle["density"][:]
        return self._density_cache

    def load_density_field(self):
        field = self._load_original_density().copy()
        norm_factor = (self.boxsize * 1e3 / self.resolution) ** 3
        field /= norm_factor
        return field

    def load_velocity_field(self):
        rho = self._load_original_density()
        with h5py.File(self._file_path(), "r") as handle:
            vx = handle["p0"][:] / rho
            vy = handle["p1"][:] / rho
            vz = handle["p2"][:] / rho
        return np.stack([vx, vy, vz], axis=0)


class Carrick2015Loader(FieldLoader):
    """
    Loader for Carrick 2015 velocity field data.

    Parameters
    ----------
    velocity_field_path : str or Path
        Path to the .npy file containing the velocity field.
    beta : float, optional
        The beta parameter to scale the velocity field. Default is 0.43.
    """

    def __init__(self, velocity_field_path, beta=0.43):
        self.velocity_field_path = Path(velocity_field_path)
        velocity_field = np.load(self.velocity_field_path)
        self._velocity_field = velocity_field
        resolution = velocity_field.shape[1]
        super().__init__(
            boxsize=400,
            coordinate_system="galactic",
            resolution=resolution,
        )
        self.beta = float(beta)

    def load_density_field(self):
        raise NotImplementedError(
            "Density field not available for Carrick 2015 data."
        )

    def load_velocity_field(self):
        return self.beta * self._velocity_field
