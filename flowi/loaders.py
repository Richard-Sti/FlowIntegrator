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

    def __init__(self, boxsize):
        self._boxsize = float(boxsize)

    @property
    def boxsize(self):
        """Return the simulation box size."""
        return self._boxsize

    @abstractmethod
    def load_density_field(self):
        """Return the density field as a NumPy array."""

    @abstractmethod
    def load_velocity_field(self):
        """Return the velocity field as a NumPy array."""


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
        super().__init__(boxsize)
        self.base_folder = Path(base_folder)
        self.simulation_number = int(simulation_number)

    def _file_path(self):
        name = f"mcmc_{self.simulation_number}.hdf5"
        return self.base_folder / name

    def load_density_field(self):
        with h5py.File(self._file_path(), "r") as handle:
            density = handle["density"][:]
        return density

    def load_velocity_field(self):
        with h5py.File(self._file_path(), "r") as handle:
            rho = handle["density"][:]
            vx = handle["p0"][:] / rho
            vy = handle["p1"][:] / rho
            vz = handle["p2"][:] / rho
        return np.stack([vx, vy, vz], axis=0)
