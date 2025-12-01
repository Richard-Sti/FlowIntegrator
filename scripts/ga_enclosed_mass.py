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
"""Compute GA enclosed mass profiles using a fixed GA position."""

import flowi
import numpy as np
from h5py import File

from config import data_root, results_root


def main():
    output_file = results_root / "Norma_enclosed_mass.hdf5"

    sigma_target = 0.0
    radii = np.linspace(5.0, 100.0, 100)  # Mpc / h
    n_rand = 11
    rng_seed = 42
    # ga_center = np.array([310.20243187, 327.82457008, 317.06044731])  # GA
    ga_center = np.array([328.77752601, 318.06299589, 296.20206674])    # Norma
    # ga_center = np.array([315.55712222, 334.76738225, 318.1171954])   # Cent.
    n_fields = 80  # number of fields to process
    n_rand_fields = 1  # only compute random profiles for first N fields

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with File(output_file, "w") as out:
        out.attrs["sigma_target"] = sigma_target
        out.attrs["ga_center"] = ga_center
        out.create_dataset("radii", data=radii)

        for field_id in range(n_fields):
            loader = flowi.ManticoreLoader(data_root, field_id)
            density = loader.load_density_field()

            if sigma_target > 0.0:
                density = flowi.smooth_scalar_field_gaussian(
                    density, loader.boxsize, sigma_target
                )

            integrator = flowi.SphericalIntegrator(
                loader.boxsize, loader.resolution
            )
            m_enclosed = integrator.integrated_density_single(
                density, ga_center, radii
            )
            m_rand = None
            if field_id < n_rand_fields:
                m_rand = integrator.integrated_density_random_points(
                    density, radii, num_points=n_rand, seed=rng_seed
                )

            grp = out.create_group(f"field_{field_id}")
            grp.attrs["sigma"] = sigma_target
            grp.attrs["centroid_x"] = float(ga_center[0])
            grp.attrs["centroid_y"] = float(ga_center[1])
            grp.attrs["centroid_z"] = float(ga_center[2])
            grp.create_dataset("mass_enclosed", data=m_enclosed)
            if m_rand is not None:
                grp.create_dataset("mass_random", data=m_rand)

    print(f"Wrote enclosed masses to {output_file}")


if __name__ == "__main__":
    main()
