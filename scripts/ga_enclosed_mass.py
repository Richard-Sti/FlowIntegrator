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
"""Compute GA enclosed mass profiles from GA centroids."""

import flowi
import numpy as np
from h5py import File

from config import data_root, results_root


def main():
    ga_file = results_root / "GA_analysis.hdf5"
    output_file = results_root / "GA_enclosed_mass.hdf5"

    sigma_target = 0.0
    radii = np.linspace(5.0, 50.0, 100)  # Mpc / h
    n_rand = 1000
    rng_seed = 42

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with File(ga_file, "r") as gaf, File(output_file, "w") as out:
        out.attrs["sigma_target"] = sigma_target
        out.create_dataset("radii", data=radii)

        fields = [k for k in gaf.keys() if k.startswith("field_")]

        for field_name in fields:
            field_id = int(field_name.split("_")[1])
            field_ga = gaf[field_name]

            sigma_key = f"sigma_{sigma_target}"
            if sigma_key not in field_ga:
                continue
            if not field_ga[sigma_key].attrs.get("matched", False):
                continue

            centroid = field_ga[sigma_key]["centroid"][()]

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
                density, centroid, radii
            )
            m_rand = integrator.integrated_density_random_points(
                density, radii, num_points=n_rand, seed=rng_seed
            )

            grp = out.create_group(field_name)
            grp.attrs["sigma"] = sigma_target
            grp.attrs["centroid_x"] = float(centroid[0])
            grp.attrs["centroid_y"] = float(centroid[1])
            grp.attrs["centroid_z"] = float(centroid[2])
            grp.create_dataset("mass_enclosed", data=m_enclosed)
            grp.create_dataset("mass_random", data=m_rand)


if __name__ == "__main__":
    main()
