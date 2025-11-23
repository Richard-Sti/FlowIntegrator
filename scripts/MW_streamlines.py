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
"""Generate Milky Way streamlines across Manticore velocity fields."""

import flowi
import h5py
import jax
import numpy as np
from jax import numpy as jnp
from tqdm import trange

from config import data_root, results_root


def load_manticore_velocity(base_folder, simulation_number):
    loader = flowi.ManticoreLoader(base_folder, simulation_number)
    return loader.load_velocity_field(), loader.boxsize


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    n_fields = 80
    num_steps = 20_000
    ds_factor = 0.05
    smoothing_scales = np.arange(0, 17)  # Mpc/h
    output = results_root / "MW_streamlines.hdf5"

    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as h5f:
        h5f.attrs["ds_factor"] = ds_factor
        h5f.attrs["num_steps"] = num_steps
        h5f.attrs["smoothing_scales"] = smoothing_scales

        for i in trange(n_fields, desc="Fields"):
            velocity_field, box_size = load_manticore_velocity(data_root, i)

            resolution = velocity_field.shape[-1]
            ds = ds_factor * box_size / resolution

            intg = flowi.TrajectoryFollower(
                velocity_field,
                box_size,
                num_steps=num_steps,
                ds=ds
            )
            x0 = jnp.full((3,), box_size / 2)

            results = intg.follow_multiple_smoothing(
                x0, smoothing_scales=smoothing_scales, verbose=False
            )
            field_grp = h5f.create_group(f"field_{i}")
            for sigma, res in zip(smoothing_scales, results):
                t_s, x_s, v_s = (np.array(res[0]), np.array(res[1]),
                                 np.array(res[2]))
                s_grp = field_grp.create_group(f"sigma_{sigma}")
                s_grp.attrs["sigma"] = sigma
                s_grp.attrs["box_size"] = box_size
                s_grp.attrs["grid_resolution"] = resolution
                s_grp.attrs["ds"] = ds
                s_grp.create_dataset("time", data=t_s, compression="gzip")
                s_grp.create_dataset("trajectory", data=x_s,
                                     compression="gzip")
                s_grp.create_dataset("speed", data=v_s, compression="gzip")
    print(f"Saved streamlines to {output.resolve()}")


if __name__ == "__main__":
    main()
