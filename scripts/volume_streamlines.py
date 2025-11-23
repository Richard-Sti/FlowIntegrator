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
"""Cluster converged streamlines in Manticore velocity fields."""

import flowi
import jax
import numpy as np
from flowi import fprint
from h5py import File, vlen_dtype
from jax import numpy as jnp
from tqdm import trange

from config import data_root, results_root


def load_manticore_velocity(base_folder, simulation_number):
    loader = flowi.ManticoreLoader(base_folder, simulation_number)
    return loader.load_velocity_field(), loader.boxsize, loader.resolution


def write_sigma_group(field_group, sigma, metadata, attractors):
    name = f"sigma_{sigma}"
    if name in field_group:
        raise ValueError(f"Group {name} already exists in output file.")

    grp = field_group.create_group(name)
    grp.attrs["sigma"] = sigma
    for key, value in metadata.items():
        grp.attrs[key] = value

    if len(attractors) == 0:
        grp.create_dataset("centroids", data=np.empty((0, 3)))
        grp.create_dataset("counts", data=np.empty((0,), dtype=np.int64))
        grp.create_dataset(
            "members",
            data=np.empty((0,), dtype=vlen_dtype(np.int64))
        )
        return

    centroids = np.stack([np.asarray(a.centroid) for a in attractors])
    counts = np.array([int(a.count) for a in attractors], dtype=np.int64)
    members = [np.asarray(a.members, dtype=np.int64) for a in attractors]

    grp.create_dataset("centroids", data=centroids, compression="gzip")
    grp.create_dataset("counts", data=counts, compression="gzip")
    grp.create_dataset(
        "members",
        data=np.array(members, dtype=vlen_dtype(np.int64)),
        compression="gzip"
    )


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    output = results_root / "manticore_voxel_clusters.hdf5"

    n_fields = 80
    num_steps = 25_000
    ds_factor = 0.05
    ngrid_particles = None
    max_distance = 200  # Mpc / h
    min_cluster_fraction = 1e-4
    smoothing_scales = [0.0, 2.0, 4.0, 16.0]  # Mpc / h

    output.parent.mkdir(parents=True, exist_ok=True)

    ngrid_attr = -1 if ngrid_particles is None else int(ngrid_particles)

    run_metadata = {
        "num_steps": num_steps,
        "ds_factor": ds_factor,
        "ngrid_particles": ngrid_attr,
        "max_distance": max_distance,
        "min_cluster_fraction": min_cluster_fraction,
        "smoothing_scales": np.asarray(smoothing_scales, dtype=np.float32),
    }

    with File(output, "w") as h5f:
        for key, value in run_metadata.items():
            if isinstance(value, np.ndarray):
                h5f.attrs.create(key, data=value)
            else:
                h5f.attrs[key] = value

    for sim in trange(n_fields, desc="Fields"):
        velocity_field, box_size, resolution = load_manticore_velocity(
            data_root, sim
        )
        ds = ds_factor * box_size / resolution
        obs_pos = jnp.full((3,), box_size / 2)

        x0 = flowi.create_initial_positions(
            box_size,
            resolution,
            N=ngrid_particles,
            observer_location=obs_pos,
            max_distance=max_distance
        )

        min_count = max(1, int(np.ceil(min_cluster_fraction * x0.shape[0])))

        with File(output, "a") as h5f:
            field_name = f"field_{sim}"
            if field_name in h5f:
                raise ValueError(f"{field_name} already exists in output.")
            field_grp = h5f.create_group(field_name)
            field_metadata = {
                "simulation_number": sim,
                "box_size": box_size,
                "resolution": resolution,
                "ds": ds,
                "observer_x": float(obs_pos[0]),
                "observer_y": float(obs_pos[1]),
                "observer_z": float(obs_pos[2]),
                "initial_particle_count": int(x0.shape[0]),
                "min_cluster_count": min_count,
            }
            for key, value in field_metadata.items():
                field_grp.attrs[key] = value

            for sigma in smoothing_scales:
                if sigma > 0.0:
                    v_field_sigma = flowi.smooth_velocity_field_gaussian(
                        velocity_field, box_size, sigma
                    )
                else:
                    v_field_sigma = velocity_field

                integrator = flowi.Integrator(
                    v_field_sigma,
                    box_size,
                    num_steps=num_steps,
                    ds=ds
                )

                fprint(f"Integrating streamlines for field {sim}, "
                       f"sigma={sigma}...")
                xf, _, dx_step = integrator.run(x0, verbose=True)

                fprint(f"Starting attractor clustering for field {sim}, "
                       f"sigma={sigma}...")
                attractors = integrator.get_cluster_info_voxel(
                    xf,
                    dx_step,
                    min_count=min_count
                )
                fprint(f"Finished attractor clustering for field {sim}, "
                       "sigma={sigma}. Found {len(attractors)} attractors.")
                kept_attractors = [attractors[idx]
                                   for idx in range(len(attractors))]

                resolution_element = (box_size / resolution) / 2.0
                converged_fraction = float(
                    np.mean(np.asarray(dx_step) < resolution_element)
                )

                fprint(
                    f"Field {sim}, sigma={sigma}: "
                    f"{len(kept_attractors)} clusters kept "
                    f"with min_count={min_count}; "
                    f"{converged_fraction:.2%} particles converged."
                )

                write_sigma_group(
                    field_grp,
                    sigma=sigma,
                    metadata={"converged_fraction": converged_fraction},
                    attractors=kept_attractors
                )


if __name__ == "__main__":
    main()
