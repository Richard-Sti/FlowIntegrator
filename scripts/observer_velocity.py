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
"""Sample observer-frame velocities across Manticore realizations."""

import flowi
import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import trange

from config import data_root, results_root


def load_velocity(sim_number):
    loader = flowi.ManticoreLoader(data_root, sim_number)
    return loader.load_velocity_field(), loader.boxsize, loader.resolution


def velocity_at_positions(v_field, box_size, positions):
    positions = np.asarray(positions, dtype=float)
    resolution = v_field.shape[1]
    coords = (positions / box_size) * resolution - 0.5
    coords = coords.T
    vx = map_coordinates(v_field[0], coords, order=1, mode="wrap")
    vy = map_coordinates(v_field[1], coords, order=1, mode="wrap")
    vz = map_coordinates(v_field[2], coords, order=1, mode="wrap")
    return np.stack([vx, vy, vz], axis=1)


def velocity_at_position(v_field, box_size, position):
    return velocity_at_positions(v_field, box_size, position[None, :])[0]


def main():
    n_fields = 80
    v_ext = np.array([-29.389921, -17.852118, -82.94362],
                     dtype=np.float32)  # km/s, to add to all velocities
    sample_radius = None  # Mpc/h; if set, average velocities within sphere
    n_samples = 100       # points for sphere sampling
    outfile = results_root / "observer_velocities.hdf5"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(outfile, "w") as h5f:
        vel_ds = h5f.create_dataset(
            "velocity", shape=(n_fields, 3), dtype=np.float32
        )
        speed_ds = h5f.create_dataset(
            "speed", shape=(n_fields,), dtype=np.float32
        )
        h5f.attrs["description"] = (
            "Velocity at observer (box center) for all Manticore "
            "realizations."
        )

        rows = []
        box_size = None
        resolution = None
        for sim in trange(n_fields, desc="Realizations"):
            v_field, box_size, resolution = load_velocity(sim)
            obs_pos = np.full(3, box_size / 2.0, dtype=np.float32)
            if sample_radius is None:
                v_obs = velocity_at_position(
                    v_field, box_size, obs_pos
                )
            else:
                directions = np.random.normal(size=(n_samples, 3))
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                radii = sample_radius * np.cbrt(np.random.random(n_samples))
                pts = obs_pos + radii[:, None] * directions
                pts = np.mod(pts, box_size)
                v_obs = velocity_at_positions(v_field, box_size, pts).mean(
                    axis=0
                )
            v_obs = v_obs + v_ext

            vel_ds[sim] = v_obs
            speed = np.linalg.norm(v_obs)
            speed_ds[sim] = speed

            if speed > 0:
                _, ell, b = flowi.cartesian_icrs_to_galactic_spherical(
                    v_obs[None, :], np.zeros(3, dtype=np.float32)
                )
                ell_val, b_val = ell[0], b[0]
            else:
                ell_val = b_val = np.nan

            rows.append((sim, speed, ell_val, b_val))

        h5f.attrs["box_size"] = box_size
        h5f.attrs["resolution"] = resolution
        h5f.attrs["v_ext"] = v_ext
        h5f.attrs["sample_radius"] = (
            np.nan if sample_radius is None else float(sample_radius)
        )
        h5f.attrs["n_samples"] = n_samples

    if rows:
        print(f"{'Realization':>12} {'|v| [km/s]':>12} "
              f"{'l [deg]':>10} {'b [deg]':>10}")
        print("-" * 48)
        for sim, speed, ell_val, b_val in rows:
            print(f"{sim:12d} {speed:12.3f} {ell_val:10.2f} {b_val:10.2f}")

        speeds = np.array([r[1] for r in rows], dtype=float)
        ell = np.array([r[2] for r in rows], dtype=float)
        b = np.array([r[3] for r in rows], dtype=float)
        valid = np.isfinite(speeds) & np.isfinite(ell) & np.isfinite(b)

        if valid.any():
            speeds = speeds[valid]
            ell = ell[valid]
            b = b[valid]

            print("-" * 48)
            print(f"{'Mean |v| [km/s]':>24}: {speeds.mean():8.3f} "
                  f"± {np.std(speeds):6.3f}")
            print(f"{'Mean l [deg]':>24}: {ell.mean():8.3f} "
                  f"± {np.std(ell):6.3f}")
            print(f"{'Mean b [deg]':>24}: {b.mean():8.3f} "
                  f"± {np.std(b):6.3f}")

    print(f"Wrote observer velocities to {outfile.resolve()}")


if __name__ == "__main__":
    main()
