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
"""
Compute velocity sourced by GA basin voxels for the observer at box center.
"""

import numpy as np
from h5py import File
from tqdm import trange

import flowi
from config import data_root, results_root


def load_ga_voxel_indices(cluster_file, ga_file, sigma, field_id):
    """
    Load GA member voxel indices for a specific field.

    Parameters
    ----------
    cluster_file : str or Path
        Path to manticore_voxel_clusters.hdf5.
    ga_file : str or Path
        Path to GA_analysis.hdf5.
    sigma : float
        Smoothing scale.
    field_id : int
        Field index.

    Returns
    -------
    members : ndarray or None
        Flat indices of GA member voxels, or None if not matched.
    """
    key = f"sigma_{sigma}"
    fname = f"field_{field_id}"

    with File(cluster_file, "r") as clf, File(ga_file, "r") as gaf:
        if fname not in gaf or fname not in clf:
            return None

        g_ga = gaf[fname]
        g_cl = clf[fname]

        if key not in g_ga or key not in g_cl:
            return None

        sg = g_ga[key]
        if not sg.attrs.get("matched", False):
            return None

        ga_idx = int(sg.attrs["index"])
        members_ds = g_cl[key]["members"]

        if ga_idx >= members_ds.shape[0]:
            return None

        members = members_ds[ga_idx]
        if members.size == 0:
            return None

        return members


def create_ga_mask_from_indices(members, resolution, max_distance, box_size):
    """
    Create a 3D binary mask from GA member flat indices.

    Parameters
    ----------
    members : ndarray
        Flat indices into the initial positions grid.
    resolution : int
        Grid resolution.
    max_distance : float
        Maximum distance used when creating initial positions.
    box_size : float
        Box size in Mpc/h.

    Returns
    -------
    mask : ndarray, shape (resolution, resolution, resolution)
        Binary mask where 1 = GA voxel.
    """
    observer = np.full(3, box_size / 2.0)
    x0_full = np.asarray(flowi.create_initial_positions(
        box_size, resolution, N=None, observer_location=observer,
        max_distance=max_distance, verbose=False))

    ga_positions = x0_full[members]

    mask = np.zeros((resolution, resolution, resolution), dtype=float)
    voxel_size = box_size / resolution
    ga_indices = np.floor(ga_positions / voxel_size).astype(int) % resolution

    for idx in ga_indices:
        mask[idx[0], idx[1], idx[2]] = 1.0

    return mask


def compute_velocities(sigma, Omega_m, h, a, smooth_scale, output_file):
    """Compute GA-sourced and full-field velocities."""
    cluster_file = results_root / "manticore_voxel_clusters.hdf5"
    ga_file = results_root / "GA_analysis.hdf5"

    # Get max_distance from cluster file
    with File(cluster_file, "r") as f:
        max_distance = float(f.attrs.get("max_distance", 150.0))

    # Find all matched fields
    with File(ga_file, "r") as f:
        field_ids = []
        for name in sorted(k for k in f if k.startswith("field_")):
            key = f"sigma_{sigma}"
            if key in f[name] and f[name][key].attrs.get("matched", False):
                field_ids.append(int(name.split("_")[1]))

    print(f"Found {len(field_ids)} matched fields for sigma={sigma}")

    v_ga_all = []
    v_full_all = []

    for field_id in (pbar := trange(len(field_ids), desc="Processing fields")):
        fid = field_ids[field_id]
        pbar.set_postfix(field=fid)

        # Load GA voxel indices
        members = load_ga_voxel_indices(cluster_file, ga_file, sigma, fid)
        if members is None:
            continue

        # Load density field
        loader = flowi.ManticoreLoader(data_root, fid)
        rho = loader.load_density_field()
        rho = flowi.smooth_scalar_field_gaussian(rho, loader.boxsize,
                                                 smooth_scale)
        delta = rho / rho.mean() - 1.0

        # Create GA mask
        mask = create_ga_mask_from_indices(members, loader.resolution,
                                           max_distance, loader.boxsize)

        # Compute velocity from full field
        v_full = flowi.delta_to_velocity(delta, loader.boxsize, Omega_m, h, a)
        center_idx = loader.resolution // 2
        v_obs_full = v_full[:, center_idx, center_idx, center_idx]

        # Compute velocity from GA-masked field only
        delta_ga = delta * mask
        v_ga = flowi.delta_to_velocity(delta_ga, loader.boxsize, Omega_m, h, a)
        v_obs_ga = v_ga[:, center_idx, center_idx, center_idx]

        v_ga_all.append(v_obs_ga)
        v_full_all.append(v_obs_full)

    v_ga_all = np.array(v_ga_all)
    v_full_all = np.array(v_full_all)

    np.savez(output_file,
             v_ga=v_ga_all,
             v_full=v_full_all,
             field_ids=np.array(field_ids),
             sigma=sigma,
             Omega_m=Omega_m,
             smooth_scale=smooth_scale)

    print(f"\nSaved results to {output_file}")
    return v_ga_all, v_full_all, np.array(field_ids)


def plot_velocity_directions(v_ga, v_full, output_file):
    """Plot velocity directions in Galactic coordinates."""
    import matplotlib.pyplot as plt
    import scienceplots  # noqa

    # Convert ICRS Cartesian velocities to Galactic spherical
    # The function expects positions relative to observer, so velocity vectors
    # can be treated as positions from origin
    origin = np.zeros(3)
    _, ell_ga, b_ga = flowi.cartesian_icrs_to_galactic_spherical(v_ga, origin)
    _, ell_full, b_full = flowi.cartesian_icrs_to_galactic_spherical(
        v_full, origin)

    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        # GA-sourced velocity directions
        ax = axes[0]
        ax.scatter(ell_ga, b_ga, s=5, alpha=0.5)
        ax.set_xlabel(r"$\ell~[^\circ]$")
        ax.set_ylabel(r"$b~[^\circ]$")
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_title("GA-sourced")

        # Full field velocity directions
        ax = axes[1]
        ax.scatter(ell_full, b_full, s=5, alpha=0.5)
        ax.set_xlabel(r"$\ell~[^\circ]$")
        ax.set_ylabel(r"$b~[^\circ]$")
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_title("Full field")

        fig.tight_layout()
        fig.savefig(output_file, dpi=300)
        plt.close(fig)

    print(f"Saved direction plot to {output_file}")

    # Print median directions
    print(f"\nGA-sourced velocity direction:")
    print(f"  Median (l, b): ({np.median(ell_ga):.1f}, {np.median(b_ga):.1f})")
    print(f"  Std (l, b): ({np.std(ell_ga):.1f}, {np.std(b_ga):.1f})")

    print(f"\nFull field velocity direction:")
    print(f"  Median (l, b): ({np.median(ell_full):.1f}, "
          f"{np.median(b_full):.1f})")
    print(f"  Std (l, b): ({np.std(ell_full):.1f}, {np.std(b_full):.1f})")


def main():
    sigma = 3.0
    Omega_m = 0.306
    h = 1.0
    a = 1.0
    smooth_scale = 2.5  # Mpc/h

    output_file = results_root / "ga_sourced_velocity.npz"
    plot_file = results_root / "ga_sourced_velocity_directions.png"

    # Load cached results if available, otherwise compute
    if output_file.exists():
        print(f"Loading cached results from {output_file}")
        data = np.load(output_file)
        v_ga_all = data["v_ga"]
        v_full_all = data["v_full"]
        field_ids = data["field_ids"]
        print(f"Loaded {len(field_ids)} fields")
    else:
        v_ga_all, v_full_all, field_ids = compute_velocities(
            sigma, Omega_m, h, a, smooth_scale, output_file)

    print(f"Shape: v_ga={v_ga_all.shape}, v_full={v_full_all.shape}")

    # Print summary statistics
    v_ga_mag = np.linalg.norm(v_ga_all, axis=1)
    v_full_mag = np.linalg.norm(v_full_all, axis=1)

    print("\nGA-sourced velocity magnitude:")
    print(f"  Mean: {v_ga_mag.mean():.1f} km/s")
    print(f"  Std:  {v_ga_mag.std():.1f} km/s")

    print("\nFull field velocity magnitude:")
    print(f"  Mean: {v_full_mag.mean():.1f} km/s")
    print(f"  Std:  {v_full_mag.std():.1f} km/s")

    print("\nRatio |v_GA|/|v_full|:")
    ratio = v_ga_mag / v_full_mag
    print(f"  Mean: {ratio.mean():.3f}")
    print(f"  Std:  {ratio.std():.3f}")

    # Plot velocity directions
    plot_velocity_directions(v_ga_all, v_full_all, plot_file)


if __name__ == "__main__":
    main()
