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
"""Minimal GA projection script."""

from shutil import rmtree

import flowi
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa
from h5py import File

from config import data_root, results_root


def matched_centers(ga_file, sigma):
    centers = []
    box_sizes = []
    field_ids = []
    with File(ga_file, "r") as h5f:
        for name in sorted(k for k in h5f if k.startswith("field_")):
            grp = h5f[name]
            key = f"sigma_{sigma}"
            if key not in grp:
                continue
            sg = grp[key]
            if not sg.attrs.get("matched", False):
                continue
            centers.append(sg["centroid"][()])
            box_sizes.append(float(grp.attrs["box_size"]))
            field_ids.append(int(name.split("_")[1]))
    if not centers:
        raise RuntimeError(f"No matched GA entries for sigma={sigma}")
    centers = np.vstack(centers)
    box_size = float(np.median(box_sizes))
    center = np.median(centers, axis=0)
    print(f"Found {centers.shape[0]} matched centroids; median={center}")
    return center, field_ids, box_size


def cube_indices(center, half_width, box_size, resolution):
    voxel = box_size / resolution
    half_n = int(np.ceil(half_width / voxel))
    ctr = np.mod((center / box_size * resolution).astype(int), resolution)
    rng = np.arange(-half_n, half_n + 1)
    return (ctr[:, None] + rng[None, :]) % resolution


def extract_cube(density, center, half_width, box_size):
    res = density.shape[0]
    idx = cube_indices(center, half_width, box_size, res)
    return density[np.ix_(idx[0], idx[1], idx[2])]


def project_cube(cube):
    # Integrate over x, y, z respectively
    return (
        cube.mean(axis=0),  # y-z
        cube.mean(axis=1),  # x-z
        cube.mean(axis=2),  # x-y
    )


def save_projection(proj, labels, outfile, half_width):
    extent = (-half_width, half_width, -half_width, half_width)
    with plt.style.context("science"):
        fig, ax = plt.subplots()
        im = ax.imshow(
            proj.T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        xlab = fr"$\mathrm{{{labels[0]}}} ~ [h^{{-1}}\,\mathrm{{Mpc}}]$"
        ylab = fr"$\mathrm{{{labels[1]}}} ~ [h^{{-1}}\,\mathrm{{Mpc}}]$"
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.0)
        cbar.set_label(r"$\rho\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$")
        fig.tight_layout()
        fig.savefig(outfile, dpi=450)
        plt.close(fig)


def main():
    center_sigma = 4.0
    plot_sigma = 1.0
    half_width = 50.0
    out_dir = results_root / "GA_plots"

    ga_file = results_root / "GA_analysis.hdf5"
    if out_dir.exists():
        print(f"Cleaning output directory {out_dir}")
        rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    center, field_ids, box_size = matched_centers(ga_file, center_sigma)
    cube_sum = None
    n_used = 0

    for fid in field_ids:
        density = flowi.ManticoreLoader(data_root, fid).load_density_field()
        if plot_sigma > 0.0:
            density = flowi.smooth_scalar_field_gaussian(
                density, box_size, plot_sigma
            )
        cube = extract_cube(density, center, half_width, box_size)
        if cube_sum is None:
            cube_sum = np.zeros_like(cube, dtype=float)
        cube_sum += cube
        n_used += 1

    cube_mean = cube_sum / n_used
    proj_yz, proj_xz, proj_xy = project_cube(cube_mean)

    base = out_dir / (
        f"GA_projection_center{center_sigma:.1f}_plot{plot_sigma:.1f}"
    )
    save_projection(
        proj_yz, ("SGY", "SGZ"),
        base.with_name(f"{base.name}_yz.png"), half_width
    )
    save_projection(
        proj_xz, ("SGX", "SGZ"),
        base.with_name(f"{base.name}_xz.png"), half_width
    )
    save_projection(
        proj_xy, ("SGX", "SGY"),
        base.with_name(f"{base.name}_xy.png"), half_width
    )
    print(f"Used {n_used} realizations.")
    print(f"Median centroid (Mpc/h): {center}")
    print(f"Wrote projections to {out_dir}")


if __name__ == "__main__":
    main()
