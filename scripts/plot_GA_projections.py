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
"""Plot GA projections in supergalactic and Galactic coordinates."""

from pathlib import Path

import flowi
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from h5py import File


def voxel_indices(center, width, box_size, resolution):
    """Return wrapped voxel indices around center for a given width."""
    voxel_size = box_size / resolution
    half = int(np.ceil(0.5 * width / voxel_size))
    ctr_idx = np.mod(
        (center / box_size * resolution).astype(int), resolution
    )
    rng = np.arange(-half, half + 1)
    idx = np.mod(ctr_idx[:, None] + rng[None, :], resolution).astype(int)
    return idx


def project_density(density, box_size, center, width, axis):
    """Project density along one axis within a finite slab."""
    res = density.shape[0]
    idx = voxel_indices(center, width, box_size, res)
    voxel_size = box_size / res
    extent_sym = [
        -0.5 * voxel_size * (idx.shape[1] - 1),
        0.5 * voxel_size * (idx.shape[1] - 1),
        -0.5 * voxel_size * (idx.shape[1] - 1),
        0.5 * voxel_size * (idx.shape[1] - 1),
    ]
    if axis == 0:
        sub = density[idx[0], :, :]
        return sub.sum(axis=0), extent_sym
    elif axis == 1:
        sub = density[:, idx[1], :]
        return sub.sum(axis=1), extent_sym
    elif axis == 2:
        sub = density[:, :, idx[2]]
        return sub.sum(axis=2), extent_sym
    raise ValueError("axis must be 0, 1, or 2")


def members_to_positions(members, box_size, resolution):
    """Convert flat voxel indices to Cartesian positions."""
    idx = np.asarray(members, dtype=int)
    ix = idx // (resolution * resolution)
    rem = idx % (resolution * resolution)
    iy = rem // resolution
    iz = rem % resolution
    voxel_size = box_size / resolution
    pos = np.stack(
        [
            (ix + 0.5) * voxel_size,
            (iy + 0.5) * voxel_size,
            (iz + 0.5) * voxel_size,
        ],
        axis=-1,
    )
    return pos, (ix, iy, iz)


def voxel_masses(ix, iy, iz, density, voxel_volume):
    """Mass per voxel given indices."""
    return density[ix, iy, iz] * voxel_volume


def plot_projection(img, extent, labels, outfile):
    """Save a 2D projection image."""
    with plt.style.context("science"):
        fig, ax = plt.subplots()
        ax.imshow(
            img.T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_xlabel(f"{labels[0]} [Mpc/h]")
        ax.set_ylabel(f"{labels[1]} [Mpc/h]")
        fig.tight_layout()
        fig.savefig(outfile, dpi=200)
        plt.close(fig)


def plot_healpy_mass(l_deg, b_deg, mass, nside, outfile):
    """Save a HEALPix mass map."""
    theta = np.deg2rad(90.0 - b_deg)
    phi = np.deg2rad(l_deg % 360.0)
    pix = hp.ang2pix(nside, theta, phi)
    npix = hp.nside2npix(nside)
    hmap = np.bincount(pix, weights=mass, minlength=npix)
    with plt.style.context("science"):
        hp.mollview(hmap, unit="Msun/h", title="", cbar=True)
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close()
    return hmap


def main():
    ga_file = Path("../results/GA_analysis.hdf5")
    cluster_file = Path("../results/manticore_voxel_clusters.hdf5")
    data_root = Path("/Users/rstiskalek/Data/Manticore/N256")
    out_dir = Path("../results/GA_plots")
    slab_width = 50.0  # Mpc / h
    nside = 64
    sigma_target = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)

    with File(ga_file, "r") as gaf, File(cluster_file, "r") as clf:
        fields = [k for k in gaf.keys() if k.startswith("field_")]
        sum_x = sum_y = sum_z = None
        hmap_sum = None
        n_used = 0

        for field_name in fields:
            field_id = int(field_name.split("_")[1])
            if field_name not in clf:
                continue
            field_ga = gaf[field_name]
            field_cl = clf[field_name]

            box_size = float(field_ga.attrs["box_size"])
            resolution = int(field_ga.attrs["resolution"])
            obs = np.array(
                [
                    field_cl.attrs["observer_x"],
                    field_cl.attrs["observer_y"],
                    field_cl.attrs["observer_z"],
                ],
                dtype=float,
            )

            loader = flowi.ManticoreLoader(data_root, field_id)
            base_density = loader.load_density_field()

            sigma = sigma_target
            sigma_key = f"sigma_{sigma}"
            if sigma_key not in field_ga:
                continue
            if not field_ga[sigma_key].attrs.get("matched", False):
                continue
            if sigma_key not in field_cl:
                continue

            ga_idx = int(field_ga[sigma_key].attrs["index"])
            centroid = field_ga[sigma_key]["centroid"][()]

            if sigma > 0.0:
                density = flowi.smooth_scalar_field_gaussian(
                    base_density, box_size, sigma
                )
            else:
                density = base_density

            img_x, ext_x = project_density(
                density, box_size, centroid, slab_width, axis=0
            )
            img_y, ext_y = project_density(
                density, box_size, centroid, slab_width, axis=1
            )
            img_z, ext_z = project_density(
                density, box_size, centroid, slab_width, axis=2
            )

            base = f"plot_field{field_id}_sigma{sigma}"
            plot_projection(
                img_x,
                ext_x,
                labels=("SGY", "SGZ"),
                outfile=out_dir / f"{base}_sgx.png",
            )
            plot_projection(
                img_y,
                ext_y,
                labels=("SGX", "SGZ"),
                outfile=out_dir / f"{base}_sgy.png",
            )
            plot_projection(
                img_z,
                ext_z,
                labels=("SGX", "SGY"),
                outfile=out_dir / f"{base}_sgz.png",
            )

            if sum_x is None:
                sum_x = np.zeros_like(img_x)
                sum_y = np.zeros_like(img_y)
                sum_z = np.zeros_like(img_z)

            sum_x += img_x
            sum_y += img_y
            sum_z += img_z

            members = field_cl[sigma_key]["members"][ga_idx]
            pos, (ix, iy, iz) = members_to_positions(
                members, box_size, resolution
            )
            voxel_volume_mpc3 = (box_size / resolution) ** 3
            voxel_volume_kpc3 = voxel_volume_mpc3 * 1.0e9
            mass = voxel_masses(ix, iy, iz, density, voxel_volume_kpc3)
            r, l_deg, b_deg = flowi.cartesian_icrs_to_galactic_spherical(
                pos, obs
            )
            _ = r  # unused distance
            hmap = plot_healpy_mass(
                l_deg, b_deg, mass, nside,
                outfile=out_dir / f"{base}_hpmass.png"
            )
            if hmap_sum is None:
                hmap_sum = np.zeros_like(hmap)
            hmap_sum += hmap
            n_used += 1

        if n_used > 0:
            avg_x = sum_x / n_used
            avg_y = sum_y / n_used
            avg_z = sum_z / n_used
            base = f"plot_mean_sigma{sigma_target}"
            plot_projection(
                avg_x,
                ext_x,
                labels=("SGY", "SGZ"),
                outfile=out_dir / f"{base}_sgx.png",
            )
            plot_projection(
                avg_y,
                ext_y,
                labels=("SGX", "SGZ"),
                outfile=out_dir / f"{base}_sgy.png",
            )
            plot_projection(
                avg_z,
                ext_z,
                labels=("SGX", "SGY"),
                outfile=out_dir / f"{base}_sgz.png",
            )
            hmap_avg = hmap_sum / n_used
            with plt.style.context("science"):
                hp.mollview(
                    hmap_avg, unit="Msun/h", title="", cbar=True
                )
                plt.savefig(
                    out_dir / f"{base}_hpmass.png",
                    dpi=200, bbox_inches="tight"
                )
                plt.close()


if __name__ == "__main__":
    main()
