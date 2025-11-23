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
"""Analyse Great Attractor volume from voxel clustering outputs."""

from pathlib import Path

import flowi
import numpy as np
from h5py import File


def enforce_full_grid(h5_handle):
    """
    Ensure the source run used a full particle grid
    (ngrid_particles=None).
    """
    if "ngrid_particles" not in h5_handle.attrs:
        raise RuntimeError(
            "Missing ngrid_particles attr; rerun volume_streamlines with "
            "ngrid_particles=None."
        )
    ngp = h5_handle.attrs["ngrid_particles"]
    try:
        ngp_val = float(ngp)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ngrid_particles attr not numeric.") from exc
    if ngp_val > 0:
        raise RuntimeError(
            "Expected ngrid_particles=None; rerun volume_streamlines with "
            "ngrid_particles=None so the attr is <= 0."
        )


def load_manticore_density(base_folder, simulation_number):
    """Load Manticore density field, box size, and resolution."""
    loader = flowi.ManticoreLoader(base_folder, simulation_number)
    return loader.load_density_field(), loader.boxsize, loader.resolution


def list_fields(h5_handle):
    """List field indices present in the HDF5 file."""
    return sorted(
        (int(k.split("_")[1]) for k in h5_handle.keys()
         if k.startswith("field_"))
    )


def load_sigma_group(h5_handle, field_id, sigma, result_file):
    """Fetch a sigma subgroup for a given field."""
    field_name = f"field_{field_id}"
    if field_name not in h5_handle:
        raise KeyError(f"{field_name} not found in {result_file}.")
    field_grp = h5_handle[field_name]
    sigma_name = f"sigma_{sigma}"
    if sigma_name not in field_grp:
        raise KeyError(f"{sigma_name} not found under {field_name}.")
    return field_grp, field_grp[sigma_name]


def pick_ga_index(centroids, target, tol):
    """Pick attractor index closest to GA within tolerance."""
    if centroids.size == 0:
        return None, None
    dists = np.linalg.norm(centroids - target, axis=1)
    within = np.where(dists <= tol)[0]
    if within.size == 0:
        return None, None
    best = within[np.argmin(dists[within])]
    return best, dists[best]


def ga_mass_from_members(members, density, voxel_volume_kpc3):
    """
    Compute GA mass (Msun/h) by summing density over member voxels.

    Density is assumed in h^2 Msun / kpc^3; volume_kpc3 carries the
    kpc^3 / h^3 factor, yielding Msun / h.
    """
    res = density.shape[0]
    flat_idx = np.unique(members)
    ix = flat_idx // (res * res)
    rem = flat_idx % (res * res)
    iy = rem // res
    iz = rem % res
    return float(np.sum(density[ix, iy, iz] * voxel_volume_kpc3))


def analyse_sigma(
    sigma_grp, ga_pos, ga_tolerance, voxel_volume_mpc3, density
):
    """Analyse GA volume and mass for a given sigma group."""
    centroids = sigma_grp["centroids"][()]
    counts = sigma_grp["counts"][()]
    members = sigma_grp["members"][()]

    idx, dist = pick_ga_index(centroids, ga_pos, ga_tolerance)
    if idx is None:
        return {"matched": False}

    # Members correspond to particle indices; with full grid it is one per
    # voxel
    n_voxels_ga = len(np.unique(members[idx]))
    ga_volume = n_voxels_ga * voxel_volume_mpc3
    voxel_volume_kpc3 = voxel_volume_mpc3 * 1.0e9
    ga_mass = ga_mass_from_members(
        members[idx], density, voxel_volume_kpc3
    )

    return {
        "matched": True,
        "index": int(idx),
        "distance": float(dist),
        "centroid": centroids[idx],
        "member_count": int(counts[idx]),
        "voxel_count": int(n_voxels_ga),
        "voxel_volume": float(voxel_volume_mpc3),
        "voxel_volume_kpc3": float(voxel_volume_kpc3),
        "ga_volume": float(ga_volume),
        "ga_mass": ga_mass,
    }


def main():
    """Loop over all fields/smoothing scales and save GA metrics."""
    result_file = Path("../results/manticore_voxel_clusters.hdf5")
    output_file = Path("../results/GA_analysis.hdf5")
    data_root = Path("/Users/rstiskalek/Data/Manticore/N256")
    ga_position = np.array([340.0, 340.0, 340.0])  # Mpc / h
    ga_tolerance = 10.0  # Mpc / h

    with File(result_file, "r") as src, File(output_file, "w") as dst:
        enforce_full_grid(src)

        smoothing_scales = np.asarray(
            src.attrs["smoothing_scales"], dtype=float
        )
        fields = list_fields(src)

        dst.attrs["source"] = str(result_file)
        dst.attrs["ga_position"] = ga_position
        dst.attrs["ga_tolerance"] = ga_tolerance
        dst.attrs["smoothing_scales"] = smoothing_scales

        for field_id in fields:
            field_name = f"field_{field_id}"
            if field_name not in src:
                continue
            field_grp_src = src[field_name]
            box_size = float(field_grp_src.attrs["box_size"])
            resolution = int(field_grp_src.attrs["resolution"])
            voxel_volume_mpc3 = (box_size / resolution) ** 3
            voxel_volume_kpc3 = voxel_volume_mpc3 * 1.0e9

            field_out = dst.create_group(field_name)
            field_out.attrs["box_size"] = box_size
            field_out.attrs["resolution"] = resolution
            field_out.attrs["voxel_volume"] = voxel_volume_mpc3
            field_out.attrs["voxel_volume_kpc3"] = voxel_volume_kpc3
            field_out.attrs["simulation_number"] = field_id

            density_field, _, _ = load_manticore_density(
                data_root, field_id
            )

            for sigma in smoothing_scales:
                try:
                    _, sigma_grp_src = load_sigma_group(
                        src, field_id, sigma, result_file
                    )
                except KeyError:
                    sigma_out = field_out.create_group(f"sigma_{sigma}")
                    sigma_out.attrs["matched"] = False
                    sigma_out.attrs["missing"] = True
                    continue

                if sigma > 0.0:
                    density_sigma = flowi.smooth_scalar_field_gaussian(
                        density_field, box_size, sigma
                    )
                else:
                    density_sigma = density_field

                res = analyse_sigma(
                    sigma_grp_src,
                    ga_pos=ga_position,
                    ga_tolerance=ga_tolerance,
                    voxel_volume_mpc3=voxel_volume_mpc3,
                    density=density_sigma
                )

                sigma_out = field_out.create_group(f"sigma_{sigma}")
                sigma_out.attrs["matched"] = res["matched"]
                sigma_out.attrs["sigma"] = sigma
                if not res["matched"]:
                    continue

                sigma_out.attrs["index"] = res["index"]
                sigma_out.attrs["distance"] = res["distance"]
                sigma_out.attrs["member_count"] = res["member_count"]
                sigma_out.attrs["voxel_count"] = res["voxel_count"]
                sigma_out.attrs["voxel_volume"] = res["voxel_volume"]
                sigma_out.attrs["voxel_volume_kpc3"] = res["voxel_volume_kpc3"]
                sigma_out.attrs["ga_volume"] = res["ga_volume"]
                sigma_out.attrs["ga_mass"] = res["ga_mass"]
                sigma_out.create_dataset("centroid", data=res["centroid"])

                print(
                    f"field {field_id}, sigma {sigma}: "
                    f"GA volume {res['ga_volume']:.6e}, "
                    f"mass {res['ga_mass']:.6e} (matched)"
                )


if __name__ == "__main__":
    main()
