#!/usr/bin/env python
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
"""Precompute heavy GA analysis data: density fields and GA masks."""

from functools import lru_cache

import flowi
import numpy as np
from h5py import File
from scipy.stats import gaussian_kde
from tqdm import tqdm

from config import data_root, results_root


def contour_level_for_fraction(arr, frac=0.95):
    """Value threshold enclosing given fraction of total counts."""
    flat = arr.ravel()
    total = flat.sum()
    if total <= 0:
        return None
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    idx = np.searchsorted(csum, frac * total, side="left")
    if idx >= flat.size:
        return None
    return flat[order[idx]]


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


def _observer_key(observer):
    obs = np.asarray(observer, dtype=float)
    return tuple(obs.tolist())


@lru_cache(maxsize=None)
def initial_positions_grid(box_size, resolution, observer_key, max_distance):
    observer = np.asarray(observer_key, dtype=float)
    grid = np.asarray(
        flowi.create_initial_positions(
            box_size,
            resolution,
            N=None,
            observer_location=observer,
            max_distance=max_distance,
            verbose=False,
        )
    )
    grid.setflags(write=False)
    return grid


def create_ga_mask(ga_positions, box_size, resolution):
    """
    Create a 3D binary mask of GA member voxels.

    Parameters
    ----------
    ga_positions : ndarray
        GA member positions (n, 3) in ICRS.
    box_size : float
        Box size.
    resolution : int
        Grid resolution.
    observer : array-like
        Observer position.
    max_distance : float
        Maximum distance for creating initial positions grid.

    Returns
    -------
    ndarray
        3D binary mask where 1 = GA member, 0 = not GA.
    """
    # ga_positions already reference the grid used upstream.

    # Create binary mask
    mask = np.zeros(resolution**3, dtype=float)

    # Find which grid points match GA positions; accumulate counts
    voxel_size = box_size / resolution
    ga_indices = (np.floor(ga_positions / voxel_size).astype(int)
                  % resolution)
    ga_flat = (ga_indices[:, 0] * resolution**2 +
               ga_indices[:, 1] * resolution +
               ga_indices[:, 2])

    np.add.at(mask, ga_flat, 1.0)
    return mask.reshape((resolution, resolution, resolution))


def main():
    # USER PARAMETERS
    center_sigma = 4.0
    plot_sigma = 2
    half_width = 90  # Mpc / h
    contour_downsample = 25
    contour_frac = 0.99
    kde_grid = 50
    nside_map = 128

    ga_file = results_root / "GA_analysis.hdf5"
    cluster_file = results_root / "manticore_voxel_clusters.hdf5"
    fname = (f"GA_precomputed_center{center_sigma:.1f}_"
             f"plot{plot_sigma:.1f}.hdf5")
    output_file = results_root / fname

    print(f"Precomputing GA data for center_sigma={center_sigma}, "
          f"plot_sigma={plot_sigma}")
    print(f"Output will be saved to: {output_file}")

    # Get matched centers and field IDs
    center, field_ids, box_size = matched_centers(ga_file, center_sigma)
    box_center = np.full(3, box_size / 2.0)
    print(f"Center: {center}")
    print(f"Box size: {box_size}")
    print(f"Number of fields: {len(field_ids)}")

    resolution = None
    density_mean = None
    ga_positions_all = []
    ga_masks_data = []
    sigma_key = f"sigma_{center_sigma}"

    with File(ga_file, "r") as ga_f, File(cluster_file, "r") as cl_f:
        max_distance = float(cl_f.attrs.get("max_distance", 0))
        print(f"Max distance: {max_distance}")

        # First pass: collect GA positions
        print("First pass: collecting GA member positions...")
        for fid in tqdm(field_ids, desc="Collecting positions"):
            fkey = f"field_{fid}"
            has_data = (fkey in ga_f and fkey in cl_f and
                        sigma_key in ga_f[fkey] and
                        sigma_key in cl_f[fkey])
            if has_data:
                if resolution is None:
                    resolution = int(ga_f[fkey].attrs["resolution"])
                    print(f"Grid resolution: {resolution}")

                ga_idx = int(ga_f[fkey][sigma_key].attrs["index"])
                members_ds = cl_f[fkey][sigma_key]["members"]
                if ga_idx < members_ds.shape[0]:
                    members = members_ds[ga_idx]
                    if members.size > 0:
                        max_d = max_distance if max_distance > 0 else None
                        x0 = initial_positions_grid(
                            box_size,
                            resolution,
                            _observer_key(box_center),
                            max_d,
                        )
                        ga_pos = np.asarray(x0)[np.asarray(members,
                                                           dtype=int)]
                        ga_positions_all.append(ga_pos)
                        ga_masks_data.append((fid, ga_pos))

    # Compute Rmax_ga from all collected positions
    if ga_positions_all:
        stacked_positions = np.vstack(ga_positions_all)
        distances = np.sqrt(
            ((stacked_positions - box_center) ** 2).sum(axis=1))
        Rmax_ga = np.percentile(distances, 99.99)
        print(f"Rmax_ga (99.99th percentile): {Rmax_ga:.2f} Mpc/h")
    else:
        Rmax_ga = max_distance
        print(f"No GA positions, using max_distance: {Rmax_ga}")

    # Compute KDE contours per field
    print("Computing KDE contours per field...")
    kde_data = []
    grid = np.linspace(-half_width, half_width, kde_grid)
    xg, yg, zg = np.meshgrid(grid, grid, grid, indexing="ij")
    coords = np.vstack([xg.ravel(), yg.ravel(), zg.ravel()])

    for fid, ga_pos in tqdm(ga_masks_data, desc="Computing KDEs"):
        rel_center = ga_pos - center
        rel_center = ((rel_center + box_size / 2) % box_size
                      - box_size / 2)
        mask_pts = np.all(np.abs(rel_center) <= half_width, axis=1)
        rel_center = rel_center[mask_pts]

        if rel_center.shape[0] >= 10:
            points = rel_center
            if (contour_downsample > 1 and
                    points.shape[0] > contour_downsample):
                points = points[::contour_downsample]

            kde = gaussian_kde(points.T)
            dens = kde(coords).reshape(kde_grid, kde_grid, kde_grid)

            kde_yz = dens.sum(axis=0)
            kde_xz = dens.sum(axis=1)
            kde_xy = dens.sum(axis=2)
            lvl_yz = contour_level_for_fraction(kde_yz, frac=contour_frac)
            lvl_xz = contour_level_for_fraction(kde_xz, frac=contour_frac)
            lvl_xy = contour_level_for_fraction(kde_xy, frac=contour_frac)

            kde_data.append((fid, kde_yz, kde_xz, kde_xy,
                            lvl_yz, lvl_xz, lvl_xy))

    print(f"Computed KDE contours for {len(kde_data)} fields")

    # Second pass: load densities and create masks
    print("Second pass: loading densities and creating masks...")
    ga_masks = []
    for fid in tqdm(field_ids, desc="Processing fields"):
        # Load and smooth density field
        density = flowi.ManticoreLoader(
            data_root, fid).load_density_field()
        if plot_sigma > 0.0:
            density = flowi.smooth_scalar_field_gaussian(
                density, box_size, plot_sigma
            )

        # Accumulate mean density
        if density_mean is None:
            density_mean = np.zeros_like(density, dtype=float)
        density_mean += density

        # Create GA mask for this field if we have data
        for mask_fid, ga_pos in ga_masks_data:
            if mask_fid == fid:
                ga_mask_field = create_ga_mask(ga_pos, box_size, resolution)
                ga_masks.append((fid, ga_mask_field))
                break

    # Compute mean density
    n_fields = len(field_ids)
    density_mean /= n_fields
    print(f"Processed {n_fields} fields")
    print(f"Computed {len(ga_masks)} GA masks")

    # Compute sky depth maps for each GA mask
    print("Computing sky depth maps for each GA mask...")
    voxel = box_size / resolution
    ga_sky_maps = []
    for fid, ga_mask_field in tqdm(ga_masks, desc="Computing sky maps"):
        sky_fraction = flowi.utils.grid_ngp_projection(
            nside=nside_map,
            rho=ga_mask_field,
            boxsize=box_size,
            observer=box_center,
            Rmax=Rmax_ga,
            dr=0.1 * voxel,
            Rmin=0,
            coords="icrs->galactic",
            r_power=0,
            verbose=False
        )
        # Convert fraction to depth by multiplying by Rmax
        sky_depth = sky_fraction * Rmax_ga
        ga_sky_maps.append((fid, sky_depth))
    print(f"Computed {len(ga_sky_maps)} sky depth maps")

    # Save to HDF5
    print(f"Saving to {output_file}...")
    with File(output_file, "w") as h5f:
        # Metadata
        h5f.attrs["center_sigma"] = center_sigma
        h5f.attrs["plot_sigma"] = plot_sigma
        h5f.attrs["box_size"] = box_size
        h5f.attrs["resolution"] = resolution
        h5f.attrs["n_fields"] = n_fields
        h5f.attrs["center"] = center
        h5f.attrs["max_distance"] = max_distance
        h5f.attrs["Rmax_ga"] = Rmax_ga
        h5f.attrs["half_width"] = half_width
        h5f.attrs["kde_grid"] = kde_grid
        h5f.attrs["contour_frac"] = contour_frac
        h5f.attrs["nside_map"] = nside_map
        h5f.create_dataset("field_ids", data=field_ids)

        # Mean density field
        h5f.create_dataset(
            "density_mean",
            data=density_mean,
            compression="gzip",
            compression_opts=4
        )

        # Individual GA masks
        masks_grp = h5f.create_group("ga_masks")
        for fid, mask in tqdm(ga_masks, desc="Saving masks"):
            masks_grp.create_dataset(
                f"field_{fid}",
                data=mask,
                compression="gzip",
                compression_opts=4
            )

        # KDE contours per field
        kde_grp = h5f.create_group("kde_contours")
        for (fid, kde_yz, kde_xz, kde_xy, lvl_yz, lvl_xz,
             lvl_xy) in tqdm(kde_data, desc="Saving KDEs"):
            field_grp = kde_grp.create_group(f"field_{fid}")
            field_grp.create_dataset("kde_yz", data=kde_yz,
                                     compression="gzip", compression_opts=4)
            field_grp.create_dataset("kde_xz", data=kde_xz,
                                     compression="gzip", compression_opts=4)
            field_grp.create_dataset("kde_xy", data=kde_xy,
                                     compression="gzip", compression_opts=4)
            if lvl_yz is not None:
                field_grp.attrs["lvl_yz"] = lvl_yz
            if lvl_xz is not None:
                field_grp.attrs["lvl_xz"] = lvl_xz
            if lvl_xy is not None:
                field_grp.attrs["lvl_xy"] = lvl_xy

        # GA sky maps per field
        sky_grp = h5f.create_group("ga_sky_maps")
        sky_grp.attrs["nside"] = nside_map
        for fid, sky_map in tqdm(ga_sky_maps, desc="Saving sky maps"):
            sky_grp.create_dataset(
                f"field_{fid}",
                data=sky_map,
                compression="gzip",
                compression_opts=4
            )

    print(f"Done! Saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
