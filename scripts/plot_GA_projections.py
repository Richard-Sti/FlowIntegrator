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
import healpy as hp
from scipy.stats import gaussian_kde
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


def collect_ga_member_positions(cluster_file, ga_file, sigma):
    """
    Stack Cartesian positions of GA-member voxels across realizations.

    Parameters
    ----------
    cluster_file : str or Path
        Path to manticore_voxel_clusters.hdf5.
    ga_file : str or Path
        Path to GA_analysis.hdf5.
    sigma : float
        Smoothing scale key to select (sigma_X in the files).

    Returns
    -------
    tuple
        (positions, n_fields_used) where positions is (N, 3) array.
    """
    positions = []
    used = 0
    key = f"sigma_{sigma}"
    with File(cluster_file, "r") as clf, File(ga_file, "r") as gaf:
        max_distance = float(clf.attrs.get("max_distance", 0))
        for fname in sorted(k for k in gaf if k.startswith("field_")):
            if fname not in clf:
                continue
            g_ga = gaf[fname]
            g_cl = clf[fname]
            if key not in g_ga or key not in g_cl:
                continue
            sg = g_ga[key]
            if not sg.attrs.get("matched", False):
                continue
            ga_idx = int(sg.attrs["index"])
            members_ds = g_cl[key]["members"]
            if ga_idx >= members_ds.shape[0]:
                continue
            members = members_ds[ga_idx]
            if members.size == 0:
                continue
            box_size = float(g_ga.attrs["box_size"])
            resolution = int(g_ga.attrs["resolution"])
            observer = np.full(3, box_size / 2.0, dtype=float)
            x0 = flowi.create_initial_positions(
                box_size,
                resolution,
                N=None,
                observer_location=observer,
                max_distance=max_distance if max_distance > 0 else None,
                verbose=False,
            )
            x0 = np.asarray(x0)[members]
            positions.append(x0)
            used += 1

    if not positions:
        raise RuntimeError(f"No GA members found for sigma={sigma}")

    return np.vstack(positions), used


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


def save_projection(proj, labels, outfile, half_width,
                    scatter=None, contour_data=None, contour_level=None,
                    offset=None, observer=None, ga_center=None):
    if offset is None:
        offset = (0.0, 0.0)
    extent = (
        offset[0] - half_width,
        offset[0] + half_width,
        offset[1] - half_width,
        offset[1] + half_width,
    )
    with plt.style.context("science"):
        fig, ax = plt.subplots()
        im = ax.imshow(
            proj.T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        xlab = fr"${labels[0]} ~ [h^{{-1}}\,\mathrm{{Mpc}}]$"
        ylab = fr"${labels[1]} ~ [h^{{-1}}\,\mathrm{{Mpc}}]$"
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if scatter is not None and scatter.size > 0:
            ax.scatter(
                scatter[:, 0], scatter[:, 1],
                s=2, c="red", alpha=0.3, linewidths=0
            )
        if observer is not None:
            ax.plot(observer[0], observer[1], "kx", ms=4, alpha=0.8,
                    label="Observer")
        if ga_center is not None:
            ax.plot(ga_center[0], ga_center[1], "rx", ms=4, alpha=0.8,
                    label="GA center")
        if contour_data is not None and contour_level is not None:
            ax.contour(
                contour_data.T,
                levels=[contour_level],
                colors="red",
                linewidths=0.5,
                origin="lower",
                extent=extent,
            )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.0)
        cbar.set_label(r"$\rho\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$")
        fig.tight_layout()
        fig.savefig(outfile, dpi=450)
        plt.close(fig)


def create_ga_mask(ga_positions, box_size, resolution, observer, max_distance):
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
    # Create initial positions grid
    x0 = flowi.create_initial_positions(
        box_size, resolution, N=None,
        observer_location=observer,
        max_distance=max_distance if max_distance > 0 else None,
        verbose=False
    )
    x0 = np.asarray(x0)

    # Create binary mask
    mask = np.zeros(resolution**3, dtype=bool)

    # Find which grid points match GA positions
    # Use a spatial approach: round positions to voxel indices
    voxel_size = box_size / resolution
    ga_indices = np.floor(ga_positions / voxel_size).astype(int) % resolution
    ga_flat = (ga_indices[:, 0] * resolution**2 +
               ga_indices[:, 1] * resolution +
               ga_indices[:, 2])

    mask[ga_flat] = True
    return mask.reshape((resolution, resolution, resolution))


def plot_ga_sky_map_from_grid(rho, box_size, observer, outfile,
                              Rmax, dr, nside=32, Rmin=0,
                              unit=r"$\langle \rho \rangle\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$",  # noqa
                              r_power=0,
                              coords=None,
                              highlight_gal=None,
                              scatter_positions=None,
                              scatter_center=None):
    """
    Project a 3D density grid to a HEALPix map using nearest-grid-point rays.

    Parameters
    ----------
    rho : ndarray
        3D density grid (h^2 Msun / kpc^3).
    box_size : float
        Box size (h^-1 Mpc).
    observer : array-like
        Observer position in ICRS Cartesian.
    outfile : Path
        Output image path.
    Rmax : float
        Maximum radius to integrate along each line of sight.
    dr : float
        Radial step for integration.
    nside : int, optional
        HEALPix nside.
    Rmin : float, optional
        Minimum radius to integrate from.
    unit : str, optional
        Unit string for the colorbar.
    """
    # Use the existing utility
    m = flowi.utils.grid_ngp_projection(
        nside, rho, box_size, np.asarray(observer, dtype=float),
        Rmax, dr, Rmin=Rmin, coords=coords, verbose=True,
        r_power=r_power,
    )
    with plt.style.context("science"):
        hp.mollview(m, title="", unit=unit, cbar=True)

        if scatter_positions is not None and scatter_center is not None:
            # Convert scatter positions to Galactic coordinates
            r_s, ell_s, b_s = flowi.cartesian_icrs_to_galactic_spherical(
                scatter_positions, scatter_center)
            theta_s = np.deg2rad(90.0 - b_s)
            phi_s = np.deg2rad(ell_s % 360.0)
            hp.projscatter(theta_s, phi_s, lonlat=False, s=1, c='red',
                           alpha=0.3, linewidths=0)

        if highlight_gal is not None:
            ell_h, b_h = highlight_gal
            theta_h = np.deg2rad(90.0 - b_h)
            phi_h = np.deg2rad(ell_h)
            hp.projplot(theta_h, phi_h, "rx", markersize=6, alpha=0.9,
                        lonlat=False)
        plt.savefig(outfile, dpi=450, bbox_inches="tight")
        plt.close()


def main():
    center_sigma = 4.0
    plot_sigma = 2
    half_width = 90
    nside_map = 128
    out_dir = results_root / "GA_plots"
    cluster_file = results_root / "manticore_voxel_clusters.hdf5"

    contour_downsample = 25
    contour_frac = 0.99
    kde_grid = 50
    show_scatter = False
    box_center = None

    ga_file = results_root / "GA_analysis.hdf5"
    if out_dir.exists():
        print(f"Cleaning output directory {out_dir}")
        rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    center, field_ids, box_size = matched_centers(ga_file, center_sigma)
    cube_sum = None
    n_used = 0
    print(f"Found center at {center} in box of size {box_size} Mpc/h")
    box_center = np.full(3, box_size / 2.0)

    # Stack GA-member voxel positions across realizations
    stacked_positions, n_pos_fields = collect_ga_member_positions(
        cluster_file, ga_file, center_sigma)

    resolution = None
    density_mean = None
    for fid in field_ids[:10]:
        density = flowi.ManticoreLoader(data_root, fid).load_density_field()
        if resolution is None:
            resolution = density.shape[0]
        if plot_sigma > 0.0:
            density = flowi.smooth_scalar_field_gaussian(
                density, box_size, plot_sigma
            )
        cube = extract_cube(density, center, half_width, box_size)
        if cube_sum is None:
            cube_sum = np.zeros_like(cube, dtype=float)

        if density_mean is None:
            density_mean = np.zeros_like(density, dtype=float)

        cube_sum += cube
        density_mean += density
        n_used += 1
        print(f"Processed field {fid}")

    cube_mean = cube_sum / n_used
    density_mean /= n_used
    proj_yz, proj_xz, proj_xy = project_cube(cube_mean)

    # Scatter overlays: keep points within the displayed cube
    box_center = np.full(3, box_size / 2.0)
    rel_center = stacked_positions - center
    rel_center = (rel_center + box_size / 2) % box_size - box_size / 2
    mask_pts = np.all(np.abs(rel_center) <= half_width, axis=1)
    rel_center = rel_center[mask_pts]
    rel_box = stacked_positions[mask_pts] - box_center

    # 3D KDE evaluated on a grid, then projected
    def kde_projection(points):
        if points.shape[0] < 10:
            return None, None, None, None, None, None
        if contour_downsample > 1 and points.shape[0] > contour_downsample:
            points = points[::contour_downsample]
        kde = gaussian_kde(points.T)
        print(f"Evaluating 3D KDE on grid {kde_grid}^3 ...")
        grid = np.linspace(-half_width, half_width, kde_grid)
        xg, yg, zg = np.meshgrid(grid, grid, grid, indexing="ij")
        coords = np.vstack([xg.ravel(), yg.ravel(), zg.ravel()])
        dens = kde(coords).reshape(kde_grid, kde_grid, kde_grid)
        proj_yz = dens.sum(axis=0)
        proj_xz = dens.sum(axis=1)
        proj_xy = dens.sum(axis=2)
        lvl_yz = contour_level_for_fraction(proj_yz, frac=contour_frac)
        lvl_xz = contour_level_for_fraction(proj_xz, frac=contour_frac)
        lvl_xy = contour_level_for_fraction(proj_xy, frac=contour_frac)
        return proj_yz, proj_xz, proj_xy, lvl_yz, lvl_xz, lvl_xy

    dens_yz, dens_xz, dens_xy, lvl_yz, lvl_xz, lvl_xy = kde_projection(
        rel_center)

    base = out_dir / (
        f"GA_projection_center{center_sigma:.1f}_plot{plot_sigma:.1f}"
    )

    save_projection(
        proj_yz, ("y", "z"),
        base.with_name(f"{base.name}_yz.png"), half_width,
        scatter=rel_box[:, [1, 2]] if show_scatter else None,
        contour_data=dens_yz,
        contour_level=lvl_yz,
        offset=(center[1] - box_center[1], center[2] - box_center[2]),
        observer=(0.0, 0.0),
        ga_center=(center[1] - box_center[1], center[2] - box_center[2])
    )
    save_projection(
        proj_xz, ("x", "z"),
        base.with_name(f"{base.name}_xz.png"), half_width,
        scatter=rel_box[:, [0, 2]] if show_scatter else None,
        contour_data=dens_xz,
        contour_level=lvl_xz,
        offset=(center[0] - box_center[0], center[2] - box_center[2]),
        observer=(0.0, 0.0),
        ga_center=(center[0] - box_center[0], center[2] - box_center[2])
    )
    save_projection(
        proj_xy, ("x", "y"),
        base.with_name(f"{base.name}_xy.png"), half_width,
        scatter=rel_box[:, [0, 1]] if show_scatter else None,
        contour_data=dens_xy,
        contour_level=lvl_xy,
        offset=(center[0] - box_center[0], center[1] - box_center[1]),
        observer=(0.0, 0.0),
        ga_center=(center[0] - box_center[0], center[1] - box_center[1])
    )

    # Sky map of mean density within spherical cut about observer using NGP
    # projection
    if resolution is None:
        resolution = cube_mean.shape[0]

    voxel = box_size / resolution

    # Compute max distance from observer to GA positions
    distances = np.sqrt(((stacked_positions - box_center) ** 2).sum(axis=1))
    Rmax_ga = distances.max()
    print(f"Maximum GA distance from observer: {Rmax_ga:.2f} Mpc/h")

    sky_density_out = out_dir / f"GA_sky_density_sigma{center_sigma:.1f}.png"

    plot_ga_sky_map_from_grid(
        density_mean,
        box_size=box_size,
        observer=box_center,
        outfile=sky_density_out,
        Rmax=Rmax_ga,
        dr=0.1 * voxel,
        nside=nside_map,
        Rmin=0,
        r_power=2,
        unit=r"$\langle \rho \rangle\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$",
        coords="icrs->galactic",
        highlight_gal=None,
    )

    # Create GA fraction sky map
    print("Creating GA mask...")
    ga_mask = create_ga_mask(stacked_positions, box_size, resolution,
                             box_center, max_distance=Rmax_ga)
    print(f"GA mask has {ga_mask.sum()} voxels marked as GA members")

    print("Computing GA fraction map...")
    ga_fraction_map = flowi.utils.grid_ngp_projection(
        nside=nside_map,
        rho=ga_mask.astype(float),
        boxsize=box_size,
        observer=box_center,
        Rmax=Rmax_ga,
        dr=0.1 * voxel,
        Rmin=0,
        coords="icrs->galactic",
        r_power=0,
        verbose=True
    )
    print(f"GA map range: [{ga_fraction_map.min():.4e}, "
          f"{ga_fraction_map.max():.4e}]")

    sky_fraction_out = out_dir / f"GA_fraction_sigma{center_sigma:.1f}.png"

    # Convert GA center to Galactic coordinates for plotting
    (r_center, ell_center,
     b_center) = flowi.cartesian_icrs_to_galactic_spherical(
        center[None, :], box_center)
    print("GA center Galactic (l, b):", ell_center[0], b_center[0])
    theta_center = np.deg2rad(90.0 - b_center[0])
    phi_center = np.deg2rad(ell_center[0] % 360.0)

    with plt.style.context("science"):
        hp.mollview(ga_fraction_map, title="", unit="Relative GA depth",
                    cbar=True, cmap="inferno")
        hp.projplot(theta_center, phi_center, 'o', markersize=8,
                    markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=0.5, lonlat=False)
        plt.savefig(sky_fraction_out, dpi=450, bbox_inches="tight")
        plt.close()

    # Plot histogram of GA radial distances
    print("Plotting GA radial distance histogram...")
    ga_distances = np.sqrt(((stacked_positions - box_center) ** 2).sum(axis=1))
    hist_out = out_dir / f"GA_distance_histogram_sigma{center_sigma:.1f}.png"

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ga_distances, bins=50, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel(r"$r ~ [h^{-1}\,\mathrm{Mpc}]$")
        ax.set_ylabel(r"Count")
        ax.axvline(Rmax_ga, color='red', linestyle='--', linewidth=1,
                   label=f'Max: {Rmax_ga:.1f}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(hist_out, dpi=450)
        plt.close(fig)

    print(f"Used {n_used} realizations.")
    print(f"Median centroid (Mpc/h): {center}")
    print(f"Wrote projections to {out_dir}")


if __name__ == "__main__":
    main()
