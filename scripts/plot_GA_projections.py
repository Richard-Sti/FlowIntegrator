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

from functools import lru_cache
from shutil import rmtree

import astropy.units as u
import flowi
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
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
        (positions_list, n_fields_used) where positions_list is a list of
        per-field (n_i, 3) arrays.
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
            x0_full = initial_positions_grid(
                box_size,
                resolution,
                _observer_key(observer),
                max_distance if max_distance > 0 else None,
            )
            x0 = np.asarray(x0_full)[members]
            positions.append(x0)
            used += 1

    if not positions:
        raise RuntimeError(f"No GA members found for sigma={sigma}")

    return positions, used


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
                    scatter=None, contours_list=None,
                    offset=None, observer=None, ga_center=None,
                    cluster_data=None, in_ga=None, box_center=None, ax=None,
                    distance_limit=75.0):
    if offset is None:
        offset = (0.0, 0.0)
    extent = (
        offset[0] - half_width,
        offset[0] + half_width,
        offset[1] - half_width,
        offset[1] + half_width,
    )

    # Create figure if ax not provided
    if ax is None:
        with plt.style.context("science"):
            fig, ax = plt.subplots()
            own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    im = ax.imshow(
        proj.T,
        origin="lower",
        extent=extent,
        cmap="inferno",
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
        ax.plot(observer[0], observer[1], "kx", ms=1, alpha=0.8,
                label="Observer")
    if ga_center is not None:
        ax.scatter(
            [ga_center[0]], [ga_center[1]],
            s=18, c="#00FFFF", edgecolors='black',
            linewidths=0.8, zorder=12, label="GA center",
        )
        ax.text(
            ga_center[0] + 4.0, ga_center[1] + 4.0,
            "GA center", color="#00FFFF", ha="left", va="bottom",
            fontsize='xx-small', weight="bold", zorder=12
        )
    if contours_list is not None:
        for contour_data, contour_level in contours_list:
            if contour_data is not None and contour_level is not None:
                ax.contour(
                    contour_data.T,
                    levels=[contour_level],
                    colors="red",
                    linewidths=0.5,
                    origin="lower",
                    extent=extent,
                    alpha=0.3
                )
    # Plot clusters
    has_cluster_data = (cluster_data is not None and in_ga is not None and
                        box_center is not None)
    if has_cluster_data:
        distances = cluster_data.get('distances', None)
        if distances is not None:
            within_distance = distances <= distance_limit
        else:
            within_distance = np.ones_like(in_ga, dtype=bool)

        # Determine which axes we're plotting (x=0, y=1, z=2)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        idx_x = axis_map[labels[0]]
        idx_y = axis_map[labels[1]]

        # Plot non-GA clusters (in orange)
        non_ga_nearby = (~in_ga) & within_distance
        if non_ga_nearby.any():
            cluster_pos = cluster_data['positions'][non_ga_nearby]
            cluster_names = [
                cluster_data['names'][i]
                for i in range(len(non_ga_nearby)) if non_ga_nearby[i]
            ]

            # Extract 2D coordinates relative to box center
            cluster_2d = (
                cluster_pos[:, [idx_x, idx_y]]
                - box_center[[idx_x, idx_y]])

            # Check if within subbox boundaries
            x_in = ((cluster_2d[:, 0] >= extent[0]) &
                    (cluster_2d[:, 0] <= extent[1]))
            y_in = ((cluster_2d[:, 1] >= extent[2]) &
                    (cluster_2d[:, 1] <= extent[3]))
            in_bounds = x_in & y_in

            # Plot markers
            if in_bounds.any():
                ax.scatter(
                    cluster_2d[in_bounds, 0], cluster_2d[in_bounds, 1],
                    s=7.5, c='#7FFF00', marker='o', zorder=10,
                    linewidths=0)
                for i, (name, is_in) in enumerate(zip(cluster_names,
                                                      in_bounds)):
                    if not is_in:
                        continue
                    ax.text(
                        cluster_2d[i, 0] + 2, cluster_2d[i, 1] + 2,
                        name, fontsize='xx-small', color='#7FFF00',
                        ha='left', va='bottom', weight='bold')

        # Plot GA clusters (in cyan)
        ga_mask = in_ga & within_distance
        if ga_mask.any():
            cluster_pos = cluster_data['positions'][ga_mask]
            cluster_names = [
                cluster_data['names'][i]
                for i in range(len(ga_mask)) if ga_mask[i]
            ]

            # Extract 2D coordinates relative to box center
            cluster_2d = (
                cluster_pos[:, [idx_x, idx_y]]
                - box_center[[idx_x, idx_y]])

            # Check if within subbox boundaries
            x_in = ((cluster_2d[:, 0] >= extent[0]) &
                    (cluster_2d[:, 0] <= extent[1]))
            y_in = ((cluster_2d[:, 1] >= extent[2]) &
                    (cluster_2d[:, 1] <= extent[3]))
            in_bounds = x_in & y_in

            # Plot markers
            if in_bounds.any():
                ax.scatter(
                    cluster_2d[in_bounds, 0], cluster_2d[in_bounds, 1],
                    s=7.5, c='#00FFFF', marker='o', zorder=10, linewidths=0)

                # Add labels
                for i, (name, is_in) in enumerate(zip(cluster_names,
                                                      in_bounds)):
                    if not is_in:
                        continue
                    ax.text(
                        cluster_2d[i, 0] + 2, cluster_2d[i, 1] + 2,
                        name, fontsize='xx-small', color='#00FFFF',
                        ha='left', va='bottom', weight='bold')

    # Handle colorbar and saving only if we created our own figure
    if own_fig:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.0)
        cbar.set_label(r"$\rho\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$")
        fig.tight_layout()
        fig.savefig(outfile, dpi=450)
        plt.close(fig)
    else:
        return im


def read_cluster_catalog(cluster_file, box_size, H0=100.0):
    """
    Read local cluster catalog and convert to Cartesian ICRS coordinates.

    Parameters
    ----------
    cluster_file : str or Path
        Path to cluster catalog file.
    box_size : float
        Box size (to account for observer position at box center).
    H0 : float
        Hubble constant in km/s/Mpc (default: 100 for h=1 units).

    Returns
    -------
    dict
        Dictionary with 'names', 'positions', 'velocities', 'ell', 'b'.
    """
    names = []
    ell = []
    b = []
    v_kms = []

    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # Name can be multiple words, last 3 are numbers
            name = ' '.join(parts[:-3])
            ell_deg = float(parts[-3])
            b_deg = float(parts[-2])
            vel = float(parts[-1])

            names.append(name)
            ell.append(ell_deg)
            b.append(b_deg)
            v_kms.append(vel)

    ell = np.array(ell)
    b = np.array(b)
    v_kms = np.array(v_kms)

    # Convert velocities to distances using FlatLambdaCDM cosmology
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    z = v_kms / 299792.458  # Convert velocity to redshift (v/c)
    distances = cosmo.comoving_distance(z).value  # Mpc (comoving)

    # Convert Galactic spherical to ICRS Cartesian
    coords_gal = SkyCoord(l=ell*u.deg, b=b*u.deg, distance=distances*u.Mpc,
                          frame='galactic')
    coords_icrs = coords_gal.icrs

    # Convert to Cartesian and add box_size/2 to account for observer position
    positions = np.column_stack([
        coords_icrs.cartesian.x.value,
        coords_icrs.cartesian.y.value,
        coords_icrs.cartesian.z.value
    ]) + box_size / 2.0

    return {
        'names': names,
        'positions': positions,
        'velocities': v_kms,
        'ell': ell,
        'b': b,
        'distances': distances
    }


def check_clusters_in_ga(cluster_data, ga_mask, box_size, resolution):
    """
    Check which clusters are within the GA by sampling the GA mask.

    Parameters
    ----------
    cluster_data : dict
        Cluster data from read_cluster_catalog.
    ga_mask : ndarray
        3D binary mask (1 = GA, 0 = not GA).
    box_size : float
        Box size.
    resolution : int
        Grid resolution.

    Returns
    -------
    ndarray
        Boolean array indicating which clusters are in GA.
    """
    voxel = box_size / float(resolution)
    idx = np.floor(cluster_data['positions'] / voxel).astype(int) % resolution
    offsets = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=int)
    neigh_idx = (idx[:, None, :] + offsets[None, :, :]) % resolution
    flat = (neigh_idx[:, :, 0] * resolution ** 2 +
            neigh_idx[:, :, 1] * resolution +
            neigh_idx[:, :, 2])
    mask_flat = ga_mask.reshape(-1)
    return (mask_flat[flat] >= 0.5).any(axis=1)


def create_ga_mask(ga_positions, box_size, resolution, observer, max_distance,
                   n_fields=None):
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
    # Create binary mask
    mask = np.zeros(resolution**3, dtype=float)

    # Find which grid points match GA positions; accumulate counts
    voxel_size = box_size / resolution
    ga_indices = np.floor(ga_positions / voxel_size).astype(int) % resolution
    ga_flat = (ga_indices[:, 0] * resolution**2 +
               ga_indices[:, 1] * resolution +
               ga_indices[:, 2])

    np.add.at(mask, ga_flat, 1.0)
    if n_fields is not None and n_fields > 0:
        mask /= float(n_fields)
    return mask.reshape((resolution, resolution, resolution))


def plot_zone_of_avoidance(b_min=-10, b_max=10, color='white', alpha=0.8):
    """
    Plot the Galactic zone of avoidance on current HEALPix map.

    Parameters
    ----------
    b_min : float
        Minimum Galactic latitude in degrees.
    b_max : float
        Maximum Galactic latitude in degrees.
    color : str
        Color for the zone.
    alpha : float
        Transparency (0=transparent, 1=opaque).
    """
    # Plot multiple horizontal lines to fill the band; hp handles wrap
    ell_deg = np.linspace(0, 360, 800)
    for b_deg in [b_min, b_max]:
        hp.projplot(
            ell_deg, np.full_like(ell_deg, b_deg),
            lonlat=True, color=color, alpha=alpha, zorder=10)


def plot_clusters_on_healpy(cluster_data, in_ga, observer, max_distance=100.0):
    """
    Plot clusters on current HEALPix map.

    Parameters
    ----------
    cluster_data : dict
        Cluster data from read_cluster_catalog.
    in_ga : ndarray
        Boolean array indicating which clusters are in GA.
    observer : array-like
        Observer position.
    max_distance : float
        Maximum distance to plot clusters (Mpc/h).
    """
    def _label_info(name, theta, phi):
        name_l = name.lower()
        short_name = name
        if name_l.startswith('coma') or 'perseus' in name_l:
            text_phi = (phi + np.deg2rad(5.0)) % (2 * np.pi)
            text_theta = theta + np.deg2rad(2.0)
            va = 'top'
        else:
            text_phi = (phi + np.deg2rad(5.0)) % (2 * np.pi)
            text_theta = theta - np.deg2rad(2.0)
            va = 'bottom'
        return short_name, text_theta, text_phi, va

    # Plot non-GA clusters within distance (in orange)
    distances = cluster_data.get('distances', None)
    if distances is not None:
        within_distance = distances <= max_distance
        non_ga_nearby = within_distance & ~in_ga
        if non_ga_nearby.any():
            nearby_names = [
                cluster_data['names'][i]
                for i in range(len(in_ga)) if non_ga_nearby[i]]
            nearby_ell = cluster_data['ell'][non_ga_nearby]
            nearby_b = cluster_data['b'][non_ga_nearby]

            theta_nearby = np.deg2rad(90.0 - nearby_b)
            phi_nearby = np.deg2rad(nearby_ell % 360.0)

            for theta, phi, name in zip(theta_nearby, phi_nearby, nearby_names):  # noqa
                hp.projplot(theta, phi, 'o', markersize=6,
                            markerfacecolor='#7FFF00', markeredgecolor='black',
                            markeredgewidth=0.8, lonlat=False)
                short_name, text_theta, text_phi, va = _label_info(
                    name, theta, phi)
                hp.projtext(text_theta, text_phi, short_name, lonlat=False,
                            fontsize='small', color='#7FFF00', ha='left',
                            va=va)

    # Plot GA clusters (in cyan)
    ga_cluster_names = [
        cluster_data['names'][i] for i in range(len(in_ga)) if in_ga[i]]
    ga_cluster_ell = cluster_data['ell'][in_ga]
    ga_cluster_b = cluster_data['b'][in_ga]

    # Convert to HEALPix coordinates
    theta_clusters = np.deg2rad(90.0 - ga_cluster_b)
    phi_clusters = np.deg2rad(ga_cluster_ell % 360.0)

    # Plot each cluster
    for i, (theta, phi, name) in enumerate(zip(theta_clusters,
                                               phi_clusters,
                                               ga_cluster_names)):
        hp.projplot(theta, phi, 'o', markersize=6,
                    markerfacecolor='#00FFFF', markeredgecolor='black',
                    markeredgewidth=0.8, lonlat=False)

        short_name, text_theta, text_phi, va = _label_info(name, theta, phi)
        hp.projtext(text_theta, text_phi, short_name, lonlat=False,
                    fontsize='small', color='#00FFFF', ha='left', va=va)


def annotate_ga_center_and_clusters(theta_center, phi_center, cluster_data,
                                    in_ga, observer, max_distance=100.0):
    hp.projplot(theta_center, phi_center, 'o', markersize=7.5,
                markerfacecolor='#00FFFF', markeredgecolor='black',
                markeredgewidth=0.8, lonlat=False, zorder=12)
    text_phi_center = (phi_center + np.deg2rad(3.0)) % (2 * np.pi)
    text_theta_center = theta_center - np.deg2rad(2.0)
    hp.projtext(text_theta_center, text_phi_center, "GA center",
                lonlat=False, fontsize='small', color='#00FFFF', ha='left',
                va='bottom')
    plot_clusters_on_healpy(cluster_data, in_ga, observer,
                            max_distance=max_distance)


def plot_ga_sky_map_from_grid(rho, box_size, observer, outfile,
                              Rmax, dr, nside=32, Rmin=0,
                              unit=None,
                              r_power=0,
                              coords=None,
                              highlight_gal=None,
                              scatter_positions=None,
                              scatter_center=None,
                              cluster_data=None,
                              in_ga=None,
                              ga_center_gal=None):
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
    if unit is None:
        unit = r"$\langle \rho \rangle\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$"

    # Use the existing utility
    m = flowi.utils.grid_ngp_projection(
        nside, rho, box_size, np.asarray(observer, dtype=float),
        Rmax, dr, Rmin=Rmin, coords=coords, verbose=True,
        r_power=r_power,
    )
    with plt.style.context("science"):
        hp.mollview(m, title="", unit=unit, cbar=True, xsize=1500)

        # Plot zone of avoidance
        plot_zone_of_avoidance()

        if ga_center_gal is not None:
            ell_c, b_c = ga_center_gal
            theta_c = np.deg2rad(90.0 - b_c)
            phi_c = np.deg2rad(ell_c % 360.0)
            hp.projplot(theta_c, phi_c, 'o', markersize=7.5,
                        markerfacecolor='#00FFFF', markeredgecolor='black',
                        markeredgewidth=0.8, lonlat=False, zorder=12)
            text_phi_c = (phi_c + np.deg2rad(3.0)) % (2 * np.pi)
            text_theta_c = theta_c - np.deg2rad(2.0)
            hp.projtext(text_theta_c, text_phi_c, "GA center",
                        lonlat=False, fontsize='small', color='#00FFFF',
                        ha='left', va='bottom')

        if scatter_positions is not None and scatter_center is not None:
            # Convert scatter positions to Galactic coordinates
            r_s, ell_s, b_s = flowi.cartesian_icrs_to_galactic_spherical(
                scatter_positions, scatter_center)
            theta_s = np.deg2rad(90.0 - b_s)
            phi_s = np.deg2rad(ell_s % 360.0)
            hp.projscatter(theta_s, phi_s, lonlat=False, s=7.5, c='red',
                           alpha=0.3, linewidths=0)

        if highlight_gal is not None:
            ell_h, b_h = highlight_gal
            theta_h = np.deg2rad(90.0 - b_h)
            phi_h = np.deg2rad(ell_h)
            hp.projplot(theta_h, phi_h, "rx", markersize=6, alpha=0.9,
                        lonlat=False)

        if cluster_data is not None and in_ga is not None:
            plot_clusters_on_healpy(cluster_data, in_ga, observer)

        plt.savefig(outfile, dpi=450, bbox_inches="tight")
        plt.close()


def main():
    center_sigma = 4.0
    plot_sigma = 2
    half_width = 90
    nside_map = 128
    out_dir = results_root / "GA_plots"

    # Load precomputed data
    fname = (f"GA_precomputed_center{center_sigma:.1f}_"
             f"plot{plot_sigma:.1f}.hdf5")
    precomputed_file = results_root / fname
    print(f"Loading precomputed data from {precomputed_file}")

    if out_dir.exists():
        print(f"Cleaning output directory {out_dir}")
        rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with File(precomputed_file, "r") as h5f:
        center = h5f.attrs["center"]
        box_size = h5f.attrs["box_size"]
        resolution = h5f.attrs["resolution"]
        Rmax_ga = h5f.attrs["Rmax_ga"]
        field_ids = h5f["field_ids"][:]
        density_mean = h5f["density_mean"][:]
        ga_sky_depth_list = []
        if "ga_sky_maps" in h5f:
            for fid in field_ids:
                fkey = f"field_{fid}"
                if fkey in h5f["ga_sky_maps"]:
                    ga_sky_depth_list.append(h5f["ga_sky_maps"][fkey][:])
        if not ga_sky_depth_list:
            raise RuntimeError("No GA sky depth maps loaded.")
        ga_depth_map = np.mean(ga_sky_depth_list, axis=0)
        ga_depth_std = np.std(ga_sky_depth_list, axis=0)
        print(f"GA depth map range: [{ga_depth_map.min():.4e}, "
              f"{ga_depth_map.max():.4e}] Mpc/h")
        print(f"GA depth std range: [{ga_depth_std.min():.4e}, "
              f"{ga_depth_std.max():.4e}] Mpc/h")

        # Load GA masks and compute membership probability
        print("Loading GA masks and computing membership probabilities...")
        ga_mask_sum = None
        n_masks = 0
        for fid in field_ids:
            fkey = f"field_{fid}"
            if fkey in h5f["ga_masks"]:
                mask = h5f["ga_masks"][fkey][:]
                if ga_mask_sum is None:
                    ga_mask_sum = np.zeros_like(mask, dtype=float)
                ga_mask_sum += mask
                n_masks += 1

        ga_membership = ga_mask_sum / n_masks if n_masks > 0 else None
        print(f"Computed membership from {n_masks} masks")

        # Load KDE contours from all realizations
        print("Loading KDE contours from all realizations...")
        kde_contours_list = []
        for fid in field_ids:
            fkey = f"field_{fid}"
            if fkey in h5f["kde_contours"]:
                kde_grp = h5f["kde_contours"][fkey]
                kde_yz = kde_grp["kde_yz"][:]
                kde_xz = kde_grp["kde_xz"][:]
                kde_xy = kde_grp["kde_xy"][:]
                lvl_yz = kde_grp.attrs.get("lvl_yz", None)
                lvl_xz = kde_grp.attrs.get("lvl_xz", None)
                lvl_xy = kde_grp.attrs.get("lvl_xy", None)
                kde_contours_list.append({
                    "kde_yz": kde_yz,
                    "kde_xz": kde_xz,
                    "kde_xy": kde_xy,
                    "lvl_yz": lvl_yz,
                    "lvl_xz": lvl_xz,
                    "lvl_xy": lvl_xy
                })
        print(f"Loaded {len(kde_contours_list)} KDE contour sets")

    n_used = len(field_ids)
    print(f"Center: {center}")
    print(f"Box size: {box_size}")
    print(f"Rmax_ga: {Rmax_ga}")
    box_center = np.full(3, box_size / 2.0)

    # Extract cube from mean density for projection
    cube_mean = extract_cube(density_mean, center, half_width, box_size)
    proj_yz, proj_xz, proj_xy = project_cube(cube_mean)

    voxel = box_size / resolution

    # Check which clusters are within the GA using membership probability
    print("\nChecking which clusters are within the GA...")
    cluster_file = data_root / "local_clusters.txt"
    cluster_data = read_cluster_catalog(cluster_file, box_size, H0=100.0)
    theta_clusters = np.deg2rad(90.0 - cluster_data["b"])
    phi_clusters = np.deg2rad(cluster_data["ell"] % 360.0)
    pix_clusters = hp.ang2pix(nside_map, theta_clusters, phi_clusters)
    cluster_depth_mean = ga_depth_map[pix_clusters]
    cluster_depth_std = ga_depth_std[pix_clusters]

    print("Computing cluster membership fractions...")
    if ga_membership is not None:
        membership_fraction = np.zeros(len(cluster_data["names"]))
        for i in range(len(cluster_data["names"])):
            pos = cluster_data['positions'][i]
            # Sample membership at cluster position
            nx, ny, nz = ga_membership.shape
            ix = int(np.floor(pos[0] / box_size * nx)) % nx
            iy = int(np.floor(pos[1] / box_size * ny)) % ny
            iz = int(np.floor(pos[2] / box_size * nz)) % nz
            membership_fraction[i] = ga_membership[ix, iy, iz]

        in_ga = membership_fraction > 0.5
        for i, name in enumerate(cluster_data['names']):
            print(f"  {name:25s}  membership fraction: "
                  f"{membership_fraction[i]:.2f}  "
                  f"d={cluster_data['distances'][i]:6.1f} Mpc/h  "
                  f"depth={cluster_depth_mean[i]:6.1f}±{cluster_depth_std[i]:6.1f} Mpc/h")  # noqa

        print(f"\nFound {in_ga.sum()} clusters within the GA "
              f"(out of {len(in_ga)}):")
        for i, name in enumerate(cluster_data['names']):
            if in_ga[i]:
                dist = cluster_data['distances'][i]
                ell = cluster_data['ell'][i]
                b = cluster_data['b'][i]
                frac = membership_fraction[i]
                depth_m = cluster_depth_mean[i]
                depth_s = cluster_depth_std[i]
                print(f"  {name:25s}  d={dist:6.1f} Mpc/h  "
                      f"(l={ell:6.1f}°, b={b:6.1f}°)  f={frac:.2f}  "
                      f"depth={depth_m:6.1f}±{depth_s:6.1f} Mpc/h")
    else:
        membership_fraction = np.zeros(len(cluster_data["names"]))
        in_ga = np.zeros(len(cluster_data["names"]), dtype=bool)

    # Prepare contours from all realizations
    contours_yz = [(c["kde_yz"], c["lvl_yz"]) for c in kde_contours_list]
    contours_xz = [(c["kde_xz"], c["lvl_xz"]) for c in kde_contours_list]
    contours_xy = [(c["kde_xy"], c["lvl_xy"]) for c in kde_contours_list]

    base = out_dir / (
        f"GA_projection_center{center_sigma:.1f}_plot{plot_sigma:.1f}"
    )

    # Convert GA center to Galactic coordinates for plotting
    (r_center, ell_center,
     b_center) = flowi.cartesian_icrs_to_galactic_spherical(
        center[None, :], box_center)
    print("GA center Galactic (l, b):", ell_center[0], b_center[0])
    theta_center = np.deg2rad(90.0 - b_center[0])
    phi_center = np.deg2rad(ell_center[0] % 360.0)

    # Create 3-panel figure
    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

        # YZ projection
        im0 = save_projection(
            proj_yz, ("y", "z"),
            None, half_width,
            scatter=None,
            contours_list=contours_yz,
            offset=(center[1] - box_center[1], center[2] - box_center[2]),
            observer=(0.0, 0.0),
            ga_center=(center[1] - box_center[1], center[2] - box_center[2]),
            cluster_data=cluster_data,
            in_ga=in_ga,
            box_center=box_center,
            ax=axes[0]
        )

        # XZ projection
        im1 = save_projection(
            proj_xz, ("x", "z"),
            None, half_width,
            scatter=None,
            contours_list=contours_xz,
            offset=(center[0] - box_center[0], center[2] - box_center[2]),
            observer=(0.0, 0.0),
            ga_center=(center[0] - box_center[0], center[2] - box_center[2]),
            cluster_data=cluster_data,
            in_ga=in_ga,
            box_center=box_center,
            ax=axes[1]
        )

        # XY projection
        im2 = save_projection(
            proj_xy, ("x", "y"),
            None, half_width,
            scatter=None,
            contours_list=contours_xy,
            offset=(center[0] - box_center[0], center[1] - box_center[1]),
            observer=(0.0, 0.0),
            ga_center=(center[0] - box_center[0], center[1] - box_center[1]),
            cluster_data=cluster_data,
            in_ga=in_ga,
            box_center=box_center,
            ax=axes[2]
        )

        # Add colorbar to each panel
        for i, (ax, im) in enumerate(zip(axes, [im0, im1, im2])):
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if i == 2:  # Only add label to rightmost panel
                cbar.set_label(r"$\rho\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$")

        fig.tight_layout()
        fig.savefig(base.with_name(f"{base.name}_combined.pdf"), dpi=450)
        plt.close(fig)

    # Plot density sky map with clusters
    sky_density_out = out_dir / f"GA_sky_density_sigma{center_sigma:.1f}.pdf"
    plot_ga_sky_map_from_grid(
        density_mean,
        box_size=box_size,
        observer=box_center,
        outfile=sky_density_out,
        Rmax=Rmax_ga,
        dr=0.3 * voxel,
        nside=nside_map,
        Rmin=0,
        r_power=2,
        unit=r"$\langle \rho \rangle\ [h^2\,M_\odot\,\mathrm{kpc}^{-3}]$",
        coords="icrs->galactic",
        # highlight_gal=(319.62680107, 26.51202313),
        cluster_data=cluster_data,
        in_ga=in_ga,
        ga_center_gal=(ell_center[0], b_center[0]),
    )

    sky_fraction_out = out_dir / f"GA_fraction_sigma{center_sigma:.1f}.pdf"
    sky_std_out = out_dir / f"GA_fraction_sigma{center_sigma:.1f}_std.pdf"

    with plt.style.context("science"):
        hp.mollview(ga_depth_map, title="",
                    unit=r"Mean of GA depth $[h^{-1}\,\mathrm{Mpc}]$",
                    cbar=True, cmap="inferno", xsize=1500)

        # Plot zone of avoidance
        plot_zone_of_avoidance()

        annotate_ga_center_and_clusters(
            theta_center, phi_center, cluster_data, in_ga, box_center,
            max_distance=100.0
        )
        plt.savefig(sky_fraction_out, dpi=450, bbox_inches="tight")
        plt.close()

    with plt.style.context("science"):
        hp.mollview(ga_depth_std, title="",
                    unit=r"Std of GA depth $[h^{-1}\,\mathrm{Mpc}]$",
                    cbar=True, cmap="magma", xsize=1500)

        plot_zone_of_avoidance()
        annotate_ga_center_and_clusters(
            theta_center, phi_center, cluster_data, in_ga, box_center,
            max_distance=100.0
        )
        plt.savefig(sky_std_out, dpi=450, bbox_inches="tight")
        plt.close()

    print(f"Used {n_used} realizations.")
    print(f"Median centroid (Mpc/h): {center}")
    print(f"Wrote projections to {out_dir}")


if __name__ == "__main__":
    main()
