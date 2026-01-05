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
Compute velocity field from density field using linear perturbation theory.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import flowi
from config import data_root, results_root


def plot_projections(v_actual, v_linear, output_file, vlim=3000,
                     residual_lim=300):
    """Plot 2D slices through center of velocity components for comparison."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    labels = [r'$v_x$', r'$v_y$', r'$v_z$']

    N = v_actual[0].shape[0]
    center_idx = N // 2

    for i, label in enumerate(labels):
        slice_actual = v_actual[i][:, :, center_idx]
        slice_linear = v_linear[i][:, :, center_idx]

        vmin = -vlim
        vmax = vlim

        im0 = axes[i, 0].imshow(slice_actual.T, origin='lower',
                                vmin=vmin, vmax=vmax)
        axes[i, 0].set_ylabel(r'$y$ [pixel]')
        if i == 0:
            axes[i, 0].set_title(f'Manticore (z={center_idx})')
        if i == 2:
            axes[i, 0].set_xlabel(r'$x$ [pixel]')
        axes[i, 0].text(0.05, 0.95, label, transform=axes[i, 0].transAxes,
                        va='top', color='white', fontsize=12)
        plt.colorbar(im0, ax=axes[i, 0], label='km/s')

        im1 = axes[i, 1].imshow(slice_linear.T, origin='lower',
                                vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 1].set_title(f'Linear Theory (z={center_idx})')
        if i == 2:
            axes[i, 1].set_xlabel(r'$x$ [pixel]')
        plt.colorbar(im1, ax=axes[i, 1], label='km/s')

        im2 = axes[i, 2].imshow(
            (slice_actual - slice_linear).T, origin='lower',
            cmap='RdBu_r', vmin=-residual_lim, vmax=residual_lim)
        if i == 0:
            axes[i, 2].set_title('Residual')
        if i == 2:
            axes[i, 2].set_xlabel(r'$x$ [pixel]')
        plt.colorbar(im2, ax=axes[i, 2], label='km/s')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved projection plot to {output_file}")


def plot_histograms(v_actual, v_linear, output_file, vlim=1000):
    """Plot 1D histograms of velocity component voxel values."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for i, label in enumerate(['$v_x$', '$v_y$', '$v_z$']):
        ax = axes[i]

        v_act_clipped = np.clip(v_actual[i].ravel(), -vlim, vlim)
        v_lin_clipped = np.clip(v_linear[i].ravel(), -vlim, vlim)

        bins = np.linspace(-vlim, vlim, 50)
        ax.hist(v_act_clipped, bins=bins, alpha=0.6,
                label='Manticore', density=True)
        ax.hist(v_lin_clipped, bins=bins, alpha=0.6,
                label='Linear Theory', density=True)
        ax.set_xlabel(f'{label} [km/s]')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_xlim(-vlim, vlim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved histogram plot to {output_file}")


def plot_scatter(v_actual, v_linear, output_file):
    """Plot scatter between actual and linear velocity voxel values."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, label in enumerate(['$v_x$', '$v_y$', '$v_z$']):
        ax = axes[i]

        sample_idx = np.random.choice(v_actual[i].size,
                                      size=min(10000, v_actual[i].size),
                                      replace=False)
        x = v_actual[i].ravel()[sample_idx]
        y = v_linear[i].ravel()[sample_idx]

        ax.hexbin(x, y, gridsize=50, cmap='viridis', mincnt=1)

        lim_min = min(x.min(), y.min())
        lim_max = max(x.max(), y.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=1)

        corr = np.corrcoef(v_actual[i].ravel(), v_linear[i].ravel())[0, 1]
        ax.text(0.05, 0.95, f'$r = {corr:.3f}$',
                transform=ax.transAxes, va='top')

        ax.set_xlabel(f'{label} Manticore [km/s]')
        ax.set_ylabel(f'{label} Linear [km/s]')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved scatter plot to {output_file}")


def compute_observer_velocity_vs_radius(delta, boxsize, observer_pos, radii,
                                        Omega_m, h, a, cumulative=True,
                                        pad_fraction=None):
    """
    Compute observer velocity as a function of truncation radius.

    For each radius, zero out delta outside that radius from the observer,
    compute the linear velocity field, and extract velocity at observer
    position.

    Parameters
    ----------
    delta : ndarray, shape (N, N, N)
        Overdensity field.
    boxsize : float
        Box size in Mpc/h.
    observer_pos : array-like, shape (3,)
        Observer position in Mpc/h.
    radii : array-like
        Array of radii in Mpc/h to evaluate.
    Omega_m : float
        Matter density parameter.
    h : float
        Reduced Hubble constant.
    a : float
        Scale factor.
    cumulative : bool
        If True (default), use cumulative thresholds (all voxels within
        radius). If False, use radial bins (voxels between consecutive radii).
    pad_fraction : float, optional
        Fractional zero-padding for FFT.

    Returns
    -------
    velocities : ndarray, shape (len(radii), 3) or (len(radii)-1, 3)
        Observer velocity [vx, vy, vz] for each radius (cumulative) or each
        bin (if cumulative=False).
    """
    N = delta.shape[0]
    cell_size = boxsize / N

    # Create coordinate grid
    coords = np.linspace(cell_size / 2, boxsize - cell_size / 2, N)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')

    # Distance from observer
    dx = x - observer_pos[0]
    dy = y - observer_pos[1]
    dz = z - observer_pos[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    if cumulative:
        velocities = np.zeros((len(radii), 3))

        for i, radius in enumerate(radii):
            delta_masked = delta.copy()
            delta_masked[dist > radius] = 0.0

            velocities[i] = flowi.delta_to_velocity(
                delta_masked, boxsize, Omega_m, h, a, pad_fraction,
                center_only=True)
    else:
        n_bins = len(radii) - 1
        velocities = np.zeros((n_bins, 3))

        for i in range(n_bins):
            r_min, r_max = radii[i], radii[i + 1]
            delta_masked = delta.copy()
            delta_masked[(dist < r_min) | (dist > r_max)] = 0.0

            velocities[i] = flowi.delta_to_velocity(
                delta_masked, boxsize, Omega_m, h, a, pad_fraction,
                center_only=True)

    return velocities


def main():
    n_fields = 80
    Omega_m = 0.306
    h = 1.0
    a = 1.0
    pad_fraction = 0.5
    smooth_scale = 2.5  # Mpc/h
    compute_radius_analysis = True
    cumulative = True
    radii = np.arange(10, 200, 15)  # Mpc/h

    if cumulative:
        output_file = results_root / "linear_velocity_fields.hdf5"
    else:
        output_file = results_root / "linear_velocity_fields_bins.hdf5"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if compute_radius_analysis:
        v_obs_all_fields = []

    with h5py.File(output_file, "w") as f:
        f.attrs["Omega_m"] = Omega_m
        f.attrs["h"] = h
        f.attrs["a"] = a
        f.attrs["smooth_scale"] = smooth_scale
        f.attrs["cumulative"] = cumulative
        f.attrs["description"] = (
            "Velocity fields from density using linear perturbation theory"
        )
        f.create_dataset("radii", data=radii)

        for i in trange(n_fields, desc="Processing fields"):
            loader = flowi.ManticoreLoader(data_root, i)
            rho = loader.load_density_field()
            rho = flowi.smooth_scalar_field_gaussian(
                rho, loader.boxsize, smooth_scale)
            delta = rho / rho.mean() - 1.0

            v_center = flowi.delta_to_velocity(
                delta, loader.boxsize, Omega_m, h, a, pad_fraction,
                verbose=i == 0, center_only=True)

            grp = f.create_group(f"field_{i}")
            grp.create_dataset("velocity_center", data=v_center)
            grp.attrs["boxsize"] = loader.boxsize
            grp.attrs["resolution"] = loader.resolution

            if i == 0:
                f.attrs["boxsize"] = loader.boxsize
                f.attrs["resolution"] = loader.resolution

            if compute_radius_analysis:
                observer_pos = np.array([loader.boxsize / 2] * 3)
                v_obs_vs_r = compute_observer_velocity_vs_radius(
                    delta, loader.boxsize, observer_pos, radii,
                    Omega_m, h, a, cumulative, pad_fraction)

                grp.create_dataset("observer_velocity_vs_radius",
                                   data=v_obs_vs_r)
                v_obs_all_fields.append(v_obs_vs_r)

    print(f"Wrote linear velocity fields to {output_file}")

    if compute_radius_analysis:
        v_obs_all_fields = np.array(v_obs_all_fields)  # (n_fields, n_radii, 3)
        v_mag = np.linalg.norm(v_obs_all_fields, axis=2)  # (n_fields, n_radii)

        mean_vmag = v_mag.mean(axis=0)
        std_vmag = v_mag.std(axis=0)

        print("\nObserver velocity magnitude vs radius:")
        print(f"{'Radius [Mpc/h]':>15} {'Mean |v| [km/s]':>18} {'Std [km/s]':>15}")  # noqa
        print("-" * 50)
        for r, vm, vs in zip(radii, mean_vmag, std_vmag):
            print(f"{r:15.1f} {vm:18.3f} {vs:15.3f}")


if __name__ == "__main__":
    main()
