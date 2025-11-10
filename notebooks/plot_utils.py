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

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_galactic(xf_gal, t):
    """
    Visualizes the results of a single trajectory in Galactic coordinates.

    Parameters
    ----------
    xf_gal : numpy.ndarray
        A 2D array of shape (num_steps, 3) containing the trajectory in
        Galactic coordinates. The columns are expected to be [r, l, b],
        where r is the distance, l is the Galactic longitude, and b is the
        Galactic latitude.
    t : numpy.ndarray
        A 1D array of shape (num_steps,) containing the time steps.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    # xf_gal columns: [r, ℓ, b]
    r = xf_gal[:, 0]
    ell = xf_gal[:, 1]
    b = xf_gal[:, 2]

    # If angles look like radians, convert to degrees
    if (np.nanmax(np.abs(ell)) <= 2 * np.pi + 1e-6 and
            np.nanmax(np.abs(b)) <= np.pi / 2 + 1e-6):
        ell = np.degrees(ell)
        b = np.degrees(b)

    # Wrap longitude to [0, 360)
    ell = np.mod(ell, 360.0)

    # Mask finite values
    mask = np.isfinite(r) & np.isfinite(ell) & np.isfinite(b) & np.isfinite(t)
    r, ell, b = r[mask], ell[mask], b[mask]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=False)

    # Shared color normalisation
    axes[0].plot(r, ell)
    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$\ell\ [^\circ]$")

    axes[1].plot(r, b)
    axes[1].set_xlabel(r"$r$")
    axes[1].set_ylabel(r"$b\ [^\circ]$")

    fig.tight_layout()
    plt.close()

    return fig, axes


def plot_multiple_trajectories_galactic(trajectories, sigmas, intg,
                                        observer_location, input_frame,
                                        panel_height=3, plot_attractors=True):
    """
    Visualizes multiple trajectories with different smoothing scales in
    Galactic coordinates.

    Parameters
    ----------
    trajectories : list of tuple
        A list of trajectory results from `follow_multiple_smoothing`.
    sigmas : list of float
        The list of smoothing scales corresponding to the trajectories.
    intg : TrajectoryFollower
        The TrajectoryFollower instance used to generate the trajectories.
    observer_location : numpy.ndarray
        The 3D position of the observer.
    input_frame : str
        The Astropy frame of the input Cartesian coordinates.
    panel_height : float, optional
        The height of each individual panel in inches. Default is 3.
    plot_attractors : bool, optional
        If True, plot the positions of Virgo, Great Attractor, and Shapley
        superclusters. Default is True.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    n_sigmas = len(sigmas)
    fig, axes = plt.subplots(n_sigmas, 2, figsize=(8, panel_height * n_sigmas),
                             sharex=True, sharey='col')

    if n_sigmas == 1:
        axes = np.array([axes])  # Make it 2D for consistent indexing

    for i, (t, xf, vmag) in enumerate(trajectories):
        xf_gal = intg.to_galactic(xf, observer_location, input_frame)

        r = xf_gal[:, 0]
        ell = xf_gal[:, 1]
        b = xf_gal[:, 2]

        axes[i, 0].plot(r, ell)
        axes[i, 1].plot(r, b)

    if plot_attractors:
        attractors = {
            'Virgo': (12, 284, 74),
            'Great Attractor': (49, 308, 29),
            'Shapley': (138, 312.5, 30.3)
        }
        offset_y = 5  # Adjust this value as needed for proper spacing

        for i, (name, (r, l, b)) in enumerate(attractors.items()):
            for j in range(n_sigmas):
                # Plot circle
                axes[j, 0].plot(r, l, 'o', color='black', markersize=5)
                axes[j, 1].plot(r, b, 'o', color='black', markersize=5)

                # Add name above circle
                axes[j, 0].text(r, l + offset_y, name, color='black',
                                ha='center', va='bottom', fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                          lw=0, alpha=0.7))
                axes[j, 1].text(r, b + offset_y, name, color='black',
                                ha='center', va='bottom', fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                          lw=0, alpha=0.7))

    for i in range(n_sigmas):
        axes[i, 0].set_ylabel(r"$\ell ~ [^\circ]$")
        axes[i, 1].set_ylabel(r"$b ~ [^\circ]$")

        secax = axes[i, 1].secondary_yaxis('right')
        if sigmas[i] == 0:
            sigma_label = "No smoothing"
        else:
            sigma_label = (fr"$\sigma = {sigmas[i]} h^{{-1}} \mathrm{{Mpc}}$")
        secax.set_ylabel(sigma_label)
        secax.set_ticks([])

        if i == n_sigmas - 1:  # Only for the last row
            axes[i, 0].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")
            axes[i, 1].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")

    fig.tight_layout()
    plt.close()

    return fig, axes
