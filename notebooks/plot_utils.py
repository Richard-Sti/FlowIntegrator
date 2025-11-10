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
    r, ell, b, tc = r[mask], ell[mask], b[mask], t[mask]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=False)

    # Shared color normalisation
    axes[0].scatter(r, ell, c=tc, s=10, alpha=0.8, cmap="viridis")
    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$\ell\ [^\circ]$")
    axes[0].set_title(r"Longitude vs Radius")

    axes[1].scatter(r, b, c=tc, s=10, alpha=0.8, cmap="viridis")
    axes[1].set_xlabel(r"$r$")
    axes[1].set_ylabel(r"$b\ [^\circ]$")
    axes[1].set_title(r"Latitude vs Radius")

    # Add a single shared colorbar
    cbar = fig.colorbar(axes[1].collections[0], ax=axes.ravel().tolist(),
                        shrink=0.85, pad=0.03)
    cbar.set_label("Time")

    fig.tight_layout()
    plt.close()

    return fig, axes
