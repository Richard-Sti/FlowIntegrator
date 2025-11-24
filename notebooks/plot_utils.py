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

import h5py
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa
from pathlib import Path

import flowi


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


def plot_multiple_trajectories_galactic(
    trajectories, smoothing_scales, intg, observer_location,
    input_frame, panel_height=3, plot_attractors=True
):
    """
    Visualizes multiple trajectories with different smoothing scales in
    Galactic coordinates.

    Parameters
    ----------
    trajectories : list of tuple
        A list of trajectory results from `follow_multiple_smoothing`.
    smoothing_scales : list of float
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
    n_sigmas = len(smoothing_scales)
    fig, axes = plt.subplots(
        n_sigmas, 2, figsize=(8, panel_height * n_sigmas),
        sharex=True, sharey='col'
    )

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
            'Great Attractor': (49, 325, -7),
            'Shapley': (138, 312.5, 30.3)
        }
        offset_y = 5  # Adjust this value as needed for proper spacing

        for i, (name, (r, l, b)) in enumerate(attractors.items()):
            for j in range(n_sigmas):
                # Plot circle
                axes[j, 0].plot(r, l, 'o', color='black', markersize=5)
                axes[j, 1].plot(r, b, 'o', color='black', markersize=5)

                # Add name above circle
                axes[j, 0].text(
                    r, l + offset_y, name, color='black',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              lw=0, alpha=0.7)
                )
                axes[j, 1].text(
                    r, b + offset_y, name, color='black',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              lw=0, alpha=0.7)
                )

    for i in range(n_sigmas):
        axes[i, 0].set_ylabel(r"$\ell ~ [^\circ]$")
        axes[i, 1].set_ylabel(r"$b ~ [^\circ]$")

        secax = axes[i, 1].secondary_yaxis('right')
        if smoothing_scales[i] == 0:
            sigma_label = "No smoothing"
        else:
            sigma_label = (
                fr"$\sigma = {smoothing_scales[i]} h^{{-1}} \mathrm{{Mpc}}$"
            )
        secax.set_ylabel(sigma_label)
        secax.set_ticks([])

        if i == n_sigmas - 1:  # Only for the last row
            axes[i, 0].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")
            axes[i, 1].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")

    fig.tight_layout()
    plt.close()

    return fig, axes


def plot_realization_trajectories(realization_trajectories,
                                  smoothing_scales, intg,
                                  observer_location, input_frame,
                                  panel_height=3, plot_attractors=True,
                                  downsample=10):
    """
    Visualizes trajectories from multiple field realizations.

    Parameters
    ----------
    realization_trajectories : list of list of tuple
        A list where each element is the output of
        `follow_multiple_smoothing` for a single realization.
    smoothing_scales : list of float
        The list of smoothing scales.
    intg : TrajectoryFollower
        The TrajectoryFollower instance.
    observer_location : numpy.ndarray
        The 3D position of the observer.
    input_frame : str
        The Astropy frame of the input Cartesian coordinates.
    panel_height : float, optional
        The height of each individual panel. Default is 3.
    plot_attractors : bool, optional
        If True, plot the positions of attractors. Default is True.
    downsample : int, optional
        Plot every nth trajectory point to reduce plotting load. If None,
        no downsampling is applied. Default is 10.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    n_sigmas = len(smoothing_scales)
    fig, axes = plt.subplots(
        n_sigmas, 2, figsize=(8, panel_height * n_sigmas),
        sharex=True, sharey='col'
    )

    if n_sigmas == 1:
        axes = np.array([axes])

    for i, sigma in enumerate(smoothing_scales):
        ax_row = axes[i]
        for trajectories in realization_trajectories:
            # `trajectories` is the output of `follow_multiple_smoothing`.
            # It's a list of (t, xf, vmag) for each sigma.
            # Get the trajectory for the current sigma
            sigma_idx = smoothing_scales.index(sigma)
            t, xf, vmag = trajectories[sigma_idx]

            if downsample is not None and downsample > 1:
                xf = xf[::downsample]

            xf_gal = intg.to_galactic(xf, observer_location, input_frame)

            r = xf_gal[:, 0]
            ell = xf_gal[:, 1]
            b = xf_gal[:, 2]

            ax_row[0].plot(r, ell, color='black', alpha=0.5, lw=0.5)
            ax_row[1].plot(r, b, color='black', alpha=0.5, lw=0.5)

    if plot_attractors:
        attractors = {
            'Virgo': (12, 284, 74),
            'Great Attractor': (49, 308, 29),
            'Shapley': (138, 312.5, 30.3)
        }
        offset_y = 5

        for i, (name, (r, l, b)) in enumerate(attractors.items()):
            for j in range(n_sigmas):
                axes[j, 0].plot(r, l, 'o', color='black', markersize=5)
                axes[j, 1].plot(r, b, 'o', color='black', markersize=5)

                axes[j, 0].text(
                    r, l + offset_y, name, color='black',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              lw=0, alpha=0.7)
                )
                axes[j, 1].text(
                    r, b + offset_y, name, color='black',
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              lw=0, alpha=0.7)
                )

    for i in range(n_sigmas):
        axes[i, 0].set_ylabel(r"$\ell ~ [^\circ]$")
        axes[i, 1].set_ylabel(r"$b ~ [^\circ]$")

        secax = axes[i, 1].secondary_yaxis('right')
        if smoothing_scales[i] == 0:
            sigma_label = "No smoothing"
        else:
            sigma_label = (
                fr"$\sigma = {smoothing_scales[i]} h^{{-1}} \mathrm{{Mpc}}$"
            )
        secax.set_ylabel(sigma_label)
        secax.set_ticks([])

        if i == n_sigmas - 1:
            axes[i, 0].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")
            axes[i, 1].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")

    fig.tight_layout()
    plt.close()

    return fig, axes


def _format_sigma_key(sigma):
    sigma_float = float(sigma)
    if sigma_float.is_integer():
        sigma_str = str(int(sigma_float))
    else:
        sigma_str = str(sigma_float).rstrip("0").rstrip(".")
    return f"sigma_{sigma_str}"


def plot_mw_streamlines(filepath, smoothing_scales, input_frame='icrs',
                        downsample=10, fields=None):
    """
    Plots the results of the MW_streamlines.py script.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the HDF5 file with the results.
    smoothing_scales : list of float
        A list of smoothing scales to plot.
    input_frame : str, optional
        The Astropy frame of the input Cartesian coordinates.
        Default is 'icrs'.
    downsample : int, optional
        Plot every nth trajectory point to reduce plotting load. If None,
        no downsampling is applied. Default is 10.
    fields : array-like of int, optional
        Only load the specified field indices (matching HDF5 groups
        `field_<idx>`). If None, load all fields.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    smoothing_scales = np.asarray(smoothing_scales, dtype=float)
    if downsample is None:
        downsample = 1
    else:
        downsample = int(downsample)
        if downsample < 1:
            raise ValueError("downsample must be >= 1")
    if fields is not None:
        fields = np.asarray(fields, dtype=int)
        field_names = {f"field_{int(f)}" for f in fields}
    else:
        field_names = None

    with h5py.File(filepath, "r") as h5f:
        available_scales = np.asarray(h5f.attrs["smoothing_scales"],
                                      dtype=float)
        has_match = np.isclose(
            smoothing_scales[:, None], available_scales[None, :]
        ).any(axis=1)
        if not np.all(has_match):
            missing = smoothing_scales[~has_match]
            raise ValueError(
                f"Invalid smoothing scales requested. "
                f"Requested missing: {missing}. "
                f"Available scales: {available_scales}"
            )

        realization_trajectories = []
        box_size = None
        ds = None
        num_steps = int(h5f.attrs.get("num_steps", 1))
        found_fields = set()
        for field_key in h5f.keys():
            if not field_key.startswith("field_"):
                continue
            if field_names is not None and field_key not in field_names:
                continue
            found_fields.add(field_key)

            field_grp = h5f[field_key]
            trajectories = []
            for sigma in smoothing_scales:
                sigma_key = _format_sigma_key(sigma)
                if sigma_key not in field_grp:
                    raise KeyError(
                        f"{sigma_key} not found in {field_key}. "
                        "Use available smoothing scales from file."
                    )

                sigma_grp = field_grp[sigma_key]
                t = sigma_grp["time"][::downsample]
                x = sigma_grp["trajectory"][::downsample]
                v = sigma_grp["speed"][::downsample]
                trajectories.append((t, x, v))
                if box_size is None:
                    box_size = float(sigma_grp.attrs["box_size"])
                if ds is None:
                    ds = float(sigma_grp.attrs.get("ds", 1.0))
            realization_trajectories.append(trajectories)

    if field_names is not None:
        missing_fields = field_names - found_fields
        if missing_fields:
            raise KeyError(
                f"Requested fields not found: {sorted(missing_fields)}")

    if box_size is None:
        raise ValueError("No trajectories found in the provided file.")
    if ds is None:
        ds = 1.0

    smoothing_scales_list = smoothing_scales.tolist()

    # Mock a TrajectoryFollower instance to use the to_galactic method
    # A dummy velocity field is sufficient.
    dummy_velocity_field = np.zeros((3, 2, 2, 2))
    intg = flowi.TrajectoryFollower(
        dummy_velocity_field, box_size, num_steps=num_steps, ds=ds
    )
    observer_location = np.full(3, box_size / 2)

    return plot_realization_trajectories(
        realization_trajectories,
        smoothing_scales_list,
        intg,
        observer_location,
        input_frame,
        downsample=downsample,
    )


def plot_ga_positions(filepaths, box_size, r_min=None):
    """
    Plot GA positions (r, ell, b) from a text file of endpoints.

    Parameters
    ----------
    filepaths : str, pathlib.Path, or sequence
        One or more text files produced by `find_ga_from_streamlines.py`
        containing columns [field] x y z (or x y z).
    box_size : float
        Simulation box size (h^-1 Mpc). Observer is assumed at box_size / 2.
    r_min : float, optional
        If provided, only plot entries with r > r_min.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    if not isinstance(filepaths, (list, tuple, np.ndarray)):
        filepaths = [filepaths]

    datasets = []
    for fp in filepaths:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] >= 4:
            positions = data[:, 1:4]
        elif data.shape[1] == 3:
            positions = data
        else:
            raise ValueError(
                "Text input must have columns [field] x y z or x y z"
            )
        datasets.append((path, positions))

    center = np.full(3, box_size / 2)

    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        colors = plt.cm.tab10.colors

        for i, (path, positions) in enumerate(datasets):
            r, ell, b = flowi.cartesian_icrs_to_galactic_spherical(
                positions, center
            )
            if r_min is not None:
                m = r > r_min
                r, ell, b = r[m], ell[m], b[m]

            # Try to parse sigma from filename suffix: ..._sigma_<val>.txt
            parts = path.stem.split("sigma_")
            if len(parts) > 1:
                try:
                    label = f"$\\sigma={float(parts[-1])}\\ h^{{-1}}\\ \\mathrm{{Mpc}}$"
                except ValueError:
                    label = path.stem
            else:
                label = path.stem

            color = colors[i % len(colors)]
            kwargs = dict(bins="auto", color=color, histtype='stepfilled',
                          alpha=0.5)
            axes[0].hist(r, **kwargs, label=label)
            axes[1].hist(ell, **kwargs)
            axes[2].hist(b, **kwargs)

            p16_r, med_r, p84_r = np.percentile(r, [16, 50, 84])
            p16_l, med_l, p84_l = np.percentile(ell, [16, 50, 84])
            p16_b, med_b, p84_b = np.percentile(b, [16, 50, 84])
            err_r_minus, err_r_plus = med_r - p16_r, p84_r - med_r
            err_l_minus, err_l_plus = med_l - p16_l, p84_l - med_l
            err_b_minus, err_b_plus = med_b - p16_b, p84_b - med_b
            print(
                f"{label}: "
                f"r = {med_r:.3g} -{err_r_minus:.3g}/+{err_r_plus:.3g}; "
                f"ell = {med_l:.3g} -{err_l_minus:.3g}/+{err_l_plus:.3g}; "
                f"b = {med_b:.3g} -{err_b_minus:.3g}/+{err_b_plus:.3g}"
            )

        axes[0].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")
        axes[0].set_ylabel("Counts per bin")
        axes[0].legend()

        axes[1].set_xlabel(r"$\ell ~ [^\circ]$")
        axes[1].set_ylabel("")

        axes[2].set_xlabel(r"$b ~ [^\circ]$")
        axes[2].set_ylabel("")

        fig.tight_layout()
        plt.close()
        return fig, axes
