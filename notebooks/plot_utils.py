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
from scipy.stats import gaussian_kde
from posterior_agreement import compute_agreement
from tqdm import trange
from astropy.coordinates import (ICRS, Galactic,
                                 CartesianRepresentation,
                                 CartesianDifferential)
import astropy.units as u

import flowi

COLS = [
    "#083d77ff",  # regal-navy
    "#db162fff",  # flag-red
    "#1b998bff",  # verdigris
    "#440d0fff",  # rich-mahogany
    "#f0f757ff",  # canary-yellow
]


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
                                  panel_height=3., plot_attractors=True,
                                  downsample=10):
    """
    Visualizes trajectories from multiple field realizations, overplotting
    all smoothing scales on shared axes with distinct colors.

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
        Height of the figure in inches. Default is 2.
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
    with plt.style.context("science"):
        colors = np.asarray(COLS)
        plot_order = np.argsort(smoothing_scales)[::-1]  # largest sigma first
        r_min = np.inf
        fig, axes = plt.subplots(
            1, 2, figsize=(9, panel_height), sharex=True, sharey='col'
        )

        for rank, sigma_idx in enumerate(plot_order):
            sigma = smoothing_scales[sigma_idx]
            sigma_val = int(sigma) if float(sigma).is_integer() else sigma
            label = "No smoothing" if sigma == 0 else (
                fr"$\sigma = {sigma_val} ~ h^{{-1}} \mathrm{{Mpc}}$"
            )
            color = colors[sigma_idx % colors.size]
            zorder = rank  # largest smoothing at lowest z-order

            for traj_idx, trajectories in enumerate(realization_trajectories):
                # `trajectories` is the output of `follow_multiple_smoothing`.
                # It's a list of (t, xf, vmag) for each sigma.
                t, xf, vmag = trajectories[sigma_idx]

                if downsample is not None and downsample > 1:
                    xf = xf[::downsample]

                xf_gal = intg.to_galactic(xf, observer_location, input_frame)

                r = xf_gal[:, 0]
                ell = xf_gal[:, 1]
                b = xf_gal[:, 2]
                r_min = min(r_min, np.nanmin(r))

                axes[0].plot(
                    r, ell, color=color, alpha=0.6, lw=0.6,
                    label=label if traj_idx == 0 else None,
                    zorder=zorder
                )
                axes[1].plot(
                    r, b, color=color, alpha=0.6, lw=0.6, zorder=zorder
                )

        if plot_attractors:
            attractors = {
                'Virgo': (12, 284, 74),
                'Great Attractor': (49, 308, 29),
                'Shapley': (138, 312.5, 30.3)
            }
            offset_y = 5
            offset_y_left = 8
            offset_y_left_down = -15
            offset_y_right_down = -10

            for i, (name, (r, l, b)) in enumerate(attractors.items()):
                axes[0].plot(r, l, 'o', color='black', markersize=5)
                axes[1].plot(r, b, 'o', color='black', markersize=5)

                axes[0].text(
                    r,
                    l + (offset_y_left_down
                         if name == "Great Attractor"
                         else offset_y_left),
                    name, color='black',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              lw=0, alpha=0.7)
                )
                axes[1].text(
                    r,
                    b + (offset_y_right_down
                         if name == "Great Attractor"
                         else offset_y),
                    name, color='black',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              lw=0, alpha=0.7)
                )

        axes[0].set_ylabel(r"$\ell ~ [^\circ]$")
        axes[1].set_ylabel(r"$b ~ [^\circ]$")
        axes[0].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")
        axes[1].set_xlabel(r"$r ~ [h^{-1} \mathrm{Mpc}]$")
        if np.isfinite(r_min):
            axes[0].set_xlim(left=r_min)
        legend = axes[0].legend(loc="best")
        for line in legend.get_lines():
            line.set_linewidth(1.4)

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


def plot_ga_enclosed_mass(filepaths, fields=None, labels=None):
    """
    Stack enclosed GA mass profiles across one or more files and plot median
    with 16/84 percentiles, comparing to random pointings when available.

    Parameters
    ----------
    filepaths : str, pathlib.Path, or sequence
        One or more HDF5 files produced by ga_enclosed_mass.py.
    fields : array-like of int, optional
        Only plot the specified field indices (matching HDF5 groups
        `field_<idx>`). If None, plot all fields.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    if not isinstance(filepaths, (list, tuple, np.ndarray)):
        filepaths = [filepaths]
    filepaths = [Path(fp) for fp in filepaths]
    if labels is None:
        labels = [None] * len(filepaths)
    if len(labels) != len(filepaths):
        raise ValueError("labels must match length of filepaths")
    for fp in filepaths:
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

    if fields is not None:
        fields = np.asarray(fields, dtype=int)
        field_names = {f"field_{int(f)}" for f in fields}
    else:
        field_names = None

    actual_per_file = []
    random_samples = []
    found_field_names = set()
    radii = None
    for fp, label in zip(filepaths, labels):
        file_actual = []
        with h5py.File(fp, "r") as h5f:
            if "radii" not in h5f:
                raise KeyError(f"Dataset `radii` not found in file: {fp}")
            fp_radii = np.asarray(h5f["radii"], dtype=float)
            if radii is None:
                radii = fp_radii
            else:
                if (radii.shape != fp_radii.shape or
                        not np.allclose(radii, fp_radii)):
                    raise ValueError("Radii mismatch across input files.")

            for key in sorted(h5f.keys()):
                if key == "radii" or not key.startswith("field_"):
                    continue
                if field_names is not None and key not in field_names:
                    continue

                grp = h5f[key]
                mass_enclosed = np.asarray(grp["mass_enclosed"], dtype=float)
                file_actual.append(mass_enclosed[None, :])
                found_field_names.add(key)
                if "mass_random" in grp:
                    mass_random = np.asarray(grp["mass_random"], dtype=float)
                    random_samples.append(mass_random)
        if file_actual:
            actual_per_file.append((np.concatenate(file_actual, axis=0),
                                    label))

    if field_names is not None:
        missing = field_names - found_field_names
        if missing:
            raise KeyError(f"Requested fields not found: {sorted(missing)}")

    if not actual_per_file:
        raise ValueError("No mass datasets found in the provided file.")

    if random_samples:
        random = np.concatenate(random_samples, axis=0)
        q3_r, q5_r, q16_r, q84_r, q95_r, q997_r = np.percentile(
            random, [0.135, 5, 16, 84, 95, 99.865], axis=0
        )
    else:
        q3_r = q5_r = q16_r = q84_r = q95_r = q997_r = None

    with plt.style.context("science"):
        fig, ax = plt.subplots()
        colors = np.asarray(COLS)
        for i, (actual, label) in enumerate(actual_per_file):
            q16_a, q50_a, q84_a = np.percentile(actual, [16, 50, 84], axis=0)
            color = colors[i % colors.size]
            base_label = label or "GA"
            ax.fill_between(radii, q16_a, q84_a, color=color,
                            alpha=0.5, label=f"{base_label}")

        if q16_r is not None:
            # Plot all bands but only label the 1σ region
            ax.fill_between(radii, q3_r, q997_r, color="tab:gray",
                            alpha=0.15)
            ax.fill_between(radii, q5_r, q95_r, color="tab:gray",
                            alpha=0.25)
            ax.fill_between(radii, q16_r, q84_r, color="tab:gray",
                            alpha=0.4, label="Random")
        ax.set_xlabel(r"$r~[h^{-1} \mathrm{Mpc}]$")
        ax.set_ylabel(r"$M(<r)~[h^{-1} \mathrm{M}_{\odot}]$")
        ax.legend(fontsize="small", loc="upper left")

        ax.set_xlim(radii.min(), radii.max())
        ax.set_ylim(3e14)
        ax.set_xscale("log")
        ax.set_yscale("log")

        fig.tight_layout()
        plt.close()

    return fig, ax


def load_ga_metrics(filepath, fields=None, value="ga_mass"):
    """
    Load GA mass or volume as a function of smoothing scale for matched
    entries.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the GA_analysis.hdf5 file.
    fields : array-like of int, optional
        Only include these field indices. If None, include all.
    value : {'ga_mass', 'ga_volume'}
        Which quantity to return.

    Returns
    -------
    sigmas : numpy.ndarray
        Smoothing scales (float) for which at least one match exists.
    vals : numpy.ndarray
        Array of shape (n_fields, n_sigmas) with the requested metric.
        Entries with no match are NaN.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if value not in {"ga_mass", "ga_volume"}:
        raise ValueError("value must be 'ga_mass' or 'ga_volume'")

    if fields is not None:
        fields = np.asarray(fields, dtype=int)
        field_names = {f"field_{int(f)}" for f in fields}
    else:
        field_names = None

    with h5py.File(filepath, "r") as h5f:
        smoothing_scales = np.asarray(h5f.attrs["smoothing_scales"],
                                      dtype=float)
        all_fields = sorted(
            int(k.split("_")[1]) for k in h5f.keys() if k.startswith("field_")
        )
        if field_names is not None:
            chosen_fields = [
                f for f in all_fields if f"field_{f}" in field_names]
        else:
            chosen_fields = all_fields

        vals = np.full((len(chosen_fields), smoothing_scales.size),
                       np.nan, dtype=float)

        for i, fid in enumerate(chosen_fields):
            field_key = f"field_{fid}"
            field_grp = h5f.get(field_key, None)
            if field_grp is None:
                continue
            for j, sigma in enumerate(smoothing_scales):
                sigma_key = f"sigma_{sigma}"
                if sigma_key not in field_grp:
                    continue
                sg = field_grp[sigma_key]
                if not sg.attrs.get("matched", False):
                    continue
                if value not in sg.attrs:
                    continue
                vals[i, j] = float(sg.attrs[value])

    return smoothing_scales, vals


def plot_ga_positions(filepaths, box_size, r_min=None, kde=False):
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
    kde : bool, optional
        If True, plot 1D KDEs instead of histograms.

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
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
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
                    sigma_val = float(parts[-1])
                    label = (f"$\\sigma={sigma_val}\\ h^{{-1}}\\ "
                             r"\\mathrm{{Mpc}}$")
                except ValueError:
                    label = path.stem
            else:
                label = path.stem

            color = colors[i % len(colors)]
            if kde:
                kde_r = gaussian_kde(r)
                kde_l = gaussian_kde(ell)
                kde_b = gaussian_kde(b)
                x_r = np.linspace(r.min(), r.max(), 256)
                x_l = np.linspace(ell.min(), ell.max(), 256)
                x_b = np.linspace(b.min(), b.max(), 256)
                axes[0].plot(x_r, kde_r(x_r), color=color, label=label)
                axes[1].plot(x_l, kde_l(x_l), color=color)
                axes[2].plot(x_b, kde_b(x_b), color=color)
            else:
                kwargs = dict(bins="auto", color=color,
                              histtype='stepfilled', alpha=0.5)
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
        axes[0].set_ylabel("Density" if kde else "Counts per bin")
        axes[0].legend(loc='upper left')

        axes[1].set_xlabel(r"$\ell ~ [^\circ]$")
        axes[1].set_ylabel("")

        axes[2].set_xlabel(r"$b ~ [^\circ]$")
        axes[2].set_ylabel("")

        fig.tight_layout()
        plt.close()
        return fig, axes


def galactic_to_cartesian(v, ell_deg, b_deg):
    """Convert Galactic spherical velocity to Cartesian."""
    ell = np.deg2rad(ell_deg)
    b = np.deg2rad(b_deg)

    vx = v * np.cos(b) * np.cos(ell)
    vy = v * np.cos(b) * np.sin(ell)
    vz = v * np.sin(b)

    return np.stack([vx, vy, vz], axis=-1)


def cartesian_to_galactic(vxyz):
    """Convert Cartesian velocity to Galactic spherical."""
    vx, vy, vz = vxyz.T
    v = np.sqrt(vx**2 + vy**2 + vz**2)

    ell = np.rad2deg(np.arctan2(vy, vx)) % 360.0
    b = np.rad2deg(np.arcsin(vz / v))

    return v, ell, b


def load_observer_velocity_convergence(filepath):
    """
    Load and process observer velocity convergence data from HDF5.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to linear_velocity_fields.hdf5.

    Returns
    -------
    dict
        Dictionary with keys:
        - radii: array of radii
        - v_obs: (n_fields, n_radii, 3) observer velocities vs radius
        - v_obs_full: (n_fields, 3) observer velocities from full field
        - v_obs_vmag, v_obs_ell, v_obs_b: Galactic spherical coords vs radius
        - v_obs_full_vmag, v_obs_full_ell, v_obs_full_b: full field Galactic
        - mean_vmag, std_vmag, mean_ell, std_ell, mean_b, std_b: statistics
        - boxsize, Omega_m, smooth_scale: metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading observer velocity convergence from: {filepath}")

    with h5py.File(filepath, "r") as f:
        n_fields = len([k for k in f.keys() if k.startswith("field_")])
        radii = f["radii"][:]
        n_radii = len(radii)
        cumulative = f.attrs.get("cumulative", True)

        print(f"Found {n_fields} fields")
        print(f"Radii: {n_radii} values from {radii.min():.1f} "
              f"to {radii.max():.1f} Mpc/h")
        print(f"Cumulative: {cumulative}")

        Omega_m = f.attrs["Omega_m"]
        smooth_scale = f.attrs["smooth_scale"]
        boxsize = f.attrs["boxsize"]

        # Check if pre-computed observer velocities exist
        has_precomputed = "observer_velocity_vs_radius" in f["field_0"]

        if not has_precomputed:
            raise ValueError(
                "File does not contain pre-computed observer velocities. "
                "Regenerate with compute_radius_analysis=True in "
                "linear_velocity_field.py")

        if cumulative:
            v_obs = np.zeros((n_fields, n_radii, 3))
        else:
            v_obs = np.zeros((n_fields, n_radii - 1, 3))

        for i in trange(n_fields, desc="Loading observer velocities"):
            v_obs[i] = f[f"field_{i}/observer_velocity_vs_radius"][:]
        print(f"Loaded pre-computed observer velocities: shape {v_obs.shape}")

    print(f"Metadata: Ω_m={Omega_m}, smooth_scale={smooth_scale} Mpc/h, "
          f"boxsize={boxsize} Mpc/h")

    print("\nLoading center velocities...")
    v_obs_full = np.zeros((n_fields, 3))
    with h5py.File(filepath, "r") as f:
        for i in trange(n_fields, desc="Loading center velocities"):
            v_obs_full[i] = f[f"field_{i}/velocity_center"][:]

    print(f"Extracted full field velocities: shape {v_obs_full.shape}")

    def icrs_velocity_to_galactic_spherical(v_icrs):
        original_shape = v_icrs.shape[:-1]
        v_icrs = v_icrs.reshape(-1, 3)

        vmag = np.zeros(len(v_icrs))
        ell = np.zeros(len(v_icrs))
        b = np.zeros(len(v_icrs))

        for i in range(len(v_icrs)):
            vel_diff = CartesianDifferential(
                v_icrs[i, 0] * u.km/u.s,
                v_icrs[i, 1] * u.km/u.s,
                v_icrs[i, 2] * u.km/u.s
            )
            pos_rep = CartesianRepresentation(0*u.kpc, 0*u.kpc, 0*u.kpc)
            icrs = ICRS(pos_rep.with_differentials(vel_diff))
            gal = icrs.transform_to(Galactic())

            vx_gal = gal.velocity.d_x.to(u.km/u.s).value
            vy_gal = gal.velocity.d_y.to(u.km/u.s).value
            vz_gal = gal.velocity.d_z.to(u.km/u.s).value

            vmag[i] = np.sqrt(vx_gal**2 + vy_gal**2 + vz_gal**2)
            ell[i] = np.arctan2(vy_gal, vx_gal) * 180 / np.pi
            vxy = np.sqrt(vx_gal**2 + vy_gal**2)
            b[i] = np.arctan2(vz_gal, vxy) * 180 / np.pi

        vmag = vmag.reshape(original_shape)
        ell = ell.reshape(original_shape)
        b = b.reshape(original_shape)
        ell = (ell + 360) % 360

        return vmag, ell, b

    print("\nConverting to Galactic spherical coordinates...")
    v_obs_vmag, v_obs_ell, v_obs_b = (
        icrs_velocity_to_galactic_spherical(v_obs))
    v_obs_full_vmag, v_obs_full_ell, v_obs_full_b = (
        icrs_velocity_to_galactic_spherical(v_obs_full))

    print("\nGalactic spherical (full field):")
    print(f"  |v| = {v_obs_full_vmag.mean():.1f} ± "
          f"{v_obs_full_vmag.std():.1f} km/s")
    print(f"  l   = {v_obs_full_ell.mean():.1f} ± "
          f"{v_obs_full_ell.std():.1f} deg")
    print(f"  b   = {v_obs_full_b.mean():.1f} ± {v_obs_full_b.std():.1f} deg")

    mean_vmag = v_obs_vmag.mean(axis=0)
    std_vmag = v_obs_vmag.std(axis=0)
    mean_ell = v_obs_ell.mean(axis=0)
    std_ell = v_obs_ell.std(axis=0)
    mean_b = v_obs_b.mean(axis=0)
    std_b = v_obs_b.std(axis=0)

    print("\nComputed statistics over all fields and radii")
    print("Data loaded successfully!\n")

    return {
        'radii': radii,
        'v_obs': v_obs,
        'v_obs_full': v_obs_full,
        'v_obs_vmag': v_obs_vmag,
        'v_obs_ell': v_obs_ell,
        'v_obs_b': v_obs_b,
        'v_obs_full_vmag': v_obs_full_vmag,
        'v_obs_full_ell': v_obs_full_ell,
        'v_obs_full_b': v_obs_full_b,
        'mean_vmag': mean_vmag,
        'std_vmag': std_vmag,
        'mean_ell': mean_ell,
        'std_ell': std_ell,
        'mean_b': mean_b,
        'std_b': std_b,
        'boxsize': boxsize,
        'Omega_m': Omega_m,
        'smooth_scale': smooth_scale,
    }


def plot_observer_velocity_convergence(data, R_max=160.0, output_file=None):
    """
    Plot observer velocity convergence with reference values.

    Parameters
    ----------
    data : dict
        Output from load_observer_velocity_convergence.
    R_max : float, optional
        Maximum radius to plot. Default: 160.0 Mpc/h.
    output_file : str or pathlib.Path, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    radii = data['radii']
    mean_vmag = data['mean_vmag']
    std_vmag = data['std_vmag']
    mean_ell = data['mean_ell']
    std_ell = data['std_ell']
    mean_b = data['mean_b']
    std_b = data['std_b']

    mask = radii < R_max
    R_ref = R_max
    R_vec = R_ref + 3.75

    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), sharex=True)

        panel_data = [
            (mean_vmag, std_vmag, r'$V_{\rm inner}~[\mathrm{km\,s^{-1}}]$'),
            (mean_ell, std_ell, r'$\ell~[\mathrm{deg}]$'),
            (mean_b, std_b, r'$b~[\mathrm{deg}]$'),
        ]

        for ax, (mean, std, ylabel) in zip(axes, panel_data):
            ax.errorbar(radii[mask], mean[mask], yerr=std[mask],
                        fmt='o', ms=2, capsize=3, color=COLS[0],
                        label=r'$\mathrm{Manticore \ (linear)}$')
            ax.set_ylabel(ylabel)
            ax.set_xlabel(r'$R~[h^{-1}\,\mathrm{Mpc}]$')

        # Use values from data dict if available, otherwise use defaults
        cmb = data.get('cmb', np.array([620.0, 271.9, 29.6]))
        cmb_err = data.get('cmb_err', np.array([15.0, 2.0, 1.4]))

        inner = data.get('inner', np.array([443.377, 229.613, 42.318]))
        inner_err = data.get('inner_err', np.array([56.843, 13.095, 9.939]))

        rng = np.random.default_rng(42)
        n_samp = 50_000
        sigma_v = 150.0

        v0 = galactic_to_cartesian(*inner)
        dv = rng.normal(scale=sigma_v, size=(n_samp, 3))
        v_samp = v0[None, :] + dv

        v_mag, ell, b = cartesian_to_galactic(v_samp)
        vec_mu = np.array([v_mag.mean(), ell.mean(), b.mean()])
        vec_std = np.array([v_mag.std(), ell.std(), b.std()])

        # CMB-LG as horizontal bands
        for i, (cmb_val, cmb_err_val) in enumerate(zip(cmb, cmb_err)):
            axes[i].axhspan(cmb_val - cmb_err_val,
                            cmb_val + cmb_err_val,
                            color=COLS[1], alpha=0.2, zorder=2,
                            label=r'$\mathrm{CMB–LG \ velocity}$' if i == 0
                            else "_nolegend_")

        axes[0].errorbar(R_ref, inner[0], yerr=inner_err[0],
                         fmt='s', ms=2, capsize=3, color=COLS[2],
                         zorder=7, label=r'$\mathrm{Manticore \ (obs)}$')
        axes[1].errorbar(R_ref, inner[1], yerr=inner_err[1],
                         fmt='s', ms=2, capsize=3, color=COLS[2],
                         zorder=7)
        axes[2].errorbar(R_ref, inner[2], yerr=inner_err[2],
                         fmt='s', ms=2, capsize=3, color=COLS[2],
                         zorder=7)

        axes[0].errorbar(R_vec, vec_mu[0], yerr=vec_std[0],
                         fmt='D', ms=2, capsize=3, color=COLS[3],
                         zorder=8,
                         label=r'$\mathrm{Manticore \ (obs} + \sigma_v)$')
        axes[1].errorbar(R_vec, vec_mu[1], yerr=vec_std[1],
                         fmt='D', ms=2, capsize=3, color=COLS[3],
                         zorder=8)
        axes[2].errorbar(R_vec, vec_mu[2], yerr=vec_std[2],
                         fmt='D', ms=2, capsize=3, color=COLS[3],
                         zorder=8)

        for ax in axes:
            ax.set_xlim(0.0, R_vec + 0.05 * R_max)

        x0 = 41.3
        dx_minus = 4.7
        dx_plus = 2.0

        for i, ax in enumerate(axes):
            ax.axvspan(x0 - dx_minus, x0 + dx_plus, color='gray',
                       alpha=0.5,
                       label=r'$\mathrm{cGA}$' if i == 0 else "_nolegend_")

        handles, labels = axes[0].get_legend_handles_labels()
        handles = handles[1:] + handles[:1]
        labels = labels[1:] + labels[:1]

        fig.legend(handles, labels, loc='upper center', ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        if output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")

        plt.close()

    # Compute ratio and angular offset between Manticore (obs) and CMB-LG
    n_mc = 100_000
    rng_mc = np.random.default_rng(42)

    # Sample CMB-LG (shared for both cases)
    cmb_v_samples = rng_mc.normal(cmb[0], cmb_err[0], n_mc)
    cmb_ell_samples = rng_mc.normal(cmb[1], cmb_err[1], n_mc)
    cmb_b_samples = rng_mc.normal(cmb[2], cmb_err[2], n_mc)

    cmb_cart_samples = np.zeros((n_mc, 3))
    for i in range(n_mc):
        cmb_cart_samples[i] = galactic_to_cartesian(
            cmb_v_samples[i], cmb_ell_samples[i], cmb_b_samples[i]
        )

    def compute_comparison(label, v_cart_samples):
        """Compute magnitude ratio and angular offset."""
        v_mag = np.linalg.norm(v_cart_samples, axis=1)
        cmb_mag = np.linalg.norm(cmb_cart_samples, axis=1)

        # Magnitude ratio
        ratio_samples = v_mag / cmb_mag
        ratio_med = np.percentile(ratio_samples, 50)
        ratio_low = np.percentile(ratio_samples, 16)
        ratio_high = np.percentile(ratio_samples, 84)

        # Angular offset
        dot_products = np.sum(cmb_cart_samples * v_cart_samples, axis=1)
        cos_theta = dot_products / (cmb_mag * v_mag)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta_samples = np.rad2deg(np.arccos(cos_theta))

        theta_med = np.percentile(theta_samples, 50)
        theta_low = np.percentile(theta_samples, 16)
        theta_high = np.percentile(theta_samples, 84)

        print(f"\n{label}:")
        print(f"  Magnitude ratio: {ratio_med:.3f} "
              f"-{ratio_med - ratio_low:.3f}/+{ratio_high - ratio_med:.3f}")
        print(f"  Angular offset:  {theta_med:.2f} "
              f"-{theta_med - theta_low:.2f}/"
              f"+{theta_high - theta_med:.2f} deg")

    print("\n" + "="*60)
    print("Comparison vs CMB-LG dipole")
    print("="*60)

    # Case 1: Manticore (obs) without sigma_v
    inner_v_samples = rng_mc.normal(inner[0], inner_err[0], n_mc)
    inner_ell_samples = rng_mc.normal(inner[1], inner_err[1], n_mc)
    inner_b_samples = rng_mc.normal(inner[2], inner_err[2], n_mc)

    inner_cart_samples = np.zeros((n_mc, 3))
    for i in range(n_mc):
        inner_cart_samples[i] = galactic_to_cartesian(
            inner_v_samples[i], inner_ell_samples[i], inner_b_samples[i]
        )
    compute_comparison("Manticore (obs)", inner_cart_samples)

    # Case 2: Manticore (obs) with sigma_v
    v0 = galactic_to_cartesian(*inner)
    dv = rng_mc.normal(scale=sigma_v, size=(n_mc, 3))
    inner_sigmav_cart_samples = v0[None, :] + dv
    compute_comparison(f"Manticore (obs + σ_v={sigma_v} km/s)",
                       inner_sigmav_cart_samples)

    print("="*60 + "\n")

    return fig, axes


def plot_observer_velocity_shells(data, R_max=200.0, output_file=None):
    """
    Plot observer velocity contributions from radial shells.

    Parameters
    ----------
    data : dict
        Output from load_observer_velocity_convergence (with cumulative=False).
    R_max : float, optional
        Maximum radius to plot. Default: 200.0 Mpc/h.
    output_file : str or pathlib.Path, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    radii = data['radii']
    mean_vmag = data['mean_vmag']
    std_vmag = data['std_vmag']
    mean_ell = data['mean_ell']
    std_ell = data['std_ell']
    mean_b = data['mean_b']
    std_b = data['std_b']

    # Compute bin centers
    bin_centers = 0.5 * (radii[:-1] + radii[1:])
    mask = bin_centers < R_max

    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), sharex=True)

        panel_data = [
            (mean_vmag, std_vmag,
             r'$|\mathbf{V}_{\rm shell}|~[\mathrm{km\,s^{-1}}]$'),
            (mean_ell, std_ell, r'$\ell~[\mathrm{deg}]$'),
            (mean_b, std_b, r'$b~[\mathrm{deg}]$'),
        ]

        for ax, (mean, std, ylabel) in zip(axes, panel_data):
            ax.errorbar(bin_centers[mask], mean[mask], yerr=std[mask],
                        fmt='o', ms=3, capsize=3, color=COLS[0],
                        label='Linear theory')
            ax.set_ylabel(ylabel)
            ax.set_xlabel(r'$R~[h^{-1}\,\mathrm{Mpc}]$')

        # cGA region
        x0 = 41.3
        dx_minus = 4.7
        dx_plus = 2.0

        for i, ax in enumerate(axes):
            ax.axvspan(x0 - dx_minus, x0 + dx_plus, color='gray',
                       alpha=0.5,
                       label=r'$\mathrm{cGA}$' if i == 0 else "_nolegend_")
            ax.set_xlim(0, R_max)

        axes[0].legend(loc='upper right')

        plt.tight_layout()

        if output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")

        plt.close()

    return fig, axes


def plot_velocity_convergence(filepath, vmag_range=None, sigma_v=None,
                              theta_ylim=3):
    """
    Plot velocity convergence as a function of radius: amplitude
    ratio and angular misalignment.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to .npz file with keys: radii, v_full, v_radii.
        - radii: 1D array of radii
        - v_full: (n_sims, 3) array of velocities at full box
        - v_radii: (n_sims, n_radii, 3) array of velocities
          vs radius
    vmag_range : tuple of float, optional
        If provided, filter simulations to only those with velocity
        magnitude within (vmin, vmax). If None, include all
        simulations.
    sigma_v : float or list of float, optional
        If provided, add 3D velocity uncertainty to v_full. Each
        component (vx, vy, vz) gets a random perturbation drawn from
        N(0, sigma_v). Units: km/s. If None, no perturbation is added.
        If a list is provided, multiple bands will be plotted, one for
        each sigma_v value.
    theta_ylim : float, optional
        Lower y-limit for the angular misalignment plot (right
        panel). Upper limit is fixed at 180 deg. Default: 3 deg.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plots.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    f = np.load(filepath)
    R = f["radii"]
    v_full = f["v_full"]
    v_radii = f["v_radii"]
    n_total = len(v_full)

    if vmag_range is not None:
        vmin, vmax = vmag_range
        v_full_mag = np.linalg.norm(v_full, axis=-1)
        mask = (v_full_mag > vmin) & (v_full_mag < vmax)
        v_full = v_full[mask]
        v_radii = v_radii[mask]
        n_selected = len(v_full)
        pct = 100 * n_selected / n_total
        print(f"vmag_range = ({vmin}, {vmax}) km/s: "
              f"{n_selected}/{n_total} ({pct:.1f}%)")

    # Store original v_full for multiple sigma_v cases
    v_full_orig = v_full.copy()

    # Convert sigma_v to list for uniform handling
    if sigma_v is None:
        sigma_v_list = [None]
    elif isinstance(sigma_v, (list, tuple, np.ndarray)):
        sigma_v_list = list(sigma_v)
        if len(sigma_v_list) > 2:
            raise ValueError("At most 2 sigma_v values are allowed")
    else:
        sigma_v_list = [sigma_v]

    # Compute statistics for each sigma_v
    results = []
    for sv in sigma_v_list:
        if sv is not None:
            rng = np.random.default_rng(42)
            dv = rng.normal(0, sv, size=v_full_orig.shape)
            v_full_pert = v_full_orig + dv
            print(f"Added 3D velocity uncertainty: σ_v = {sv} km/s")
        else:
            v_full_pert = v_full_orig

        vR_mag = np.linalg.norm(v_radii, axis=-1)
        vbox_mag = np.linalg.norm(v_full_pert, axis=-1)

        # Projection: v_inner · v_box / |v_box|^2
        dot_product = np.sum(v_radii * v_full_pert[:, None, :], axis=-1)
        vbox_mag_sq = vbox_mag[:, None]**2
        projection_ratio = dot_product / vbox_mag_sq

        # Cosine for angle calculation
        cos_theta = dot_product / (vR_mag * vbox_mag[:, None])

        amplitude_ratio = vR_mag / vbox_mag[:, None]

        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_deg = np.degrees(np.arccos(cos_theta))

        amp_low2, _, amp_high2 = np.percentile(
            amplitude_ratio, [2.5, 50, 97.5], axis=0)
        amp_low1, amp_med, amp_high1 = np.percentile(
            amplitude_ratio, [16, 50, 84], axis=0)

        th_low2, _, th_high2 = np.percentile(
            theta_deg, [2.5, 50, 97.5], axis=0)
        th_low1, th_med, th_high1 = np.percentile(
            theta_deg, [16, 50, 84], axis=0)

        # Projection ratio statistics
        proj_low2, _, proj_high2 = np.percentile(
            projection_ratio, [2.5, 50, 97.5], axis=0)
        proj_low1, proj_med, proj_high1 = np.percentile(
            projection_ratio, [16, 50, 84], axis=0)

        results.append({
            'sigma_v': sv,
            'amp_low2': amp_low2,
            'amp_high2': amp_high2,
            'amp_low1': amp_low1,
            'amp_med': amp_med,
            'amp_high1': amp_high1,
            'th_low2': th_low2,
            'th_high2': th_high2,
            'th_low1': th_low1,
            'th_med': th_med,
            'th_high1': th_high1,
        })

    with plt.style.context("science"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))

        # Plot each sigma_v case
        for i, res in enumerate(results):
            color = COLS[i % len(COLS)]
            sv = res['sigma_v']
            if sv is not None:
                label = rf"$\sigma_v = {sv}~\mathrm{{km\,s^{{-1}}}}$"
            else:
                label = r"$\mathrm{No} ~ \sigma_v$"

            if i == 0:
                # First case: filled bands, higher zorder
                # Panel 0: Amplitude ratio
                axes[0].fill_between(R, res['amp_low2'], res['amp_high2'],
                                     color=color, alpha=0.2, zorder=3)
                axes[0].fill_between(R, res['amp_low1'], res['amp_high1'],
                                     color=color, alpha=0.4, label=label,
                                     zorder=3)
                axes[0].plot(R, res['amp_med'], color=color, linestyle="--",
                             zorder=4)

                # Panel 1: Angular misalignment
                axes[1].fill_between(R, res['th_low2'], res['th_high2'],
                                     color=color, alpha=0.2, zorder=3)
                axes[1].fill_between(R, res['th_low1'], res['th_high1'],
                                     color=color, alpha=0.4, zorder=3)
                axes[1].plot(R, res['th_med'], color=color, linestyle="--",
                             zorder=4)
            else:
                # Second case: outlines only (1σ and 2σ), lower zorder
                # Panel 0: Amplitude ratio
                axes[0].plot(R, res['amp_low2'], color=color, linestyle='-',
                             linewidth=1, alpha=0.3, zorder=2)
                axes[0].plot(R, res['amp_high2'], color=color, linestyle='-',
                             linewidth=1, alpha=0.3, zorder=2)
                axes[0].plot(R, res['amp_low1'], color=color, linestyle='-',
                             linewidth=1, alpha=0.7, zorder=2)
                axes[0].plot(R, res['amp_high1'], color=color, linestyle='-',
                             linewidth=1, alpha=0.7, zorder=2)
                axes[0].plot(R, res['amp_med'], color=color, linestyle="--",
                             label=label, zorder=2)

                # Panel 1: Angular misalignment
                axes[1].plot(R, res['th_low2'], color=color, linestyle='-',
                             linewidth=1, alpha=0.3, zorder=2)
                axes[1].plot(R, res['th_high2'], color=color, linestyle='-',
                             linewidth=1, alpha=0.3, zorder=2)
                axes[1].plot(R, res['th_low1'], color=color, linestyle='-',
                             linewidth=1, alpha=0.7, zorder=2)
                axes[1].plot(R, res['th_high1'], color=color, linestyle='-',
                             linewidth=1, alpha=0.7, zorder=2)
                axes[1].plot(R, res['th_med'], color=color, linestyle="--",
                             zorder=2)

        spine_lw = axes[0].spines['bottom'].get_linewidth()

        # Panel 0: Amplitude ratio
        axes[0].axhline(1, color='black', linestyle='--',
                        linewidth=spine_lw, zorder=1)
        axes[0].axvline(150, color='black', linestyle='--',
                        linewidth=spine_lw, zorder=1)
        axes[0].set_xlabel(r"$R~[h^{-1}\,\mathrm{Mpc}]$")
        axes[0].set_ylabel(
            r"$|\mathbf{V}_{\rm inner}(R)| / |\mathbf{V}_{\rm box}|$")
        axes[0].set_ylim(0)
        axes[0].set_xlim(R.min(), R.max())
        if len(results) > 1:
            axes[0].legend(loc='best')

        # Panel 1: Angular misalignment
        axes[1].axvline(150, color='black', linestyle='--',
                        linewidth=spine_lw, zorder=1)
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$R~[h^{-1}\,\mathrm{Mpc}]$")
        axes[1].set_ylabel(r"$\theta(R)~[\mathrm{deg}]$")
        axes[1].set_ylim(theta_ylim, 180)
        axes[1].set_xlim(R.min(), R.max())

        fig.tight_layout()
        plt.close()

    # Print statistics for each case
    idx_150 = np.argmin(np.abs(R - 150.0))
    R_150 = R[idx_150]

    for res in results:
        sv = res['sigma_v']
        sv_str = f"σ_v = {sv} km/s" if sv is not None else "No σ_v"

        amp_med_150 = res['amp_med'][idx_150]
        amp_err_minus = amp_med_150 - res['amp_low1'][idx_150]
        amp_err_plus = res['amp_high1'][idx_150] - amp_med_150

        th_med_150 = res['th_med'][idx_150]
        th_err_minus = th_med_150 - res['th_low1'][idx_150]
        th_err_plus = res['th_high1'][idx_150] - th_med_150

        print(f"\nAt R = {R_150:.1f} Mpc/h ({sv_str}):")
        print(f"  |v_inner|/|v_box| = {amp_med_150:.3f} "
              f"-{amp_err_minus:.3f}/+{amp_err_plus:.3f}")
        print(f"  θ = {th_med_150:.2f} "
              f"-{th_err_minus:.2f}/+{th_err_plus:.2f} deg")

    return fig, axes


def plot_velocity_convergence_scatter(filepath, data, radius, sigma_v=None,
                                      reference_radius=None, vmag_range=None,
                                      gaussianize_manticore=False):
    """
    Scatter plot of magnitude ratio vs angular offset at a given radius.

    Shows two distributions:
    1. Blue contour: Quijote random simulations - ratio of velocity at `radius`
       relative to full box velocity.
    2. Red contour: Manticore - ratio of velocity at `reference_radius` (or
       Manticore observer velocity if not provided) relative to CMB dipole.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to .npz file with Quijote simulation data:
        - radii: 1D array of radii
        - v_full: (n_sims, 3) velocities at full box
        - v_radii: (n_sims, n_radii, 3) velocities vs radius
    data : dict
        Output from load_observer_velocity_convergence containing:
        - radii: array of radii
        - v_obs: (n_fields, n_radii, 3) velocities vs radius from Manticore
        - cmb: (3,) CMB-LG velocity [vmag, ell, b]
        - cmb_err: (3,) CMB-LG errors
        - inner: (3,) Manticore obs velocity [vmag, ell, b]
        - inner_err: (3,) Manticore obs errors
    radius : float
        Radius in Mpc/h at which to evaluate for the Quijote randoms.
    sigma_v : float or list of float, optional
        If provided, propagate 3D velocity uncertainty:
        - For Quijote: add to observer velocity (center of box)
        - For Manticore: add to Manticore (linear) velocity before ratio
        If a list is provided, create multiple panels (one per sigma_v value).
    reference_radius : float, optional
        If provided, use Manticore velocity at this radius from `data['v_obs']`
        instead of Manticore observer velocity (`inner`).
    vmag_range : tuple of float, optional
        If provided, filter Quijote simulations to only those with full box
        velocity magnitude within (vmin, vmax) km/s.
    gaussianize_manticore : bool, optional
        If True, fit a 2D Gaussian to the Manticore (ratio, log10(theta))
        distribution and use samples from that Gaussian for plotting.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Convert sigma_v to list
    if sigma_v is None:
        sigma_v_list = [None]
    elif isinstance(sigma_v, (list, tuple, np.ndarray)):
        sigma_v_list = list(sigma_v)
    else:
        sigma_v_list = [sigma_v]

    n_panels = len(sigma_v_list)

    # Load Quijote data
    f = np.load(filepath)
    R = f["radii"]
    v_full_orig = f["v_full"]
    v_radii_orig = f["v_radii"]
    n_total = len(v_full_orig)

    # Filter based on full box velocity magnitude
    if vmag_range is not None:
        vmin, vmax = vmag_range
        v_full_mag = np.linalg.norm(v_full_orig, axis=-1)
        mask = (v_full_mag >= vmin) & (v_full_mag <= vmax)
        v_full_orig = v_full_orig[mask]
        v_radii_orig = v_radii_orig[mask]
        n_selected = len(v_full_orig)
        pct = 100 * n_selected / n_total
        print(f"Quijote: vmag_range = ({vmin}, {vmax}) km/s: "
              f"{n_selected}/{n_total} ({pct:.1f}%)")

    # Find closest radius index for Quijote
    idx_r = np.argmin(np.abs(R - radius))
    R_actual = R[idx_r]
    print(f"Quijote: Using R = {R_actual:.1f} Mpc/h (requested {radius})")

    v_at_r = v_radii_orig[:, idx_r, :]

    # Manticore setup (common across panels)
    cmb = data.get('cmb', np.array([620.0, 271.9, 29.6]))
    cmb_err = data.get('cmb_err', np.array([15.0, 2.0, 1.4]))
    inner = data.get('inner', np.array([443.377, 229.613, 42.318]))
    inner_err = data.get('inner_err', np.array([56.843, 13.095, 9.939]))

    n_mc = 100_000

    # Get Manticore velocity (pre-computed Galactic coords)
    if reference_radius is not None:
        radii_manticore = data['radii']
        idx_ref = np.argmin(np.abs(radii_manticore - reference_radius))
        R_ref_actual = radii_manticore[idx_ref]
        print(f"Manticore: Using R = {R_ref_actual:.1f} Mpc/h "
              f"(requested {reference_radius})")

        v_mag_manticore = data['v_obs_vmag'][:, idx_ref]
        v_ell_manticore = data['v_obs_ell'][:, idx_ref]
        v_b_manticore = data['v_obs_b'][:, idx_ref]
        print(f"Manticore (linear) at R={R_ref_actual:.1f} Mpc/h "
              f"({len(v_mag_manticore)} realizations):")
        print(f"  |V| = {np.mean(v_mag_manticore):.1f} ± "
              f"{np.std(v_mag_manticore):.1f} km/s")
        print(f"  ell = {np.mean(v_ell_manticore):.1f} ± "
              f"{np.std(v_ell_manticore):.1f} deg")
        print(f"  b   = {np.mean(v_b_manticore):.1f} ± "
              f"{np.std(v_b_manticore):.1f} deg")

        v_manticore_gal_base = np.array([
            galactic_to_cartesian(vm, el, bb)
            for vm, el, bb in zip(v_mag_manticore, v_ell_manticore,
                                  v_b_manticore)])
        manticore_label = r"Manticore"
    else:
        print("Manticore (obs):")
        print(f"  |V| = {inner[0]:.1f} ± {inner_err[0]:.1f} km/s")
        print(f"  ell = {inner[1]:.1f} ± {inner_err[1]:.1f} deg")
        print(f"  b   = {inner[2]:.1f} ± {inner_err[2]:.1f} deg")
        v_manticore_gal_base = None
        manticore_label = r"Manticore"

    def find_contour_levels(x, y, fractions):
        """Find KDE contour levels enclosing given fractions of data."""
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x = x[valid]
        y = y[valid]

        log_y = np.log10(y)
        xy_data = np.vstack([x, log_y])

        try:
            kde = gaussian_kde(xy_data)
        except np.linalg.LinAlgError:
            jitter_x = 1e-6 * np.std(x) * np.random.randn(len(x))
            jitter_y = 1e-6 * np.std(log_y) * np.random.randn(len(log_y))
            xy_data = np.vstack([x + jitter_x, log_y + jitter_y])
            kde = gaussian_kde(xy_data)

        x_grid = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
        y_grid = np.linspace(log_y.min() - 0.2, log_y.max() + 0.2, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)

        Z_flat = Z.ravel()
        Z_sorted = np.sort(Z_flat)[::-1]
        Z_cumsum = np.cumsum(Z_sorted) / Z_sorted.sum()

        levels = []
        for frac in fractions:
            idx = np.searchsorted(Z_cumsum, frac)
            levels.append(Z_sorted[min(idx, len(Z_sorted) - 1)])

        return X, Y, Z, sorted(levels)

    # Compute distributions for each sigma_v
    results = []
    for sv in sigma_v_list:
        rng = np.random.default_rng(42)
        rng_mc = np.random.default_rng(42)

        # Quijote
        if sv is not None:
            dv = rng.normal(0, sv, size=v_full_orig.shape)
            v_full_pert = v_full_orig + dv
        else:
            v_full_pert = v_full_orig.copy()

        vR_mag = np.linalg.norm(v_at_r, axis=-1)
        vfull_mag = np.linalg.norm(v_full_pert, axis=-1)
        quijote_ratio = vR_mag / vfull_mag

        dot_product = np.sum(v_at_r * v_full_pert, axis=-1)
        cos_theta = np.clip(dot_product / (vR_mag * vfull_mag), -1.0, 1.0)
        quijote_theta = np.degrees(np.arccos(cos_theta))

        # Sample CMB-LG
        cmb_v_samples = rng_mc.normal(cmb[0], cmb_err[0], n_mc)
        cmb_ell_samples = rng_mc.normal(cmb[1], cmb_err[1], n_mc)
        cmb_b_samples = rng_mc.normal(cmb[2], cmb_err[2], n_mc)
        cmb_cart = np.array([
            galactic_to_cartesian(v, ell, b) for v, ell, b
            in zip(cmb_v_samples, cmb_ell_samples, cmb_b_samples)])
        cmb_mag = np.linalg.norm(cmb_cart, axis=1)

        # Manticore
        if reference_radius is not None:
            v_manticore_gal = v_manticore_gal_base.copy()
            if sv is not None:
                dv = rng_mc.normal(0, sv, size=v_manticore_gal.shape)
                v_manticore_gal = v_manticore_gal + dv

            v_manticore_mag = np.linalg.norm(v_manticore_gal, axis=-1)
            n_fields = len(v_manticore_gal)
            indices = rng_mc.choice(n_fields, size=n_mc, replace=True)
            v_manticore_resampled = v_manticore_gal[indices]
            v_manticore_mag_resampled = v_manticore_mag[indices]

            manticore_ratio = v_manticore_mag_resampled / cmb_mag
            dot_products = np.sum(cmb_cart * v_manticore_resampled, axis=1)
            cos_theta_m = np.clip(
                dot_products / (cmb_mag * v_manticore_mag_resampled), -1, 1)
            manticore_theta = np.rad2deg(np.arccos(cos_theta_m))
        else:
            if sv is not None:
                v0 = galactic_to_cartesian(*inner)
                dv = rng_mc.normal(scale=sv, size=(n_mc, 3))
                inner_cart = v0[None, :] + dv
            else:
                inner_v_samples = rng_mc.normal(inner[0], inner_err[0], n_mc)
                inner_ell_samples = rng_mc.normal(inner[1], inner_err[1], n_mc)
                inner_b_samples = rng_mc.normal(inner[2], inner_err[2], n_mc)
                inner_cart = np.array([
                    galactic_to_cartesian(v, ell, b) for v, ell, b
                    in zip(inner_v_samples, inner_ell_samples, inner_b_samples)
                ])

            inner_mag = np.linalg.norm(inner_cart, axis=1)
            manticore_ratio = inner_mag / cmb_mag
            dot_products = np.sum(cmb_cart * inner_cart, axis=1)
            cos_theta_m = np.clip(dot_products / (cmb_mag * inner_mag), -1, 1)
            manticore_theta = np.rad2deg(np.arccos(cos_theta_m))

        # Gaussianize if requested
        if gaussianize_manticore:
            valid = (np.isfinite(manticore_ratio) &
                     np.isfinite(manticore_theta) & (manticore_theta > 0))
            ratio_valid = manticore_ratio[valid]
            log_theta_valid = np.log10(manticore_theta[valid])

            mean = np.array([np.mean(ratio_valid), np.mean(log_theta_valid)])
            cov = np.cov(ratio_valid, log_theta_valid)

            rng_gauss = np.random.default_rng(42)
            samples = rng_gauss.multivariate_normal(
                mean, cov, size=len(ratio_valid))
            manticore_ratio = samples[:, 0]
            manticore_theta = 10**samples[:, 1]

        # Label for this sigma_v
        if sv is not None:
            sv_label = rf"$\sigma_v = {sv} ~ \mathrm{{km \, s^{{-1}}}}$"
        else:
            sv_label = r"$\mathrm{No} ~ \sigma_v$"

        results.append({
            'sigma_v': sv,
            'sv_label': sv_label,
            'quijote_ratio': quijote_ratio,
            'quijote_theta': quijote_theta,
            'manticore_ratio': manticore_ratio,
            'manticore_theta': manticore_theta,
        })

    # Print statistics
    for res in results:
        print("\n" + "="*60)
        print(f"{res['sv_label']}")
        print("-"*60)
        print("Quijote randoms:")
        print(f"  Magnitude ratio: {np.median(res['quijote_ratio']):.3f} ± "
              f"{np.std(res['quijote_ratio']):.3f}")
        print(f"  Angular offset:  {np.median(res['quijote_theta']):.2f} ± "
              f"{np.std(res['quijote_theta']):.2f} deg")
        print(f"\n{manticore_label}:")
        print(f"  Magnitude ratio: {np.median(res['manticore_ratio']):.3f} ± "
              f"{np.std(res['manticore_ratio']):.3f}")
        print(f"  Angular offset:  {np.median(res['manticore_theta']):.2f} ± "
              f"{np.std(res['manticore_theta']):.2f} deg")
        print("="*60)

    # Plotting
    with plt.style.context("science"):
        default_width, default_height = plt.rcParams['figure.figsize']
        if n_panels == 1:
            fig, axes = plt.subplots(
                figsize=(default_width, default_height))
            axes = [axes]
        else:
            fig, axes = plt.subplots(
                n_panels, 1,
                figsize=(default_width, 0.85 * default_height * n_panels),
                sharex=True)

        color_quijote = COLS[0]
        color_manticore = COLS[1]

        for i, (res, ax) in enumerate(zip(results, axes)):
            # Quijote contours (filled)
            X, Y, Z, levels = find_contour_levels(
                res['quijote_ratio'], res['quijote_theta'], [0.68, 0.95])
            ax.contourf(X, 10**Y, Z, levels=[levels[0], levels[1], Z.max()],
                        colors=[color_quijote], alpha=[0.2, 0.4])
            ax.contour(X, 10**Y, Z, levels=levels, colors=[color_quijote],
                       linewidths=1, linestyles=['-', '--'])

            # Manticore contours (red, unfilled)
            X_m, Y_m, Z_m, levels_m = find_contour_levels(
                res['manticore_ratio'], res['manticore_theta'], [0.68, 0.95])
            ax.contour(X_m, 10**Y_m, Z_m, levels=levels_m,
                       colors=[color_manticore], linewidths=1.5,
                       linestyles=['-', '--'], zorder=10)

            # Set axis limits based on 2σ contours
            mask_q = Z >= levels[0]
            mask_m = Z_m >= levels_m[0]

            x_min = min(X[mask_q].min(), X_m[mask_m].min())
            x_max = max(X[mask_q].max(), X_m[mask_m].max())
            y_min = min(Y[mask_q].min(), Y_m[mask_m].min())
            y_max = max(Y[mask_q].max(), Y_m[mask_m].max())

            x_pad = 0.15 * (x_max - x_min)
            y_pad = 0.15 * (y_max - y_min)

            ax.axvline(1, color='black', linestyle='--',
                       linewidth=ax.spines['bottom'].get_linewidth())
            ax.set_ylabel(r"$\mathrm{Angular \ alignment} ~ [\mathrm{deg}]$")
            ax.set_yscale("log")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(10**(y_min - y_pad), 10**(y_max + y_pad))

            # Panel label
            ax.text(0.95, 0.95, res['sv_label'], transform=ax.transAxes,
                    ha='right', va='top')

            # Legend only on first panel
            if i == 0:
                ax.plot([], [], color=color_quijote,
                        label=r"$\mathrm{Random}$")
                ax.plot([], [], color=color_manticore,
                        label=r"$\mathrm{Manticore}$")
                ax.legend(loc='lower left')

            # Posterior agreement
            chain_quijote = np.column_stack(
                [res['quijote_ratio'], np.log10(res['quijote_theta'])])
            chain_manticore = np.column_stack(
                [res['manticore_ratio'], np.log10(res['manticore_theta'])])
            agreement = compute_agreement([chain_quijote, chain_manticore])
            print(f"{res['sv_label']}: Posterior agreement = "
                  f"{agreement.sigma:.2f}σ")

        # Only set xlabel on last panel
        axes[-1].set_xlabel(r"$\mathrm{Magnitude \ ratio}$")

        fig.tight_layout()
        plt.close()

    return fig, axes


def plot_observer_vmag_histogram(filepath):
    """
    Plot a KDE of observer velocity magnitudes with CMB reference.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to .npz file with key v_full: (n_sims, 3) array of velocities.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    f = np.load(filepath)
    v_full = f["v_full"]
    vmag = np.linalg.norm(v_full, axis=-1)

    with plt.style.context("science"):
        fig, ax = plt.subplots()

        # KDE
        kde = gaussian_kde(vmag)
        x = np.linspace(0, vmag.mean() + 4 * vmag.std(), 512)
        ax.plot(x, kde(x), color=COLS[0])

        # CMB-LG velocity band: 620 ± 15 km/s
        ax.axvspan(620 - 15, 620 + 15, color=COLS[1], alpha=0.3,
                   label=r"$\mathrm{CMB–LG}$")
        ax.legend(loc='upper right')

        ax.set_xlabel(
            r"$|\mathbf{V}_{\rm box}|~[\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax.set_ylabel(r"$\mathrm{Probability \ density}$")
        ax.set_xlim(0, vmag.mean() + 4 * vmag.std())
        ax.set_ylim(0)

        fig.tight_layout()
        plt.close()

    return fig, ax
