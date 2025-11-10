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
This module provides utility functions for the FlowIntegrator package,
including a timestamped print function and functions for loading and
preparing data.
"""

import datetime
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as ndi


def fprint(*args, verbose=True, **kwargs):
    """
    Print a message with a timestamp prepended.

    Parameters
    ----------
    *args
        Variable length argument list.
    verbose : bool, optional
        If True, print the message. Default: True.
    **kwargs
        Arbitrary keyword arguments.
    """
    if verbose:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp}", *args, **kwargs)


def create_initial_positions(box_size, resolution, N=None,
                             observer_location=None, max_distance=None):
    """
    Create initial particle positions on a grid.

    By default, it places one particle at the center of each cell of
    the velocity field grid (`resolution`). If `N` is specified, it
    creates a grid of `N`^3 uniformly spaced particles.

    Parameters
    ----------
    box_size : float
        The size of the simulation box (e.g., 256.0).
    resolution : int
        The resolution of the velocity field grid.
    N : int, optional
        The resolution of the particle grid. If None, it defaults to
        `resolution`. Default: None.
    observer_location : jax.Array, optional
        A 3D JAX array (shape (3,)) representing the observer's position.
        If provided along with `max_distance`, only particles within
        `max_distance` from this location will be returned. Default: None.
    max_distance : float, optional
        The maximum distance from the `observer_location` to include particles.
        If provided along with `observer_location`, only particles within
        this distance will be returned. Default: None.

    Returns
    -------
    jax.Array
        An array of initial particle positions on the JAX device.
    """
    if N is None:
        N = resolution

    cell_size = box_size / N
    # Create coordinates for the center of each cell
    coords = jnp.linspace(cell_size / 2, box_size - cell_size / 2, N)
    x, y, z = jnp.meshgrid(coords, coords, coords, indexing='ij')

    initial_positions = jnp.stack(
        [x.ravel(), y.ravel(), z.ravel()], axis=-1
    ).astype(jnp.float32)

    if observer_location is not None and max_distance is not None:
        if observer_location.shape != (3,):
            raise ValueError(
                "observer_location must be a 1D array of shape (3,)"
            )
        distances = jnp.sqrt(
            jnp.sum((initial_positions - observer_location)**2, axis=-1)
        )
        initial_positions = initial_positions[distances <= max_distance]
        fprint(f"Filtered to {initial_positions.shape[0]} particles "
               f"within {max_distance} of observer.")

    fprint(f"Initialized {initial_positions.shape[0]} particles on device.")
    return initial_positions


def smooth_velocity_field_gaussian(v_field, box_size, sigma):
    """
    Smooths a 3D velocity field using a Gaussian kernel.

    Applies a Gaussian filter to each component (vx, vy, vz) of the
    velocity field independently.

    Parameters
    ----------
    v_field : numpy.ndarray
        The 3D velocity field as a NumPy array with shape
        (3, resolution, resolution, resolution).
    box_size : float
        The size of the simulation box in physical units (Mpc / h).
    sigma : float
        Standard deviation for Gaussian kernel in physical units (Mpc / h).

    Returns
    -------
    numpy.ndarray
        The smoothed velocity field as a NumPy array.
    """
    # v_field is expected to be a NumPy array

    # Get resolution from v_field shape
    resolution = v_field.shape[1]

    # Convert sigma from physical units (Mpc/h) to grid units (pixels)
    sigma_pixels = sigma * resolution / box_size

    smoothed_v_field_np = np.empty_like(v_field)

    # Apply Gaussian filter to each component
    for i in range(v_field.shape[0]):
        smoothed_v_field_np[i] = ndi.gaussian_filter(
            v_field[i], sigma=sigma_pixels, mode='wrap'
        )

    return smoothed_v_field_np
