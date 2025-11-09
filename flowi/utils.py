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


def create_initial_positions(box_size, resolution, N=None):
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

    fprint(f"Initialized {initial_positions.shape[0]} particles on device.")
    return initial_positions
