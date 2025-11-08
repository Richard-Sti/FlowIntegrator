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
This module provides the `Integrator` class, which is responsible for
evolving test particles in a 3D velocity field using a JAX-based
Runge-Kutta 2nd order (RK2) integrator with an adaptive spatial step.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from .utils import fprint


def _get_velocity(positions, v_field, map_coords_kwargs):
    """
    Get velocities at particle positions using interpolation.
    """
    coords = positions.T
    kwargs = dict(map_coords_kwargs)
    vx = map_coordinates(v_field[..., 0], coords, **kwargs)
    vy = map_coordinates(v_field[..., 1], coords, **kwargs)
    vz = map_coordinates(v_field[..., 2], coords, **kwargs)
    return jnp.stack([vx, vy, vz], axis=-1)


def _rk2_step(positions_carry, _, v_field, ds, map_coords_kwargs, epsilon):
    """
    Performs one RK2 step using an adaptive spatial step.
    """
    # --- RK2 Step 1 (k1) ---
    v1 = _get_velocity(positions_carry, v_field, map_coords_kwargs)
    speeds1 = jnp.sqrt(jnp.sum(v1**2, axis=-1, keepdims=True))
    dt_vector1 = ds / (speeds1 + epsilon)
    p_mid = positions_carry + v1 * 0.5 * dt_vector1

    # --- RK2 Step 2 (k2) ---
    v2 = _get_velocity(p_mid, v_field, map_coords_kwargs)
    speeds2 = jnp.sqrt(jnp.sum(v2**2, axis=-1, keepdims=True))
    dt_vector2 = ds / (speeds2 + epsilon)

    # --- Final Update ---
    final_positions = positions_carry + v2 * dt_vector2
    return final_positions, None


class Integrator:
    """
    Handles the evolution of test particles in a 3D velocity field.

    Parameters
    ----------
    v_field : jax.Array
        The 3D velocity field.
    num_steps : int
        The number of integration steps.
    ds : float
        The spatial step length.
    epsilon : float, optional
        Small value added to the denominator when calculating the
        per-particle time step (`dt_vector`). This prevents division
        by zero for particles with near-zero velocity, effectively
        stopping them from moving further. Default: 1e-6.
    **map_coords_kwargs
        Keyword arguments for `map_coordinates`.
    """
    def __init__(self, v_field, num_steps, ds, epsilon=1e-6,
                 **map_coords_kwargs):
        self.v_field = v_field
        self.num_steps = num_steps
        self.ds = ds
        self.epsilon = epsilon
        if not map_coords_kwargs:
            self.map_coords_kwargs = {
                'order': 1, 'mode': 'constant', 'cval': 0.0
            }
        else:
            self.map_coords_kwargs = map_coords_kwargs

    def run(self, initial_positions):
        """
        Runs the full integration.
        """
        fprint("Compiling and running JIT-compiled integration...")

        # Convert kwargs dict to a hashable, static tuple for JIT
        map_kwargs_tuple = tuple(self.map_coords_kwargs.items())

        # Create a partially applied step function
        step_fn = partial(
            _rk2_step,
            v_field=self.v_field,
            ds=self.ds,
            map_coords_kwargs=map_kwargs_tuple,
            epsilon=self.epsilon
        )

        # JIT-compile the step function, marking the kwargs as static
        jitted_step_fn = jax.jit(
            step_fn, static_argnames=('map_coords_kwargs', 'epsilon')
        )

        dummy_scan_array = jnp.zeros(self.num_steps)
        final_positions, _ = jax.lax.scan(
            jitted_step_fn,
            initial_positions,
            dummy_scan_array
        )

        final_positions.block_until_ready()
        fprint("Integration complete.")
        return final_positions
