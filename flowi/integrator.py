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
from tqdm.auto import tqdm

from .utils import fprint
from .clustering import find_attractors_from_convergence


def _get_velocity(positions, v_field, map_coords_kwargs, box_size):
    """
    Get velocities at particle positions using interpolation.
    Converts physical coordinates to pixel coordinates for map_coordinates.
    """
    resolution = v_field.shape[1]
    # Convert physical coordinates to pixel-center-based coordinates
    pixel_coords = (positions / box_size) * resolution - 0.5
    coords = pixel_coords.T
    kwargs = dict(map_coords_kwargs)
    vx = map_coordinates(v_field[0], coords, **kwargs)
    vy = map_coordinates(v_field[1], coords, **kwargs)
    vz = map_coordinates(v_field[2], coords, **kwargs)
    return jnp.stack([vx, vy, vz], axis=-1)


def _rk2_step(positions_carry, _, v_field, ds, map_coords_kwargs, epsilon,
              box_size):
    """
    Performs one RK2 step using an adaptive spatial step. Applies periodic
    boundary conditions.
    """
    # --- RK2 Step 1 (k1) ---
    v1 = _get_velocity(positions_carry, v_field, map_coords_kwargs, box_size)
    speeds1 = jnp.sqrt(jnp.sum(v1**2, axis=-1, keepdims=True))
    # dt_vector1: This is a "time-like" quantity. If ds is in Mpc/h and speeds
    # are in km/s, then dt_vector1 has units of (Mpc/h) / (km/s).
    # This implicitly includes the conversion factor between Mpc and km, and
    # h and s.
    dt_vector1 = ds / (speeds1 + epsilon)
    # v1 * 0.5 * dt_vector1: (km/s) * (Mpc/h)/(km/s) = Mpc/h.
    # This ensures dimensional consistency with positions_carry (Mpc/h).
    p_mid = positions_carry + v1 * 0.5 * dt_vector1

    # --- RK2 Step 2 (k2) ---
    v2 = _get_velocity(p_mid, v_field, map_coords_kwargs, box_size)
    speeds2 = jnp.sqrt(jnp.sum(v2**2, axis=-1, keepdims=True))
    # dt_vector2: Same unit interpretation as dt_vector1.
    dt_vector2 = ds / (speeds2 + epsilon)

    # --- Final Update ---
    # v2 * dt_vector2: (km/s) * (Mpc/h)/(km/s) = Mpc/h.
    # This ensures dimensional consistency with positions_carry (Mpc/h).
    final_positions = positions_carry + v2 * dt_vector2

    # Apply periodic boundary conditions
    final_positions = jnp.mod(final_positions, box_size)

    return final_positions, None


@partial(jax.jit, static_argnames=('map_coords_kwargs', 'epsilon',
                                   'box_size', 'chunk_size'))
def _chunked_scan(carry, x, v_field, ds, map_coords_kwargs, epsilon,
                  box_size, chunk_size):
    step_fn = partial(
        _rk2_step,
        v_field=v_field,
        ds=ds,
        map_coords_kwargs=map_coords_kwargs,
        epsilon=epsilon,
        box_size=box_size
    )
    return jax.lax.scan(step_fn, carry, jnp.zeros(chunk_size))


class Integrator:
    """
    Handles the evolution of test particles in a 3D velocity field.
    Assumes periodic boundary conditions.

    Parameters
    ----------
    v_field : numpy.ndarray
        The 3D velocity field as a NumPy array with shape
        (3, resolution, resolution, resolution). It will be moved to the
        JAX device upon initialization.
    box_size : float
        The size of the simulation box in physical units.
    num_steps : int
        The number of integration steps.
    ds : float
        The spatial step length. `ds` will be automatically capped to
        half the resolution element size if a larger value is provided.
    epsilon : float, optional
        Small value added to the denominator when calculating the
        per-particle time step (`dt_vector`). This prevents division
        by zero for particles with near-zero velocity, effectively
        stopping them from moving further. Default: 1e-6.
    n_steps_check : int, optional
        The number of steps after which to check the displacement of particles.
        Defaults to `num_steps // 10`.
    **map_coords_kwargs
        Keyword arguments for `map_coordinates`.
    """
    def __init__(self, v_field, box_size, num_steps, ds, epsilon=1e-6,
                 n_steps_check=None, **map_coords_kwargs):
        if v_field.shape[0] != 3 or v_field.ndim != 4:
            raise ValueError(
                "Velocity field must have shape (3, res, res, res)"
            )
        self.v_field = jax.device_put(v_field)
        self.box_size = box_size
        self.num_steps = num_steps
        # ds: The spatial step length. Its units should be consistent with
        # box_size and positions (e.g., Mpc/h).
        self.ds = ds
        # epsilon: Its units should be consistent with speeds (e.g., km/s).
        self.epsilon = epsilon

        if n_steps_check is None:
            self.n_steps_check = max(1, self.num_steps // 10)
        else:
            self.n_steps_check = n_steps_check

        self.checkpoint_positions = None
        self.step_counter = 0
        self.displacement_over_n_steps = None

        # Pop 'adaptive' from map_coords_kwargs if it exists, as it is no
        # longer a valid parameter.
        if 'adaptive' in map_coords_kwargs:
            map_coords_kwargs.pop('adaptive')

        # Enforce maximum spatial step size
        resolution = self.v_field.shape[1]
        max_ds = (self.box_size / resolution) / 2.0
        if self.ds > max_ds:
            fprint(f"Warning: Provided ds ({self.ds}) is larger than "
                   f"half the resolution element ({max_ds}). "
                   f"Capping ds to {max_ds}.")
            self.ds = max_ds

        if not map_coords_kwargs:
            self.map_coords_kwargs = {
                'order': 1, 'mode': 'wrap', 'cval': 0.0
            }
        else:
            self.map_coords_kwargs = map_coords_kwargs

    def run(self, initial_positions, chunk_size=None):
        """
        Runs the full integration with a progress indicator.

        Parameters
        ----------
        initial_positions : jax.Array
            The starting positions of the particles.
        chunk_size : int, optional
            The number of steps to run in each chunk. If None, it defaults
            to 10% of `num_steps`. Default: None.

        Returns
        -------
        tuple
            A tuple containing:
            - jax.Array: The final positions of the particles.
            - jax.Array: The speeds of the particles at the final positions.
            - jax.Array: The displacement of each particle over the
              last `n_steps_check` steps.
        """
        fprint("Compiling and running JIT-compiled integration...")

        if chunk_size is None:
            chunk_size = max(1, self.num_steps // 100)

        map_kwargs_tuple = tuple(self.map_coords_kwargs.items())

        positions = initial_positions

        # Initialize checkpointing for convergence check
        self.checkpoint_positions = initial_positions
        self.step_counter = 0
        self.displacement_over_n_steps = jnp.zeros(
            initial_positions.shape[0]
        )

        with tqdm(total=self.num_steps, desc="Integrating") as pbar:
            for _ in range(0, self.num_steps, chunk_size):
                positions, _ = _chunked_scan(
                    positions, None, self.v_field, self.ds,
                    map_kwargs_tuple, self.epsilon, self.box_size,
                    chunk_size
                )
                positions.block_until_ready()
                pbar.update(chunk_size)

                # Perform convergence check
                self.step_counter += chunk_size
                if self.step_counter >= self.n_steps_check:
                    displacement = jnp.sqrt(
                        jnp.sum((positions - self.checkpoint_positions)**2,
                                axis=-1)
                    )
                    self.displacement_over_n_steps = displacement
                    self.checkpoint_positions = positions
                    self.step_counter = 0

        fprint("Integration complete.")

        # Calculate final speeds
        final_velocities = _get_velocity(
            positions, self.v_field, map_kwargs_tuple, self.box_size
        )
        final_speeds = jnp.sqrt(jnp.sum(final_velocities**2, axis=-1))

        # Calculate resolution element size
        resolution = self.v_field.shape[1]
        half_resolution_element = (self.box_size / resolution) / 2.0

        # Check for convergence
        converged_particles_mask = (
            self.displacement_over_n_steps < half_resolution_element
        )
        fraction_converged = jnp.sum(converged_particles_mask) / len(
            converged_particles_mask
        )
        fprint(
            f"Fraction of particles with displacement over "
            f"{self.n_steps_check} steps less than "
            f"half a resolution element ({half_resolution_element:.2e}): "
            f"{fraction_converged:.2%}"
        )

        return positions, final_speeds, self.displacement_over_n_steps

    def get_cluster_info(self, positions, displacement_over_n_steps,
                         dbscan_eps=None, dbscan_min_samples=None):
        """
        Performs DBSCAN clustering on converged particles and returns cluster
        information (centroids and counts).

        Parameters
        ----------
        positions : jax.Array
            The final positions of all particles.
        displacement_over_n_steps : jax.Array
            The displacement of each particle over the last n_steps_check
            steps.
        dbscan_eps : float, optional
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other for DBSCAN clustering. If None,
            defaults to half a resolution element.
        dbscan_min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a
            point to be considered as a core point for DBSCAN clustering. If
            None, defaults to 2.

        Returns
        -------
        AttractorCollection
            An `AttractorCollection` object containing `AttractorInfo` objects,
            each with 'centroid' (mean position) and 'count' (number of
            particles) for each found attractor. Returns an empty collection if
            no attractors are found.
        
        """
        return find_attractors_from_convergence(
            positions,
            displacement_over_n_steps,
            self.box_size,
            self.v_field.shape,
            dbscan_eps,
            dbscan_min_samples,
        )
