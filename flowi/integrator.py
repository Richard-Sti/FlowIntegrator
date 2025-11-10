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

from .utils import fprint, smooth_velocity_field_gaussian
from .clustering import find_attractors_from_convergence
from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u
import numpy as np


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

    return final_positions, (final_positions, speeds2)


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


class TrajectoryFollower:
    """
    Follows the trajectory of a single particle through the velocity field.
    """
    def __init__(self, v_field, box_size, num_steps, ds, epsilon=1e-6,
                 min_steps=None, stop_on_convergence=True,
                 convergence_check_steps=None, convergence_threshold=None,
                 **map_coords_kwargs):
        if v_field.shape[0] != 3 or v_field.ndim != 4:
            raise ValueError(
                "Velocity field must have shape (3, res, res, res)"
            )
        self.v_field = jax.device_put(v_field)
        self.box_size = box_size
        self.num_steps = num_steps
        self.ds = ds
        self.epsilon = epsilon

        if min_steps is None:
            self.min_steps = max(1, int(0.3 * num_steps))
        else:
            self.min_steps = min_steps

        self.stop_on_convergence = stop_on_convergence

        if convergence_check_steps is None:
            self.convergence_check_steps = max(1, int(0.1 * num_steps))
        else:
            self.convergence_check_steps = convergence_check_steps

        # Pop 'adaptive' from map_coords_kwargs if it exists, as it is no
        # longer a valid parameter.
        if 'adaptive' in map_coords_kwargs:
            map_coords_kwargs.pop('adaptive')

        if convergence_threshold is None:
            resolution = self.v_field.shape[1]
            self.convergence_threshold = (self.box_size / resolution) / 2.0
        else:
            self.convergence_threshold = convergence_threshold

        if not map_coords_kwargs:
            self.map_coords_kwargs = {
                'order': 1, 'mode': 'wrap', 'cval': 0.0
            }
        else:
            self.map_coords_kwargs = map_coords_kwargs

    def follow(self, initial_position, smooth_sigma=None):
        """
        Follows the trajectory of a single particle.

        Parameters
        ----------
        initial_position : jax.Array
            The starting position of the particle (shape (3,)).
        smooth_sigma : float, optional
            Standard deviation for Gaussian kernel in physical units (Mpc / h)
            to smooth the velocity field. If None, no smoothing is applied.
            Default: None.

        Returns
        -------
        tuple
            A tuple containing:
            - jax.Array: A 1D array of shape (num_steps,) containing the
              time steps (from 1 to num_steps).
            - jax.Array: A 2D array of shape (num_steps, 3) containing the
              trajectory of the particle at each time step.
            - jax.Array: A 1D array of shape (num_steps,) containing the
              velocity magnitude of the particle at each time step.
        """
        if initial_position.shape != (3,):
            raise ValueError(
                "initial_position must be a 1D array of shape (3,)"
            )

        v_field = self.v_field
        if smooth_sigma is not None and smooth_sigma > 0:
            # Convert JAX array to NumPy array for smoothing
            v_field_np = np.array(self.v_field)

            # Smooth the velocity field
            smoothed_v_field_np = smooth_velocity_field_gaussian(
                v_field_np, self.box_size, smooth_sigma
            )

            # Convert back to JAX array
            v_field = jax.device_put(smoothed_v_field_np)

        # The _rk2_step function expects positions_carry to be (N, 3).
        # For a single particle, we reshape it to (1, 3) and then back.
        single_particle_position = initial_position[jnp.newaxis, :]

        # Create a step function for lax.scan
        step_fn = partial(
            _rk2_step,
            v_field=v_field,
            ds=self.ds,
            map_coords_kwargs=tuple(self.map_coords_kwargs.items()),
            epsilon=self.epsilon,
            box_size=self.box_size
        )

        if not self.stop_on_convergence:
            # Use lax.scan to perform multiple RK2 steps if not stopping on
            # convergence
            _, (trajectory_steps, speeds_steps) = jax.lax.scan(
                step_fn, single_particle_position, jnp.zeros(self.num_steps)
            )
            trajectory = trajectory_steps.squeeze()
            speeds_trajectory = speeds_steps.squeeze()
            time_steps = jnp.arange(1, self.num_steps + 1)
            return time_steps, trajectory, speeds_trajectory

        # --- Convergence stopping logic ---
        def cond_fun(state):
            step, _, _, _, converged, _ = state
            return (step < self.num_steps) & ~converged

        def body_fun(state):
            step, last_pos, trajectory, speeds, _, last_checkpoint_pos = state
            new_pos, (_, new_speed) = step_fn(last_pos, None)
            trajectory = trajectory.at[step].set(new_pos.squeeze())
            speeds = speeds.at[step].set(new_speed.squeeze())

            # Check for convergence
            is_check_step = ((step + 1) >= self.min_steps) & \
                            ((step + 1) % self.convergence_check_steps == 0)
            displacement = jnp.sqrt(
                jnp.sum((new_pos - last_checkpoint_pos)**2)
            )
            converged = is_check_step & \
                (displacement < self.convergence_threshold)

            # Update checkpoint position
            new_checkpoint_pos = jax.lax.cond(
                is_check_step,
                lambda _: new_pos,
                lambda _: last_checkpoint_pos,
                None
            )
            return (step + 1, new_pos, trajectory, speeds, converged,
                    new_checkpoint_pos)

        # Initialize state for the while_loop
        initial_state = (
            0,
            single_particle_position,
            jnp.zeros((self.num_steps, 3)),
            jnp.zeros(self.num_steps),
            jnp.array(False),
            single_particle_position
        )

        # Run the while_loop
        final_step, _, trajectory, speeds_trajectory, _, _ = jax.lax.while_loop(  # noqa
            cond_fun, body_fun, initial_state
        )

        # Trim the results to the actual number of steps
        time_steps = jnp.arange(1, final_step + 1)
        trajectory = trajectory[:final_step]
        speeds_trajectory = speeds_trajectory[:final_step]

        return time_steps, trajectory, speeds_trajectory

    def to_galactic(self, trajectory, observer_location, input_frame):
        """
        Converts the trajectory steps to Galactic coordinates.

        Parameters
        ----------
        trajectory : jax.Array
            A 2D array of shape (num_steps, 3) containing the trajectory
            of the particle.
        observer_location : numpy.ndarray
            The 3D position of the observer within the box, in the same
            Cartesian units as the trajectory. Shape must be (3,).
        input_frame : str
            The Astropy frame of the input Cartesian coordinates of the
            trajectory steps (e.g., 'icrs', 'supergalactic').

        Returns
        -------
        jax.Array
            A 2D JAX array of shape (num_steps, 3), where each row contains
            the distance, Galactic longitude (l), and Galactic latitude (b)
            for each trajectory step. Units are Mpc for distance and degrees
            for l and b.
        """
        if (not isinstance(observer_location, np.ndarray) or
                observer_location.shape != (3,)):
            raise ValueError(
                "observer_location must be a numpy array of shape (3,)"
            )

        galactic_coords_data = []
        for step_position in trajectory:
            # Convert JAX array to NumPy array for Astropy
            step_position_np = np.array(step_position) * u.Mpc

            # The vector from observer to step_position is:
            relative_position = step_position_np - observer_location * u.Mpc

            cartesian_representation = CartesianRepresentation(
                x=relative_position[0],
                y=relative_position[1],
                z=relative_position[2],
                unit=u.Mpc
            )

            # Create a SkyCoord object in the specified input_frame
            sky_coord = SkyCoord(cartesian_representation, frame=input_frame)

            # Transform to Galactic coordinates
            galactic_coord = sky_coord.galactic
            galactic_coords_data.append([
                galactic_coord.distance.value,
                galactic_coord.l.value,
                galactic_coord.b.value
            ])
        return jnp.array(galactic_coords_data)

    def follow_multiple_smoothing(self, initial_position, smoothing_scales):
        """
        Follows the trajectory of a single particle with multiple smoothing
        scales.

        Parameters
        ----------
        initial_position : jax.Array
            The starting position of the particle (shape (3,)).
        smoothing_scales : list of float
            A list of standard deviations for the Gaussian kernel to smooth
            the velocity field.

        Returns
        -------
        list of tuple
            A list of tuples, where each tuple contains the results from
            the `follow` method for each smoothing scale:
            (time_steps, trajectory, speeds_trajectory).
        """
        results = []
        for sigma in tqdm(smoothing_scales, desc="Smoothing scales"):
            result = self.follow(initial_position, smooth_sigma=sigma)
            results.append(result)
        return results
