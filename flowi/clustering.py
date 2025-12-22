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


import jax.numpy as jnp
import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
import collections

from .utils import fprint

# Import astropy for coordinate transformations
from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u


@dataclass
class AttractorInfo:
    """
    Data class to hold information about a single DBSCAN attractor.
    """
    centroid: jnp.ndarray
    count: int
    members: jnp.ndarray

    def to_galactic(self, observer_location, input_frame):
        """
        Converts the attractor centroid to Galactic coordinates.

        Parameters
        ----------
        observer_location : numpy.ndarray
            The 3D position of the observer within the box, in the same
            Cartesian units as the centroid. Shape must be (3,).
        input_frame : str
            The Astropy frame of the input Cartesian coordinates of the
            centroid.

        Returns
        -------
        astropy.coordinates.Galactic
            The Galactic coordinates of the attractor centroid relative to the
            observer.
        """
        # Convert JAX array to NumPy array for Astropy
        centroid_np = np.array(self.centroid) * u.Mpc  # Assuming Mpc units

        # Ensure observer_location is a numpy array and has correct shape
        if (not isinstance(observer_location, np.ndarray) or
                observer_location.shape != (3,)):
            raise ValueError(
                "observer_location must be a numpy array of shape (3,)"
            )

        # The vector from observer to centroid is:
        relative_centroid = centroid_np - observer_location * u.Mpc

        cartesian_representation = CartesianRepresentation(
            x=relative_centroid[0],
            y=relative_centroid[1],
            z=relative_centroid[2],
            unit=u.Mpc
        )

        # Create a SkyCoord object in the specified input_frame
        sky_coord = SkyCoord(cartesian_representation, frame=input_frame)

        # Transform to Galactic coordinates
        galactic_coord = sky_coord.galactic
        return galactic_coord


class AttractorCollection:
    """
    A collection of AttractorInfo objects, providing convenient access to
    attractor properties.
    """
    def __init__(self, attractors: list[AttractorInfo]):
        self._attractors = attractors

    def __len__(self):
        return len(self._attractors)

    def __getitem__(self, index):
        return self._attractors[index]

    @property
    def counts(self):
        """
        Returns a 1D JAX array of counts for all attractors.
        """
        return jnp.array([a.count for a in self._attractors])

    @property
    def centroids(self):
        """
        Returns a 2D JAX array of centroids for all attractors.
        Shape is (num_attractors, 3).
        """
        return jnp.array([a.centroid for a in self._attractors])

    def to_galactic(self, observer_location, input_frame):
        """
        Converts the centroids of all attractors in the collection to
        Galactic coordinates.

        Parameters
        ----------
        observer_location : numpy.ndarray
            The 3D position of the observer within the box, in the same
            Cartesian units as the centroids. Shape must be (3,).
        input_frame : str
            The Astropy frame of the input Cartesian coordinates of the
            centroids.

        Returns
        -------
        jax.Array
            A 2D JAX array of shape (n_attractors, 3), where each row
            contains the distance, Galactic longitude (l), and Galactic
            latitude (b) for an attractor centroid. Units are Mpc for
            distance and degrees for l and b.
        """
        galactic_coords_data = []
        for attractor in self._attractors:
            galactic_coord = attractor.to_galactic(
                observer_location, input_frame)
            galactic_coords_data.append([
                galactic_coord.distance.value,
                galactic_coord.l.value,
                galactic_coord.b.value
            ])
        return jnp.array(galactic_coords_data)


def find_attractors_from_convergence(
    positions,
    displacement_over_n_steps,
    box_size,
    v_field_shape,
    dbscan_eps=None,
    dbscan_min_samples=None,
):
    """
    Performs DBSCAN clustering on converged particles to find attractors.

    Parameters
    ----------
    positions : jax.Array
        The final positions of all particles.
    displacement_over_n_steps : jax.Array
        The displacement of each particle over the last n_steps_check
        steps.
    box_size : float
        The size of the simulation box in physical units.
    v_field_shape : tuple
        The shape of the velocity field, used to determine resolution.
    dbscan_eps : float, optional
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other for DBSCAN clustering. If
        None, defaults to half a resolution element.
    dbscan_min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a
        point to be considered as a core point for DBSCAN clustering. If
        None, defaults to 2.

    Returns
    -------
    AttractorCollection
        An `AttractorCollection` object containing `AttractorInfo`
        objects, each with 'centroid' (mean position) and 'count'
        (number of particles) for each found attractor. Returns an
        empty collection if no attractors are found.
    """
    attractor_info_list = []

    # Calculate resolution element size
    resolution = v_field_shape[1]
    half_resolution_element = (box_size / resolution) / 2.0

    # Identify converged particles
    converged_particles_mask = (
        displacement_over_n_steps < half_resolution_element
    )
    converged_final_positions = positions[converged_particles_mask]
    # Get the original indices of the converged particles
    converged_particle_indices = jnp.where(converged_particles_mask)[0]

    if converged_final_positions.shape[0] > 0:
        # Set default DBSCAN parameters if not provided
        eps = dbscan_eps if dbscan_eps is not None else half_resolution_element
        min_samples = (
            dbscan_min_samples if dbscan_min_samples is not None else 2)

        fprint(f"Performing DBSCAN clustering with eps={eps:.2e}, "
               f"min_samples={min_samples} on "
               f"{converged_final_positions.shape[0]} converged "
               "particles.")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # Convert JAX array to NumPy array for scikit-learn
        cluster_labels_converged = dbscan.fit_predict(
            np.array(converged_final_positions)
        )

        unique_labels = jnp.unique(cluster_labels_converged)
        n_attractors = len(unique_labels[unique_labels != -1])
        n_noise = jnp.sum(cluster_labels_converged == -1)
        fprint(f"DBSCAN found {n_attractors} attractors "
               f"and {n_noise} noise points.")

        for k in unique_labels:
            if k == -1:  # Noise points
                continue

            class_member_mask = (cluster_labels_converged == k)
            attractor_positions = converged_final_positions[
                class_member_mask]

            centroid = jnp.mean(attractor_positions, axis=0)
            count = attractor_positions.shape[0]
            # Get the indices of particles belonging to this attractor
            attractor_members = converged_particle_indices[
                class_member_mask]

            attractor_info_list.append(
                AttractorInfo(centroid=centroid, count=count,
                              members=attractor_members))
    else:
        fprint("No converged particles found for DBSCAN clustering.")

    # Sort attractors by count in descending order
    attractor_info_list.sort(key=lambda x: x.count, reverse=True)

    return AttractorCollection(attractor_info_list)


def find_attractors_by_voxel_counting(
    positions,
    displacement_over_n_steps,
    box_size,
    grid_resolution,
    min_count=1
):
    """
    Finds attractors by counting converged streamlines in voxels.

    Parameters
    ----------
    positions : jax.Array
        The final positions of all particles.
    displacement_over_n_steps : jax.Array
        The displacement of each particle over the last n_steps_check
        steps.
    box_size : float
        The size of the simulation box in physical units.
    grid_resolution : int
        The resolution of the grid to use for voxelization.
    min_count : int, optional
        Minimum number of particles required to keep a voxel-connected
        attractor. Default: 1.

    Returns
    -------
    AttractorCollection
        An `AttractorCollection` object containing `AttractorInfo`
        objects for each voxel with converged particles.
    """
    attractor_info_list = []

    # Calculate resolution element size
    half_resolution_element = (box_size / grid_resolution) / 2.0

    # Identify converged particles
    converged_particles_mask = (
        displacement_over_n_steps < half_resolution_element
    )
    converged_final_positions = positions[converged_particles_mask]
    converged_particle_indices = jnp.where(converged_particles_mask)[0]

    if converged_final_positions.shape[0] == 0:
        fprint("No converged particles found.")
        return AttractorCollection([])

    fprint(f"Voxelizing {converged_final_positions.shape[0]} "
           f"converged particles into a {grid_resolution}^3 grid with "
           f"min_count={min_count}.")

    # Voxelize converged positions with periodic wrapping on boundaries
    voxel_indices = jnp.floor(
        converged_final_positions * (grid_resolution / box_size)
    ).astype(int) % grid_resolution

    # Group particles by voxel index
    voxel_groups = collections.defaultdict(list)
    positions_np = np.asarray(converged_final_positions)
    voxel_indices_np = np.asarray(voxel_indices, dtype=int)
    particle_indices_np = np.asarray(converged_particle_indices, dtype=int)
    for pos, v_idx, member_idx in zip(positions_np,
                                      voxel_indices_np,
                                      particle_indices_np):
        voxel_groups[tuple(v_idx)].append((pos, member_idx))

    fprint(f"Found {len(voxel_groups)} occupied voxels.")

    # Union-Find to merge voxel groups across periodic boundaries
    parent = {key: key for key in voxel_groups}

    def find(voxel_key):
        while parent[voxel_key] != voxel_key:
            parent[voxel_key] = parent[parent[voxel_key]]
            voxel_key = parent[voxel_key]
        return voxel_key

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    neighbor_offsets = (
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1)
    )

    for voxel_key in voxel_groups:
        for dx, dy, dz in neighbor_offsets:
            neighbor = (
                (voxel_key[0] + dx) % grid_resolution,
                (voxel_key[1] + dy) % grid_resolution,
                (voxel_key[2] + dz) % grid_resolution
            )
            if neighbor in voxel_groups:
                union(voxel_key, neighbor)

    components = collections.defaultdict(list)
    for voxel_key in voxel_groups:
        components[find(voxel_key)].append(voxel_key)

    def _circular_mean(coords):
        angles = coords / box_size * (2.0 * jnp.pi)
        sin_sum = jnp.sin(angles).mean(axis=0)
        cos_sum = jnp.cos(angles).mean(axis=0)
        mean_angle = jnp.arctan2(sin_sum, cos_sum)
        return (mean_angle % (2.0 * jnp.pi)) * (box_size / (2.0 * jnp.pi))

    # Create AttractorInfo for each connected component
    for comp_voxels in components.values():
        member_positions = []
        member_indices = []
        for voxel_index in comp_voxels:
            for pos, m_idx in voxel_groups[voxel_index]:
                member_positions.append(pos)
                member_indices.append(m_idx)

        member_positions = jnp.array(member_positions)
        member_indices = jnp.array(member_indices)

        count = len(member_positions)
        if count < min_count:
            continue

        centroid = _circular_mean(member_positions)

        attractor_info_list.append(
            AttractorInfo(centroid=centroid,
                          count=count,
                          members=member_indices)
        )

    # Sort attractors by count in descending order
    attractor_info_list.sort(key=lambda x: x.count, reverse=True)

    return AttractorCollection(attractor_info_list)
