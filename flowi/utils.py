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

import astropy.units as u
import healpy as hp
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as ndi
from astropy.coordinates import (ICRS, CartesianRepresentation, Galactic,
                                 SkyCoord, SphericalRepresentation,
                                 Supergalactic)
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange


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
                             observer_location=None, max_distance=None,
                             verbose=True):
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
    verbose : bool, optional
        If True, print informational messages. Default: True.

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
               f"within {max_distance} Mpc/h of observer.", verbose=verbose)

    fprint(f"Initialized {initial_positions.shape[0]} particles on device.",
           verbose=verbose)
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


def smooth_scalar_field_gaussian(field, box_size, sigma):
    """
    Smooth a 3D scalar field with a Gaussian kernel in physical units.
    """
    resolution = field.shape[0]
    sigma_pixels = sigma * resolution / box_size
    return ndi.gaussian_filter(field, sigma=sigma_pixels, mode='wrap')


def _cartesian_icrs_to_spherical(pos, center, frame_cls):
    """Convert ICRS Cartesian positions to spherical in target frame."""
    pos_q = u.Quantity(pos, copy=False)
    cen_q = u.Quantity(center, copy=False)

    rel = pos_q - cen_q

    rep = CartesianRepresentation(rel[..., 0], rel[..., 1], rel[..., 2])
    icrs = ICRS(rep)
    tgt = icrs.transform_to(frame_cls())
    sph = tgt.represent_as(SphericalRepresentation)

    ell = sph.lon.to(u.deg).value
    b = sph.lat.to(u.deg).value
    r = sph.distance.value

    return r, ell, b


def cartesian_icrs_to_galactic_spherical(pos, center):
    """ICRS Cartesian to Galactic spherical about center."""
    return _cartesian_icrs_to_spherical(pos, center, Galactic)


def cartesian_icrs_to_supergalactic_spherical(pos, center):
    """ICRS Cartesian to supergalactic spherical about center."""
    return _cartesian_icrs_to_spherical(pos, center, Supergalactic)


def galactic_to_radec(l_deg, b_deg):
    """Convert Galactic longitude and latitude to equatorial coordinates."""
    ell, b = np.broadcast_arrays(np.asarray(l_deg, dtype=float),
                                 np.asarray(b_deg, dtype=float))
    coord = SkyCoord(l=ell * u.deg, b=b * u.deg, frame="galactic").icrs
    ra = coord.ra.to_value(u.deg)
    dec = coord.dec.to_value(u.deg)
    return ra, dec


def radec_to_galactic(ra_deg, dec_deg):
    """Convert equatorial coordinates to Galactic longitude and latitude."""
    ra, dec = np.broadcast_arrays(np.asarray(ra_deg, dtype=float),
                                  np.asarray(dec_deg, dtype=float))
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    ell = coord.l.to_value(u.deg)
    b = coord.b.to_value(u.deg)
    return ell, b


def grid_ngp_projection(nside, rho, boxsize, observer, Rmax,
                        dr, Rmin=0, chunksize=10_000, r_power=2,
                        coords=None, verbose=True):
    nx, ny, nz = rho.shape
    x = (np.arange(nx) + 0.5) * boxsize / nx
    y = (np.arange(ny) + 0.5) * boxsize / ny
    z = (np.arange(nz) + 0.5) * boxsize / nz

    # Interpolator (periodic handled by manual wrapping)
    fprint("building the 3D grid interpolator...", verbose=verbose)
    interp = RegularGridInterpolator((x, y, z), rho, bounds_error=True)

    # Radial samples
    r = np.arange(Rmin, Rmax + dr, dr)
    nr = r.size
    fprint(f"going to evaluate {nr} radial samples from {Rmin} to {Rmax} "
           f"with dr={dr} and r_power={r_power}...",
           verbose=verbose)

    npix = hp.nside2npix(nside)
    map_out = np.zeros(npix, dtype=np.float64)

    # Pixel directions; optional coord conversion
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    if coords == "icrs->galactic":
        ra, dec = galactic_to_radec(
            np.rad2deg(phi), 90.0 - np.rad2deg(theta))
        theta, phi = np.deg2rad(90.0 - dec), np.deg2rad(ra)
        pix_rhat = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
            ], axis=1)
    else:
        pix_rhat = np.array(hp.pix2vec(nside, np.arange(npix))).T

    norm = simpson(r**r_power, x=r)

    # Chunk over pixels to control memory
    iter_kwargs = {"desc": "Projecting grid",
                   "disable": not verbose or npix < chunksize}
    for i0 in trange(0, npix, chunksize, **iter_kwargs):
        i1 = min(i0 + chunksize, npix)
        nhat = pix_rhat[i0:i1]  # (chunk_size, 3)

        # Ray points: (nr, chunk_size, 3)
        pts = observer[None, None, :] + r[:, None, None] * nhat[None, :, :]
        pts = pts.reshape(-1, 3)

        # Interpolate rho along rays, reshape to (nr, C) and integrate
        vals = interp(pts).reshape(nr, i1 - i0)
        vals = simpson(r[:, None]**r_power * vals, x=r, axis=0) / norm

        map_out[i0:i1] = vals

    return map_out
