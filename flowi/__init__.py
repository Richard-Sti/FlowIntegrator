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


__version__ = "0.1.0"

from .apertures import SphericalIntegrator  # noqa
from .integrator import Integrator, TrajectoryFollower  # noqa: F401
from .loaders import (  # noqa: F401
    ManticoreLoader,
    Carrick2015Loader,
)
from .utils import (  # noqa: F401
    fprint,
    create_initial_positions,
    smooth_velocity_field_gaussian,
    smooth_scalar_field_gaussian,
    cartesian_icrs_to_galactic_spherical,
    cartesian_icrs_to_supergalactic_spherical,
    grid_ngp_projection,
    radec_to_galactic,
    galactic_to_radec,
)

from .velocity_convergence import (  # noqa: F401
    matter_power_spectrum_camb,
    velocity_sigma_1d_squared_within_radius,
    expected_velocity_within_radius,
)

from .linear_theory import (  # noqa: F401
    delta_to_velocity,
    growth_rate,
    hubble_parameter,
)
