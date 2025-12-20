# FlowIntegrator: Velocity Field Analysis for the Local Universe

A Python package for analysing cosmological velocity fields and streamline convergence. Developed for the study of the Great Attractor and Local Group dynamics using BORG-based reconstructions.

## Overview

FlowIntegrator (`flowi`) provides tools for:

- **Streamline integration**: Follow velocity field trajectories to identify convergence points and basins of attraction
- **Linear theory velocities**: Compute peculiar velocity fields from density fields using linear perturbation theory
- **Coordinate transformations**: Convert between ICRS Cartesian and Galactic/Supergalactic spherical coordinates
- **Field smoothing**: Gaussian smoothing of scalar and vector fields
- **Data loaders**: Interface to Manticore-Local and Carrick2015 velocity field reconstructions

This code accompanies the paper:

> R. Stiskalek, H. Desmond, S. McAlpine, G. Lavaux, J. Jasche (2025).
> *Revisiting the Great Attractor: Streamline Convergence and the Local Group's cosmic velocity and dynamical fate*

## Installation

```bash
python -m venv venv_flowi
source venv_flowi/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Usage

```python
import flowi

# Load a Manticore velocity field
loader = flowi.ManticoreLoader(data_path, field_id=0)
rho = loader.load_density_field()

# Smooth and compute velocity from density
rho_smooth = flowi.smooth_scalar_field_gaussian(rho, loader.boxsize, sigma=2.5)
delta = rho_smooth / rho_smooth.mean() - 1.0
velocity = flowi.delta_to_velocity(delta, loader.boxsize, Omega_m=0.306)

# Integrate streamlines
integrator = flowi.Integrator(velocity, loader.boxsize)
trajectories = integrator.integrate(initial_positions, t_max=1.0)

# Convert to Galactic coordinates
r, ell, b = flowi.cartesian_icrs_to_galactic_spherical(positions, observer)
```

## License

GNU General Public License v3.0
