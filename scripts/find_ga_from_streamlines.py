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
"""Extract Great Attractor positions from MW streamlines."""

from pathlib import Path

import h5py
import numpy as np


def _format_sigma_key(sigma):
    sigma_float = float(sigma)
    if sigma_float.is_integer():
        sigma_str = str(int(sigma_float))
    else:
        sigma_str = str(sigma_float).rstrip("0").rstrip(".")
    return f"sigma_{sigma_str}"


def load_final_positions(h5_path, sigma_targets):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    sigma_targets = np.atleast_1d(np.asarray(sigma_targets, dtype=float))
    records_by_sigma = {float(s): [] for s in sigma_targets}

    with h5py.File(h5_path, "r") as h5f:
        available = np.asarray(h5f.attrs.get("smoothing_scales", []),
                               dtype=float)
        if available.size:
            for sigma in sigma_targets:
                matched = np.isclose(available, sigma)
                if not matched.any():
                    raise ValueError(
                        f"smoothing scale {sigma} not in file. "
                        f"Available: {available}"
                    )

        sigma_keys = {
            sigma: _format_sigma_key(sigma) for sigma in sigma_targets}

        for name in h5f.keys():
            if not name.startswith("field_"):
                continue
            field_idx = int(name.split("_")[1])
            field_grp = h5f[name]
            for sigma, sigma_key in sigma_keys.items():
                if sigma_key not in field_grp:
                    raise KeyError(f"{sigma_key} missing in {name}")
                traj = field_grp[sigma_key]["trajectory"][()]
                if traj.shape[0] == 0:
                    final_pos = (np.nan, np.nan, np.nan)
                else:
                    final_pos = traj[-1]
                records_by_sigma[sigma].append((field_idx, *final_pos))
    return records_by_sigma


def write_records(records, output_path):
    header = "# field x[h^-1 Mpc] y[h^-1 Mpc] z[h^-1 Mpc]\n"
    records = sorted(records, key=lambda r: r[0])
    lines = [f"{field} {x} {y} {z}\n" for field, x, y, z in records]
    output_path = Path(output_path)
    output_path.write_text(header + "".join(lines))


def main():
    h5_file = Path("../results/MW_streamlines.hdf5")
    sigma_targets = [2, 3, 4, 5]  # Choose smoothing scales to extract
    output_dir = Path("../results")

    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_sigma = load_final_positions(h5_file, sigma_targets)
    for sigma, records in records_by_sigma.items():
        fname = output_dir / f"ga_positions_sigma_{sigma}.txt"
        write_records(records, fname)
        print(f"Wrote {len(records)} GA positions to {fname}")


if __name__ == "__main__":
    main()
