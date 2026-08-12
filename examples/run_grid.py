"""
Parallel parameter-grid example.

Run with

    python3 examples/run_grid.py

The calculation can be interrupted and resumed by running the
same command again.
"""

import os

import numpy as np

from compact_star_shooting.grid import (
    run_grid_parallel,
)


# ============================================================
# Parameter grid
# ============================================================

rho0_values = np.linspace(
    0.0,
    3.0e-3,
    5,
)

phi_values = np.linspace(
    0.0,
    0.10,
    5,
)


# ============================================================
# Parallel resources
# ============================================================

max_workers = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        min(
            4,
            os.cpu_count() or 1,
        ),
    )
)


# ============================================================
# Run grid
# ============================================================

grid = run_grid_parallel(
    rho0_values=rho0_values,
    phi_values=phi_values,

    max_workers=max_workers,

    checkpoint_path=(
        "mixed_star_grid_checkpoint.npz"
    ),

    checkpoint_every=5,

    K_poly=100.0,
    gamma=2.0,
    mu=1.0,

    r_max=20.0,
    n_points=3000,

    omega_low=0.8,
    omega_high=103.0,
    omega_tol=1.0e-8,

    verbose=True,
)


# ============================================================
# Summary
# ============================================================

n_success = int(
    np.sum(
        grid["status"] == 1
    )
)

n_failed = int(
    np.sum(
        grid["status"] == -1
    )
)

n_pending = int(
    np.sum(
        grid["status"] == 0
    )
)


print()
print("Grid summary")
print("============")

print(
    f"successful = {n_success}"
)

print(
    f"failed     = {n_failed}"
)

print(
    f"pending    = {n_pending}"
)


print()
print("Total gravitational mass")
print("========================")

print(
    grid["M_total"]
)


# ============================================================
# Save flat CSV table
# ============================================================

rows = []

for i, rho0_c in enumerate(
    grid["rho0_values"]
):

    for j, phi_c in enumerate(
        grid["phi_values"]
    ):

        row = [
            rho0_c,
            phi_c,
            grid["status"][i, j],
        ]

        for name in grid["observable_names"]:

            row.append(
                grid[name][i, j]
            )

        rows.append(row)


rows = np.asarray(
    rows,
    dtype=float,
)


header = (
    "rho0_c,"
    "phi_c,"
    "status,"
    + ",".join(
        grid["observable_names"]
    )
)


np.savetxt(
    "mixed_star_grid_results.csv",
    rows,
    delimiter=",",
    header=header,
    comments="",
)


print()
print(
    "Saved:"
    " mixed_star_grid_results.csv"
)