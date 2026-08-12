"""
Parallel parameter-grid utilities for mixed fermion-boson stars.

The module evaluates independent stellar configurations on a
two-dimensional grid of

    (rho0_c, phi_c)

using ProcessPoolExecutor.

Results are saved incrementally to an NPZ checkpoint so interrupted
runs can be resumed without recomputing completed models.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .integrator import integrate_star
from .shooting import shoot_omega
from .observables import compute_observables


# ============================================================
# Observable layout
# ============================================================

OBSERVABLE_NAMES = (
    "omega_shoot",
    "omega_phys",
    "alpha_inf",
    "M_total",
    "M_baryon_rest",
    "M_boson_rest",
    "N_baryon",
    "N_boson",
    "R_fluid",
    "R_baryon_99",
    "R_boson_99",
    "R_total_99",
    "rest_mass",
    "binding_energy",
    "mass_defect",
    "relative_binding_energy",
    "M_fluid_grav",
    "M_scalar_grav",
    "nodes",
    "phi_end",
)

N_OBSERVABLES = len(
    OBSERVABLE_NAMES
)


# ============================================================
# Single stellar model
# ============================================================

def compute_grid_point(args):
    """
    Compute one point of the (rho0_c, phi_c) parameter grid.

    Parameters
    ----------
    args : tuple
        Tuple containing

            i,
            j,
            rho0_c,
            phi_c,
            solver_options.

    Returns
    -------
    tuple
        On success:

            i, j, values, ""

        On failure:

            i, j, None, error_message.
    """

    (
        i,
        j,
        rho0_c,
        phi_c,
        options,
    ) = args

    i = int(i)
    j = int(j)

    rho0_c = float(rho0_c)
    phi_c = float(phi_c)

    try:

        # ----------------------------------------------------
        # Solver parameters
        # ----------------------------------------------------

        K_poly = options["K_poly"]
        gamma = options["gamma"]
        mu = options["mu"]

        r_max = options["r_max"]
        n_points = options["n_points"]

        # ----------------------------------------------------
        # Pure fluid / vacuum limit
        # ----------------------------------------------------

        if phi_c == 0.0:

            omega_shoot = 0.0
            nodes = 0

            result = integrate_star(
                rho0_c=rho0_c,
                phi_c=0.0,
                omega=0.0,
                r_max=r_max,
                n_points=n_points,
                K_poly=K_poly,
                gamma=gamma,
                mu=mu,
            )

        # ----------------------------------------------------
        # Bosonic or mixed configuration
        # ----------------------------------------------------

        else:

            shoot = shoot_omega(
                rho0_c=rho0_c,
                phi_c=phi_c,
                omega_low=options["omega_low"],
                omega_high=options["omega_high"],
                r_max=r_max,
                n_points=n_points,
                K_poly=K_poly,
                gamma=gamma,
                mu=mu,
                omega_tol=options["omega_tol"],
                max_iterations=options["max_iterations"],
                verbose=False,
            )

            if not shoot["converged"]:
                raise RuntimeError(
                    "Scalar-frequency shooting did not converge."
                )

            omega_shoot = float(
                shoot["omega"]
            )

            nodes = int(
                shoot["nodes"]
            )

            result = shoot["result"]

        # ----------------------------------------------------
        # Physical observables
        # ----------------------------------------------------

        obs = compute_observables(
            result=result,
            omega_shoot=omega_shoot,
            K_poly=K_poly,
            gamma=gamma,
            mu=mu,
            baryon_mass=options["baryon_mass"],
            radius_fraction=options["radius_fraction"],
        )

        M_fluid_grav = float(
            obs["M_fluid_grav_profile"][-1]
        )

        M_scalar_grav = float(
            obs["M_scalar_grav_profile"][-1]
        )

        phi_end = float(
            result["phi"][-1]
        )

        values = np.array(
            [
                omega_shoot,
                obs["omega_phys"],
                obs["alpha_inf"],
                obs["M_total"],
                obs["M_baryon_rest"],
                obs["M_boson_rest"],
                obs["N_baryon"],
                obs["N_boson"],
                obs["R_fluid"],
                obs["R_baryon_99"],
                obs["R_boson_99"],
                obs["R_total_99"],
                obs["rest_mass"],
                obs["binding_energy"],
                obs["mass_defect"],
                obs["relative_binding_energy"],
                M_fluid_grav,
                M_scalar_grav,
                float(nodes),
                phi_end,
            ],
            dtype=float,
        )

        return (
            i,
            j,
            values,
            "",
        )

    except Exception as error:

        return (
            i,
            j,
            None,
            str(error),
        )


# ============================================================
# Checkpoint helpers
# ============================================================

def _new_grid_storage(
    n_rho,
    n_phi,
):
    """
    Create empty arrays for a new parameter-grid calculation.
    """

    values = np.full(
        (
            n_rho,
            n_phi,
            N_OBSERVABLES,
        ),
        np.nan,
        dtype=float,
    )

    # 0  -> pending
    # 1  -> successful
    # -1 -> failed
    status = np.zeros(
        (
            n_rho,
            n_phi,
        ),
        dtype=np.int8,
    )

    errors = np.full(
        (
            n_rho,
            n_phi,
        ),
        "",
        dtype="<U500",
    )

    return (
        values,
        status,
        errors,
    )


def _load_checkpoint(
    checkpoint_path,
    rho0_values,
    phi_values,
):
    """
    Load a compatible checkpoint or create a new empty grid.
    """

    n_rho = len(
        rho0_values
    )

    n_phi = len(
        phi_values
    )

    if (
        checkpoint_path is not None
        and os.path.exists(checkpoint_path)
    ):

        try:

            data = np.load(
                checkpoint_path
            )

            values = data["values"]
            status = data["status"]
            errors = data["errors"]

            rho_saved = data["rho0_values"]
            phi_saved = data["phi_values"]

            compatible = (
                values.shape
                == (
                    n_rho,
                    n_phi,
                    N_OBSERVABLES,
                )
                and status.shape
                == (
                    n_rho,
                    n_phi,
                )
                and np.allclose(
                    rho_saved,
                    rho0_values,
                )
                and np.allclose(
                    phi_saved,
                    phi_values,
                )
            )

            if compatible:

                return (
                    values.copy(),
                    status.copy(),
                    errors.copy(),
                )

        except Exception:

            pass

    return _new_grid_storage(
        n_rho,
        n_phi,
    )


def _save_checkpoint(
    checkpoint_path,
    rho0_values,
    phi_values,
    values,
    status,
    errors,
):
    """
    Save the current grid state atomically.
    """

    if checkpoint_path is None:
        return

    checkpoint_path = str(
        checkpoint_path
    )

    if checkpoint_path.endswith(
        ".npz"
    ):
        temporary_path = (
            checkpoint_path[:-4]
            + ".tmp.npz"
        )

    else:
        temporary_path = (
            checkpoint_path
            + ".tmp.npz"
        )

    np.savez(
        temporary_path,
        rho0_values=rho0_values,
        phi_values=phi_values,
        values=values,
        status=status,
        errors=errors,
        observable_names=np.asarray(
            OBSERVABLE_NAMES,
        ),
    )

    os.replace(
        temporary_path,
        checkpoint_path,
    )


# ============================================================
# Parallel grid
# ============================================================

def run_grid_parallel(
    rho0_values,
    phi_values,
    max_workers=None,
    checkpoint_path=None,
    checkpoint_every=10,
    retry_failed=False,
    verbose=True,
    K_poly=100.0,
    gamma=2.0,
    mu=1.0,
    r_max=20.0,
    n_points=3000,
    omega_low=0.8,
    omega_high=103.0,
    omega_tol=1.0e-8,
    max_iterations=150,
    baryon_mass=1.0,
    radius_fraction=0.99,
):
    """
    Evaluate a two-dimensional stellar parameter grid in parallel.

    Parameters
    ----------
    rho0_values : array_like
        Central fluid rest-mass densities.

    phi_values : array_like
        Central scalar amplitudes.

    max_workers : int or None
        Number of worker processes.

        If None, the function first checks SLURM_CPUS_PER_TASK
        and otherwise uses the available CPU count minus one.

    checkpoint_path : str or None
        Path to an NPZ checkpoint file.

    checkpoint_every : int
        Save checkpoint after this many newly processed points.

    retry_failed : bool
        If True, previously failed points are attempted again.

    verbose : bool
        Print progress information.

    Other parameters
    ----------------
    K_poly, gamma, mu
        Physical model parameters.

    r_max, n_points
        Radial integration settings.

    omega_low, omega_high
        Initial shooting window.

    omega_tol
        Eigenfrequency tolerance.

    max_iterations
        Maximum shooting iterations.

    baryon_mass
        Fermionic constituent mass in code units.

    radius_fraction
        Fraction used for effective radii.

    Returns
    -------
    dict
        Grid axes, observable arrays, status matrix and errors.
    """

    rho0_values = np.asarray(
        rho0_values,
        dtype=float,
    )

    phi_values = np.asarray(
        phi_values,
        dtype=float,
    )

    if rho0_values.ndim != 1:
        raise ValueError(
            "rho0_values must be one-dimensional."
        )

    if phi_values.ndim != 1:
        raise ValueError(
            "phi_values must be one-dimensional."
        )

    if checkpoint_every < 1:
        raise ValueError(
            "checkpoint_every must be positive."
        )

    n_rho = len(
        rho0_values
    )

    n_phi = len(
        phi_values
    )

    # --------------------------------------------------------
    # Worker count
    # --------------------------------------------------------

    if max_workers is None:

        slurm_workers = os.environ.get(
            "SLURM_CPUS_PER_TASK"
        )

        if slurm_workers is not None:

            max_workers = int(
                slurm_workers
            )

        else:

            cpu_count = (
                os.cpu_count()
                or 2
            )

            max_workers = max(
                1,
                cpu_count - 1,
            )

    max_workers = max(
        1,
        int(max_workers),
    )

    # --------------------------------------------------------
    # Restore checkpoint
    # --------------------------------------------------------

    (
        values,
        status,
        errors,
    ) = _load_checkpoint(
        checkpoint_path,
        rho0_values,
        phi_values,
    )

    # --------------------------------------------------------
    # Solver options passed to workers
    # --------------------------------------------------------

    options = {
        "K_poly": float(K_poly),
        "gamma": float(gamma),
        "mu": float(mu),
        "r_max": float(r_max),
        "n_points": int(n_points),
        "omega_low": float(omega_low),
        "omega_high": float(omega_high),
        "omega_tol": float(omega_tol),
        "max_iterations": int(
            max_iterations
        ),
        "baryon_mass": float(
            baryon_mass
        ),
        "radius_fraction": float(
            radius_fraction
        ),
    }

    # --------------------------------------------------------
    # Build pending-task list
    # --------------------------------------------------------

    tasks = []

    for i in range(n_rho):

        for j in range(n_phi):

            point_status = int(
                status[i, j]
            )

            if point_status == 0:

                pending = True

            elif (
                point_status == -1
                and retry_failed
            ):

                pending = True

            else:

                pending = False

            if pending:

                tasks.append(
                    (
                        i,
                        j,
                        float(
                            rho0_values[i]
                        ),
                        float(
                            phi_values[j]
                        ),
                        options,
                    )
                )

    total_points = (
        n_rho
        * n_phi
    )

    already_processed = int(
        np.count_nonzero(
            status
        )
    )

    if verbose:

        print(
            "=" * 60
        )

        print(
            "Parallel compact-star grid"
        )

        print(
            f"Grid       : "
            f"{n_rho} x {n_phi} "
            f"= {total_points} points"
        )

        print(
            f"Workers    : "
            f"{max_workers}"
        )

        print(
            f"Completed  : "
            f"{already_processed}"
        )

        print(
            f"Pending    : "
            f"{len(tasks)}"
        )

        if checkpoint_path is not None:

            print(
                f"Checkpoint : "
                f"{checkpoint_path}"
            )

        print(
            "=" * 60
        )

    # --------------------------------------------------------
    # Parallel execution
    # --------------------------------------------------------

    newly_processed = 0

    if tasks:

        with ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:

            for (
                i,
                j,
                point_values,
                error,
            ) in executor.map(
                compute_grid_point,
                tasks,
            ):

                newly_processed += 1

                if error:

                    status[i, j] = -1

                    errors[i, j] = (
                        error[:500]
                    )

                    if verbose:

                        print(
                            f"FAIL "
                            f"({i:3d}, {j:3d}) "
                            f"rho0={rho0_values[i]:.6e} "
                            f"phi={phi_values[j]:.6e} "
                            f"| {error}"
                        )

                else:

                    values[
                        i,
                        j,
                        :,
                    ] = point_values

                    status[i, j] = 1
                    errors[i, j] = ""

                    if verbose:

                        done_total = int(
                            np.count_nonzero(
                                status
                            )
                        )

                        print(
                            f"DONE "
                            f"({i:3d}, {j:3d}) "
                            f"| {done_total}/{total_points}"
                        )

                # --------------------------------------------
                # Incremental checkpoint
                # --------------------------------------------

                if (
                    checkpoint_path
                    is not None
                    and newly_processed
                    % checkpoint_every
                    == 0
                ):

                    _save_checkpoint(
                        checkpoint_path,
                        rho0_values,
                        phi_values,
                        values,
                        status,
                        errors,
                    )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    if checkpoint_path is not None:

        _save_checkpoint(
            checkpoint_path,
            rho0_values,
            phi_values,
            values,
            status,
            errors,
        )

    # --------------------------------------------------------
    # Named observable matrices
    # --------------------------------------------------------

    observable_grids = {}

    for k, name in enumerate(
        OBSERVABLE_NAMES
    ):

        observable_grids[name] = (
            values[:, :, k]
        )

    return {
        "rho0_values": rho0_values,
        "phi_values": phi_values,
        "values": values,
        "status": status,
        "errors": errors,
        "observable_names": OBSERVABLE_NAMES,
        **observable_grids,
    }