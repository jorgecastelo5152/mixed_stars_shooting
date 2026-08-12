"""
Shooting algorithm for the scalar-field eigenfrequency.

For fixed central values

    rho0_c
    phi_c

the scalar eigenfrequency omega is determined by bisection.

The algorithm searches for the fundamental, nodeless scalar
configuration that decays asymptotically.

The shooting strategy follows the same basic logic as the original
research implementation, but is restricted here to

    G = c = hbar = 1,
    canonical scalar kinetic term,
    V(phi) = mu^2 phi^2.
"""

import numpy as np

from .integrator import integrate_star


def count_zero_crossings(phi):
    """
    Count sign changes in a scalar-field profile.

    Parameters
    ----------
    phi : array_like
        Scalar-field profile.

    Returns
    -------
    int
        Number of zero crossings.
    """

    phi = np.asarray(phi, dtype=float)

    if len(phi) < 2:
        return 0

    return int(
        np.sum(
            phi[:-1] * phi[1:] < 0.0
        )
    )


def trial_scalar_value(
    rho0_c,
    phi_c,
    omega,
    r_max=50.0,
    n_points=5000,
    K_poly=100.0,
    gamma=2.0,
    mu=1.0,
):
    """
    Integrate one trial frequency and extract the shooting diagnostic.

    Parameters
    ----------
    rho0_c : float
        Central fluid rest-mass density.

    phi_c : float
        Central scalar amplitude.

    omega : float
        Trial shooting frequency.

    r_max : float
        Maximum integration radius.

    n_points : int
        Number of output points.

    K_poly, gamma : float
        Polytropic EOS parameters.

    mu : float
        Boson mass parameter.

    Returns
    -------
    diagnostic : float
        Scalar-field value at the last successfully integrated point.

    result : dict
        Full integration result.

    nodes : int
        Number of zero crossings in the scalar profile.
    """

    result = integrate_star(
        rho0_c=rho0_c,
        phi_c=phi_c,
        omega=omega,
        r_max=r_max,
        n_points=n_points,
        K_poly=K_poly,
        gamma=gamma,
        mu=mu,
        raise_on_failure=False,
    )

    phi = np.asarray(
        result["phi"],
        dtype=float,
    )

    nodes = count_zero_crossings(phi)

    diagnostic = float(
        phi[-1]
    )

    return diagnostic, result, nodes


def shoot_omega(
    rho0_c,
    phi_c,
    omega_low=0.8,
    omega_high=103.0,
    r_max=50.0,
    n_points=5000,
    K_poly=100.0,
    gamma=2.0,
    mu=1.0,
    omega_tol=1.0e-8,
    max_iterations=150,
    verbose=False,
):
    """
    Determine the scalar-field eigenfrequency by bisection.

    Parameters
    ----------
    rho0_c : float
        Central fluid rest-mass density.

    phi_c : float
        Central scalar-field amplitude.

    omega_low, omega_high : float, optional
        Initial shooting bracket.

    r_max : float, optional
        Maximum radial integration domain.

    n_points : int, optional
        Number of radial output points.

    K_poly : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    mu : float, optional
        Boson mass.

    omega_tol : float, optional
        Convergence tolerance for the shooting frequency.

    max_iterations : int, optional
        Maximum number of bisection iterations.

    verbose : bool, optional
        Print iteration diagnostics.

    Returns
    -------
    dict
        Shooting result containing

            omega
            result
            converged
            iterations
            omega_low
            omega_high
            nodes
    """

    if phi_c <= 0.0:
        raise ValueError(
            "phi_c must be positive for scalar shooting."
        )

    if omega_low <= 0.0:
        raise ValueError(
            "omega_low must be positive."
        )

    if omega_high <= omega_low:
        raise ValueError(
            "omega_high must be larger than omega_low."
        )

    w_low = float(omega_low)
    w_high = float(omega_high)

    # --------------------------------------------------------
    # Evaluate the initial bracket
    # --------------------------------------------------------

    phi_low, result_low, nodes_low = trial_scalar_value(
        rho0_c=rho0_c,
        phi_c=phi_c,
        omega=w_low,
        r_max=r_max,
        n_points=n_points,
        K_poly=K_poly,
        gamma=gamma,
        mu=mu,
    )

    phi_high, result_high, nodes_high = trial_scalar_value(
        rho0_c=rho0_c,
        phi_c=phi_c,
        omega=w_high,
        r_max=r_max,
        n_points=n_points,
        K_poly=K_poly,
        gamma=gamma,
        mu=mu,
    )

    if verbose:

        print(
            "Initial shooting bracket"
        )

        print(
            f"  omega_low  = {w_low:.10f}"
            f" | phi_end = {phi_low:+.6e}"
            f" | nodes = {nodes_low}"
        )

        print(
            f"  omega_high = {w_high:.10f}"
            f" | phi_end = {phi_high:+.6e}"
            f" | nodes = {nodes_high}"
        )

    # --------------------------------------------------------
    # Bisection loop
    # --------------------------------------------------------

    final_result = None
    final_nodes = None

    for iteration in range(
        1,
        max_iterations + 1,
    ):

        w_mid = 0.5 * (
            w_low + w_high
        )

        phi_mid, result_mid, nodes_mid = trial_scalar_value(
            rho0_c=rho0_c,
            phi_c=phi_c,
            omega=w_mid,
            r_max=r_max,
            n_points=n_points,
            K_poly=K_poly,
            gamma=gamma,
            mu=mu,
        )

        final_result = result_mid
        final_nodes = nodes_mid

        if verbose:

            print(
                f"{iteration:3d}"
                f" | omega = {w_mid:.10f}"
                f" | phi_end = {phi_mid:+.6e}"
                f" | nodes = {nodes_mid}"
                f" | r_end = {result_mid['r'][-1]:.6f}"
                f" | success = {result_mid['success']}"
            )

        # ----------------------------------------------------
        # Fundamental-mode branch selection
        # ----------------------------------------------------

        # If the trial already crossed zero, omega is on the
        # oscillatory side of the fundamental eigenvalue.
        if nodes_mid > 0:

            w_high = w_mid
            phi_high = phi_mid

        else:

            # Nodeless solution.
            #
            # The sign of the outer scalar field determines
            # which side of the eigenvalue we are on.
            if phi_mid > 0.0:

                w_low = w_mid
                phi_low = phi_mid

            else:

                w_high = w_mid
                phi_high = phi_mid

        # ----------------------------------------------------
        # Frequency convergence
        # ----------------------------------------------------

        if abs(
            w_high - w_low
        ) < omega_tol:

            omega = 0.5 * (
                w_low + w_high
            )

            diagnostic, result, nodes = trial_scalar_value(
                rho0_c=rho0_c,
                phi_c=phi_c,
                omega=omega,
                r_max=r_max,
                n_points=n_points,
                K_poly=K_poly,
                gamma=gamma,
                mu=mu,
            )

            return {
                "omega": omega,
                "result": result,
                "converged": True,
                "iterations": iteration,
                "omega_low": w_low,
                "omega_high": w_high,
                "phi_end": diagnostic,
                "nodes": nodes,
            }

    # --------------------------------------------------------
    # Maximum number of iterations reached
    # --------------------------------------------------------

    omega = 0.5 * (
        w_low + w_high
    )

    return {
        "omega": omega,
        "result": final_result,
        "converged": False,
        "iterations": max_iterations,
        "omega_low": w_low,
        "omega_high": w_high,
        "phi_end": (
            float(final_result["phi"][-1])
            if final_result is not None
            else np.nan
        ),
        "nodes": final_nodes,
    }