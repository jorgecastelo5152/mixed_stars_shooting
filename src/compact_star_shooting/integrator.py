"""
Radial integration utilities for mixed fermion-boson stars.

This module integrates the coupled Einstein-fluid-Klein-Gordon
system for a fixed trial value of the scalar eigenfrequency omega.

The shooting procedure itself is implemented separately.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .equations import mixed_star_rhs
from .eos import pressure_from_rest_density


def initial_conditions(
    rho0_c,
    phi_c,
    K_poly=100.0,
    gamma=2.0,
):
    """
    Construct regular central initial conditions.

    Parameters
    ----------
    rho0_c : float
        Central rest-mass density of the fluid.

    phi_c : float
        Central scalar-field amplitude.

    K_poly : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    Returns
    -------
    ndarray
        Initial state

            [a, alpha, psi, phi, p].
    """

    if rho0_c < 0.0:
        raise ValueError(
            "Central rest-mass density must be non-negative."
        )

    if phi_c < 0.0:
        raise ValueError(
            "Central scalar amplitude must be non-negative."
        )

    p_c = float(
        pressure_from_rest_density(
            rho0_c,
            K=K_poly,
            gamma=gamma,
        )
    )

    return np.array(
        [
            1.0,    # a(0)
            1.0,    # alpha(0)
            0.0,    # phi'(0)
            phi_c,  # phi(0)
            p_c,    # p(0)
        ],
        dtype=float,
    )


def fluid_surface_radius(
    r,
    p,
    pressure_tol=1.0e-10,
):
    """
    Estimate the fluid surface radius.

    The surface is defined as the first radial point where

        p <= pressure_tol.

    Parameters
    ----------
    r : array_like
        Radial grid.

    p : array_like
        Fluid-pressure profile.

    pressure_tol : float, optional
        Numerical pressure threshold.

    Returns
    -------
    float
        Estimated fluid radius.

        Returns np.nan if the pressure does not reach the
        threshold inside the integration domain.
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    if r.shape != p.shape:
        raise ValueError(
            "r and p must have the same shape."
        )

    indices = np.where(
        p <= pressure_tol
    )[0]

    if len(indices) == 0:
        return np.nan

    return float(
        r[indices[0]]
    )


def integrate_star(
    rho0_c,
    phi_c,
    omega,
    r_max=50.0,
    n_points=5000,
    r_min=1.0e-6,
    K_poly=100.0,
    gamma=2.0,
    mu=1.0,
    method="DOP853",
    rtol=1.0e-10,
    atol=1.0e-12,
    pressure_tol=1.0e-10,
    raise_on_failure=True,
):
    """
    Integrate a mixed fermion-boson configuration for fixed omega.

    Parameters
    ----------
    rho0_c : float
        Central fluid rest-mass density.

    phi_c : float
        Central scalar amplitude.

    omega : float
        Trial scalar eigenfrequency.

    r_max : float, optional
        Maximum integration radius.

    n_points : int, optional
        Number of output grid points.

    r_min : float, optional
        Initial radius. Must be strictly positive.

    K_poly : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    mu : float, optional
        Scalar-field mass.

    method : str, optional
        SciPy solve_ivp integration method.

    rtol, atol : float, optional
        Relative and absolute solver tolerances.

    pressure_tol : float, optional
        Threshold used to estimate the fluid surface.

    raise_on_failure : bool, optional
        If True, raise RuntimeError when solve_ivp fails.

        If False, return the partial numerical solution.
        This is useful during the shooting procedure, where
        divergent trial frequencies still provide information
        about the eigenvalue bracket.

    Returns
    -------
    dict
        Dictionary containing the radial profiles and
        integration diagnostics.
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if r_min <= 0.0:
        raise ValueError(
            "r_min must be strictly positive."
        )

    if r_max <= r_min:
        raise ValueError(
            "r_max must be larger than r_min."
        )

    if n_points < 2:
        raise ValueError(
            "n_points must be at least 2."
        )

    if omega < 0.0:
        raise ValueError(
            "omega must be non-negative."
        )

    if rho0_c < 0.0:
        raise ValueError(
            "rho0_c must be non-negative."
        )

    if phi_c < 0.0:
        raise ValueError(
            "phi_c must be non-negative."
        )

    # --------------------------------------------------------
    # Central conditions
    # --------------------------------------------------------

    y0 = initial_conditions(
        rho0_c=rho0_c,
        phi_c=phi_c,
        K_poly=K_poly,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # Radial grid
    # --------------------------------------------------------

    r_eval = np.linspace(
        r_min,
        r_max,
        n_points,
    )

    # --------------------------------------------------------
    # Radial integration
    # --------------------------------------------------------

    solution = solve_ivp(
        mixed_star_rhs,
        t_span=(
            r_min,
            r_max,
        ),
        y0=y0,
        t_eval=r_eval,
        args=(
            omega,
            K_poly,
            gamma,
            mu,
        ),
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if (
        not solution.success
        and raise_on_failure
    ):
        raise RuntimeError(
            "Radial integration failed: "
            + solution.message
        )

    # --------------------------------------------------------
    # Extract profiles
    # --------------------------------------------------------

    r = np.asarray(
        solution.t,
        dtype=float,
    )

    a = np.asarray(
        solution.y[0],
        dtype=float,
    )

    alpha = np.asarray(
        solution.y[1],
        dtype=float,
    )

    psi = np.asarray(
        solution.y[2],
        dtype=float,
    )

    phi = np.asarray(
        solution.y[3],
        dtype=float,
    )

    p = np.asarray(
        solution.y[4],
        dtype=float,
    )

    # --------------------------------------------------------
    # Physical pressure profile
    # --------------------------------------------------------

    # Pressure values slightly below zero may occur because the
    # fluid surface generally lies between output grid points.
    # For physical post-processing we clip those values to zero.

    p_physical = np.maximum(
        p,
        0.0,
    )

    # --------------------------------------------------------
    # Fluid surface
    # --------------------------------------------------------

    # A pure boson star has rho0_c = 0 and therefore no
    # fermionic surface. Returning r_min in that case would be
    # numerically understandable but physically misleading.

    if rho0_c > 0.0:

        R_fluid = fluid_surface_radius(
            r,
            p,
            pressure_tol=pressure_tol,
        )

    else:

        R_fluid = np.nan

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {
        "r": r,
        "a": a,
        "alpha": alpha,
        "psi": psi,
        "phi": phi,
        "p": p_physical,

        "rho0_c": float(rho0_c),
        "phi_c": float(phi_c),
        "omega": float(omega),

        "K_poly": float(K_poly),
        "gamma": float(gamma),
        "mu": float(mu),

        "R_fluid": R_fluid,

        "success": bool(
            solution.success
        ),

        "message": solution.message,

        "solution": solution,
    }