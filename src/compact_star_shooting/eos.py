"""
Polytropic equation of state for the fluid component.

The fluid is described by

    p = K rho0^Gamma,

with total energy density

    epsilon = rho0 + p / (Gamma - 1),

in geometrized units G = c = hbar = 1.
"""

import numpy as np


def pressure_from_rest_density(
    rho0,
    K=100.0,
    gamma=2.0,
):
    """
    Compute pressure from the rest-mass density.

    Parameters
    ----------
    rho0 : float or array_like
        Rest-mass density.

    K : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    Returns
    -------
    float or ndarray
        Pressure.
    """

    rho0 = np.asarray(rho0, dtype=float)

    if np.any(rho0 < 0.0):
        raise ValueError(
            "Rest-mass density must be non-negative."
        )

    return K * rho0**gamma


def rest_density_from_pressure(
    p,
    K=100.0,
    gamma=2.0,
):
    """
    Compute rest-mass density from pressure.

    Parameters
    ----------
    p : float or array_like
        Fluid pressure.

    K : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    Returns
    -------
    float or ndarray
        Rest-mass density.
    """

    p = np.asarray(p, dtype=float)

    if K <= 0.0:
        raise ValueError(
            "Polytropic constant K must be positive."
        )

    if gamma <= 1.0:
        raise ValueError(
            "Polytropic exponent gamma must be larger than 1."
        )

    # Outside the fluid surface we use rho0 = 0.
    p_positive = np.maximum(
        p,
        0.0,
    )

    return (
        p_positive / K
    ) ** (1.0 / gamma)


def energy_density_from_pressure(
    p,
    K=100.0,
    gamma=2.0,
):
    """
    Compute the total fluid energy density.

    The polytropic model uses

        epsilon = rho0 + p / (gamma - 1),

    where rho0 is the rest-mass density.

    Parameters
    ----------
    p : float or array_like
        Fluid pressure.

    K : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    Returns
    -------
    float or ndarray
        Total energy density.
    """

    if gamma <= 1.0:
        raise ValueError(
            "Polytropic exponent gamma must be larger than 1."
        )

    p = np.asarray(p, dtype=float)

    p_positive = np.maximum(
        p,
        0.0,
    )

    rho0 = rest_density_from_pressure(
        p_positive,
        K=K,
        gamma=gamma,
    )

    return (
        rho0
        + p_positive / (gamma - 1.0)
    )


def eos_from_pressure(
    p,
    K=100.0,
    gamma=2.0,
):
    """
    Evaluate the complete polytropic equation of state.

    Parameters
    ----------
    p : float or array_like
        Fluid pressure.

    K : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    Returns
    -------
    rho0 : float or ndarray
        Rest-mass density.

    epsilon : float or ndarray
        Total energy density.
    """

    rho0 = rest_density_from_pressure(
        p,
        K=K,
        gamma=gamma,
    )

    epsilon = energy_density_from_pressure(
        p,
        K=K,
        gamma=gamma,
    )

    return rho0, epsilon