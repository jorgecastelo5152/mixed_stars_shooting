"""
Field equations for static mixed fermion-boson stars.

The model contains

    - a static spherically symmetric spacetime,
    - a perfect fluid with a polytropic equation of state,
    - a canonical complex scalar field,
    - a purely massive scalar potential.

Units
-----
Geometrized units are used throughout:

    G = c = hbar = 1.

The metric is

    ds^2 = -alpha(r)^2 dt^2
           + a(r)^2 dr^2
           + r^2 dOmega^2,

and the complex scalar field is written as

    Phi(t, r) = phi(r) exp(-i omega t).

The scalar potential is

    V(phi) = mu^2 phi^2.

No scalar self-interactions or non-canonical kinetic terms
are included.
"""

import numpy as np

from .eos import energy_density_from_pressure


FOUR_PI = 4.0 * np.pi


def scalar_potential(
    phi,
    mu=1.0,
):
    """
    Massive scalar potential.

    Parameters
    ----------
    phi : float or array_like
        Radial scalar-field amplitude.

    mu : float, optional
        Boson mass parameter.

    Returns
    -------
    float or ndarray
        Scalar potential V(phi) = mu^2 phi^2.
    """

    return mu**2 * np.asarray(phi)**2


def scalar_mass_term(
    mu=1.0,
):
    """
    Return dV/d(phi^2) for the massive potential.

    For

        V(phi) = mu^2 phi^2,

    one has

        dV / d(phi^2) = mu^2.
    """

    return mu**2


def mixed_star_rhs(
    r,
    y,
    omega,
    K_poly=100.0,
    gamma=2.0,
    mu=1.0,
):
    """
    Right-hand side of the mixed fermion-boson star equations.

    Parameters
    ----------
    r : float
        Radial coordinate. The integration must start at r > 0.

    y : array_like
        Dynamical variables

            y = [a, alpha, psi, phi, p],

        where

            a
                radial metric function,

            alpha
                lapse function,

            psi = dphi/dr
                radial derivative of the scalar field,

            phi
                scalar-field amplitude,

            p
                fluid pressure.

    omega : float
        Scalar-field eigenfrequency used in the shooting method.

    K_poly : float, optional
        Polytropic constant.

    gamma : float, optional
        Polytropic exponent.

    mu : float, optional
        Scalar-field mass.

    Returns
    -------
    ndarray
        Radial derivatives

            [da/dr,
             dalpha/dr,
             dpsi/dr,
             dphi/dr,
             dp/dr].
    """

    if r <= 0.0:
        raise ValueError(
            "The ODE system is singular at r = 0. "
            "Start the integration at a small positive radius."
        )

    a, alpha, psi, phi, p = y

    # --------------------------------------------------------
    # Fluid sector
    # --------------------------------------------------------

    # Once the fluid pressure reaches zero, the fluid contribution
    # is switched off while the bosonic field continues to evolve.
    p_fluid = max(
        float(p),
        0.0,
    )

    epsilon = float(
        energy_density_from_pressure(
            p_fluid,
            K=K_poly,
            gamma=gamma,
        )
    )

    # --------------------------------------------------------
    # Scalar sector
    # --------------------------------------------------------

    V = float(
        scalar_potential(
            phi,
            mu=mu,
        )
    )

    omega_over_alpha_sq = (
        omega / alpha
    ) ** 2

    a_sq = a**2

    # --------------------------------------------------------
    # Einstein equation for a(r)
    # --------------------------------------------------------

    da_dr = (
        a / 2.0
        * (
            (1.0 - a_sq) / r

            + FOUR_PI
            * r
            * (
                psi**2

                + a_sq
                * (
                    omega_over_alpha_sq
                    * phi**2

                    + 2.0 * epsilon

                    + V
                )
            )
        )
    )

    # --------------------------------------------------------
    # Einstein equation for alpha(r)
    # --------------------------------------------------------

    metric_source = (
        (a_sq - 1.0) / r

        + FOUR_PI
        * r
        * (
            psi**2

            + a_sq
            * (
                omega_over_alpha_sq
                * phi**2

                + 2.0 * p_fluid

                - V
            )
        )
    )

    dalpha_dr = (
        alpha / 2.0
        * metric_source
    )

    # --------------------------------------------------------
    # Klein-Gordon equation
    # --------------------------------------------------------

    geometric_friction = (
        (1.0 + a_sq) / r

        - FOUR_PI
        * a_sq
        * r
        * (
            V
            + p_fluid
            + epsilon
        )
    )

    effective_mass_term = (
        scalar_mass_term(mu=mu)
        - omega_over_alpha_sq
    )

    dpsi_dr = (
        a_sq
        * effective_mass_term
        * phi

        - psi
        * geometric_friction
    )

    dphi_dr = psi

    # --------------------------------------------------------
    # Hydrostatic-equilibrium equation
    # --------------------------------------------------------

    if p > 0.0:

        dp_dr = (
            -(p_fluid + epsilon)
            / 2.0
            * metric_source
        )

    else:

        # Outside the fluid surface the fermionic component
        # remains identically zero.
        dp_dr = 0.0

    return np.array(
        [
            da_dr,
            dalpha_dr,
            dpsi_dr,
            dphi_dr,
            dp_dr,
        ],
        dtype=float,
    )