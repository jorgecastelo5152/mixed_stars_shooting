"""
Observable quantities for static fermion-boson stars.

The module computes quantities derived from an already integrated
stellar configuration:

    - asymptotic lapse normalization,
    - physical scalar frequency,
    - gravitational mass profile,
    - fluid and scalar energy-density profiles,
    - baryonic rest-mass profile,
    - baryon number,
    - boson number,
    - bosonic rest-mass profile,
    - characteristic 99% radii,
    - binding energy and mass defect.

Units
-----
Geometrized units are used:

    G = c = hbar = 1.

The spacetime metric is

    ds^2 = -alpha(r)^2 dt^2
           + a(r)^2 dr^2
           + r^2 dOmega^2.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .eos import (
    rest_density_from_pressure,
    energy_density_from_pressure,
)


FOUR_PI = 4.0 * np.pi
EIGHT_PI = 8.0 * np.pi


# ============================================================
# Asymptotic normalization
# ============================================================

def normalize_lapse(
    alpha,
    omega_shoot,
):
    """
    Normalize the lapse so that alpha -> 1 asymptotically.

    The numerical integration uses

        alpha(0) = 1,

    while the physical time coordinate is fixed by imposing

        alpha(infinity) = 1.

    On the finite numerical domain,

        alpha_inf ~= alpha(r_max),

    so that

        alpha_phys = alpha / alpha_inf,

    and

        omega_phys = omega_shoot / alpha_inf.

    Parameters
    ----------
    alpha : array_like
        Numerical lapse profile.

    omega_shoot : float
        Frequency used during the shooting integration.

    Returns
    -------
    alpha_phys : ndarray
        Asymptotically normalized lapse.

    omega_phys : float
        Physical scalar eigenfrequency.

    alpha_inf : float
        Numerical asymptotic lapse before normalization.
    """

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    if alpha.ndim != 1:
        raise ValueError(
            "alpha must be a one-dimensional array."
        )

    if len(alpha) == 0:
        raise ValueError(
            "alpha cannot be empty."
        )

    alpha_inf = float(
        alpha[-1]
    )

    if not np.isfinite(alpha_inf):
        raise ValueError(
            "Asymptotic lapse is not finite."
        )

    if alpha_inf <= 0.0:
        raise ValueError(
            "Asymptotic lapse must be positive."
        )

    alpha_phys = (
        alpha / alpha_inf
    )

    omega_phys = (
        float(omega_shoot)
        / alpha_inf
    )

    return (
        alpha_phys,
        omega_phys,
        alpha_inf,
    )


# ============================================================
# Gravitational mass
# ============================================================

def mass_profile(
    r,
    a,
):
    """
    Compute the gravitational mass profile.

    From

        a(r)^2 = [1 - 2 m(r)/r]^{-1},

    one obtains

        m(r)
        =
        r/2 * [1 - 1/a(r)^2].

    Parameters
    ----------
    r : array_like
        Radial coordinate.

    a : array_like
        Radial metric function.

    Returns
    -------
    ndarray
        Gravitational mass profile.
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    a = np.asarray(
        a,
        dtype=float,
    )

    if r.shape != a.shape:
        raise ValueError(
            "r and a must have the same shape."
        )

    if np.any(a == 0.0):
        raise ValueError(
            "Metric function a cannot vanish."
        )

    return (
        0.5
        * r
        * (
            1.0
            - 1.0 / a**2
        )
    )


# ============================================================
# Matter energy densities
# ============================================================

def fluid_energy_density_profile(
    p,
    K_poly=100.0,
    gamma=2.0,
):
    """
    Compute the fluid energy-density profile.

    Outside the fluid surface, where p <= 0, the density
    is set to zero.
    """

    p = np.asarray(
        p,
        dtype=float,
    )

    return np.asarray(
        energy_density_from_pressure(
            p,
            K=K_poly,
            gamma=gamma,
        ),
        dtype=float,
    )


def fluid_rest_density_profile(
    p,
    K_poly=100.0,
    gamma=2.0,
):
    """
    Compute the fluid rest-mass density rho0(r).
    """

    p = np.asarray(
        p,
        dtype=float,
    )

    return np.asarray(
        rest_density_from_pressure(
            p,
            K=K_poly,
            gamma=gamma,
        ),
        dtype=float,
    )


def scalar_energy_density_profile(
    a,
    alpha,
    phi,
    psi,
    omega,
    mu=1.0,
):
    """
    Compute the canonical scalar-field energy density.

    For

        Phi(t,r) = phi(r) exp(-i omega t)

    and

        V(phi) = mu^2 phi^2,

    the scalar contribution consistent with the Einstein
    equations used in this package is

        rho_phi =
            1/2 * psi^2 / a^2
          + 1/2 * omega^2 phi^2 / alpha^2
          + 1/2 * mu^2 phi^2.
    """

    a = np.asarray(a, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)

    return (
        0.5 * psi**2 / a**2
        + 0.5 * (omega / alpha)**2 * phi**2
        + 0.5 * mu**2 * phi**2
    )


# ============================================================
# Gravitational component profiles
# ============================================================

def component_mass_profile(
    r,
    energy_density,
):
    """
    Compute the cumulative gravitational-mass contribution
    associated with an energy-density profile.

    In areal coordinates,

        dm/dr = 4 pi r^2 rho.
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    energy_density = np.asarray(
        energy_density,
        dtype=float,
    )

    if r.shape != energy_density.shape:
        raise ValueError(
            "r and energy_density must have the same shape."
        )

    integrand = (
        FOUR_PI
        * r**2
        * energy_density
    )

    return cumulative_trapezoid(
        integrand,
        r,
        initial=0.0,
    )


# ============================================================
# Baryonic quantities
# ============================================================

def baryonic_rest_mass_profile(
    r,
    a,
    rho0,
):
    """
    Compute the baryonic rest-mass profile.

    The proper-volume integral is

        M0(r)
        =
        4 pi integral a(r) rho0(r) r^2 dr.
    """

    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    rho0 = np.asarray(rho0, dtype=float)

    integrand = (
        FOUR_PI
        * r**2
        * a
        * rho0
    )

    return cumulative_trapezoid(
        integrand,
        r,
        initial=0.0,
    )


def baryon_number_profile(
    r,
    a,
    rho0,
    baryon_mass=1.0,
):
    """
    Compute the baryon-number profile.

    If rho0 is the rest-mass density,

        n = rho0 / m_b,

    and

        N_B
        =
        4 pi integral a n r^2 dr.

    Parameters
    ----------
    baryon_mass : float, optional
        Single-baryon rest mass in the same code units.

        The default baryon_mass=1 corresponds to measuring
        rest mass in units of the constituent particle mass.
    """

    if baryon_mass <= 0.0:
        raise ValueError(
            "baryon_mass must be positive."
        )

    return (
        baryonic_rest_mass_profile(
            r,
            a,
            rho0,
        )
        / baryon_mass
    )


# ============================================================
# Bosonic quantities
# ============================================================

def boson_number_profile(
    r,
    a,
    alpha_phys,
    phi,
    omega_phys,
):
    """
    Compute the conserved boson-number profile.

    For a canonical complex scalar field,

        N_phi
        =
        4 pi integral
        r^2 a(r)
        [omega_phys / alpha_phys(r)]
        phi(r)^2 dr.

    This is the canonical K(phi)=1 limit of the current integral.
    """

    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    alpha_phys = np.asarray(alpha_phys, dtype=float)
    phi = np.asarray(phi, dtype=float)

    integrand = (
        FOUR_PI
        * r**2
        * a
        * (
            omega_phys / alpha_phys
        )
        * phi**2
    )

    return cumulative_trapezoid(
        integrand,
        r,
        initial=0.0,
    )


def boson_rest_mass_profile(
    N_boson,
    mu=1.0,
):
    """
    Convert boson number into total bosonic rest mass.

    In natural units,

        M0_boson = mu * N_boson.
    """

    N_boson = np.asarray(
        N_boson,
        dtype=float,
    )

    return (
        mu
        * N_boson
    )


# ============================================================
# Characteristic radii
# ============================================================

def enclosed_fraction_radius(
    r,
    cumulative_quantity,
    fraction=0.99,
):
    """
    Radius containing a specified fraction of a cumulative quantity.

    Parameters
    ----------
    r : array_like
        Radial grid.

    cumulative_quantity : array_like
        Monotonically increasing cumulative quantity.

    fraction : float, optional
        Desired enclosed fraction. Default is 0.99.

    Returns
    -------
    float
        First radius containing the requested fraction.

        Returns np.nan if the final quantity is zero or invalid.
    """

    if not (
        0.0 < fraction <= 1.0
    ):
        raise ValueError(
            "fraction must lie in (0, 1]."
        )

    r = np.asarray(
        r,
        dtype=float,
    )

    q = np.asarray(
        cumulative_quantity,
        dtype=float,
    )

    if r.shape != q.shape:
        raise ValueError(
            "r and cumulative_quantity must have the same shape."
        )

    q_final = float(
        q[-1]
    )

    if (
        not np.isfinite(q_final)
        or q_final <= 0.0
    ):
        return np.nan

    target = (
        fraction
        * q_final
    )

    indices = np.where(
        q >= target
    )[0]

    if len(indices) == 0:
        return np.nan

    return float(
        r[indices[0]]
    )


# ============================================================
# Binding energy
# ============================================================

def binding_energy(
    gravitational_mass,
    baryonic_rest_mass=0.0,
    bosonic_rest_mass=0.0,
):
    """
    Compute total rest mass, binding energy, and mass defect.

    We define

        M_rest
        =
        M0_fermion + M0_boson,

    and

        E_bind
        =
        M_grav - M_rest.

    Therefore a gravitationally bound configuration has

        E_bind < 0.

    The positive mass defect is

        M_defect
        =
        M_rest - M_grav.
    """

    M_grav = float(
        gravitational_mass
    )

    M_rest = (
        float(baryonic_rest_mass)
        + float(bosonic_rest_mass)
    )

    E_bind = (
        M_grav
        - M_rest
    )

    mass_defect = (
        M_rest
        - M_grav
    )

    if M_rest > 0.0:

        relative_binding = (
            E_bind / M_rest
        )

        relative_mass_defect = (
            mass_defect / M_rest
        )

    else:

        relative_binding = np.nan
        relative_mass_defect = np.nan

    return {
        "rest_mass": M_rest,
        "binding_energy": E_bind,
        "mass_defect": mass_defect,
        "relative_binding_energy": relative_binding,
        "relative_mass_defect": relative_mass_defect,
    }


# ============================================================
# Complete observable extraction
# ============================================================

def compute_observables(
    result,
    omega_shoot,
    K_poly=100.0,
    gamma=2.0,
    mu=1.0,
    baryon_mass=1.0,
    radius_fraction=0.99,
):
    """
    Compute the main observables of an integrated configuration.

    Parameters
    ----------
    result : dict
        Output of integrate_star.

    omega_shoot : float
        Eigenfrequency obtained from the shooting procedure.

    K_poly, gamma : float
        Polytropic EOS parameters.

    mu : float
        Boson mass.

    baryon_mass : float
        Fermionic constituent mass in code units.

    radius_fraction : float
        Fraction used for effective radii. Default is 0.99.

    Returns
    -------
    dict
        Scalar observables and radial profiles.
    """

    r = np.asarray(
        result["r"],
        dtype=float,
    )

    a = np.asarray(
        result["a"],
        dtype=float,
    )

    alpha = np.asarray(
        result["alpha"],
        dtype=float,
    )

    phi = np.asarray(
        result["phi"],
        dtype=float,
    )

    psi = np.asarray(
        result["psi"],
        dtype=float,
    )

    p = np.asarray(
        result["p"],
        dtype=float,
    )

    # --------------------------------------------------------
    # Time normalization
    # --------------------------------------------------------

    (
        alpha_phys,
        omega_phys,
        alpha_inf,
    ) = normalize_lapse(
        alpha,
        omega_shoot,
    )

    # --------------------------------------------------------
    # Gravitational mass
    # --------------------------------------------------------

    M_profile = mass_profile(
        r,
        a,
    )

    M_total = float(
        M_profile[-1]
    )

    # --------------------------------------------------------
    # Matter densities
    # --------------------------------------------------------

    epsilon_fluid = (
        fluid_energy_density_profile(
            p,
            K_poly=K_poly,
            gamma=gamma,
        )
    )

    rho0 = (
        fluid_rest_density_profile(
            p,
            K_poly=K_poly,
            gamma=gamma,
        )
    )

    rho_scalar = (
        scalar_energy_density_profile(
            a,
            alpha_phys,
            phi,
            psi,
            omega_phys,
            mu=mu,
        )
    )

    # --------------------------------------------------------
    # Gravitational component contributions
    # --------------------------------------------------------

    M_fluid_grav_profile = (
        component_mass_profile(
            r,
            epsilon_fluid,
        )
    )

    M_scalar_grav_profile = (
        component_mass_profile(
            r,
            rho_scalar,
        )
    )

    # --------------------------------------------------------
    # Fermionic rest mass / particle number
    # --------------------------------------------------------

    M_baryon_profile = (
        baryonic_rest_mass_profile(
            r,
            a,
            rho0,
        )
    )

    N_baryon_profile = (
        baryon_number_profile(
            r,
            a,
            rho0,
            baryon_mass=baryon_mass,
        )
    )

    M_baryon = float(
        M_baryon_profile[-1]
    )

    N_baryon = float(
        N_baryon_profile[-1]
    )

    # --------------------------------------------------------
    # Boson number / rest mass
    # --------------------------------------------------------

    N_boson_profile = (
        boson_number_profile(
            r,
            a,
            alpha_phys,
            phi,
            omega_phys,
        )
    )

    M_boson_profile = (
        boson_rest_mass_profile(
            N_boson_profile,
            mu=mu,
        )
    )

    N_boson = float(
        N_boson_profile[-1]
    )

    M_boson = float(
        M_boson_profile[-1]
    )

    # --------------------------------------------------------
    # Characteristic radii
    # --------------------------------------------------------

    R_total = enclosed_fraction_radius(
        r,
        M_profile,
        fraction=radius_fraction,
    )

    R_boson = enclosed_fraction_radius(
        r,
        N_boson_profile,
        fraction=radius_fraction,
    )

    R_baryon = enclosed_fraction_radius(
        r,
        M_baryon_profile,
        fraction=radius_fraction,
    )

    R_fluid = float(
        result.get(
            "R_fluid",
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Binding energy
    # --------------------------------------------------------

    bind = binding_energy(
        gravitational_mass=M_total,
        baryonic_rest_mass=M_baryon,
        bosonic_rest_mass=M_boson,
    )

    return {
        # Main scalar observables
        "M_total": M_total,

        "M_baryon_rest": M_baryon,
        "M_boson_rest": M_boson,

        "N_baryon": N_baryon,
        "N_boson": N_boson,

        "R_fluid": R_fluid,
        "R_total_99": R_total,
        "R_baryon_99": R_baryon,
        "R_boson_99": R_boson,

        "omega_shoot": float(omega_shoot),
        "omega_phys": omega_phys,
        "alpha_inf": alpha_inf,

        # Binding quantities
        **bind,

        # Profiles
        "r": r,

        "alpha_phys": alpha_phys,

        "M_profile": M_profile,
        "M_fluid_grav_profile": M_fluid_grav_profile,
        "M_scalar_grav_profile": M_scalar_grav_profile,

        "rho0": rho0,
        "epsilon_fluid": epsilon_fluid,
        "rho_scalar": rho_scalar,

        "M_baryon_profile": M_baryon_profile,
        "N_baryon_profile": N_baryon_profile,

        "N_boson_profile": N_boson_profile,
        "M_boson_profile": M_boson_profile,
    }