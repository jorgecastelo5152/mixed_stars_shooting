import numpy as np

from compact_star_shooting.observables import (
    normalize_lapse,
    mass_profile,
    enclosed_fraction_radius,
    baryonic_rest_mass_profile,
    boson_rest_mass_profile,
    binding_energy,
)


def test_lapse_normalization():

    alpha = np.array(
        [
            1.0,
            1.1,
            1.2,
            1.25,
        ]
    )

    omega_shoot = 1.1

    alpha_phys, omega_phys, alpha_inf = normalize_lapse(
        alpha,
        omega_shoot,
    )

    assert np.isclose(
        alpha_inf,
        1.25,
    )

    assert np.isclose(
        alpha_phys[-1],
        1.0,
    )

    assert np.isclose(
        omega_phys,
        omega_shoot / alpha_inf,
    )


def test_flat_spacetime_has_zero_mass():

    r = np.linspace(
        1.0e-6,
        10.0,
        100,
    )

    a = np.ones_like(r)

    mass = mass_profile(
        r,
        a,
    )

    assert np.allclose(
        mass,
        0.0,
    )


def test_mass_profile_shape():

    r = np.linspace(
        1.0e-6,
        10.0,
        100,
    )

    a = 1.0 + 0.01 * r

    mass = mass_profile(
        r,
        a,
    )

    assert mass.shape == r.shape

    assert np.all(
        np.isfinite(mass)
    )


def test_enclosed_fraction_radius():

    r = np.linspace(
        0.0,
        10.0,
        101,
    )

    q = r.copy()

    R90 = enclosed_fraction_radius(
        r,
        q,
        fraction=0.90,
    )

    assert np.isclose(
        R90,
        9.0,
    )


def test_baryonic_rest_mass_is_positive():

    r = np.linspace(
        1.0e-6,
        5.0,
        100,
    )

    a = np.ones_like(r)

    rho0 = np.ones_like(r) * 1.0e-3

    M0 = baryonic_rest_mass_profile(
        r,
        a,
        rho0,
    )

    assert M0[-1] > 0.0

    assert np.all(
        np.diff(M0) >= 0.0
    )


def test_boson_rest_mass():

    N = np.array(
        [
            0.0,
            1.0,
            2.0,
        ]
    )

    mu = 0.5

    M0 = boson_rest_mass_profile(
        N,
        mu=mu,
    )

    assert np.allclose(
        M0,
        mu * N,
    )


def test_binding_energy():

    result = binding_energy(
        gravitational_mass=0.9,
        baryonic_rest_mass=0.6,
        bosonic_rest_mass=0.4,
    )

    assert np.isclose(
        result["rest_mass"],
        1.0,
    )

    assert np.isclose(
        result["binding_energy"],
        -0.1,
    )

    assert np.isclose(
        result["mass_defect"],
        0.1,
    )