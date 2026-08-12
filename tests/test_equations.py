import numpy as np

from compact_star_shooting.equations import (
    scalar_potential,
    scalar_mass_term,
    mixed_star_rhs,
)


def test_massive_scalar_potential():

    phi = 0.2
    mu = 1.5

    V = scalar_potential(
        phi,
        mu=mu,
    )

    expected = (
        mu**2
        * phi**2
    )

    assert np.isclose(
        V,
        expected,
    )


def test_scalar_mass_term():

    mu = 1.7

    result = scalar_mass_term(
        mu=mu,
    )

    assert np.isclose(
        result,
        mu**2,
    )


def test_rhs_returns_finite_values():

    r = 1.0

    y = np.array(
        [
            1.0,      # a
            1.0,      # alpha
            0.0,      # psi
            0.05,     # phi
            1.0e-4,   # p
        ]
    )

    dydr = mixed_star_rhs(
        r,
        y,
        omega=0.9,
        K_poly=100.0,
        gamma=2.0,
        mu=1.0,
    )

    assert dydr.shape == (5,)

    assert np.all(
        np.isfinite(dydr)
    )


def test_phi_derivative_is_psi():

    r = 1.0

    psi = 0.03

    y = np.array(
        [
            1.0,
            1.0,
            psi,
            0.05,
            1.0e-4,
        ]
    )

    dydr = mixed_star_rhs(
        r,
        y,
        omega=0.9,
    )

    assert np.isclose(
        dydr[3],
        psi,
    )


def test_fluid_switches_off_for_negative_pressure():

    r = 2.0

    y = np.array(
        [
            1.0,
            1.0,
            0.01,
            0.05,
            -1.0e-8,
        ]
    )

    dydr = mixed_star_rhs(
        r,
        y,
        omega=0.9,
    )

    assert dydr[4] == 0.0