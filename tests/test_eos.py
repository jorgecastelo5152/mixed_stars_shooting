import numpy as np

from compact_star_shooting.eos import (
    pressure_from_rest_density,
    rest_density_from_pressure,
    energy_density_from_pressure,
)


def test_polytropic_inverse_relation():

    K = 100.0
    gamma = 2.0
    rho0 = 1.5e-3

    p = pressure_from_rest_density(
        rho0,
        K=K,
        gamma=gamma,
    )

    rho0_recovered = rest_density_from_pressure(
        p,
        K=K,
        gamma=gamma,
    )

    assert np.isclose(
        rho0,
        rho0_recovered,
    )


def test_energy_density():

    K = 100.0
    gamma = 2.0
    rho0 = 1.0e-3

    p = pressure_from_rest_density(
        rho0,
        K=K,
        gamma=gamma,
    )

    epsilon = energy_density_from_pressure(
        p,
        K=K,
        gamma=gamma,
    )

    expected = (
        rho0
        + p / (gamma - 1.0)
    )

    assert np.isclose(
        epsilon,
        expected,
    )


def test_zero_pressure():

    rho0 = rest_density_from_pressure(
        0.0
    )

    epsilon = energy_density_from_pressure(
        0.0
    )

    assert rho0 == 0.0
    assert epsilon == 0.0