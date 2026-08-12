import numpy as np

from compact_star_shooting.integrator import (
    initial_conditions,
    integrate_star,
)


def test_initial_conditions():

    rho0_c = 1.0e-3
    phi_c = 0.05

    y0 = initial_conditions(
        rho0_c,
        phi_c,
    )

    assert y0.shape == (5,)

    assert np.isclose(
        y0[0],
        1.0,
    )

    assert np.isclose(
        y0[1],
        1.0,
    )

    assert np.isclose(
        y0[2],
        0.0,
    )

    assert np.isclose(
        y0[3],
        phi_c,
    )

    assert y0[4] > 0.0


def test_integration_returns_profiles():

    result = integrate_star(
        rho0_c=1.0e-3,
        phi_c=0.01,
        omega=0.9,
        r_max=5.0,
        n_points=500,
    )

    assert result["success"]

    assert len(result["r"]) == 500

    assert result["a"].shape == result["r"].shape
    assert result["alpha"].shape == result["r"].shape
    assert result["phi"].shape == result["r"].shape
    assert result["psi"].shape == result["r"].shape
    assert result["p"].shape == result["r"].shape

    assert np.all(
        np.isfinite(result["r"])
    )


def test_pressure_is_non_negative_after_postprocessing():

    result = integrate_star(
        rho0_c=1.0e-3,
        phi_c=0.0,
        omega=0.0,
        r_max=5.0,
        n_points=500,
    )

    assert np.all(
        result["p"] >= 0.0
    )