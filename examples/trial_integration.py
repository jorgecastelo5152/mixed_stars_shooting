import matplotlib.pyplot as plt

from compact_star_shooting.integrator import integrate_star


# ============================================================
# Central configuration
# ============================================================

rho0_c = 1.0e-3
phi_c = 0.05


# ============================================================
# Trial eigenfrequencies
# ============================================================

omega_values = [
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
]


# ============================================================
# Radial integration settings
# ============================================================

r_max = 15.0
n_points = 2000


# ============================================================
# Integrate trial solutions
# ============================================================

results = []

for omega in omega_values:

    print(f"\nTrying omega = {omega:.3f}")

    try:

        result = integrate_star(
            rho0_c=rho0_c,
            phi_c=phi_c,
            omega=omega,
            r_max=r_max,
            n_points=n_points,
            K_poly=100.0,
            gamma=2.0,
            mu=1.0,
        )

        results.append(result)

        print(
            f"  success"
            f" | phi(r_max) = {result['phi'][-1]:+.6e}"
            f" | R_fluid = {result['R_fluid']:.6f}"
        )

    except RuntimeError as error:

        print(
            f"  integration failed"
            f" | {error}"
        )


# ============================================================
# Scalar-field profiles
# ============================================================

if results:

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for result in results:

        ax.plot(
            result["r"],
            result["phi"],
            label=rf"$\omega={result['omega']:.2f}$",
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi(r)$")

    ax.set_title(
        "Trial scalar-field profiles"
    )

    ax.legend()

    fig.tight_layout()

    plt.show()


# ============================================================
# Fluid-pressure profiles
# ============================================================

if results:

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for result in results:

        ax.plot(
            result["r"],
            result["p"],
            label=rf"$\omega={result['omega']:.2f}$",
        )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$p(r)$")

    ax.set_title(
        "Fluid-pressure profiles"
    )

    ax.legend()

    fig.tight_layout()

    plt.show()