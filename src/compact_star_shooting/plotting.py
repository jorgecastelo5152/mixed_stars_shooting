"""
Plotting utilities for compact-star solutions.

The functions in this module provide lightweight visualization
tools for individual stellar configurations and parameter-grid
results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_solution(
    result,
    observables=None,
    show=True,
    save_path=None,
):
    """
    Plot the radial structure of a stellar configuration.

    Parameters
    ----------
    result : dict
        Output of integrate_star or the final result returned
        by the shooting procedure.

    observables : dict or None
        Optional output of compute_observables.

        If supplied, the asymptotically normalized lapse and
        mass profile are used.

    show : bool, optional
        Display the figure.

    save_path : str or None, optional
        Save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    r = np.asarray(
        result["r"],
        dtype=float,
    )

    phi = np.asarray(
        result["phi"],
        dtype=float,
    )

    p = np.asarray(
        result["p"],
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

    if observables is not None:

        alpha_plot = np.asarray(
            observables["alpha_phys"],
            dtype=float,
        )

        mass = np.asarray(
            observables["M_profile"],
            dtype=float,
        )

    else:

        alpha_plot = alpha

        mass = (
            0.5
            * r
            * (
                1.0
                - 1.0 / a**2
            )
        )

    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Scalar field
    # --------------------------------------------------------

    ax = axes[0, 0]

    ax.plot(
        r,
        phi,
    )

    ax.set_xlabel(
        r"$r$"
    )

    ax.set_ylabel(
        r"$\phi(r)$"
    )

    ax.set_title(
        "Scalar field"
    )

    ax.grid(
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Fluid pressure
    # --------------------------------------------------------

    ax = axes[0, 1]

    ax.plot(
        r,
        p,
    )

    ax.set_xlabel(
        r"$r$"
    )

    ax.set_ylabel(
        r"$p(r)$"
    )

    ax.set_title(
        "Fluid pressure"
    )

    ax.grid(
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Metric functions
    # --------------------------------------------------------

    ax = axes[1, 0]

    ax.plot(
        r,
        a,
        label=r"$a(r)$",
    )

    ax.plot(
        r,
        alpha_plot,
        label=r"$\alpha(r)$",
    )

    ax.set_xlabel(
        r"$r$"
    )

    ax.set_ylabel(
        "Metric functions"
    )

    ax.set_title(
        "Metric"
    )

    ax.legend()

    ax.grid(
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Gravitational mass
    # --------------------------------------------------------

    ax = axes[1, 1]

    ax.plot(
        r,
        mass,
        label=r"$m(r)$",
    )

    if observables is not None:

        ax.plot(
            r,
            observables[
                "M_fluid_grav_profile"
            ],
            linestyle="--",
            label=r"$M_{\rm fluid}(r)$",
        )

        ax.plot(
            r,
            observables[
                "M_scalar_grav_profile"
            ],
            linestyle=":",
            label=r"$M_{\rm scalar}(r)$",
        )

    ax.set_xlabel(
        r"$r$"
    )

    ax.set_ylabel(
        r"$M(r)$"
    )

    ax.set_title(
        "Mass profiles"
    )

    ax.legend()

    ax.grid(
        alpha=0.3,
    )

    # ========================================================
    # Save / show
    # ========================================================

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    if show:

        plt.show()

    return fig


def plot_mass_radius_grid(
    csv_path,
    show=True,
    save_path=None,
):
    """
    Plot total gravitational mass against the largest
    characteristic radius for a parameter grid.

    The effective outer radius is defined as

        R_outer =
            max(
                R_fluid,
                R_baryon_99,
                R_boson_99,
                R_total_99
            )

    ignoring NaN values.

    Parameters
    ----------
    csv_path : str
        CSV file generated from the parallel grid.

    show : bool, optional
        Display the figure.

    save_path : str or None, optional
        Save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
    )

    M_total = np.asarray(
        data["M_total"],
        dtype=float,
    )

    radius_arrays = np.vstack(
        [
            data["R_fluid"],
            data["R_baryon_99"],
            data["R_boson_99"],
            data["R_total_99"],
        ]
    )

    with np.errstate(
        all="ignore"
    ):

        R_outer = np.nanmax(
            radius_arrays,
            axis=0,
        )

    status = np.asarray(
        data["status"],
        dtype=int,
    )

    valid = (
        (status == 1)
        & np.isfinite(M_total)
        & np.isfinite(R_outer)
    )

    fig, ax = plt.subplots(
        figsize=(7, 5),
        constrained_layout=True,
    )

    ax.scatter(
        R_outer[valid],
        M_total[valid],
        s=30,
    )

    ax.set_xlabel(
        r"$R_{\rm outer}$"
    )

    ax.set_ylabel(
        r"$M_{\rm total}$"
    )

    ax.set_title(
        "Mass-radius relation"
    )

    ax.grid(
        alpha=0.3,
    )

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    if show:

        plt.show()

    return fig