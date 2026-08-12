from compact_star_shooting.shooting import shoot_omega
from compact_star_shooting.observables import compute_observables


# ============================================================
# Mixed fermion-boson configuration
# ============================================================

rho0_c = 1.0e-3
phi_c = 0.05

mu = 1.0
K_poly = 100.0
gamma = 2.0


# ============================================================
# Shooting
# ============================================================

shoot = shoot_omega(
    rho0_c=rho0_c,
    phi_c=phi_c,
    omega_low=0.8,
    omega_high=103.0,
    r_max=20.0,
    n_points=3000,
    K_poly=K_poly,
    gamma=gamma,
    mu=mu,
    omega_tol=1.0e-8,
    verbose=False,
)


# ============================================================
# Observables
# ============================================================

obs = compute_observables(
    result=shoot["result"],
    omega_shoot=shoot["omega"],
    K_poly=K_poly,
    gamma=gamma,
    mu=mu,
)



from compact_star_shooting.plotting import plot_solution

# ============================================================
# Print summary
# ============================================================

print()
print("Mixed fermion-boson star")
print("========================")

print()
print("Central values")
print("--------------")
print(f"rho0_c          = {rho0_c:.6e}")
print(f"phi_c           = {phi_c:.6e}")

print()
print("Shooting")
print("--------")
print(f"converged       = {shoot['converged']}")
print(f"iterations      = {shoot['iterations']}")
print(f"nodes           = {shoot['nodes']}")
print(
    f"phi(r_max)      = "
    f"{shoot['result']['phi'][-1]:+.6e}"
)

print()
print("Frequency")
print("---------")
print(f"omega_shoot     = {obs['omega_shoot']:.12f}")
print(f"alpha_inf       = {obs['alpha_inf']:.12f}")
print(f"omega_phys      = {obs['omega_phys']:.12f}")
print(
    f"omega_phys / mu = "
    f"{obs['omega_phys'] / mu:.12f}"
)

print()
print("Masses")
print("------")
print(f"M_total         = {obs['M_total']:.12f}")
print(f"M_baryon_rest   = {obs['M_baryon_rest']:.12f}")
print(f"M_boson_rest    = {obs['M_boson_rest']:.12f}")

print()
print("Particle numbers")
print("----------------")
print(f"N_baryon        = {obs['N_baryon']:.12f}")
print(f"N_boson         = {obs['N_boson']:.12f}")

print()
print("Radii")
print("-----")
print(f"R_fluid         = {obs['R_fluid']:.12f}")
print(f"R_baryon_99     = {obs['R_baryon_99']:.12f}")
print(f"R_boson_99      = {obs['R_boson_99']:.12f}")
print(f"R_total_99      = {obs['R_total_99']:.12f}")

print()
print("Binding")
print("-------")
print(f"M_rest          = {obs['rest_mass']:.12f}")
print(f"E_bind          = {obs['binding_energy']:.12f}")
print(f"mass_defect     = {obs['mass_defect']:.12f}")
print(
    f"E_bind / M_rest = "
    f"{obs['relative_binding_energy']:.12e}"
)

print()
print("Gravitational components")
print("------------------------")

M_fluid = obs["M_fluid_grav_profile"][-1]
M_scalar = obs["M_scalar_grav_profile"][-1]

print(f"M_fluid_grav    = {M_fluid:.12f}")
print(f"M_scalar_grav   = {M_scalar:.12f}")
print(f"M_components    = {M_fluid + M_scalar:.12f}")
print(f"M_metric        = {obs['M_total']:.12f}")
print(
    f"difference      = "
    f"{M_fluid + M_scalar - obs['M_total']:+.6e}"
)


plot_solution(
    result=shoot["result"],
    observables=obs,
    save_path="figures/mixed_star_profiles.png",
)