from compact_star_shooting.shooting import shoot_omega
from compact_star_shooting.observables import compute_observables


# ============================================================
# Boson-star configuration
# ============================================================

rho0_c = 0.0
phi_c = 0.05

mu = 1.0


# ============================================================
# Shooting
# ============================================================

shoot = shoot_omega(
    rho0_c=rho0_c,
    phi_c=phi_c,
    omega_low=0.8,
    omega_high=103.0,
    r_max=15.0,
    n_points=2000,
    mu=mu,
    omega_tol=1.0e-8,
    verbose=False,
)


# ============================================================
# Compute observables
# ============================================================

obs = compute_observables(
    result=shoot["result"],
    omega_shoot=shoot["omega"],
    mu=mu,
)


# ============================================================
# Print summary
# ============================================================

print()
print("Boson-star solution")
print("===================")

print()
print("Shooting")
print("--------")

print(f"converged       = {shoot['converged']}")
print(f"iterations      = {shoot['iterations']}")
print(f"nodes           = {shoot['nodes']}")
print(f"phi(r_max)      = {shoot['result']['phi'][-1]:+.6e}")

print()
print("Frequency")
print("---------")

print(f"omega_shoot     = {obs['omega_shoot']:.12f}")
print(f"alpha_inf       = {obs['alpha_inf']:.12f}")
print(f"omega_phys      = {obs['omega_phys']:.12f}")
print(f"omega_phys / mu = {obs['omega_phys'] / mu:.12f}")

print()
print("Masses")
print("------")

print(f"M_total         = {obs['M_total']:.12f}")
print(f"M_boson_rest    = {obs['M_boson_rest']:.12f}")
print(f"M_baryon_rest   = {obs['M_baryon_rest']:.12f}")

print()
print("Particle numbers")
print("----------------")

print(f"N_boson         = {obs['N_boson']:.12f}")
print(f"N_baryon        = {obs['N_baryon']:.12f}")

print()
print("Radii")
print("-----")

print(f"R_fluid         = {obs['R_fluid']}")
print(f"R_boson_99      = {obs['R_boson_99']:.12f}")
print(f"R_total_99      = {obs['R_total_99']:.12f}")
print(f"R_baryon_99     = {obs['R_baryon_99']}")

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
print("Consistency")
print("-----------")

M_components = (
    obs["M_fluid_grav_profile"][-1]
    + obs["M_scalar_grav_profile"][-1]
)

print(
    f"M_fluid_grav   = "
    f"{obs['M_fluid_grav_profile'][-1]:.12f}"
)

print(
    f"M_scalar_grav  = "
    f"{obs['M_scalar_grav_profile'][-1]:.12f}"
)

print(
    f"M_components   = "
    f"{M_components:.12f}"
)

print(
    f"M_metric       = "
    f"{obs['M_total']:.12f}"
)

print(
    f"difference     = "
    f"{M_components - obs['M_total']:+.6e}"
)