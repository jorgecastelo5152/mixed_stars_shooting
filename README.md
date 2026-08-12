# Compact Star Shooting

A compact and modular Python implementation of shooting methods for static mixed fermion-boson stars in General Relativity.

The code solves the coupled Einstein-fluid-Klein-Gordon system in spherical symmetry and determines the scalar-field eigenfrequency through a shooting procedure.

The project is intentionally restricted to a clean minimal model:

- static and spherically symmetric configurations,
- perfect-fluid fermionic matter,
- polytropic equation of state,
- canonical complex scalar field,
- massive scalar potential,
- geometrized units with \(G=c=\hbar=1\).

The repository supports:

- pure neutron stars,
- pure boson stars,
- mixed fermion-boson stars,
- scalar-frequency shooting,
- asymptotic lapse normalization,
- gravitational and rest masses,
- particle numbers,
- effective radii,
- binding energies,
- parallel two-dimensional parameter grids,
- incremental checkpointing and restart,
- CSV export,
- radial-profile visualization,
- mass-radius diagrams.

---

## Physical model

The spacetime metric is

\[
ds^2
=
-\alpha(r)^2 dt^2
+
a(r)^2 dr^2
+
r^2 d\Omega^2.
\]

The complex scalar field is written as

\[
\Phi(t,r)
=
\phi(r)e^{-i\omega t},
\]

with potential

\[
V(\phi)
=
\mu^2\phi^2.
\]

The fluid obeys a polytropic equation of state,

\[
p
=
K\rho_0^\Gamma,
\]

with total fluid energy density

\[
\epsilon
=
\rho_0
+
\frac{p}{\Gamma-1}.
\]

The numerical state vector is

\[
y(r)
=
\left[
a,\,
\alpha,\,
\phi',\,
\phi,\,
p
\right].
\]

Regular central conditions are imposed as

\[
a(0)=1,
\qquad
\alpha(0)=1,
\qquad
\phi'(0)=0.
\]

The scalar frequency \(\omega\) is not known a priori and must be determined as an eigenvalue.

---

## Shooting method

For fixed central values

\[
\rho_{0,c},
\qquad
\phi_c,
\]

the code searches for the scalar eigenfrequency

\[
\omega_{\rm shoot}
=
\omega_\star(\rho_{0,c},\phi_c).
\]

Trial frequencies are integrated radially and classified according to the scalar-field behavior and number of zero crossings.

The fundamental bosonic configuration is selected as the nodeless solution separating the growing and oscillatory branches.

After shooting, the lapse is normalized asymptotically,

\[
\alpha_{\rm phys}(r)
=
\frac{\alpha(r)}{\alpha_\infty},
\]

where

\[
\alpha_\infty
\simeq
\alpha(r_{\max}).
\]

The corresponding physical scalar frequency is

\[
\omega_{\rm phys}
=
\frac{\omega_{\rm shoot}}{\alpha_\infty}.
\]

For bound configurations,

\[
\omega_{\rm phys}<\mu.
\]

---

## Observables

The gravitational mass profile is obtained directly from the radial metric function,

\[
m(r)
=
\frac{r}{2}
\left(
1-\frac{1}{a(r)^2}
\right).
\]

The total gravitational mass is approximated by

\[
M
\simeq
m(r_{\max}).
\]

The fermionic rest-mass profile is

\[
M_{0,F}(r)
=
4\pi
\int_0^r
a(\tilde r)
\rho_0(\tilde r)
\tilde r^2
d\tilde r.
\]

The conserved boson number is

\[
N_B(r)
=
4\pi
\int_0^r
\tilde r^2
a(\tilde r)
\frac{\omega_{\rm phys}}
{\alpha_{\rm phys}(\tilde r)}
\phi(\tilde r)^2
d\tilde r.
\]

The associated bosonic rest mass is

\[
M_{0,B}
=
\mu N_B.
\]

The total rest mass is

\[
M_{\rm rest}
=
M_{0,F}
+
M_{0,B},
\]

and the binding energy is defined as

\[
E_{\rm bind}
=
M-M_{\rm rest}.
\]

Bound configurations therefore typically satisfy

\[
E_{\rm bind}<0.
\]

The code also computes characteristic radii containing a chosen fraction of cumulative mass or particle number, with \(99\%\) used by default.

---

## Repository structure

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── figures/
│   ├── mixed_star_profiles.png
│   └── mass_radius_grid.png
├── src/
│   └── compact_star_shooting/
│       ├── __init__.py
│       ├── eos.py
│       ├── equations.py
│       ├── integrator.py
│       ├── shooting.py
│       ├── observables.py
│       ├── grid.py
│       └── plotting.py
├── examples/
│   ├── trial_integration.py
│   ├── boson_star_shooting.py
│   ├── mixed_star.py
│   ├── run_grid.py
│   └── plot_grid.py
└── tests/
    ├── test_eos.py
    ├── test_equations.py
    ├── test_integrator.py
    └── test_observables.py
```

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repository-url>
cd mixed_stars_shooting
pip install -e .
```

For development, including the test suite:

```bash
pip install -e ".[dev]"
```

Alternatively, install the dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Running individual configurations

### Pure boson star

Run

```bash
python3 examples/boson_star_shooting.py
```

A representative configuration with

\[
\phi_c=0.05,
\qquad
\mu=1,
\]

gives approximately

```text
omega_shoot  = 1.14134
omega_phys   = 0.93427
M_total      = 0.60759
N_boson      = 0.62417
```

The solution is nodeless and satisfies

\[
\omega_{\rm phys}<\mu,
\]

as expected for a bound bosonic configuration.

The corresponding gravitational and bosonic rest masses are approximately

\[
M_{\rm grav}\simeq0.60759,
\qquad
M_{0,B}\simeq0.62417.
\]

The binding energy is therefore

\[
E_{\rm bind}
\simeq
-0.01658.
\]

---

### Mixed fermion-boson star

Run

```bash
python3 examples/mixed_star.py
```

For example,

\[
\rho_{0,c}=10^{-3},
\qquad
\phi_c=0.05,
\]

produces a configuration containing both fermionic and bosonic matter.

A representative solution is

```text
omega_phys      = 0.89825
M_total         = 0.65013
M_baryon_rest   = 0.24222
M_boson_rest    = 0.43462
R_fluid         = 6.07
R_boson_99      = 8.72
```

The configuration contains a compact fluid component together with a more extended bosonic component.

The total rest mass is approximately

\[
M_{\rm rest}
\simeq
0.67684,
\]

while

\[
M_{\rm grav}
\simeq
0.65013.
\]

The resulting binding energy is

\[
E_{\rm bind}
\simeq
-0.02671,
\]

corresponding to a relative binding energy of approximately

\[
\frac{E_{\rm bind}}{M_{\rm rest}}
\simeq
-3.95\times10^{-2}.
\]

---

## Radial profiles

The individual mixed-star example can also generate a radial-profile figure containing:

- the scalar field \(\phi(r)\),
- the fluid pressure \(p(r)\),
- the metric functions \(a(r)\) and \(\alpha(r)\),
- the total, fermionic, and bosonic gravitational-mass profiles.

Run

```bash
python3 examples/mixed_star.py
```

to generate

```text
figures/mixed_star_profiles.png
```

The plotting utilities are implemented in

```text
src/compact_star_shooting/plotting.py
```

and can also be used directly through

```python
from compact_star_shooting.plotting import plot_solution

plot_solution(
    result=shoot["result"],
    observables=obs,
    save_path="figures/mixed_star_profiles.png",
)
```

---

## Parallel parameter grids

Independent stellar configurations can be evaluated over a two-dimensional grid,

\[
(\rho_{0,c},\phi_c),
\]

using multiple CPU processes.

Run

```bash
python3 examples/run_grid.py
```

The grid solver uses

```text
concurrent.futures.ProcessPoolExecutor
```

to distribute independent stellar models across available CPUs.

Each point performs the complete workflow:

```text
central parameters
       ↓
frequency shooting
       ↓
radial integration
       ↓
observable extraction
```

The resulting quantities are stored as two-dimensional arrays with shape

```text
(N_rho, N_phi)
```

for observables such as

```python
grid["M_total"]
grid["omega_phys"]

grid["M_baryon_rest"]
grid["M_boson_rest"]

grid["N_baryon"]
grid["N_boson"]

grid["R_fluid"]
grid["R_baryon_99"]
grid["R_boson_99"]
grid["R_total_99"]

grid["binding_energy"]

grid["M_fluid_grav"]
grid["M_scalar_grav"]
```

The grid includes the limiting configurations naturally:

- \(\phi_c=0\): pure fluid stars,
- \(\rho_{0,c}=0\): pure boson stars,
- \(\rho_{0,c}>0\) and \(\phi_c>0\): mixed stars.

---

## Parallel execution

The number of worker processes can be specified directly.

For example,

```python
grid = run_grid_parallel(
    rho0_values,
    phi_values,
    max_workers=8,
)
```

When running in an HPC environment, the code also reads

```text
SLURM_CPUS_PER_TASK
```

when available.

For example,

```bash
SLURM_CPUS_PER_TASK=8 python3 examples/run_grid.py
```

uses eight worker processes.

---

## Incremental checkpointing

Long parameter-grid calculations support incremental checkpointing.

The default example creates

```text
mixed_star_grid_checkpoint.npz
```

which stores:

```text
rho0_values
phi_values
values
status
errors
observable_names
```

Each point has a status value

```text
 0 = pending
 1 = successful
-1 = failed
```

The checkpoint is updated periodically during execution.

If a calculation is interrupted, rerunning

```bash
python3 examples/run_grid.py
```

restores the previous state and computes only the remaining points.

This allows large parameter grids to be resumed without recomputing completed models.

---

## CSV export

Grid results are also exported to

```text
mixed_star_grid_results.csv
```

with one stellar configuration per row.

The table contains columns including

```text
rho0_c
phi_c
status
omega_shoot
omega_phys
alpha_inf
M_total
M_baryon_rest
M_boson_rest
N_baryon
N_boson
R_fluid
R_baryon_99
R_boson_99
R_total_99
rest_mass
binding_energy
mass_defect
relative_binding_energy
M_fluid_grav
M_scalar_grav
nodes
phi_end
```

This flat representation makes the output easy to analyze with NumPy, pandas, plotting software, or external tools.

---

## Mass-radius visualization

The parameter-grid output can be represented in the mass-radius plane.

Run

```bash
python3 examples/plot_grid.py
```

to generate

```text
figures/mass_radius_grid.png
```

For each configuration, an outer characteristic radius is defined as

\[
R_{\rm outer}
=
\max
\left(
R_{\rm fluid},
R_{F,99},
R_{B,99},
R_{99}
\right),
\]

ignoring undefined radii.

The resulting figure shows

\[
M_{\rm total}
\]

against

\[
R_{\rm outer}
\]

for all successfully computed configurations.

Because mixed fermion-boson stars depend on two central parameters, this plot represents a two-dimensional family of solutions rather than a single one-parameter mass-radius sequence.

---

## Numerical consistency checks

The implementation includes independent checks of the numerical solution.

The gravitational mass can be obtained directly from the metric,

\[
M_{\rm metric}(r)
=
\frac{r}{2}
\left(
1-\frac{1}{a(r)^2}
\right).
\]

It can also be reconstructed from the integrated matter contributions,

\[
M_{\rm matter}(r)
=
M_{\rm fluid}^{\rm grav}(r)
+
M_{\rm scalar}^{\rm grav}(r).
\]

For the representative pure boson-star configuration,

\[
M_{\rm scalar}^{\rm grav}
=
0.607592330404,
\]

while

\[
M_{\rm metric}
=
0.607592330423,
\]

corresponding to a difference of order

\[
10^{-11}.
\]

For the representative mixed configuration,

\[
M_{\rm fluid}^{\rm grav}
=
0.229357116857,
\]

and

\[
M_{\rm scalar}^{\rm grav}
=
0.420768530626.
\]

Their sum is

\[
M_{\rm matter}
=
0.650125647484,
\]

while the metric gives

\[
M_{\rm metric}
=
0.650125596794.
\]

The difference is of order

\[
10^{-8}.
\]

These comparisons provide a direct numerical consistency check between the matter sector and the Einstein equations.

---

## Numerical strategy

The code is organized into independent numerical layers.

### Equation of state

The fluid sector is defined by

\[
p
=
K\rho_0^\Gamma,
\]

with

\[
\epsilon
=
\rho_0
+
\frac{p}{\Gamma-1}.
\]

The equation-of-state implementation is independent of the field equations.

### Field equations

The coupled Einstein-fluid-Klein-Gordon equations are written as a first-order radial ODE system,

\[
y(r)
=
\left[
a,\,
\alpha,\,
\psi,\,
\phi,\,
p
\right],
\]

with

\[
\psi
=
\frac{d\phi}{dr}.
\]

### Radial integration

For fixed \(\omega\), the equations are integrated with

```text
scipy.integrate.solve_ivp
```

using `DOP853` by default.

The integration begins at a small positive radius to avoid explicit evaluation of the \(1/r\) terms at the coordinate origin.

The integrator can also return partial solutions for divergent trial frequencies, which is useful during the shooting procedure.

### Fluid surface

The fluid component has a finite surface determined by

\[
p(r)\leq p_{\rm tol}.
\]

For a pure boson star,

\[
\rho_{0,c}=0,
\]

and therefore no fermionic surface exists. In that case,

```text
R_fluid = nan
```

is returned.

### Scalar eigenvalue problem

For fixed

\[
\rho_{0,c},
\qquad
\phi_c,
\]

the scalar eigenfrequency is determined by shooting.

Trial solutions are classified according to their radial behavior and number of zero crossings.

The fundamental configuration corresponds to the nodeless solution.

---

## Characteristic radii

Mixed configurations naturally contain several spatial scales.

### Fluid radius

The physical fluid surface is defined by

\[
p(R_F)=0
\]

up to the numerical pressure tolerance.

### Fermionic effective radius

The code also computes the radius enclosing \(99\%\) of the fermionic rest mass,

\[
R_{F,99}.
\]

### Bosonic effective radius

Because the scalar field has an asymptotic tail, it has no finite surface.

The effective bosonic radius

\[
R_{B,99}
\]

is defined as the radius enclosing \(99\%\) of the conserved boson number.

### Total effective radius

The total radius

\[
R_{99}
\]

contains \(99\%\) of the gravitational mass.

---

## Binding energy

The total rest mass is

\[
M_{\rm rest}
=
M_{0,F}
+
M_{0,B},
\]

where

\[
M_{0,B}
=
\mu N_B.
\]

The binding energy is defined as

\[
E_{\rm bind}
=
M_{\rm grav}
-
M_{\rm rest}.
\]

With this convention, bound configurations typically satisfy

\[
E_{\rm bind}<0.
\]

The positive mass defect is

\[
M_{\rm defect}
=
M_{\rm rest}
-
M_{\rm grav}.
\]

The code additionally provides the relative quantities

\[
\frac{E_{\rm bind}}{M_{\rm rest}}
\]

and

\[
\frac{M_{\rm defect}}{M_{\rm rest}}.
\]

---

## Tests

Run the complete test suite with

```bash
pytest -v
```

The tests cover:

- polytropic equation-of-state relations,
- pressure and rest-density inversion,
- scalar potential,
- scalar mass term,
- regularity of the ODE right-hand side,
- central initial conditions,
- radial integration,
- pure-fluid configurations,
- pressure post-processing,
- asymptotic lapse normalization,
- gravitational mass profiles,
- enclosed-fraction radii,
- fermionic rest mass,
- bosonic rest mass,
- binding-energy definitions.

---

## Example workflow

A typical single-model workflow consists of three steps.

Choose the central parameters,

```python
rho0_c = 1.0e-3
phi_c = 0.05
```

determine the scalar eigenfrequency,

```python
shoot = shoot_omega(
    rho0_c=rho0_c,
    phi_c=phi_c,
    omega_low=0.8,
    omega_high=103.0,
    r_max=20.0,
    n_points=3000,
)
```

and compute the physical observables,

```python
obs = compute_observables(
    result=shoot["result"],
    omega_shoot=shoot["omega"],
)
```

The resulting dictionary includes

```python
obs["M_total"]

obs["M_baryon_rest"]
obs["M_boson_rest"]

obs["N_baryon"]
obs["N_boson"]

obs["R_fluid"]
obs["R_baryon_99"]
obs["R_boson_99"]
obs["R_total_99"]

obs["omega_phys"]
obs["binding_energy"]
```

as well as the corresponding radial profiles.

---

## Scope

This repository is intended as a compact numerical-method demonstration rather than a full production compact-star code.

The main focus is on:

- nonlinear coupled ODE systems,
- eigenvalue problems,
- shooting algorithms,
- finite and asymptotic boundary conditions,
- parallel scientific computing,
- checkpoint/restart workflows,
- numerical validation,
- physical post-processing,
- scientific visualization,
- modular Python design.

The physical model is deliberately kept minimal so that the numerical methodology remains transparent.

---

## Possible extensions

Natural future extensions include:

- automated one-parameter stellar sequences,
- continuation methods for the scalar eigenfrequency,
- higher-resolution parameter surveys,
- stability diagnostics,
- additional mass-radius families,
- automated visualization of parameter-space maps,
- more advanced HPC workflows.

---

## Requirements

Runtime dependencies:

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

Development additionally requires

```text
pytest>=7.0
```

---

## License

This project is distributed under the MIT License.

See the `LICENSE` file for details.