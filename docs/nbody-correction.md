# N-body correction: finite Mars encounter foundation

This increment implements the first high-fidelity development milestone:
a custom FP64 C++ Newtonian N-body propagator, its variational equations,
a two-arc multiple-shooting corrector, and independent finite-encounter
verification with direct-SPICE DOP853 and NASA GMAT R2026a RK89.

**A corrected local encounter is not yet a corrected complete mission.** The
boundary velocity changes must be reattached to the interplanetary phases.
Both Mars encounters, Ceres approach/stay/departure, and Earth endpoints still
need joint correction. Higher-order forces and spacecraft engineering models
remain necessary for the highest-quality research replacements.

## Run and reproduce

Build the CPU libraries with the existing C++17 build command; no CUDA is needed
for this correction layer. The previous CUDA Lambert and flyby libraries remain
available for broad searches.

```bash
python native/build.py
python tools/fetch_research_kernels.py --mission
python run.py nbody-flyby /path/to/validated-mission/best.json \
    --outdir results/nbody \
    --gmat-console /path/to/GMAT/R2026a/bin/GmatConsole
```

The source candidate requires a matching, passing `*.validation.json` from
`mission-validate`, plus its original mission configuration. The config is
inferred from the candidate's run directory, or supplied with `--config`.
The `--node` option defaults to the outbound Mars visit, index 1; index 3 selects
the inbound Mars visit in the EMC sequence.

Each saved run includes byte-identical source candidate and validation files
plus a standalone resolved mission configuration. Reproduce it with:

```bash
python run.py nbody-flyby RUN/source-candidate.json \
    --config RUN/source-mission-config.json --node 1 \
    --gmat-console /path/to/GmatConsole
```

Use the original node, boundary radius, margin and spacing from `RUN/config.json`.

All downloads are explicit. GMAT is an optional, separately installed external
validator; the program never installs it. Omitting `--gmat-console` records
external validation as `not_run`. The independent direct-SPICE DOP853 checks
still run. The tested distribution is NASA's
[GMAT R2026a Linux release](https://sourceforge.net/projects/gmat/files/GMAT/GMAT-R2026a/).

Options `--boundary-radius`, `--design-margin` and `--ephemeris-step` are in
metres, metres and seconds, respectively. Defaults are a 100,000 km finite
encounter boundary, an explicitly added 10 km seed periapsis margin, and 60 s
SPICE table spacing. Set `--design-margin 0` to attempt the unmodified source
encounter geometry; altitude-boundary seeds may then fail the corrected
minimum-radius constraint. Failure is preserved and never turned into success.

## Force model, coordinates and ephemerides

The spacecraft is a massless test particle in an ephemeris-driven Newtonian
point-mass field. At every integration stage,

\[
\ddot{\mathbf r}=\sum_i\mu_i
\frac{\mathbf R_i(t)-\mathbf r}{\lVert\mathbf R_i(t)-\mathbf r\rVert^3}.
\]

Positions and velocities use geometric J2000/ICRF axes, with the Solar System
Barycenter as ephemeris centre, aberration NONE and TDB ET seconds. A **constant
inertial coordinate translation** near Mars improves floating-point scaling.
It is not a moving planet-centred origin and adds no indirect-force terms.
Absolute SSB position is recovered by adding the recorded offset; velocity is
unchanged.

The force list includes the Sun, Mercury and Venus systems, Earth and Moon
separately, Mars/Phobos/Deimos separately, Jupiter/Saturn/Uranus/Neptune/Pluto
systems, and Ceres. Distant system barycentres are explicit point-mass
approximations. No resolved body is counted again through its system mass.
The original pinned SPICE/GM kernels are retained; Mars body-centre and moon
states use mar099s. Ceres's SPK ID is 20000001 while its GM is read from 2000001.

Python queries SPICE and supplies immutable position/velocity tables to C++.
The native force interpolates positions with cubic Hermite polynomials, without
extrapolation. The corrector uses derivatives of the same interpolated Mars
state for smooth epoch sensitivities. Validation queries SPICE directly and
also repeats propagation with half the table spacing. Sampled midpoint and
random-phase interpolation checks are recorded; they are not a rigorous
continuous error bound. Binary64 absolute ET resolution and SPK evaluation
also contribute to small residual differences.

## Integrator and sensitivities

`native/nbody/cpu.cpp` contains adaptive Dormand–Prince 5(4), compiled without
fast math or floating-point contraction. It integrates the six state variables
and the full 6×6 state-transition matrix together:

\[
\dot\Phi=A(t)\Phi,\qquad
A=\begin{bmatrix}0&I\\G&0\end{bmatrix},\qquad
G=\sum_i\mu_i\left(\frac{3\mathbf d_i\mathbf d_i^T}{d_i^5}-\frac{I}{d_i^3}\right).
\]

Position, velocity and STM absolute tolerances are separate. Step size, step
budget and ephemeris coverage are explicit. Collision guards protect each
listed body. Failed outputs have NaN states and a non-success status. Native
statuses distinguish invalid input, ephemeris coverage, collision, step floor,
step budget and nonfinite arithmetic. A rejected trial stage's closest distance
is a guard diagnostic, not a certified minimum altitude.

The separate `libnbody_cpu.so` C ABI has a version and checked structure sizes.
Contexts own copies of their ephemerides and release them explicitly. No native
callback into Python or SPICE occurs, and no other engine's ABI changes.

## Lifting and correcting a finite encounter

A local seed is lifted from the source's incoming asymptote and B-plane
orientation. The selected additional periapsis margin is applied explicitly.
The outgoing asymptote consequently changes: this is a **declared local seed
redesign**, not a claim to preserve the original mission's exact geometry.
Ballistic seeds have zero encounter impulse. Powered seeds retain their target
outgoing hyperbolic speed and a fixed signed tangential impulse at periapsis.
Hyperbolic boundary states and traversal times are calculated at finite radius.

The corrector varies ten quantities: entry velocity (3), the Mars-relative
joining state (6), and encounter epoch (1). It solves ten equations: state
continuity at the join (6), exit position (3), and radial-velocity tangency at
closest approach (1). Analytical Jacobians use propagated STMs, the tangential
burn's velocity derivative and endpoint/epoch derivatives. Optimizer success
is recorded separately from physical acceptance.

The entry and exit positions and their epochs remain fixed. Their velocities
can change and are exported as three-dimensional corrections. Those changes
are **not trajectory-correction burns** already placed in a complete mission;
they are revised phase-boundary conditions for subsequent global correction.
The powered impulse remains a modeled instantaneous encounter burn. Finite
thrust and mass depletion are future layers.

## Independent acceptance

Acceptance includes:

- DOP853 integration with a separately implemented gravitational force that
  queries SPICE directly, bypassing the native force and ephemeris interpolator.
- A continuous independent propagation from entry through the declared impulse
  to exit, without resetting the spacecraft to the stored shooting node.
- Closest-approach searches on both arcs for every force body, collision
  clearances, minimum/maximum Mars radius checks,
  and verification that the joining node is near the actual closest approach.
- Analytic STM columns checked against finite differences of the state map.
- Tighter integration tolerances and half-spacing ephemerides, tested separately.
- Complete shooting-Jacobian checks, including epoch partials, in the tests.

The explicit numerical thresholds are stored in every validation report.
Typical acceptance requires independent arc agreement within 5 cm and
1e-5 m/s, continuous terminal position within 10 cm, and table/integration
refinement changes within 2–3 cm. These are numerical model-consistency limits,
not claims of physical navigation accuracy.

With `--gmat-console`, the program generates and runs a complete GMAT script
with the same point masses, GMs and explicit SPKs, no harmonics, radiation
pressure or relativistic correction, and GMAT's independent RK89 integrator.
The powered case applies the actual inertial burn vector in MJ2000Eq axes.
All before/after states, generated scripts and console output are preserved.
GMAT uses Mars-centred equations while the custom engine sums barycentric
forces, so residuals also expose force-origin/ephemeris consistency effects.
Comparison uses GMAT's reported epochs and exact initial numeric state, with
separate epoch-offset checks, to distinguish time-format rounding from dynamics.
Acceptance requires <=10 cm and 2e-5 m/s at reported states.

The workflow hashes source snapshots, native library, inputs, kernels and GMAT
executable/core runtime files. Each run saves config, source candidate, lifted
seed, ephemeris table/provenance, corrected state, STMs, boundary changes,
validation residuals, external comparison and encounter figure. A matching
validation report is required for every reported acceptance.
An artifact checksum manifest covers the saved inputs, numerical tables,
results, reports and source snapshots.

## Verification and remaining quality work

```bash
python -m unittest discover -s nbody_tests -v
LAMBERT_TEST_SPICE=1 python -m unittest discover -s nbody_tests -v
```

Tests exercise analytic gravity gradients, the two-body limit, reversibility,
STM finite differences, exact cubic moving ephemerides, coordinate translation,
coverage/collision/error states, lifecycle, nonbinary terminal epochs, exact
step budgets, both real local correction cases, and the entire shooting
Jacobian. The standalone native smoke test also runs under address, leak and
undefined-behaviour sanitizers.

The next required mission-design milestone is to connect the corrected finite
encounters to N-body interplanetary phases and solve the entire sequence
jointly. Higher-fidelity force work includes Mars nonspherical gravity (a
material omission for low flybys), Earth and Ceres gravity/shape, consistent
rotational models, solar radiation pressure, relativity and finite burns as
justified by a measured error budget. Ceres arrival/descent/stay/ascent and
Earth departure/return need engineering constraints. Finally, launch/burn/
ephemeris/model uncertainty and correction-fuel margins must be quantified.
Agreement between two point-mass implementations does not validate omitted
forces, operational feasibility, or the paper's published trajectory claims.

The multiple-shooting development follows the approach described in
[NASA's high-fidelity multi-flyby rendering paper](https://ntrs.nasa.gov/citations/20190028884).
