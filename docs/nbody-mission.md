# Complete finite EMC N-body correction

`nbody-mission` joins Earth parking departure, a finite Mars encounter, arrival
and departure at one rotating Ceres surface site, the return Mars encounter,
and Earth parking capture in a single multiple-shooting problem. Ballistic and
powered cases remain separate. Both retain the configured ten-year limit.

```bash
python native/build.py
python run.py nbody-mission /path/to/validated-mission/best.json \
    --outdir results/nbody-missions
```

The candidate needs a matching passing `mission-validate` record and the source
mission configuration, inferred from its directory or passed with `--config`.
A run performs correction and independent checks, preserves failed results,
and returns exit status 2 when acceptance fails. `--max-evaluations` controls
the initial trust-region correction budget (default 160); a bounded Newton
polish may then add residual evaluations. No downloads occur in this command.

Reproduce a saved result from its `source-candidate.json`, with
`--config RUN/source-mission-config.json`. Every run preserves resolved kernels
and their hashes, source snapshots and hashes, the native library hash,
Python/package versions, initial geometry, decision vector, all event states,
all eight state-transition maps, interpolation inputs, independent trajectory
samples, convergence evidence, plots and artifact hashes.

## Physical model

All four interplanetary legs use the custom FP64 C++ propagator and its analytic
variational equations. The field contains fourteen prescribed geometric SPICE
point masses without system/body double-counting, plus the oriented Mars J2
term. Python supplies pinned ephemerides and handles scientific orchestration.
This correction layer runs on the CPU; broad Lambert/flyby searches still use
the existing CUDA kernels. No CUDA fallback is involved.

Mars J2 is 1.96045e-3 at reference radius 3,396,200 m, the approximate constants
listed on the [NASA Mars fact sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html).
This is an explicitly specified low-order model, not a modern fitted GMM gravity
field. The pole comes from `IAU_MARS` in pinned `pck00011.tpc`; normalized pole
vectors are interpolated at the ephemeris knots. Axisymmetric J2 does not depend
on the prime meridian. For Mars-relative position x, pole p, r=|x|, z=x·p:

```
a_J2 = 1.5*mu*J2*R^2/r^5 * ((5*z^2/r^2 - 1)*x - 2*z*p)
```

The C++ implementation includes the analytic spatial gradient of this term in
the STM. Independent tests differentiate a separately written scalar potential,
check the rotated acceleration and gradient, symmetry and zero trace, and
compare propagated STM columns with finite differences.

Earth endpoints are free-plane circular parking orbits at the configured
radius. Their actual impulsive injection and capture vectors are charged.
Ceres is a rotating sphere with instantaneous tangential landing and launch at
the same body-fixed point, including the full ground velocity. Solid-surface
support during the stay is prescribed. The corrector uses the exact constant
pole/constant spin model in the pinned Ceres PCK; it rejects a different model
with nonzero pole rates or spin acceleration. The native Ceres collision guard
is relaxed by one metre solely to permit integration from the spherical surface;
independent checks use the physical radius with a declared 2 cm numerical
allowance. This is not permission to fly below the surface.

Both Mars impulses, when allowed, are tangential at their common periapsis.
They are variable, explicitly exported, and included in the total cost. The
ballistic model applies exactly zero impulse at both encounters. All legs meet
at shared encounter states. No deep-space correction impulse is introduced.

## Joint correction and numerical precision

There are six event epochs. Ceres arrival and departure share one body-fixed
site. Earth parking positions, both Mars periapsis directions/radii, endpoint
speeds/directions, site, dates, and powered impulses are design variables.
Bounds permit local date shifts of at most two days and local geometry changes;
this is a feasibility correction of known mission basins. It does not minimize
cost or establish global optimality. It cannot rescue a globally inadequate
search by declaring it impossible.

Each of the four legs propagates from both finite endpoint states to a fixed
interior time. Twenty-four position/velocity match equations are solved
jointly. Tangency and fixed spherical Earth/Ceres radii are enforced by
parameterization. Mars radii remain above their configured minima through a
positive exponential gap. A 10 km seed lifting margin is explicit; the global
solve may reduce it, subject to the physical minimum and local parameter bounds.

Analytic STMs supply the propagation derivatives. Cheap central differences
supply only the endpoint coordinate and rotation derivatives. Endpoint epoch
partials include the initial vector field. A trust-region solve is followed by
a scaled Newton polish which holds the rapidly rotating Ceres epochs and active
bounds while correcting the remaining geometry and dates. Optimizer status and
physical acceptance remain separate.

Small periapsis positions are never first added to an AU-scale coordinate for
propagation. Each half-arc uses a constant inertial origin anchored at a nearby
body table knot and a local integration clock. Interpolation subtracts that
origin before combining polynomial terms. This is a coordinate translation,
not a moving body-centred approximation: no indirect-force term is added.
Explicit ephemeris knots are labelled with the representable absolute ET that
SPICE actually queried. Ceres rotates using a smooth relative-time increment
from an anchored SPICE transformation, avoiding sub-microsecond quantization in
billion-second absolute epochs. Saved global Cartesian states are convenient
for inspection; the relative states and knot arrays preserve the small boundary
data needed for precise reproduction.

The separately versioned N-body ABI is now 2, adding optional J2 and centred
propagation. Lambert and flyby ABIs remain unchanged. Rebuild the N-body library
when updating its Python binding; layout/version mismatches fail explicitly.

## Acceptance and limits

Native shooting must match every interior position within 1 m and velocity
within 1e-5 m/s. Current examples normally close much more tightly. Independent
SciPy DOP853 uses its own directly queried SPICE force implementation, including
an independently coded oriented J2 term. It starts from the same numeric
inertial state, rather than changing the boundary condition to follow a newly
queried body centre. Half-arc endpoints must agree within 100 m and 2e-4 m/s;
independently propagated pairs must meet within 200 m and 3e-4 m/s. Tighter native
integration and half-spacing ephemerides are checked separately for all eight
half-arcs (25 m, 5e-5 m/s). These are explicit numerical screening tolerances,
not navigation accuracy or an estimate of real trajectory uncertainty.

The checks also search the independent dense trajectories for closest approach
to every force body, enforce solar and Mars clearance, verify the surface site,
all date/duration/stay limits and any configured C3/burn limits, and independently
recompute the complete impulse budget. Searches sample integration nodes and
uniform/logarithmic meshes and refine every bracketed minimum; this is not a
formal interval collision certificate.

`full_mission_reattached` is true only when every numerical acceptance check
passes. It means the finite phases meet within the published shooting tolerances.
It does not certify one uninterrupted forward propagation through every encounter
or imply flight readiness. The saved original runs record whole-mission GMAT
acceptance as `not_run`. A subsequent [complete-candidate GMAT replay](gmat-global-replay.md)
executed both candidates but failed its external numerical comparison. The
earlier GMAT validation of isolated point-mass flybys does not validate this
full-chain/J2 result.

These are candidate research trajectories, not publication-ready replacements.
Modern higher-order gravity, Earth harmonics, relativity and solar radiation
pressure, finite burns and propellant/mass limits, practical launch/capture
constraints, Ceres terrain/descent/ascent, and navigation/force-model uncertainty
remain to be introduced. Low-order force consistency and numerical closure alone
do not establish engineering feasibility. The high delta-v examples are not
claimed to be efficient missions or evidence for retracting the paper.
