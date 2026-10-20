# Tenth increment: complete Python multi-revolution transfers

Native C++/CUDA multi-revolution roots and velocities are now available in
[the eleventh increment](eleventh-increment.md). The Python implementation below
remains the separate numerical reference.

The custom Python solver now returns both endpoint velocities for an explicitly
selected M>0 left or right root. `solve_multi_rev_pair` returns both branches in
fixed order while preparing geometry and the minimum only once. The cached Juno
case now validates the actual product roots/velocities as well as feasibility.

The C++/CUDA kernel and native batch ABI remain M=0 in this increment. The new
Python engine is the validated numerical basis for that subsequent port.

## API and branch contract

```python
import math
from src.lambert_geometry import LambertInput
from src.lambert_multirev_solver import solve_multi_rev, solve_multi_rev_pair
from src.lambert_solver import LambertStatus

problem = LambertInput(
    r1_m=(1., 0., 0.), r2_m=(0., 1., 0.),
    tof_s=2*math.pi+math.pi/2, mu_m3_s2=1., revolutions=1,
)
left, right = solve_multi_rev_pair(problem)
for result in (left, right):
    if result.status == LambertStatus.OK:
        print(result.branch.value, result.v1_m_s, result.v2_m_s)
        print(result.root.relative_residual, result.root.x_precision_limited)

right_only = solve_multi_rev(problem, branch='right')
```

The right solution in that example is the circular one: departure (0,1,0) and
arrival (-1,0,0), within numerical roundoff. The left branch is a distinct ellipse.
Physical inputs retain metres, seconds and mu in m^3/s^2 in one central-body
inertial frame. Accepted counts are integer 1 through 2^31-1; that input range
is distinct from the smaller revolution counts in complete physical validation.

- `long_way` chooses the geometric angle/orientation, as in M=0.
- `branch='left'` selects x below the minimum; `'right'` selects x above it.
  These labels do not depend on x being negative/positive or on another
  library's low/high naming.
- `solve_multi_rev_pair` always returns (left,right), including failures. Each
  slot has an independent status and optional velocities. A failed branch does
  not suppress a successful sibling.
- `solve_x_multi_rev` and `solve_x_multi_rev_pair` expose the dimensionless
  scalar root layer in `src/lambert_multirev_root.py`.

The existing `solve_zero_rev`, geometry helper and native input contracts remain.
The new full solver validates positive M, then reuses the M-independent geometry
calculation through an internal M=0 geometry adapter. It retains the original
count in the root and full result. Velocity reconstruction is extracted into one
shared helper in `lambert_solver.py`, preserving its arithmetic order and tiny-arc
fixes. No external Lambert solver, ephemeris or numerical package is called by
the runtime; it uses Python's standard library.

## Brackets, stopping and conditioning

The [previous increment](ninth-increment.md) supplies minimum-time detection and
its estimated uncertainty. `NO_SOLUTION`, `NEAR_MINIMUM` and `MINIMUM_FAILURE`
produce no root or velocity. A target inside the uncertainty band remains
unresolved; no approximate double root is silently returned.

For a separated feasible target, the root intervals exclude the entire computed
minimum bracket: left uses (-1,minimum.lower) and right uses
(minimum.upper,1). Each finite inner endpoint is freshly evaluated and must have
T below the requested time. The analytic infinite-TOF endpoints +/-1 are open
and never evaluated. Izzo's Eq. (31) starters are computed with logs and tanh to
avoid ratio overflow. An excluded/unusable starter selects the bracket midpoint
and records used_fallback_guess.

The shared Householder formula now accepts the correct slope direction for each
branch. Proposals must stay strictly inside the branch bracket and move at most
75% of its width. Every fourth iteration checks progress: bisection is forced if
the residual has not decreased by a factor of four over the cycle and the
residual tolerance is not already met. Unusable derivatives or rejected proposals
also bisect. This avoids throwing an almost-converged solution back across a
wide bracket after three rapidly improving updates. Before that adjustment,
Juno's left branch took 38 updates; it now takes five with the same accuracy checks.
The default budget is 128 updates and all exits are explicit.

Success requires both:

- abs(dx) <= x_atol+x_rtol*abs(x), default 1e-14 for each coefficient.
- abs(T(x)-target) <= tof_atol+tof_rtol*target, default 0 and 5e-13.

A qualifying zero/rounded step is freshly evaluated. Exhausted representable
x values return RESOLUTION_LIMIT when no endpoint meets the residual tolerance.
A final available derivative must have the selected branch's slope sign.
Root-stage evaluation counts include actual nested TOF work. Minimum counts are
separate in root.minimum; a pair shares that object and its cost is counted once.
RootOptions.max_bracket_expansions is unused for these fixed finite branch
domains; MinimumOptions independently controls the minimum search budget.

`estimated_x_error` is twice the slope-based estimate
(abs(residual)+16*ulp(T))/abs(T'), adding a local curvature allowance. The
True/False/None precision flag retains its meaning: True exceeds the requested
x tolerance, and None means the combined derivative API could not supply a
finite estimate. Such diagnostics are empirical estimates, not interval proofs
or velocity error certificates. Higher-derivative failure can still permit
bisection to converge, with precision explicitly unknown.

## Independent validation

All **63 standard-library Python tests** pass, including ten new root/velocity
tests. They exercise circular multi-orbit transfers, both geometric directions,
Juno's two positive roots, time reversal, dimensional similarity, truthful shared
counts, bisection with unavailable derivatives, invalid inputs, zero budgets,
open endpoints, unresolved minima, no fabricated velocities and a successful
sibling retained when the other branch exhausts its budget.

The full independent validator uses Lagrange's anomaly equation in at least
120 decimal digits (up to 403 with retained subnormal q) and bracketed bisection.
It does not invert the product TOF implementation.

| Scalar root outcome | Count |
| --- | ---: |
| Successful roots | 457 |
| Deliberate binary64 resolution limits near an open endpoint | 25 |
| Near-minimum multiplicity unresolved | 36 |
| Infeasible requests | 36 |

The resolution cases check neighboring representable x values independently;
none meets the requested residual tolerance. Among the 457 successes, 186 have
precision warnings and zero have unknown precision. Maximum absolute x error
is 1.090e-9 on a flagged ill-conditioned case. Maximum unflagged error is
4.615e-16. Actual error / reported estimate stays below 0.410, and forward error
at returned roots stays below 3.609 ulps of T versus the retained 16-ulp model.
The corpus includes M=1,2,5,20,100 and 2^31-1, both shape signs, retained q down to
the smallest subnormal, flat curves and targets close to minimum time.

Complete physical validation separately covers:

| Check | Corpus / maximum measured error |
| --- | --- |
| Independently generated elliptic conics | 48 problems, 96 branch solutions, M=1,2,5,10 |
| Velocity against known conic | 1.081e-15 scaled |
| Velocity against Gooding, both solutions | 3.422e-15 scaled |
| Endpoint energy difference | 8.096e-16 scaled |
| Angular momentum vector difference | 5.689e-16 scaled |
| Multiprecision Kepler propagated position / velocity | 1.298e-11 / 1.293e-11 scaled |
| Actual-input multiprecision velocity checks | 40 difficult problems, 80 branches, maximum 4.564e-16 scaled |

Conic states and elapsed times come from independent true-anomaly equations and
80-digit quadrature. The corpus uses eccentricities 0,.2,.7,.9, arbitrary 3D
planes, both angular-momentum senses and short/long orientations, and normalized,
Earth and AU/Sun dimensions. At least one branch must reproduce the independently
known conic, and both must match the two Gooding solutions. The hard actual-input
cases rebuild the received geometry in 140 digits and include tiny angles down
to 2e-14 radians, near-antiparallel directions, unequal radii and M=1,3.

Scaled velocity errors use a nonzero characteristic speed and reference endpoint
speeds; they are not m/s. Propagated position uses max(p,r2) and velocity uses
max(sqrt(mu/p),known endpoint speeds). Independent Kepler propagation solves the
unwrapped eccentric-anomaly equation from the returned initial state at 100 digits.
Known-state controls close to 4.168e-12 position / 4.153e-12 velocity on those scales.
Both Kepler propagation and adaptive DOP853 sampling verify the full angular
sweep M*2*pi+transfer_angle. Endpoint closure alone would not establish M.

A DOP853 check exposed an integration floor in a high-eccentricity five-revolution
case: scaled position closure was 3.959e-9 for the product and 3.665e-9 using the
multiprecision-reference velocity rounded to binary64. Product velocity error
was only 7.017e-16 scaled. Smaller steps/tolerances gave about 3e-9 closure rather
than consistent convergence. The primary endpoint acceptance therefore uses
the independent 100-digit Kepler propagation with the original 3e-9 bound.
DOP853 closure remains a separately reported diagnostic, with maximum
3.959e-9 position / 3.958e-9 velocity; its winding error is below 2.342e-9 rad.
These integration errors are not attributed to Lambert's arithmetic.

A separate near-minimum physical corpus has 24 successful branch solutions,
six infeasible and six unresolved slots at M=1,2,20. Every successful slot is warned.
Maximum velocity error is 3.947e-10 at the normalized example's unit scale,
showing the conditioning limit of binary64 inversion. Each error is checked
against an independent MP velocity-sensitivity envelope derived from the
reported root estimate, with a strict ordinary-error floor. The largest fraction
of that envelope is 0.021. These flagged cases are reported separately from the
ordinary high-accuracy results.

## Completed Juno mission regression

The cached Jupiter-centered Juno states and their provenance are unchanged from
the [ninth increment](ninth-increment.md): February 10 to April 15, 2017 TDB,
64 days, one observed complete revolution. No new network request is required.
M=2 and M=3 remain infeasible; product calls now verify both failure slots have
no velocities. Both M=1 product branches are checked against independent
actual-input MP velocities, Gooding, Kepler propagation and ODE winding.

| Check | Left branch | Right branch |
| --- | ---: | ---: |
| x | 0.0321116404383114 | 0.2479384830168246 |
| Root updates | 5 | 5 |
| Absolute relative TOF residual | 4.256e-16 | 2.128e-16 |
| Velocity difference from Gooding | 2.566e-12 m/s | 2.979e-12 m/s |
| Velocity difference from actual-input MP | 7.569e-13 m/s | 2.399e-12 m/s |
| Independent Kepler position closure | 0.566 micrometres | 1.266 micrometres |
| DOP853 position closure | 5.042 mm | 1.609 mm |

Both roots have a conservative precision warning: estimated x errors are
2.077e-14 and 1.837e-14, just above the requested x tolerance. Their measured
errors remain small; the diagnostic is preserved rather than suppressed.
The point-mass propagated angle is 364.839021625 degrees on both branches.
Its slight difference from the sampled spacecraft's projected angle is consistent
with using a fixed Keplerian plane and endpoints rather than fitting the actual
perturbed trajectory. Differences from actual spacecraft velocities remain the
physical model comparisons recorded earlier, about 134/95 m/s on the closer
right branch, separate from numerical solver validation.

## Reproduce

The independent M=0 regression also passed: 3,541 native scalar TOF/root cases,
1,411 derivative cases, 440 full transfers per native backend and the three
Mars 2020 boundaries, with actual CUDA execution. The 11 native integration
tests pass. These preserve the existing zero-revolution path; they do not claim
multi-revolution GPU support.

```bash
python3 -m unittest discover -s tests -v
.venv-validation/bin/python validation/validate_multirev_solver.py --json multirev-solver.json
.venv-validation/bin/python validation/validate_juno.py --json juno-multirev.json
# Independent M=0 regression after sharing reconstruction:
.venv-validation/bin/python validation/validate_native.py --gpu
LAMBERT_TEST_CUDA=1 .venv-validation/bin/python -m unittest discover -s native/tests -v
```

Use validation/requirements.txt and the existing native build instructions for
those optional references/GPU checks. `--section scalar|conics|actual|conditioning`
is available for targeted independent checks. Raw current results are committed
under validation/results/multirev-solver* and juno-multirev*.

The next port should add the now-tested M>0 minimum/root mathematics and explicit
branch/status fields to a versioned C++/CUDA interface, then repeat these
independent checks on the actual device before tuning throughput.
