# Fifth increment: complete zero-revolution CPU transfer

Historical validation snapshot. The [sixth increment](sixth-increment.md)
fixes the tiny-arc limitation described below and removes the temporary
geometry warning properties. Use that increment for current precision results.

Built on verified `major-revision` commit `07d4b29`. This increment adds one
velocity reconstruction module and a complete scalar entry point. The earlier
geometry, TOF, derivatives/starter and root modules are unchanged. There is no
legacy GUI integration, ephemeris access, CUDA kernel or multirevolution logic.

## What the equations do

The published attachment was checked directly, particularly Algorithm 1 on
published p. 12 of Izzo's [Revisiting Lambert's problem](https://doi.org/10.1007/s10569-014-9587-y).
Geometry already supplies normalized radial vectors and transverse vectors
`t_i = normal cross radial_i`, with the selected short/long arc normal.
The root supplies the same x,y pair for both endpoints. Define

```text
gamma = sqrt(mu*s/2)
A = lambda*y - x
B = lambda*y + x
W = y + lambda*x

Vr1 =  gamma/r1 * (A-rho*B)
Vr2 = -gamma/r2 * (A+rho*B)
Vt1 =  gamma/r1 * sigma*W
Vt2 =  gamma/r2 * sigma*W

v1 = Vr1*radial1 + Vt1*transverse1
v2 = Vr2*radial2 + Vt2*transverse2
```

The radial components describe motion toward or away from the central body;
the transverse components carry the selected angular momentum direction. Both
velocities belong to the same two-body transfer. Arrival velocity is derived
from that orbit, rather than assigned from a circular-orbit assumption.

The implementation computes `gamma/r_i` using binary exponents (`frexp/ldexp`)
so an intermediate `mu*s` need not overflow when the final speed is finite.
Circular tests include r=1e150, mu=1e300, whose speed is representable although
the direct product overflows. A nonrepresentable scale or velocity is an explicit
range failure, including underflow of a required positive transverse component.

The identity `(y-lambda*x)*(y+lambda*x)=q`, where
`q=(1-lambda)*(1+lambda)`, evaluates the subtractive member by division. A and B
are also rearranged according to the sign of lambda*x, using compensated sums.
These changes avoid unnecessary cancellation during reconstruction. They do
not recover information already lost in the geometry or root inputs.
No external solver code was ported into the mathematical runtime.

## Contract, statuses and use

Positions are metres, elapsed time is positive dynamical seconds, and mu is
m^3/s^2 for one central body. Both positions and returned velocities use the
same right-handed, central-body inertial frame. The caller supplies consistent
origins, frames, time conversions and ephemerides. Velocities are m/s.

`long_way=False` selects an arc below pi with normal along r1 cross r2;
`True` selects the arc above pi with the opposite normal. This flag is not
fixed-axis prograde/retrograde. Only M=0 is supported. Collinear inputs,
including exact antiparallel Hohmann endpoints, remain explicitly unsupported
without an additional transfer-plane convention.

```python
import math
from src.lambert_geometry import LambertInput
from src.lambert_solver import LambertStatus, solve_zero_rev

radius, mu = 7e6, 3.986004418e14
speed = math.sqrt(mu/radius)
problem = LambertInput((radius, 0., 0.), (0., radius, 0.),
                       (math.pi/2)*radius/speed, mu)
result = solve_zero_rev(problem)
if result.status != LambertStatus.OK:
    print(result.status, result.geometry.status,
          None if result.root is None else result.root.status)
elif result.root.x_precision_limited is not False:
    print('Review precision before using these velocities', result)
else:
    print(result.v1_m_s, result.v2_m_s)
    print(result.root.iterations, result.root.residual)
```

The example's conservative caller policy is illustrative. Even False flags
are not a certified physical velocity error bound; mission tolerances require
their own validation.

| LambertStatus | Meaning |
| --- | --- |
| OK | Root stopping tests passed; both velocities are finite and representable |
| GEOMETRY_FAILURE | Inspect the nested GeometryStatus, including invalid input, degeneracy, unsupported revolutions or range |
| ROOT_FAILURE | Inspect the complete RootResult for numerical failure and diagnostics |
| NUMERICAL_RANGE | Root succeeded but velocity reconstruction exceeded supported range |

All failures have both velocity fields None. `geometry` always retains the
GeometryResult; `root` is None only when geometry failed. Successful results
retain the original RootResult object, including its actual completed updates,
signed nondimensional residual, stopping options' effects, error estimate and
tri-state x-precision diagnostic. Unknown precision remains None.
Default root tolerances remain x_atol=x_rtol=1e-14 and tof_rtol=5e-13.

## Separate geometry precision warning and unresolved limit

The full solve exposed a precision limit that scalar root tests alone cannot
detect. For almost coincident endpoint directions, lambda is close to +/-1.
Rounding lambda then loses relative precision in q, despite a tiny residual in
the equation expressed using that rounded lambda. The geometric identity is
`q=c/s`. The result therefore exposes

```text
geometry_q_relative_mismatch = |(1-lambda)*(1+lambda) - c/s| / (c/s)
geometry_precision_warning = mismatch > 64*binary64_epsilon
```

The threshold is 1.4210854715202004e-14. This is a consistency diagnostic,
not an interval proof, a velocity error estimate, or a guarantee that every
ill-conditioned input is detected. It is separate from root.x_precision_limited.
It can conservatively warn on a long arc whose measured velocities are accurate.
Four of the 248 accepted cases carry this geometry warning; none carries a
root warning or unknown root diagnostic.

A separate 10-case circular characterization uses angles 1e-4, 1e-6, 1e-8,
1e-10, 1e-12 with both short and long arcs. All ten carry geometry warnings,
while their root warnings are False. The short-arc velocity error divided by
circular speed grows from 3.8425e-12 to **8.8901e-5** at angle 1e-12. Long-arc
errors in this family remain at most 2.221e-16. These cases characterize an
unresolved precision limit; they do not satisfy or extend the main corpus's
velocity-accuracy claim. OK alone must not be used to infer accurate velocities.

A subsequent focused CPU increment should carry stable geometric q=c/s through
the TOF/root/reconstruction chain, with independent revalidation. That change
crosses the earlier scalar interfaces and is deferred to keep this increment
coherent and reviewable. CUDA translation should follow that precision work.

## Independent validation and observed errors

All 41 standard-library tests pass, including eight new full-solver tests.
They cover circular quarter/three-quarter arcs at four scales and in two
planes, elliptic eccentric-anomaly timing, parabolic Barker timing, hyperbolic
anomaly timing, endpoint time reversal, explicit failure outputs, dimensional
range failure, preservation of tri-state root diagnostics, and the new geometry
warning. Analytical circular maximum scaled velocity error is 3.519e-16.

`validation/validate_solver.py` uses 80-digit conic states and independent
quadrature of `dt/dnu=r^2/sqrt(mu*p)` to generate physical inputs and expected
velocities. It does not use the production geometry, TOF, derivatives, starter,
root or velocity formula for its reference. Rotations are proper 3D orbital
bases rounded to binary64; endpoints, time and parameters are rounded to the
actual inputs. This comparison includes that input rounding.

The 248 accepted cases comprise 128 elliptic, 20 parabolic and 100 hyperbolic
transfers; 110 long and 138 short arcs; 120 positive and 128 negative angular
momentum z directions. Eccentricities range from zero to five, including
1+/-1e-8 and 1+/-1e-12. Scales cover normalized units, Earth-orbit dimensions
and AU/Sun dimensions. Eight circular stress cases cover angles 1e-4,
pi+/-1e-6 and 2*pi-1e-4, with both directions. Polar circular tests are in the
standard-library suite; the Gooding corpus uses nonpolar normals to make its
fixed-z direction convention unambiguous.

Each case is also compared with
[LambertHub 1.0.0's Gooding implementation](https://github.com/jorgepiloto/lamberthub/blob/v1.0.0/src/lamberthub/universal_solvers/gooding.py),
using maxiter=100 and atol=rtol=1e-13. This supplies an independent root-solving
algorithm. Comparison inputs are nondimensionalized with p and sqrt(mu/p),
giving mu=1. Gooding's prograde flag is mapped from the intended normal's z sign,
and exhaustion/nonfinite output is rejected explicitly.

Gooding's angle routine uses acos of a normalized dot product, which loses
accuracy near zero/pi. For the eight circular stress cases its differences are
reported diagnostically and acceptance uses the independent analytical
velocities. Its maximum stress error against the analytical velocity is
2.6221e-9, while custom/Gooding disagreement is 2.6259e-9. The strict Gooding
bound applies to the other 240 cases; no custom accuracy bound was relaxed to
accommodate the comparison solver. These observations concern the pinned Python
implementation, not a general limitation of Gooding's mathematical method.

For every accepted case, the returned departure state is independently
integrated under `r''=-mu*r/|r|^3` with SciPy
[solve_ivp/DOP853](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
The analytical departure state is integrated as a control. Both integrations
are repeated at tighter tolerances and smaller maximum steps, for four
propagations per case. Initial r,v use units p and sqrt(mu/p); time uses
p/sqrt(mu/p). First-pass rtol=2e-13, atol=2e-14, max_step=tof/64; refined
rtol=3e-14, atol=3e-15, max_step=tof/128.

Velocity error is divided by max(sqrt(mu/p), both analytical endpoint speeds).
Position error is divided by max(p, arrival radius). Energy differences use
max(mu/p, both kinetic-energy magnitudes, both potential-energy magnitudes).
Angular momentum difference uses max(sqrt(mu*p), departure momentum norm).
These scaled errors remain meaningful at the parabolic zero-energy limit.

| Check | Observed maximum | Acceptance bound |
| --- | ---: | ---: |
| Custom velocity vs independent conic state, all 248 | 3.843e-12 | 2e-11 |
| Custom velocity vs Gooding, ordinary 240 | 1.391e-13 | 2e-11 |
| Endpoint specific energy difference | 9.377e-16 | 2e-12 |
| Endpoint angular momentum vector difference | 1.470e-15 | 2e-12 |
| Propagated arrival position | 2.921e-10 | 2e-9 |
| Propagated arrival velocity | 2.159e-10 | 2e-9 |
| Control arrival position | 2.887e-10 | 2e-9 |
| Control arrival velocity | 2.134e-10 | 2e-9 |
| Refined propagated arrival position | 5.052e-11 | 5e-10 |
| Refined propagated arrival velocity | 3.735e-11 | 5e-10 |
| Refined control arrival position | 4.838e-11 | 5e-10 |
| Refined control arrival velocity | 3.576e-11 | 5e-10 |

Propagation errors include integrator error. The largest errors occur in an
e=0.9 long arc, and the independently known control state has comparable error;
the tighter integrations reduce both. Endpoint invariants and independent
analytical velocities provide separate checks. Every case has the intended
angular-momentum direction. No unexpected solver/comparator/integrator failures
occurred. Maximum computed relative root residual is 1.808e-15 and maximum
completed root updates is three in this physical corpus. These are not
throughput measurements or a universal iteration bound.

Confidence is high in the reported tested domain and explicit failure behavior.
Uniform binary64 velocity accuracy, all extreme geometries, collision avoidance,
finite-body constraints, perturbations and multirevolution solutions are not
established. The point-mass two-body core does not impose a central-body surface.

## Reproduce

From the repository root, using Python 3.12.3 in this session:

```bash
python3 -m unittest discover -s tests -v
python3 -m venv .venv-validation
.venv-validation/bin/python -m pip install -r validation/requirements.txt
.venv-validation/bin/python validation/validate_solver.py
```

The session used the same packages in the task's external validation venv.
Pinned validation packages are mpmath 1.3.0, LambertHub 1.0.0, NumPy 2.5.4,
SciPy 1.18.1, Numba 0.68.0 and llvmlite 0.50.0. None is imported by the custom
scalar mathematical runtime, which remains standard-library-only.
