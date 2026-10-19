# Ninth increment: multi-revolution scalar preparation and Juno ephemerides

`src/lambert_multirev.py` now evaluates multi-revolution time-of-flight and its
first three derivatives, locates the minimum flight time, and classifies a
requested time relative to that minimum. This is the Python scalar preparation
increment proposed before implementing the two velocity branches.

The existing full Python/C++/CUDA solvers still support M=0 only and explicitly
reject M>0. The new functions do not return Lambert roots or velocities. Existing
native binaries, APIs and the [eighth increment's optimization](eighth-increment.md)
are unchanged.

## Mathematical implementation

Time remains dimensionless, T=sqrt(2*mu/s^3)*elapsed_seconds. Inputs use the
existing signed lambda and retained q=c/s convention. Only integer M>=1 through
the signed-int32 maximum and elliptic -1<x<1 are accepted. Boolean/fractional
counts, excluded endpoints and inconsistent shapes fail explicitly.

The forward calculation uses Eq. (18) of
[Izzo, Revisiting Lambert's problem](https://doi.org/10.1007/s10569-014-9587-y):

\[
T_M(x,\lambda)=T_0(x,\lambda)+\frac{M\pi}{(1-x^2)^{3/2}}.
\]

The implementation directly includes M*pi in the elliptic angle numerator.
This avoids unnecessary M=0 evaluation and its near-parabolic cancellation.
The finite M=0 parabolic limits are not applied to M>0. Retained-q eta and atan2
keep the existing quadrant and tiny-shape safeguards. A tiny eta may underflow
without invalidating the representable M>0 time, which is dominated by its
revolution contribution.

Derivatives use the paper's Eq. (22) with T_M. Scaled products of sqrt(q)/y and
lambda/y avoid forming overflowing ratio powers when the final derivative fits.
At x=0, T'=-2 and T'''=-16 are used directly. Genuine zero derivatives are valid;
a derivative exceeding binary64 range returns NUMERICAL_RANGE with no partial
values. One such deliberately extreme case is independently verified.

`find_minimum_multi_rev` solves T'=0 in 0<x<1. It starts with computed derivative
signs at 0 and 0.5, with bounded expansion toward 1 if necessary. Safeguarded
Halley proposals remain inside that bracket. Every fourth update bisects, and
unusable/repeated proposals also bisect. A computed zero derivative triggers
two quarter-point probes; it does not manufacture a zero-width bracket.
Termination requires the full bracket width to meet x_atol+x_rtol*abs(midpoint).
A fresh midpoint evaluation supplies the returned time and stationarity residual.

Counts include all actual evaluations and distinguish Halley/bisection updates,
zero-derivative probes and bracket expansion. Failure leaves x_min/time absent,
with the actual bracket, counters and evaluation status available for diagnosis.
Default tolerances are x_atol=x_rtol=1e-14 and the update budget is 128.

The estimated minimum-time error is the larger endpoint-to-midpoint time spread
plus 16 ulps of the returned time. This is an empirically checked roundoff
estimate, not an interval certificate. Classification is deliberately:

| Requested T relative to reported minimum | Result |
| --- | --- |
| Below the estimated uncertainty band | NO_SOLUTION |
| Above the estimated uncertainty band | TWO_SOLUTIONS |
| Within the band, including equality | NEAR_MINIMUM, multiplicity unresolved |

Targets within numerical uncertainty are not silently declared an exact merged
root. As in the zero-revolution diagnostic, these labels are numerical estimates
with measured validation limits.

## API example

```python
from src.lambert_multirev import (
    MinimumStatus, classify_multirev_time,
    evaluate_derivatives_multi_rev, find_minimum_multi_rev,
)

value = evaluate_derivatives_multi_rev(x=0.2, lambda_=0.5, revolutions=1)
minimum = find_minimum_multi_rev(lambda_=0.5, revolutions=1)
if minimum.status == MinimumStatus.OK:
    feasibility = classify_multirev_time(target_time=5.0, minimum=minimum)
```

All values here are dimensionless. Optional q and MinimumOptions expose the
retained geometry and numerical budgets. The runtime uses only Python's standard
library; there are no network, ephemeris or third-party solver calls in it.

## Independent numerical validation

All **53 standard-library Python tests** pass, including nine new tests. The
11 existing native CPU/CUDA integration tests also pass, including the explicit
M>0 unsupported status of the current complete solver. No multi-revolution CUDA
execution is claimed.

`validation/validate_multirev.py` independently evaluates Lagrange's eccentric
anomaly time equation, differentiates it with mpmath, and bisects that independent
derivative to locate minima. Precision is at least 100 decimal digits and grows
to 403 digits for retained subnormal q. Minimum references use 220 bisections,
or 500 for extremely small q. The lambda=0, M=1 printed fixture uses 160 digits.

| Check | Measured result |
| --- | ---: |
| Forward TOF cases | 685 |
| Maximum relative TOF error | 4.291e-16 |
| Maximum forward error | 2.736 ulps |
| First/second/third derivative scaled errors | 1.197e-15 / 2.228e-15 / 9.922e-15 |
| Expected unrepresentable derivative cases | 1, explicitly rejected |
| Independent minimum cases | 49 |
| Maximum absolute x_min error | 5.466e-15 |
| Maximum minimum-time error | 2.575 ulps |
| Largest actual time error / reported estimate | 0.144 |
| Maximum minimum updates | 48 |

Derivative errors use abs(actual-reference)/max(1,abs(reference)), so near-zero
derivatives are tested with an absolute scale. Cases span M=1,2,5,20,100 and
2^31-1, both lambda signs, endpoint neighbors and positive retained q down to
the smallest subnormal. For each of the 49 minima, five targets around its
uncertainty band are checked against the independent reference. Status tests
cover invalid inputs, exhausted budgets, exhausted floating-point resolution,
unknown near-minimum multiplicity and a disabled-Halley bisection fallback.
These are bounded numerical tests, not an exhaustive proof for all input tuples.

## Real mission case: Juno around Jupiter

Use **Juno, Horizons ID -61**, centered on **Jupiter, 500@599**, from
**2017-02-10 00:00:00 to 2017-04-15 00:00:00 TDB**, a 64-day interval.
NASA documents Juno's early approximately 53-day Jupiter orbits and a perijove
on March 27, 2017 in its [mission orbit chronology](https://science.nasa.gov/mission/juno/juno-orbits/).
The [Horizons Lookup documentation](https://ssd-api.jpl.nasa.gov/doc/horizons_lookup.html)
identifies -61 as the spacecraft.

The three public [Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html)
responses are committed under `validation/data/juno/`, with full query URLs,
parameters, retrieval UTC, API response and SHA-256 of the returned text:

- Jupiter object data supplies GM=126686531.900 km^3/s^2.
- `orbit-hourly.json`: 1,537 hourly Juno vector states for the full interval.
- `perijove-refined.json`: 97 five-minute states around the sampled closest
  approach, March 27 05:00-13:00 TDB, avoiding angular undersampling at perijove.

Queries explicitly request VECTORS, ICRF, REF_PLANE=FRAME, KM-S, geometric
VEC_CORR=NONE and TIME_TYPE=TDB. Runtime validation checks those conventions,
target/center identities, redundant calendar/JD time tags, sample ordering,
cadence, time bounds and agreement at overlapping epochs. Positions and
velocities convert to SI; elapsed time comes from continuous TDB calendar tags.
The Horizons vector source is Juno_merged.

The merged 1,625 unique samples sweep **364.838484766 degrees** when projected
into the initial orbital plane. The largest sample increment is 13.319 degrees;
the minimum projected radius / actual radius is 0.9999663. This resolves one
complete observed turn plus about 4.84 degrees in that reference plane, with
small departure from it. The test checks positive increments, adequate angular
sampling and a projection safely away from its singularity.

For the endpoint geometry, lambda=0.8151730690824357, q=0.3354928674427224 and
T=4.174730431091548. The validated M=0 geometry helper prepares the shape only;
M is then supplied to the new scalar preparation functions.

| Requested revolutions | Minimum time for these endpoints | 64-day classification |
| --- | ---: | --- |
| 1 | 62.703263506 days | Two solutions |
| 2 | 111.741044281 days | No solution |
| 3 | 160.271586220 days | No solution |

Every minimum and classification is checked against multiprecision Lagrange
references. The estimated M=1 minimum-time uncertainty is about 21 nanoseconds
in this nondimensionalization; this is a numerical diagnostic for the specified
point-mass problem, not uncertainty in the actual mission ephemeris or dynamics.

For future branch-solver regressions, the harness independently constructs both
M=1 roots by multiprecision bisection and compares their orbital energies with
Gooding's two solutions. This does not call a product multi-revolution root
solver. Both positive x values emphasize that branch identity must be relative
to x_min, not the sign of x:

| Reference branch | x | Semimajor axis | Gooding low_path for this geometry |
| --- | ---: | ---: | --- |
| Left of x_min=0.141796914738 | 0.0321116404383 | 3.819868346e9 m | False |
| Right of x_min | 0.247938483017 | 4.065872860e9 m | True |

The product TOF/derivative checks at those independently found roots have maximum
relative time residual 2.221e-16 and the expected opposite slope signs.
Reference semimajor axes agree to 4.441e-16 relative. A separate eccentric-anomaly
elapsed-time check includes the requested complete revolution and agrees within
1.863e-9 seconds. The low/high mapping above is measured for this geometry and
is not assumed as a general API rule.

The right reference branch is closer to the actual spacecraft velocities:
133.735 m/s departure and 94.797 m/s arrival differences, versus 367.102 and
1,209.240 m/s on the left branch. Those are point-mass model comparisons. Actual
Juno motion includes Jupiter's nonspherical gravity, other bodies and any
maneuvers in the trajectory record. They are separate from numerical correctness
of the Lambert calculation; this increment does not fit or reproduce that full
dynamical history.

## Reproduction and next increment

From the repository root, with the existing validation dependencies installed:

```bash
python3 -m unittest discover -s tests -v
.venv-validation/bin/python validation/validate_multirev.py --json multirev.json
.venv-validation/bin/python validation/validate_juno.py --json juno.json
# Optional network refresh, performed one Horizons request at a time:
.venv-validation/bin/python validation/validate_juno.py --refresh --json juno.json
LAMBERT_TEST_CUDA=1 .venv-validation/bin/python -m unittest discover -s native/tests -v
```

Ordinary mission validation uses the committed snapshots offline. Raw tests and
numerical results are saved in `validation/results/multirev-preparation*` and
`juno-preparation*`. The local validation environment is the existing
`../../work/tof-validation-env` relative to this checkout.

The next increment is the bracketed left/right root solver and velocity
reconstruction in Python, including near-minimum branch identity and Juno
reference comparisons. Only after that validation should the extension enter
the shared C++/CUDA mathematics and its batch API.
