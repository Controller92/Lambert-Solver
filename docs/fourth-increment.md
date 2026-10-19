# Fourth increment: safeguarded zero-revolution root finder

Built on verified `major-revision` commit `32080f1`. This increment implements
only inversion of the validated scalar T(x,lambda) curve. Existing geometry,
TOF, derivative/starter modules and legacy trajectory functions are unchanged.
No terminal velocities, multirevolution logic, ephemerides or CUDA code is added.

## Contract and basic use

```python
from src.lambert_root import RootOptions, RootStatus, solve_x_zero_rev

result = solve_x_zero_rev(0.75, -0.5)
assert result.status == RootStatus.OK
print(result.x, result.residual, result.iterations)  # 1.0, 0.0, 1
```

Inputs are a positive finite target nondimensional T and finite |lambda|<1.
M=0 is fixed. The root domain is x>-1, including elliptic x<1, parabolic x=1
and hyperbolic x>1. The sign of lambda preserves the selected short/long arc.
There is no separate multirevolution branch or root selector.

The physical conventions remain metres, seconds, mu in m^3/s^2, one central body
and one right-handed inertial frame at both endpoints, and a consistent dynamical
elapsed time. Geometry supplies `T=sqrt(2*mu/s^3)*tof`. The root layer itself
needs no physical units, dates or coordinate conversions.

Default RootOptions are:

| Option | Default | Meaning |
| --- | ---: | --- |
| x_atol | 1e-14 | Absolute step tolerance |
| x_rtol | 1e-14 | Relative step tolerance |
| tof_atol | 0 | Absolute nondimensional TOF residual tolerance |
| tof_rtol | 5e-13 | Relative TOF residual tolerance |
| max_iterations | 128 | Completed root-update budget |
| max_bracket_expansions | 1024 | Finite upper-bound expansion budget |

Both tests must pass on a newly evaluated candidate:

```text
|x_new-x_old| <= x_atol+x_rtol*|x_new|
|T(x_new)-target| <= tof_atol+tof_rtol*target.
```

The step test measures stabilization; it is not a certificate of actual x error
on an ill-conditioned curve. The separate conditioning diagnostics below expose
that distinction. Absolute and relative tolerances must be finite and nonnegative;
relative tolerances must be below one. Explicit zero tolerances are allowed.
Budgets must be nonnegative integers. Booleans are rejected as numerical options.

## Mathematics and safeguards explained before implementation

The published Izzo paper was checked directly, especially section 4.1 on p. 10:
[DOI 10.1007/s10569-014-9587-y](https://doi.org/10.1007/s10569-014-9587-y).
For f=T(x)-target, its Householder correction is

```text
dx = f*(f'^2-f*f''/2) / (f'*(f'^2-f*f'') + f'''*f^2/6).
```

The source agrees with the update in the
[author's implementation](https://github.com/esa/pykep/blob/master/src/lambert_problem.cpp).
No solver code was ported. This implementation factors the correction using
`u=f/f'`, `a=u*f''/f'`, `b=u*(f'''/f')*u/6`, giving
`dx=u*(1-a/2)/(1-a+b)`. This avoids unnecessary squares/cubes of derivatives.
A nonfinite result or near-cancelled denominator is rejected.

For admitted M=0 geometry, T(x) decreases from infinity as x approaches -1 to
zero as x tends to infinity. There is one mathematical root for every positive
target T. The finite-precision implementation maintains a sign bracket using
evaluated T values:

- If target>=T(0), start with the open analytic lower limit -1 and upper bound
  0. The code never evaluates x=-1.
- Otherwise start at lower bound 0 and an upper bound at least 1, using the
  verified starter when available. Expand the upper bound by `hi=2*hi+1` until
  T(hi)<=target, subject to explicit range/evaluation/budget checks.
- Use a starter within the bracket. If it is unavailable or outside the final
  bracket, use the bracket midpoint and record used_fallback_guess=True.
- Retain Householder candidates only inside the bracket and with correction
  at most 75% of its width. Failed derivative evaluations and unsafe corrections
  use bisection. Every fourth update also bisects if the residual is still large,
  ensuring periodic bracket contraction rather than relying on the starter.
- If a proposed Householder point cannot be evaluated, attempt the midpoint.
  If that also cannot be evaluated, report the underlying evaluation failure.
- When the current f is exactly zero, make a genuine zero update and freshly
  evaluate T. It counts as one verification step, rather than inventing a prior
  iteration or claiming convergence solely from an initial residual.
- If no interior binary64 point remains, a qualifying endpoint can still be
  verified. If neither endpoint meets the residual tolerance, return
  RESOLUTION_LIMIT. A small or zero step cannot override a bad time residual.

Bracketing uses binary64 function values, not rigorous interval arithmetic.
The independent checks below quantify equation and inverse error.

## Results and diagnostics

RootResult has x and y only on OK. On failure those fields remain None, while
last_x, computed time, signed residual, signed relative residual, last accepted
step and bracket bounds retain available diagnostics. No velocity or substitute
root is returned on failure.

| Status | Meaning |
| --- | --- |
| OK | Both requested stopping tests passed on a freshly evaluated candidate |
| INVALID_INPUT | Invalid domain, nonfinite input or invalid options |
| BRACKET_LIMIT | Upper-bound expansion budget exhausted |
| ITERATION_LIMIT | Root-update budget exhausted |
| RESOLUTION_LIMIT | Available neighboring x values cannot meet the time tolerance |
| EVALUATION_FAILURE | Required forward evaluation failed, with its TofStatus recorded |
| NUMERICAL_RANGE | Constructing a finite upper bracket exceeded representable range |

These are numerical statuses, not claims of Lambert infeasibility. In particular,
the five old unrepresentable starters now recover roots by explicit bisection.

iterations counts completed x updates; householder_steps+bisection_steps+
verification_steps equals iterations. Bracket expansions are separate.
tof_evaluations includes two calls inside the starter, one inside every invoked
derivative evaluation, direct forward calls and the final conditioning check.
derivative_failures counts failed derivative attempts used during iteration;
the final optional diagnostic can be unavailable without undoing a valid root.
The root state is fixed-size and local to one call; no iteration history or
batch arrays are allocated. This is a CPU reference, with no throughput claim.

## Conditioning near long-arc degeneracy

Very negative lambda near -1 and small negative x can give an almost flat time
curve. Even an evaluated residual of zero then leaves uncertainty in x because
the forward T was rounded to binary64. Input target rounding can also shift the
true root away from the generating x. Validation distinguishes those errors by
inverting the actual rounded target independently.

On success, when the derivative API is available at the returned x, report

```text
estimated_x_error = (|computed residual| + 16*ulp(computed T)) / |T'(x)|
x_precision_limited = estimated_x_error > x_atol+x_rtol*|x|.
```

The 16-ulp allowance is a diagnostic roundoff model calibrated against the
independent forward checks, not an interval proof or a universal error bound.
At all 3,175 returned roots in the main and conditioning corpora, the maximum
forward T error measured against the 160-digit equation is 14.397 ulps, below
this allowance. The validation script reproduces this calibration.
It gives a first-order sensitivity estimate. A True flag calls attention to
possible x accuracy worse than the requested step scale. Both fields remain
None when the combined derivative API cannot evaluate the point, such as large
hyperbolic x with underflowing higher derivatives. Unknown is not reported as
an assurance of accuracy. OK certifies the stopping criteria, not a uniform
forward-x accuracy promise; consumers needing a value-accuracy certificate must
inspect this diagnostic and validate their physical tolerances separately.

## Independent validation

All 33 standard-library tests pass. New tests cover exact x=0 and x=1 roots,
circular short/long boundary data, all five formerly unrepresentable starters,
the poor near-lambda=1 starter, derivative-underflow fallback, unsafe step
injection, actual evaluation counts, exhausted iteration and bracket budgets,
unattainable residual tolerances, invalid inputs and conditioning diagnostics.

`validation/validate_root.py` generates target times from the independent
Lagrange anomaly formula, rounds them to binary64 as actual inputs, and inverts
those rounded targets with independent multiprecision bisection. It uses no
production starter or derivatives. Generating x supplies a small independent
reference bracket; it does not seed the production solve.

Each reference bracket is narrowed by 100 bisections. Lagrange precision starts
at 140 digits and increases according to distance from x=1; this handles
multiprecision midpoints far closer to the removable parabolic singularity than
binary64 neighbors. Final bracket signs are checked at 60 additional digits.
The maximum scaled reference width is 1.578e-40.

The principal corpus contains 3,127 generated cases: 500 deterministic random
pairs plus elliptic, near-parabolic, hyperbolic, endpoint and method-switch
cases, including x up to 1e150 and lambda immediately inside +/-1. No unexpected
failures occurred. Error scales below use denominator 1+|reference x|.

| Check | Observed maximum | Acceptance bound |
| --- | ---: | ---: |
| Scaled x error against independent inverse | 7.805e-15 | 5e-14 |
| Scaled x error against generating x | 7.577e-15 | Reported separately from rounded-target inverse error |
| Independent relative TOF residual | 4.517e-13 | 6e-13 |
| Computed relative TOF residual | 4.520e-13 | 5e-13 |

Mean completed updates: 5.53118; median 2, 95th percentile 38, 99th percentile
49 and maximum 51. The corpus deliberately includes extreme geometries and
derivative underflow; these counts are not a benchmark on a representative
mission batch and cannot establish a GPU speedup. Across the corpus, 395 cases
use bisection, 45 use a replacement starter, and 74 encounter derivative failure
and fall back. At most one finite upper-bracket expansion is used.

The principal corpus has 12 conservative x-precision warnings and 74 unavailable
diagnostics where combined derivatives underflow. All roots still meet the
measured principal-corpus bounds above.

An additional 48-case conditioning corpus probes small negative x and lambda
near -1. All cases meet the stopping criteria, with 11 x-precision warnings.
Maximum scaled error against the independent inverse is 5.420e-12, at generating
x=-1e-5 and lambda=-0.9999999999999999. The returned x is
-1.0000005574415106e-5 and its reported first-order absolute error estimate is
7.366e-11. The greatest actual absolute error divided by the reported estimate
is 0.15027. All measured errors beyond the ordinary 5e-14 scaled bound are
flagged. This supplemental acceptance uses the reported conditioning model;
a uniform 5e-14 x-error bound is explicitly not claimed there.

Confidence is high for the tested domains and explicit failure behavior. The
solver does not promise success for every finite binary64 input or every user
tolerance. Forward evaluation range and input/function conditioning remain
real limits. No propagation, terminal-velocity or multirevolution accuracy has
been established by this increment.

## Reproduce and next step

From the repository root:

```bash
python3 -m unittest discover -s tests -v
.venv-validation/bin/python validation/validate_root.py
```

The existing validation/requirements.txt supplies mpmath 1.3.0. The session uses
Python 3.12.3. Runtime math requires only the standard library. The independent
root validation is separate and fails explicitly if its dependency is missing.

The next increment should reconstruct v1/v2 from this root and the verified
geometry, then test circular solutions, two-body propagation, arrival velocity,
energy, angular momentum and an established independent Lambert implementation.
GPU work follows the validated complete CPU solve; device allocation/transfers,
execution mapping and timing remain separate concerns.
