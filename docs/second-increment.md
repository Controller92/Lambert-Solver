# Second increment: forward zero-revolution time of flight

Implemented on local `major-revision`, following verified geometry commit
`c3aa36b`. Scope: one scalar forward mathematical layer. No changes to the legacy
trajectory functions, no x inversion, derivatives, starting guesses, terminal
velocities, multirevolution logic or CUDA kernel. This remains a custom solver
development path, not a library wrapper.

## What this layer means

Geometry supplies lambda and the target nondimensional time. The next solver
layer will seek an x for which the forward function equals that target. This
increment implements only that forward function:

```python
from src.lambert_tof import TofStatus, evaluate_tof_zero_rev

result = evaluate_tof_zero_rev(x=1.0, lambda_=0.5)
assert result.status == TofStatus.OK
print(result.time_nondim)  # 7/12, a known parabolic value
```

Both inputs and outputs are dimensionless. The previous contract remains:
`T=sqrt(2*mu/s^3)*elapsed_seconds`, positions in metres relative to one central
body in a shared right-handed inertial frame, mu in m^3/s^2, and a consistent
dynamical time interval. The sign of lambda carries the chosen short/long arc
orientation. No ephemerides or date conversions enter this function. There is
exactly zero complete revolutions; no multi-revolution branch selector is exposed.

Domain: finite x > -1 and -1 < lambda < 1. Ellipses use -1 < x < 1, the parabola
uses x=1, and hyperbolas use x>1. x=-1 has infinite TOF. Degenerate |lambda|=1 is
excluded. OK indicates forward evaluation, not convergence of a Lambert solve.

## Mathematics checked before implementation

The attached published Izzo paper, Celest Mech Dyn Astr (2015) 121:1-15,
[DOI](https://doi.org/10.1007/s10569-014-9587-y), was read directly. Relevant
relations are on published p. 6, Eqs. (17)-(21):

- `y=sqrt(1-lambda^2*(1-x^2))` and `eta=y-lambda*x`.
- `T=(psi/sqrt(abs(1-x^2))-x+lambda*y)/(1-x^2)` for M=0.
- Elliptic `sin(psi)=eta*sqrt(1-x^2)` and
  `cos(psi)=x*y+lambda*(1-x^2)`.
- Hyperbolic `sinh(psi)=eta*sqrt(x^2-1)`.
- Near x=1: `z=(1-lambda-x*eta)/2`,
  `T=(2/3)*eta^3*F + 2*lambda*eta`, with `F=2F1(3,1;5/2;z)`.
- At x=1 exactly: `T=(2/3)*(1-lambda^3)`.

The printed Eq. (20) labels the Gaussian function as 1F2. The Gaussian function
with numerator parameters 3,1 and denominator parameter 5/2 is conventionally
2F1. Its definition is given by
[NIST DLMF 15.2.1](https://dlmf.nist.gov/15.2.E1). The author's
[PyKEP TOF/series source](https://github.com/esa/pykep/blob/master/src/lambert_problem.cpp)
also confirms these parameters and recurrence. The implementation was derived
from the equations and Gauss series; no solver source was ported.

## Important code choices

`src/lambert_tof.py` contains the entire new runtime layer. It uses Python
binary64 floats and only the standard library, with scalar fixed-size state.

1. Compute `q=(1-lambda)*(1+lambda)` and `y=hypot(sqrt(q),lambda*x)`. These
   avoid cancellation of 1-lambda^2 and avoid squaring a large x prematurely.
2. When lambda*x>0, compute `eta=q/(y+lambda*x)` using
   `y^2-(lambda*x)^2=q`. Otherwise evaluate the noncancelling subtraction.
3. At the parabola use `(2/3)*(1-lambda)*(1+lambda+lambda^2)`, avoiding
   cancellation of 1-lambda^3 near positive lambda=1.
4. For 0.8<=x<=1.2, use the Battin expression. Across the admitted lambda
   interval, z lies between -0.44 and 0.36. Thus |z|<=0.44<1 and the series
   converges rapidly. The exact x=1 case still takes its analytic limit.
5. The Gauss series uses t0=1 and
   `t_(n+1)=t_n*z*(n+3)/(n+5/2)`, with compensated summation. Its decreasing
   absolute term ratios give a geometric bound on the uncomputed tail.
   Stop when that tail is <= binary64 epsilon times the sum. The maximum is
   128 nonconstant terms. This bounds truncation in F, not total rounding in T.
6. Away from this window, elliptic psi uses atan2 and hyperbolic psi uses asinh.
   atan2 preserves the elliptic quadrant; neither requires an inverse function
   evaluated at a nearly rounded-to-one argument. Use
   `lambda*y-x=lambda*eta-q*x` to avoid another cancellation. Hyperbolic
   division is performed twice by sqrt(x^2-1), avoiding a squared denominator.

Errors never yield substitute values. INVALID_INPUT rejects invalid domains,
nonfinite inputs and booleans. NUMERICAL_RANGE reports nonrepresentable
intermediate operations or nonpositive/nonfinite outputs. SERIES_NONCONVERGENCE
reports an exhausted term budget. Failures have no T or y. Successful output
includes y, evaluation method and actual series-term count. There is no root
iteration count or residual at this layer.

## Analytical checks and independent authority

The standard-library suite now has 15 tests, all passing, including the prior
seven geometry tests. The eight new tests check:

- Eq. (19): x=0 gives `acos(lambda)+lambda*sqrt(1-lambda^2)`.
- Eq. (21): x=1 gives the parabolic limit, compared against 80-digit Decimal
  arithmetic, including the floats immediately inside lambda=+/-1.
- Known circular short/long arcs at Earth scale: a=R fixes |x|, the orientation
  fixes its sign, and independently known circular elapsed times are reproduced.
  Absolute nondimensional T errors: 2.220e-16 and 4.441e-16, bound 2e-15.
- The closed hyperbolic case lambda=0, x=sqrt(2).
- Both floating-point neighbors of x=1, and continuity at the two method switches.
- Positive decreasing sampled M=0 curves from x=-0.999999 through x=1e150.
- Invalid/range failures and deliberately exhausted series diagnostics.

`validation/validate_tof.py` independently evaluates the earlier Lagrange anomaly
formula from Eqs. (9)-(15), not the production Eq. (18)/Battin implementation:

- Elliptic: alpha=2*acos(x), beta=2*asin(lambda*sqrt(1-x^2));
  `T=((alpha-sin(alpha))-(beta-sin(beta)))/(2*(1-x^2)^(3/2))`.
- Hyperbolic: alpha=2*acosh(x), beta=2*asinh(lambda*sqrt(x^2-1));
  `T=((sinh(alpha)-alpha)-(sinh(beta)-beta))/(2*(x^2-1)^(3/2))`.
- The parabolic point uses its exact limit. Input floats are converted to
  multiprecision without rounding through short decimal strings.

It uses mpmath 1.3.0 at 100 digits, then checks every reference value at 160
digits. This independent formulation and implementation is the numerical
authority for this layer. It shares the two-body theory, and is not a complete
independent Lambert solver or a substitute for eventual propagation validation.

## Observed numerical results

3,127 distinct inputs passed with no evaluation failures: 1,100 elliptic,
1,375 Battin, 37 exact parabolic and 615 hyperbolic evaluations. The corpus
includes 500 deterministic random inputs (seed 9587), representable neighbors
of x=-1, x=1 and lambda=+/-1, switch boundaries, and hyperbolic x up to 1e150.

| Evaluation method | Maximum relative T error |
| --- | ---: |
| Elliptic | 1.1661e-15 |
| Battin | 2.2477e-15 |
| Parabolic | 2.9464e-16 |
| Hyperbolic | 1.1592e-15 |

Overall bound used: relative error <=5e-14. Worst relative error is at
x=1.1999999999999997, lambda=-0.999999, reference T=1.1897857126568745.
Maximum nonconstant series terms: 47 of 128.
Maximum relative reference change from 100 to 160 digits: 4.830e-71,
below the explicit 1e-70 bound. Maximum relative y error is 2.255e-16,
against a separate 1e-15 bound.

The tested T range is 2.2204460492503131e-166 to 9.4948811840908474e23.
The maximum absolute T error is 1.1607344618487486e8 at the largest-T case,
x=-0.9999999999999999, lambda=-0.9999999999999999. This is about 1.22e-16
relative to its T, illustrating why a fixed absolute tolerance is inappropriate
over this range. T is dimensionless; multiply by the geometry time scale to
express an error in seconds for a particular physical problem.

Confidence is high within this tested corpus. These are observed bounds, not
a proof of an error bound for every admitted binary64 input. Very large
hyperbolic inputs may overflow intermediate products despite a representable
mathematical T. For example x=1e300, lambda=-0.9 explicitly returns NUMERICAL_RANGE.
Scaling for that extreme regime is deferred. No clipping or approximate
asymptotic substitute is used to hide it.

## Reproduce

From the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m venv .venv-validation
.venv-validation/bin/python -m pip install -r validation/requirements.txt
.venv-validation/bin/python validation/validate_tof.py
```

The actual session used a task-local validation environment outside the checkout.
The dependency is for independent validation only. No production dependency or
GUI/ephemeris requirement was added. GPU performance was not measured.

## Next mathematical step

Verify the first three analytic T derivatives against independent numerical
differentiation, including their parabolic limits. Then validate the T0/T1
starting guesses. Root iteration, its step/residual stopping criteria, velocity
reconstruction and physical propagation checks remain subsequent increments.
