# Third increment: derivatives and zero-revolution initial guess

This increment prepares for root iteration on local `major-revision`, following
`8e60de0`. It adds `src/lambert_root_preparation.py`, analytical tests and an
independent validation script. The validated geometry and forward T code are
unchanged. No root step, solved velocities, multirevolution branch or CUDA kernel
is implemented here. No legacy trajectory entry point is rerouted.

## Inputs, outputs and purpose

`evaluate_derivatives_zero_rev(x, lambda_)` returns T, y and the first three
derivatives with respect to x, holding lambda fixed. `initial_guess_zero_rev(T,
lambda_)` returns an approximate x0 and the T0/T1 thresholds. Both reuse TofStatus:
OK, INVALID_INPUT, NUMERICAL_RANGE and SERIES_NONCONVERGENCE. Failures contain no
partial mathematical values; the derivative-series failure records its actual
term count. OK does not mean that a Lambert root was found.

All values are dimensionless. The unchanged physical contract uses metres,
seconds, mu in m^3/s^2, positions relative to the same central body in the same
right-handed inertial frame, and elapsed dynamical time. Geometry defines
`T=sqrt(2*mu/s^3)*tof`. lambda retains the short/long arc sign, and M=0 is fixed.
The derivative domain is finite x>-1, |lambda|<1; the starter instead accepts
positive finite target T and the same lambda domain. Ephemerides, frame changes,
dates, allocation/launch management and perturbations remain separate.

The first derivative describes how transfer time changes with x. It is negative
for the tested M=0 problems: a shorter target time requires a larger x. Higher
derivatives will permit the paper's Householder update, but that update and its
stopping rules require a separate verified increment. A starter is only a place
to begin that process.

## Equations checked before editing

The attached published paper was read directly, including complete published
pp. 7 and 11: Izzo, Celest Mech Dyn Astr (2015) 121:1-15,
[DOI](https://doi.org/10.1007/s10569-014-9587-y). Equation (22), with d=1-x^2
and q=1-lambda^2, gives:

```text
d*T'   = 3*x*T - 2 + 2*lambda^3*x/y
d*T''  = 3*T + 5*x*T' + 2*q*lambda^3/y^3
d*T''' = 7*x*T'' + 8*T' - 6*q*lambda^5*x/y^5
```

Repeated division by d amplifies rounding near x=1. The production code therefore
uses a separately derived local series there. For x>0, the earlier Lagrange
anomaly expression can be written as the real integral

```text
T = 2 * integral(lambda to 1) u^2 / sqrt(1-d*u^2) du.
```

Expanding the binomial denominator for |d|<1, then integrating term by term:

```text
T = sum(n=0..infinity) a_n*d^n
a_n = 2*c_n*(1-lambda^(2*n+3))/(2*n+3)
c_0 = 1; c_n/c_(n-1) = (n-1/2)/n.
```

Differentiate this convergent series with respect to d. With S1=dT/dd,
S2=d^2T/dd^2 and S3=d^3T/dd^3, the chain rule gives

```text
T'   = -2*x*S1
T''  = 4*x^2*S2 - 2*S1
T''' = -8*x^3*S3 + 12*x*S2.
```

These S1/S2/S3 symbols are local derivative sums, not the Battin S1 argument in
the paper. At x=1 exactly, only the first three coefficients contribute:

```text
T'(1)   = -(2/5)*(1-lambda^5)
T''(1)  = (6/7)*(1-lambda^7) - (2/5)*(1-lambda^5)
T'''(1) = (18/7)*(1-lambda^7) - (10/3)*(1-lambda^9).
```

The first agrees with Eq. (23); the other two are derived here and checked
independently. At x=0, |lambda|<1 ensures y>0, and Eq. (22) is regular:
T'(0)=-2 and T'''(0)=-16.

## Important numerical choices

- Near x=1 use the differentiated series over 0.7<=x<=1.3, so |d|<=0.69.
  This is wider than the forward-TOF Battin window because the derivative
  recurrences amplify cancellation more strongly. The series uses three
  compensated sums, scalar state, and at most 256 nonconstant coefficients.
- Stable coefficient complements start from factored 1-lambda^3 and update
  `1-lambda^(m+2)=(1-lambda^m)+lambda^m*q`. This avoids subtracting near-equal
  values when positive lambda is close to one. Exact parabolic powers use
  expm1/log for positive lambda and addition for negative lambda.
- For n>=3, `r=|d|*(n+1/2)/(n-2)` bounds all three next absolute term ratios;
  subsequent ratios decrease. Once r<1, each remaining tail is bounded by
  `|current_term|*r/(1-r)`. All three must be below binary64 epsilon times their
  sum. This controls truncation, not every source of floating-point rounding.
- Away from the local region use Eq. (22) and the already validated forward
  T/y. In the first numerator replace
  `-2+2*lambda^3*x/y` with `-2*(q+lambda^2*eta/y)`, where eta=y-lambda*x is
  rationalized when needed. This avoids cancellation near positive lambda=1.
- Return NUMERICAL_RANGE if any required derivative is nonfinite or underflows
  to zero. All three derivatives must be representable for this combined API.
  Thus the admitted practical range is narrower than forward T alone. For
  example, T(1e100,0) is representable while its third derivative is not.

These are scalar CPU reference routines, with no reduced precision, approximate
math settings, device buffers or performance claim. A derivative evaluation
reuses one forward T/y evaluation. No production dependency was added.

## Starter derivation and correction to the printed expression

Use T0=T(0,lambda) and T1=T(1,lambda), both obtained from the verified evaluator:

```text
T >= T0:       x0 = (T0/T)^(2/3)-1
T <= T1:       x0 = 1 + (5/2)*T1*(T1-T)/(T*(1-lambda^5))
T1 < T < T0:   x0 = exp(log(2)*log(T0/T)/log(T0/T1))-1.
```

The middle branch is rederived from the paper's preceding log-time relation
`xi=log(2)*(tau-tau0)/(tau1-tau0)`, with x=exp(xi)-1. The printed expression
on p. 11 uses exponent log2(T1/T0) on base T0/T, which is negative and fails
the required x=1 endpoint. The correct positive exponent on T0/T is
`log(2)/log(T0/T1)`, the reciprocal of log2(T0/T1).
This is also consistent with the middle starter in the
[author's implementation](https://github.com/esa/pykep/blob/master/src/lambert_problem.cpp),
written there using base T/T0 and exponent log(2)/log(T1/T0). No source was ported.

Production arithmetic uses expm1 and log1p near boundaries to retain small
offsets from x=0. The short-time factor cancels (1-lambda) analytically, giving
`(5/3)*(1+lambda+lambda^2)/(1+lambda+lambda^2+lambda^3+lambda^4)`.
Boundary equality is explicit: T=T0 returns x0=0, and T=T1 returns x0=1.
The routine rejects an x0 rounded to -1 rather than silently clipping it.

## Tests and independent numerical validation

All 22 standard-library tests pass, including the previous 15. New analytical
tests cover x=0 identities, all three parabolic limits against 80-digit Decimal
arithmetic, floating-point neighbors of x=1, derivative method-switch continuity,
starter endpoints, monotonicity, inverse slope at the parabola and failure paths.
At T=sqrt(T0*T1), the correct middle branch gives x0=sqrt(2)-1 for every lambda;
this specifically catches the printed exponent defect.

`validation/validate_root_preparation.py` performs numerical central
differentiation of the independent Lagrange anomaly formula from the previous
increment. It does not differentiate the production code or use its derivative
recurrence/series. Steps are scaled by distances to x=-1 and the complex zeros
of y. Each reference is evaluated at 100 decimal digits with h proportional to
1e-10, then at 160 digits with h/100. Their maximum relative change is
4.526e-19, below the 1e-17 reference-stability bound.

1,059 inputs passed, including 500 deterministic random pairs and endpoint,
near-parabolic and method-switch cases, lambda at the floats immediately inside
+/-1, and hyperbolic x up to 1e40. There were no unexpected derivative failures.
Method counts: 773 recurrence, 273 series and 13 exact parabolic evaluations.

| Derivative | Maximum relative error | Acceptance bound |
| --- | ---: | ---: |
| First | 7.734e-15 | 5e-14 |
| Second | 6.783e-14 | 2e-13 |
| Third | 6.260e-13 | 2e-12 |

All three maxima occur in the recurrence immediately above x=1.3, where some
cancellation remains. The first two occur at lambda=-0.9999999999999999; the
third at lambda=-0.9. The local series maxima are 8.938e-16, 3.766e-15 and
3.002e-14 respectively. Exact parabolic maxima are 2.094e-16, 3.660e-16 and
8.966e-16. Maximum derivative-series index was 138 of 256.

Confidence is high within this tested corpus. The bounds are measured and
validated against independent numerics; they are not a proof covering every
binary64 input. In particular, the third derivative is less accurate than T.

## Starter correctness versus usefulness

Of the same generated problems, 1,054 produced representable starters and
matched a high-precision implementation of the derived formula. Maximum
scaled formula error `|x0-x0_reference|/(1+|x0_reference|)` is 4.600e-16 against
a 5e-14 bound. Regions: 142 long-time, 274 intermediate and 638 short-time.

Guess quality is a separate measurement. Compared with the known x used to
generate each target time, `|x0-x|/(1+|x|)` has mean 0.03243 and maximum 0.66969.
The worst case is x=0.01, lambda=0.9999999999999999: x0=0.6863818585732864 for
T=2.2204460492490802e-14. A correct starter formula is not necessarily close
to the root near extreme geometry, and no convergence claim follows.

Five additional cases correctly report NUMERICAL_RANGE because the mathematical
starter rounds to -1. Four have generating x=-0.9999999999999999 and lambda
0.99, 0.999, 0.999999 or 0.9999999999999999, with T=9.494881184090847e23.
The fifth has x=-0.999999999999 and lambda=0.9999999999999999,
T=1.1107575921394486e18. The generating roots are representable: this limitation
belongs to the starter, not Lambert feasibility. No replacement seed is hidden.

## Reproduce and next step

From the repository root, use the existing validation dependency:

```bash
python3 -m unittest discover -s tests -v
.venv-validation/bin/python validation/validate_root_preparation.py
```

If needed, create .venv-validation and install validation/requirements.txt as
documented in the second increment. Actual session runtime: Python 3.12.3,
mpmath 1.3.0. Tests and independent validation are separate commands; no critical
check is silently skipped when the validation dependency is missing.

The next increment should implement a safeguarded M=0 root finder with explicit
bracketing, validated alternatives for unrepresentable starters, both step and
TOF-residual criteria, real iteration counts and nonconvergence diagnostics.
It must recover known x values across all branches of the M=0 curve before
adding velocity reconstruction or CUDA. Nothing here establishes root convergence,
physical arrival accuracy or GPU speed.
