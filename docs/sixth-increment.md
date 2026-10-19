# Sixth increment: preserve tiny arcs throughout the CPU solve

The subsequent [seventh increment](seventh-increment.md) implements the shared
C++ CPU/CUDA engine and batch interface. The Python mathematics described here
remains its independent reference.

This increment fixes the precision limit identified at `8acc65f` and cleans
up shared mathematical operations before CUDA translation. It preserves the
metre/second contract, central-body frame, short/long arc convention, M=0 scope,
root stopping criteria and explicit failure outputs. Legacy GUI code is untouched.

## Why a small residual was insufficient

Geometry supplies lambda with `1-lambda^2=c/s`. For nearly coincident directions
and similar radii, lambda rounds very close to one. Reconstructing c/s from that
single rounded number loses information about the chord. A root can then satisfy
the wrong rounded geometry accurately. Tightening its stopping tolerance cannot
recover the discarded information.

The geometry now retains `q=c/s` as a separate scalar. It is carried through
the forward TOF, derivatives, initial guess, root, final diagnostic and velocity
reconstruction. For positive lambda, the small complement is evaluated as
`1-lambda=q/(1+lambda)`. This also appears in the Battin-series argument, exact
parabolic T, differentiated-series coefficients and parabolic derivatives.
Updating only y or only the final velocities would leave inconsistent equations.

The equations were checked against the attached published Izzo paper, pp. 6-7
and Algorithm 1 on p. 12:
[Revisiting Lambert's problem](https://doi.org/10.1007/s10569-014-9587-y).
The retained representation is an algebraic rearrangement, with no change to
the two-body model. No external solver code was copied.

## Arbitrary 3D planes needed an additional fix

The original axis-aligned circular probes did not reveal every geometry loss.
Testing a tiny arc in a general orientation also exposed cancellation in the
cross product of normalized radials and in `(r1_norm-r2_norm)/c`.

The geometry helper now:

- Scales each original position by an exact power of two before finding its
  transfer-plane normal. Compensated differences of products preserve the
  small cross product. Python 3.12 lacks math.fma, so a bounded Dekker product
  remainder is used. Ordinary perpendicular transverse bases still use the
  direct cross product. The splitting technique is described in
  [Dekker's original work](https://ir.cwi.nl/pub/9159).
- Resolves one half-angle through a non-subtractive vector norm, then obtains
  the other from `sin(theta)=2*sin(theta/2)*cos(theta/2)`. This protects both
  nearly parallel and nearly antiparallel directions.
- Computes rho using
  `dot((R1-R2)/c, (R1+R2)/(r1_norm+r2_norm))`, with bounded scaling. This
  avoids subtracting rounded endpoint norms and dividing their error by c.

The collinearity rejection threshold remains `64*binary64_epsilon` in the sine
of the angle. Exact parallel/antiparallel inputs still require an additional
plane convention and remain unsupported. The threshold now uses a more accurate
plane calculation. This does not remove physical sensitivity to uncertain inputs.

## Interface and cleanup

`solve_zero_rev(LambertInput(...), options)` is unchanged. Its geometry result
now contains q. The lower-level functions accept an optional keyword-only q:

```python
g = prepare_geometry(problem).geometry
root = solve_x_zero_rev(g.time_nondim, g.lambda_, q=g.q)
tof = evaluate_tof_zero_rev(root.x, g.lambda_, q=g.q)
```

`evaluate_derivatives_zero_rev` and `initial_guess_zero_rev` accept the same
keyword. Omission preserves the earlier scalar-lambda interpretation. Explicit
q must be finite, positive, <=1 and consistent with lambda within 64 binary64
epsilons absolutely. An absolute check is necessary because tiny q deliberately
contains information absent from lambda. Positive q can represent a nondegenerate
shape even if its accompanying lambda rounds to +/-1. Scalar-only +/-1 inputs
remain invalid; vector geometry still applies its collinearity cutoff.

Two small shared helpers validate the shape representation and evaluate
`eta=y-lambda*x` stably. Derivatives and reconstruction reuse the already
computed y. Parabolic odd-power complements use bounded polynomial evaluation.
The temporary `geometry_precision_warning`, `geometry_q_relative_mismatch` and
their threshold were removed because the calculation now retains the missing
quantity. They are not replaced with an always-False assurance.

Root residuals, actual evaluation/update counts, failure statuses, estimated
x error and tri-state x_precision_limited remain available. The root warning
still matters on genuinely flat long-arc time curves; this increment does not
claim to cure that separate conditioning problem. Fixed-size scalar state and
separation from host I/O remain appropriate for a CUDA C++ translation.

## Accuracy must refer to the received inputs

For tiny angles, a rounded endpoint need not remain exactly on the ideal circle
that generated it. For example `(cos(a),sin(a))` becomes `(1,a)` when a is tiny.
For elapsed time a with mu=1, the latter boundary data requires a radial departure
velocity approximately a/2, while the ideal circle has zero radial departure
velocity. Both statements are compatible: they describe different boundary data.

The strongest new physical validation therefore rebuilds all geometry from the
actual binary64 inputs in 120-digit arithmetic. It solves the independent
Lagrange anomaly equation by 180 bisections, then reconstructs both velocities
in multiprecision. No production geometry or root helper constructs that
reference. This avoids mistaking input quantization for implementation error.

## Validation results

All **44 standard-library tests** pass. The three new test methods cover
invalid/inconsistent shape pairs at every scalar layer, independently computed
parabolic limits including lambda rounded to +/-1, and arbitrary-plane geometry
against Decimal arithmetic. The old temporary-warning test now verifies actual
tiny-arc velocities. One old exact rho assertion allows its mathematically
equivalent calculation's one-ulp rounding difference.

| Independent suite | Cases | Observed maximum error |
| --- | ---: | --- |
| Original scalar TOF | 3,127 | 2.248e-15 relative T |
| Original first/second/third derivatives | 1,059 | 7.734e-15 / 6.783e-14 / 6.260e-13 relative |
| Original root inversions | 3,127 | 7.805e-15 scaled x error |
| New retained-q TOF/root inputs | 366 | 1.544e-15 relative T |
| New retained-q derivative inputs | 352 | 3.463e-15 / 2.434e-14 / 1.601e-13 relative |
| New actual-input physical transfers | 144 | 8.965e-16 scaled endpoint velocity error |
| Existing full-transfer conic corpus | 248 | 2.623e-13 scaled known-state velocity difference |
| Existing strict Gooding comparisons | 240 | 1.135e-13 scaled endpoint velocity difference |

The 144 new physical cases include angles 1e-4, 1e-8, 1e-12, 2e-14,
pi-1e-8 and pi-2e-14; equal, slightly unequal and 2:1 radii; axis-aligned and
general 3D planes; normalized and Earth-orbit scales; short and long arcs.
Velocity errors are divided by max(sqrt(mu/r1), both reference speeds).
Their acceptance bound is 2e-12. Maximum normal-vector error is 1.382e-16,
rho error 2.356e-16, sigma error 3.931e-16 and relative q error 1.974e-16,
all below 5e-14. No unexpected failures occurred.

The retained-q scalar corpus reaches q=1e-16, both lambda signs, parabolic
neighbors and hyperbolic x=1e150. Its 352 representable derivative cases meet
the original derivative acceptance bounds. Fourteen roots have unavailable
higher-derivative diagnostics; they retain None and use the established
safeguards. Three carry precision warnings. Maximum scaled inverse error is
6.467e-12 on a flat long-arc curve and is accepted only under the reported
conditioning model, with the warning required beyond the ordinary 5e-14 bound.
No uniform x-accuracy claim is made for such cases.

The original 48 flat-curve conditioning cases also pass unchanged, with maximum
scaled inverse error 5.420e-12. The 16-ulp forward-roundoff diagnostic allowance
is retained: measured maxima at returned roots are 14.397 ulps in the original
3,175-case calibration and 11.074 ulps in the new 366-case corpus. This remains
a calibrated first-order model, not a rigorous interval bound.

The 248 physical cases still pass 992 independent two-body integrations at
two tolerance/step settings, including analytical controls. Refined maximum
scaled position/velocity errors are 4.882e-11 / 3.608e-11. Controls show comparable
integrator error, as before. The comparison with unrounded known conics includes
input quantization; it is distinct from the 144 actual-input reference checks.

The [Mars 2020 report](mars2020-validation.md) adds three real-ephemeris boundary
problems, independent Gooding comparisons and dimensional propagation checks.
Confidence is high in the tested domains and explicit failures. No universal
accuracy certificate, physical-radius constraint or multiple-revolution support
is implied.

## Reproduce and next milestone

Use Python 3.12 and validation/requirements.txt as in the fifth increment:

```bash
python3 -m unittest discover -s tests -v
.venv-validation/bin/python validation/validate_tof.py
.venv-validation/bin/python validation/validate_root_preparation.py
.venv-validation/bin/python validation/validate_root.py
.venv-validation/bin/python validation/validate_solver.py
.venv-validation/bin/python validation/validate_precision.py
.venv-validation/bin/python validation/validate_mars2020.py
```

The mathematical runtime remains standard-library-only. The next milestone is
the first double-precision CUDA C++ implementation with a minimal Python batch
interface, preserving these contracts and validating against this reference.
Kernel optimization follows measurements on the actual GPU; no performance or
speedup claim is established here.
