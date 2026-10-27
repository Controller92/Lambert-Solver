# Terminal-velocity Jacobian

The optional native Jacobian supplies the first derivatives needed for local
trajectory optimization and uncertainty analysis. It supports zero revolutions
and either explicit multi-revolution root in both C++ and CUDA. It differentiates
the received endpoint positions and elapsed time, holding `mu`, direction,
revolution count and root branch fixed.

For `z = [r1x,r1y,r1z,r2x,r2y,r2z,dt]` and
`v = [v1x,v1y,v1z,v2x,v2y,v2z]`, the returned matrix is `dv/dz`, with shape
`(6,7)`. Positions are metres, time seconds and velocities metres per second.
Position columns have units `1/s`; the final column has units `m/s²`.

## API

Build as usual with `python native/build.py`, optionally adding
`--cuda-root /path/to/toolkit --arch 120`. Rebuild both libraries after updating.
The velocity result layout and existing default calls retain ABI version 2.
The optional Jacobian layout has a separate size check.

```python
import numpy as np
from src.lambert_geometry import LambertInput
from src.lambert_native import NativeEngine, pack_problems

rows = pack_problems([
    LambertInput((7e6, 0., 0.), (0., 7e6, 0.), 1500., 3.986004418e14)
])
solutions, sensitivities = NativeEngine().cpu_with_jacobian(rows)
usable = (solutions['status'] == 0) & (sensitivities['status'] == 0)
J = sensitivities['values'][usable]  # (number usable, 6, 7)

gpu = NativeEngine('cuda')
with gpu.cuda_batch(len(rows)) as batch:
    batch.upload(rows)
    milliseconds = batch.run(jacobian=True)
    solutions = batch.download()
    sensitivities = batch.download_jacobian()
```

Use `pack_problems(..., branches='right')` or `pack_multi_rev_pairs(...)` for
multi-revolution inputs. Left/right are relative to the minimum of the TOF
curve, independent of the sign of `x`.

`run(jacobian=True)` solves once per repetition and differentiates the stored
solutions with a second kernel. Its CUDA-event timing includes the velocity
and derivative kernels. Device allocation is lazy on the first nonempty
Jacobian run, outside that event timing. It remains allocated until the batch
closes. Ordinary batches allocate only the existing velocity buffers.
Uploads and ordinary `run()` calls invalidate any prior Jacobian download.
Empty batches and noncontiguous host inputs follow the existing batch contract.
The default 128-thread block is used in the recorded checks and benchmarks.

Native C entry points are `lb_cpu_jacobian`, `lb_cuda_run_jacobian` and
`lb_cuda_download_jacobian`. `lambert::terminal_jacobian(input,result)` is the
shared host/device implementation; its result must come from that same input.

## Method and conditioning

The scalar equation is `F(x,z)=T_M(x,lambda(z))-target(z)=0`. Differentiating
the converged equation gives

```
dx/dz = (dtarget/dz - dT_shape/dz) / T_x
dT_shape = -2 lambda² / y * dlambda
         = lambda / y * dq,  q = 1 - lambda².
```

The production TOF derivatives supply `T_x`. A small seven-component forward
differential type handles smooth geometry and velocity reconstruction. Neither
the initial guess, iterative solver, bracketing logic nor branch selector is
differentiated. No perturbed Lambert solves are used to compute the Jacobian.
Each evaluation uses fixed local length/time units before converting back to SI.

The geometry retains `q=c/s` independently of the rounded `lambda`. For
`abs(lambda) > 0.5`, the shape differential uses `q`; elsewhere it uses
`lambda`. Compensated cross-product primal values preserve the same physical
plane as the velocity calculation.

For short local arcs, direct differentiation of a normalized plane caused large
cancelling terms in arbitrary orientations. The derivative implementation uses
the equivalent Lagrange identities

```
v1 = [r2-r1 + s eta² radial1] / g
v2 = [r2-r1 - s eta² radial2] / g
eta = y-lambda*x
g = dt - (2/3) sqrt(s³/(2 mu)) eta³ H
H = 2F1(1/2,3/2;5/2;(1-x²)eta²).
```

The bounded convergent small-anomaly series is regular across the parabolic
limit. This path is used only for M=0 with positive half-anomaly cosine,
`abs(eta)<0.01` and series argument magnitude below 0.1. Other inputs use the
existing radial/transverse reconstruction differentiated directly. The
velocity solver itself is unchanged.

Check the **Jacobian status as well as the velocity status**:

| Status | Meaning | Matrix |
|---|---|---|
| 0 | Derivative evaluation passed its conditioning/arithmetic checks | finite |
| 1 | Velocity solve failed, including infeasible/ambiguous roots | NaN |
| 2 | Scalar root sensitivity is ill-conditioned | NaN |
| 3 | Derivative evaluation exceeded its numerical range | NaN |

Three dimensionless diagnostics accompany every evaluated root:

* `root_condition = abs(T/T_x)`: change of `x` per relative target-time change.
* `slope_ratio = abs(T_x)/max(T,abs(T_xx))`.
* `estimated_relative_slope_error = abs(T_xx/T_x) * estimated_x_error`, using
  the larger of the solver estimate and a 32-epsilon local rounding floor.

Status 2 is returned if the first diagnostic is nonfinite, `slope_ratio` is at
most `64*sqrt(epsilon)`, or estimated relative slope error is at least `1e-4`.
These are conservative local diagnostics, **not a rigorous bound on all 42
matrix entries**. Near the multi-revolution minimum, `T_x` tends to zero and
the true derivatives diverge. An accepted matrix can lose digits as that point
is approached. Exactly collinear geometries retain the velocity solver's
rejection; no transfer-plane derivative is invented.

The generic implicit-differentiation approach is discussed by Russell,
“Complete Lambert Solver Including Second-Order Sensitivities,” JGCD 45(2),
2022, DOI [10.2514/1.G006089](https://doi.org/10.2514/1.G006089). This implementation
keeps Izzo's parameterization and independently implements first derivatives.
It adds no ivLam source code, coefficient tables or runtime dependency.

## Validation and measured overhead

Reproduce the checks with the validation environment's NumPy, SciPy and mpmath:

```sh
python -m unittest discover -s tests -v
LAMBERT_TEST_CUDA=1 python -m unittest discover -s native/tests -v
python validation/validate_jacobian.py --gpu --output validation/results/jacobian-accuracy.json
python validation/benchmark_jacobian.py --output validation/results/jacobian-benchmark.json
```

`validate_jacobian.py` uses two independent numerical references:

1. DOP853 integrates the Newtonian trajectory and its variational equations.
   With STM blocks A,B,C,D, it obtains the departure blocks from
   `[-B^-1 A, B^-1, -B^-1 v2]`, then the arrival blocks from the final flow.
   Propagated endpoint position and velocity are checked separately.
2. Multiprecision Lagrange-anomaly inversion uses independently bracketed roots
   and central differences of the complete physical velocities, rather than
   native partials. The nominal reference uses 100 digits and relative steps
   `1e-25`; selected tiny-arc and both multi-revolution branch cases are checked
   again with 130 digits and steps `1e-28`.

The 60-case corpus includes elliptic/hyperbolic transfers, both directions, M=1,2,5,
both roots, randomized 3D geometries, parabolic neighbors, tiny arcs to `2e-14`
radians, arbitrary-plane tiny arcs, nearly antiparallel endpoints, and explicit
near-minimum sensitivity degradation. Unit tests additionally check finite
difference step sweeps, quadratic Taylor remainders, rotations, physical units,
Kepler similarity, reversal, failures, empty batches and CUDA buffer lifetimes.
Near-minimum reference comparisons use a separately reported conditioning
allowance; they do not share the ordinary-case accuracy bound.

For 32,768 warmed rows on the RTX PRO 2000 Blackwell laptop GPU, the recorded
matched medians are:

| Workload | Velocity kernels | Velocity + Jacobian kernels | Both outputs, transfer-inclusive |
|---|---:|---:|---:|
| M=0 rotated search | 3.499 ms | 5.951 ms (5.51 million rows/s) | 9.968 ms |
| M=1..5, one selected branch per row | 5.621 ms | 7.573 ms (4.33 million rows/s) | 11.180 ms |

These timings exclude compilation and allocation. The last column includes
upload, one run and downloads of the 368-byte solution and 368-byte sensitivity
rows. Multi-revolution rates count branch solutions, not complete root pairs.
This is a first correctness implementation: the optional derivative kernel
uses 255 registers and the compiler reports local-memory spills. No claim of
optimal throughput or superiority to legacy solvers is made.

## Research scope

This is a two-body endpoint Jacobian. A date derivative additionally needs the
ephemeris velocities and the derivative of elapsed time. A full mission-cost
gradient also needs flyby, parking, surface and cost chain rules, with branch
policies fixed locally. Those integrations, Hessians, navigation covariance,
force-model sensitivities and N-body second-order tensors are subsequent work.
The existing N-body mission STM correction and its acceptance criteria remain
separate from this Lambert derivative API.
