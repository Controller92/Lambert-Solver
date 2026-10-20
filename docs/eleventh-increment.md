# Native multi-revolution roots and velocities

The shared CUDA C++ numerical core now solves M>0 transfers, including both
stationary-point branches and their departure/arrival velocities. The C++ CPU
backend uses the same core. The Python engine remains a separate implementation
for validation and scientific use. No external Lambert library, ephemeris query,
or CPU fallback runs inside the CUDA product path.

## Python batch API

Rebuild both libraries because this increment changes the native ABI to version 2:

```bash
python3 native/build.py
python3 native/build.py --cuda-root /path/to/cuda --arch 120
```

Positions use meters, elapsed time seconds, and GM m^3/s^2, as before. M is a
positive signed-int32 count. Branch selection is independent of `long_way`,
prograde/retrograde orientation, and the sign of x.

```python
import math
from src.lambert_geometry import LambertInput
from src.lambert_native import NativeEngine, pack_problems, pack_multi_rev_pairs
from src.lambert_root import RootOptions
from src.lambert_multirev import MinimumOptions

problem = LambertInput((1., 0., 0.), (0., 1., 0.),
                       2 * math.pi + math.pi / 2, 1., revolutions=1)

# One selected branch per problem. A sequence of branch names selects per row.
right_rows = pack_problems([problem], branches='right')
right = NativeEngine().cpu(right_rows)

# Both branches, fixed order: left then right for each problem.
rows = pack_multi_rev_pairs([problem])
engine = NativeEngine('cuda')
with engine.cuda_batch(len(rows)) as batch:
    batch.upload(rows)
    elapsed_ms = batch.run(
        RootOptions(), minimum_options=MinimumOptions())
    pair = batch.download().reshape(-1, 2)

for result in pair[0]:
    if result['status'] == 0:
        print(result['branch'], result['v1'], result['v2'])
    else:
        print(result['geometry_status'], result['root']['status'],
              result['minimum']['status'])
```

`pack_problems` defaults to the left branch for M>0. M=0 ignores the branch
selector and retains its single-root behavior. Raw `INPUT_DTYPE` arrays encode
left=0 and right=1. Existing persistent buffers support homogeneous and mixed
M=0/M>0 batches, arbitrary tails, empty uploads, repeated solves and reuse.
`minimum_options` is keyword-only on `NativeEngine.cpu` and `CudaBatch.run`.
It controls the minimum search separately from the root budget.

Each native branch occupies one GPU lane and independently computes its minimum.
The pair helper packs two rows; unlike Python's pair call, it does not share a
minimum. This preserves independent branch scheduling and statuses, at the cost
of repeated geometry/minimum work when both branches are requested. A shared
minimum stage is a future performance option, not part of this increment.

## Numerical port

`native/multirev.hpp` ports the validated Python sequence:

1. Evaluate the elliptic TOF with retained q, including subnormal q in scalar
   validation probes. Genuine zero derivatives are allowed; overflow is explicit.
2. Locate the minimum by safeguarded Halley steps and bisection on T'=0.
   Require the whole computed sign bracket to meet the minimum x tolerance.
3. Compare the target time with the minimum and its estimated uncertainty.
   Leave multiplicity unresolved in the near-minimum uncertainty band.
4. Bracket each root outside the entire computed minimum bracket. The analytic
   x=-1 and x=1 infinite limits remain open and are never evaluated.
5. Use Izzo's logarithmic/tanh starter and direction-aware Householder steps,
   bounded by the branch bracket. Restore bisection when residual progress stalls.
6. Require both residual and step stopping checks, verify the final slope sign,
   and reconstruct velocities with the existing stable q/eta arithmetic.

The reconstruction is shared with M=0 without changing its arithmetic order.
Explicit FMA is retained only for compensated product differences; implicit
contraction, fast math and flush-to-zero remain disabled by the build.
Widening M before M+1 avoids overflow at INT32_MAX.

The native derivative loop reuses its already evaluated TOF. `root.evaluations`
therefore counts actual native TOF calls rather than copying Python's repeated
calls. `minimum.evaluations` counts minimum-stage calls separately. Pair rows
have independent minimum counts, so sum both when measuring native pair work.
Iterations count successful evaluated updates. For each root:
`iterations == householder + bisection + verification`.

### Roundoff regression found during batch validation

A deterministic 32,768-row transfer-search distribution exposed three feasible
rows that failed minimum detection on either CPU or GPU. The computed T' had
rounded to zero, but a nearby quarter probe crossed the computed derivative sign
change. The previous Python rule treated that crossing as an immediate resolution
failure even though a narrower bracket was available.

Both Python and C++ now keep the existing opposite-sign endpoint and narrow the
bracket with the evaluated quarter-probe sign. The ordinary quarter-bracket rule
is retained when both probes have the expected signs. No zero-width bracket or
invented stationary root is substituted. The original width test and estimated
16-ulp time allowance remain unchanged. All 32,768 feasible rows now succeed on
both backends. The three physical inputs are preserved in
`validation/data/multirev-roundoff.json`; four scalar shape variants, including
CPU/GPU q rounding, have independent Lagrange minimum checks and unit regressions.

## Result contract and ABI

Top-level statuses remain OK=0, invalid geometry/input=1, root failure=2,
velocity failure=3. Only OK publishes endpoint velocities. Inspect nested status
codes and diagnostics when a row fails; successful siblings remain usable.

M>0 root statuses match `MultiRootStatus`:

| Code | Meaning |
|---:|---|
| 0 | OK |
| 1 | Invalid root input/options |
| 2 | Branch bracket failure |
| 3 | Root iteration limit |
| 4 | Binary64 resolution limit |
| 5 | TOF evaluation failure |
| 6 | Numerical range (reserved by the shared status contract) |
| 7 | Minimum search failed |
| 8 | No solution for this M |
| 9 | Near minimum, multiplicity unresolved |
| 10 | Final derivative contradicts the requested branch |

Minimum statuses match `MinimumStatus`: OK=0, invalid input=1, evaluation
failure=2, bracket limit=3, iteration limit=4, resolution limit=5.
The -1 sentinel means the stage did not run. Failed stages keep root/minimum
solution values NaN while retaining diagnostic brackets and counters.
M>0 metadata includes `branch`, `revolutions`, and the nested `minimum` record.

Root `precision_limited` stays tri-state: 1 warns, 0 indicates the slope-based
estimate meets the requested x tolerance, -1 means conditioning is unknown.
The M>0 estimate includes the same factor-of-two local inverse-curvature allowance
as Python. Neither the x estimate nor the minimum-time uncertainty is a certified
interval or a velocity accuracy guarantee. Near-minimum successes may lose x and
velocity precision while satisfying the time residual; inspect the warning.

ABI v2 input rows occupy 80 bytes (previously 72), results 368 (previously 240),
and minimum records 120. The original ABI handshake retains its original return
structure so old/new bindings can reject a version mismatch safely; a separate
handshake checks minimum layouts. Old libraries are explicitly rejected.
Direct C ABI callers must supply `MinimumOptions` to `lb_cpu` and `lb_cuda_run`;
`lb_cuda_info` now also selects zero or multi kernel resources.
The Python wrapper and both existing benchmark tools are updated together.

## CUDA execution and performance

Separate zero- and multi-revolution kernels preserve the zero kernel's register
footprint. Upload scans revolution counts on the host to select required kernels.
Homogeneous batches launch one kernel; mixed batches launch both in the same
stream, with disjoint row ownership. Invalid nonzero counts receive explicit
per-row errors. Event timing includes both launches for a mixed batch. Upload
classification is part of pipeline timing, not resident kernel timing.

On the NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, CUDA 12.8:

| Kernel | Registers/thread | Local frame | Compiler spill loads/stores |
|---|---:|---:|---:|
| M=0 | 197 | 32 bytes | 0 |
| M>0 | 212 | 64 bytes | 0 |

At 128 threads/block, both allow two active blocks per SM, a theoretical
occupancy of 16.7%. Wider diagnostics increase transfer traffic even for M=0;
unchanged registers do not imply unchanged transfer-inclusive performance.

| Distribution | Branch rows | CPU ms | CUDA kernel ms | Pipeline ms | Kernel branches/s |
|---|---:|---:|---:|---:|---:|
| M=1..5 both branches | 1,024 | 4.540 | 0.616 | 0.825 | 1,662,752 |
| M=1..5 both branches | 32,768 | 110.114 | 5.611 | 8.440 | 5,840,155 |
| mixed M=0/M>0 | 1,024 | 4.691 | 0.702 | 0.979 | 1,458,989 |
| mixed M=0/M>0 | 32,768 | 150.509 | 6.545 | 9.539 | 5,006,909 |

The benchmark uses general 3D rotated inputs, M=1..5, both branch selectors and
both geometric directions. Each row counts one branch solution, not a pair.
CPU timing is a scalar one-thread C++ baseline. CUDA events exclude compilation,
allocation, transfers and input generation. Pipeline wall timing includes the
host dispatch scan, upload, solve and download using existing allocations.
Clocks are uncontrolled; the saved samples describe this machine/run rather than
a guaranteed throughput. FP64 arithmetic, minimum refinement, repeated minimum
work for pairs and the diagnostic payload remain optimization opportunities.

## Independent validation

All checks passed on actual native outputs:

- 64 standard-library Python tests, 23 native integration tests with CUDA enabled.
- 685 independent C++ TOF/derivative cases, including one expected derivative
  range failure; maximum forward error 2.736 ulps.
- 53 independent minimum/feasibility cases; maximum absolute minimum x error
  5.465e-15, with time errors inside the estimated allowance.
- 554 scalar root slots: 457 OK, 25 resolution limits, 36 near-minimum,
  36 no-solution. The maximum unflagged x error was 4.559e-16.
  Ill-conditioned successful roots reached 1.09e-09 absolute x error and were warned;
  these errors stayed inside the independently checked conditioning diagnostic.
- On each physical backend: 96 conic branches, 86 actual-input multiprecision
  branches (including the six new roundoff regressions), and 24 successful
  near-minimum branches. That is 206 successful complete branches per backend,
  plus both Juno branches. Unresolved/absent branches were also checked.
- CUDA maximum scaled velocity differences: 5.642e-15 versus Gooding on
  ordinary conics, 8.783e-16 versus multiprecision actual-input velocities.
- CUDA maximum scaled Kepler endpoint errors on conics: position
  9.205e-12, velocity 9.172e-12; maximum ODE winding error
  2.333e-09 radians. All original acceptance bounds passed.
- CUDA near-minimum maximum velocity error: 5.317e-10 in normalized units;
  all 24 successes warned, with maximum independent sensitivity-envelope ratio
  0.02328. This is not ordinary full-precision performance.
- CUDA Juno: both branches used five updates, maximum Gooding velocity difference
  2.978e-12 m/s, maximum multiprecision Kepler position closure
  1.266e-06 m. Both carry conservative x-precision warnings.
- Existing M=0 independent C++/CUDA validation passed, including Mars 2020.
- Compute Sanitizer memcheck and initcheck: zero errors; memcheck: zero leaked bytes.
- Both existing native benchmark tools passed ABI-v2 smoke runs. A cached ABI-v1
  library was rejected before solving.

Saved raw evidence is in `validation/results/native-multirev-*`, including build
output, numerical results, tests, sanitizer logs and performance samples. The
Python minimum correction also passed the complete independent Python validation.


The ordinary conic checks use independent Gooding solves, analytical conic states,
energy/momentum conservation, 100-digit Kepler propagation and DOP853 winding.
The actual-input corpus reconstructs geometry and both velocities with 140-digit
Lagrange inversion. No product root or velocity supplies those references.
Near-minimum sensitivity checks are reported separately from ordinary accuracy.
As in increment ten, adaptive ODE endpoint errors are diagnostics; multiprecision
Kepler endpoint checks enforce the original closure bounds.

The Juno case uses the previously cached, hash-checked Horizons geometric ICRF
vectors centered on Jupiter over February 10 to April 15, 2017 TDB. No network
refresh occurred. M=1 has two roots; M=2 and M=3 correctly return no solution.
Ephemeris velocity differences reflect the limitations of a point-mass arc and
are not numerical error criteria.

## Reproduction

Use the validation dependencies documented in earlier increments, then:

```bash
python3 -m unittest discover -s tests -v
LAMBERT_TEST_CUDA=1 python3 -m unittest discover -s native/tests -v
python3 validation/validate_native_multirev.py --gpu --json native-multirev.json
python3 validation/validate_native.py --gpu
python3 validation/validate_multirev.py
python3 validation/validate_multirev_solver.py
python3 validation/benchmark_multirev.py --json native-multirev-benchmark.json
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    python3 -m unittest discover -s native/tests -v
```

Set `LAMBERT_TEST_CUDA=1` for sanitizer invocations as well. Repeat with
`--tool initcheck`. The independent native validation adapter translates only
statuses and fields; complete physical solves execute on the requested backend.
Scalar preparation/root probes execute C++ on the CPU, including when provided
by the CUDA shared library. CUDA is never inferred from CPU/GPU agreement alone.

The legacy GUI is unchanged. Exact collinear transfers remain outside the
geometry contract. This increment establishes a functioning, validated native
M>0 baseline; it does not claim exhaustive numerical certification or final GPU
tuning.
