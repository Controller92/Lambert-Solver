# First CPU increment and repository audit

Audit date: 2026-10-19. Confidence is high for the code paths and numerical
reproductions below. No original failing input or historical convergence log was
supplied, so a particular past run cannot be attributed to one defect.

## Scope and preserved state

A fresh clone was fetched; no existing checkout was modified. No applicable
AGENTS.md was found in its ancestors or repository tree. Remote branch tips:

| Branch | Commit | Finding |
| --- | --- | --- |
| main | `3a6f47cfbdb990fe50e89097d369273728296fa2` | Approximate scalar/batch CuPy calculations, no x iteration |
| feature/gravity-assists | `ea6117945ef8df51f170fb86be419c46a308c07c` | Same approximate Lambert functions, plus gravity-assist workflows |
| izzo-improvement-in-progress | `60a3f6d0a5f74f888f7cc2366a2e0cefea91b91c` | Attempted x helpers, duplicate definitions, no connection to solver entry points |

Local `major-revision` was created directly from fetched `origin/main`. Its initial
upstream association with main was removed to avoid an unintended default push. It did
not already exist locally or remotely. Existing production files are unchanged.
History inspection locates the attempted x helpers in commit `55a4dd8`, not a
separate complete correct implementation. No mathematics was ported wholesale.

## Confirmed defects

Links below pin the audited commits; line numbers refer to those versions.

1. **No actual Lambert inversion in the public solver.** On
   [main, trajectory.py:67-111](https://github.com/Controller92/Lambert-Solver/blob/3a6f47cfbdb990fe50e89097d369273728296fa2/src/trajectory.py#L67-L111),
   the code estimates an ellipse's semimajor axis from a period approximation,
   clamps it to the minimum-energy value, and scales it by revolution count.
   It never evaluates T(x), never solves the requested endpoint time, and cannot
   represent hyperbolic semimajor axes. `numiter` and `rtol` are unused.
   The gravity-assist branch retains these functions at the same lines.
2. **Arrival velocity is dimensionally incorrect.**
   [main:83-84 and 207-208](https://github.com/Controller92/Lambert-Solver/blob/3a6f47cfbdb990fe50e89097d369273728296fa2/src/trajectory.py#L83-L84)
   computes `(g*r2-r1)/g`. Since g has units of seconds, its numerator adds
   incompatible dimensions. It is not Lagrange's arrival-velocity expression.
   The departure f/g coefficients also use a in place of the appropriate orbital
   geometry. An analytical circular quarter-orbit test confirms substantial errors.
3. **Direction differs between scalar and batch.**
   [main:75-77 versus 200-201](https://github.com/Controller92/Lambert-Solver/blob/3a6f47cfbdb990fe50e89097d369273728296fa2/src/trajectory.py#L200-L201)
   uses signed sin(theta) in the scalar zero-revolution g, but abs(sin(theta))
   in the batch g. The long-way flag therefore does not produce the same arc.
4. **Unreported replacement velocities.**
   [main:121-129 and 218-229](https://github.com/Controller92/Lambert-Solver/blob/3a6f47cfbdb990fe50e89097d369273728296fa2/src/trajectory.py#L121-L129)
   replaces high speeds with a fixed +Y circular-velocity vector, without a
   failure status. That vector need not even be transverse to the input radius.
5. **The attempted iteration solves the wrong function.**
   [improvement:1203-1265](https://github.com/Controller92/Lambert-Solver/blob/60a3f6d0a5f74f888f7cc2366a2e0cefea91b91c/src/trajectory.py#L1203-L1265)
   uses f=x+psi-T and an unrelated polynomial psi. Izzo Eq. (18) instead gives
   `T=( (psi+M*pi)/sqrt(abs(1-x*x)) - x + lambda*y )/(1-x*x)` with psi
   from Eq. (17). The update omits the third derivative in the paper's
   fourth-order Householder formula. A small absolute step alone is accepted;
   no final TOF residual is required. On exhaustion it prints that it falls back
   to Newton but simply returns x. The starter at 1177-1200 is unrelated to the
   paper's T0/T1 piecewise starter and does not implement M>0 branches.
6. **Duplicate definitions and broken surviving derivative helpers.**
   [improvement:1053-1289](https://github.com/Controller92/Lambert-Solver/blob/60a3f6d0a5f74f888f7cc2366a2e0cefea91b91c/src/trajectory.py#L1053-L1289)
   defines get_izzo_initial_guess, householder_iteration, compute_psi, and S1
   twice. Python uses the later definitions. compute_dT_dx still calls
   compute_psi with one argument, while the final definition requires two,
   producing TypeError. The earlier shadowed iteration also has sqrt(1-x*x)
   on its hyperbolic path and is not a valid alternative. The S1 functions do
   not implement the Battin near-parabolic expression in Eq. (20).
7. **The enhanced entry point never invokes these helpers.**
   [improvement:1292-1379](https://github.com/Controller92/Lambert-Solver/blob/60a3f6d0a5f74f888f7cc2366a2e0cefea91b91c/src/trajectory.py#L1292-L1379)
   repeats the semimajor-axis approximation and hard-codes iterations=1,
   converged=True, final_error=0.0. It unconditionally calls cp.asnumpy, which
   fails when the fallback backend is NumPy. Printed convergence here is not
   evidence of x convergence or a solved boundary problem.
8. **Host and math are coupled, and central-body conventions disagree.**
   [main trajectory.py:14-17, 309-336](https://github.com/Controller92/Lambert-Solver/blob/3a6f47cfbdb990fe50e89097d369273728296fa2/src/trajectory.py#L309-L336)
   imports ephemeris/time backends at module import, uses Earth's mu for
   interplanetary positions, and substitutes made-up positions on exceptions.
   [skyfield_interface.py:141-174](https://github.com/Controller92/Lambert-Solver/blob/3a6f47cfbdb990fe50e89097d369273728296fa2/src/skyfield_interface.py#L141-L174)
   labels its positions heliocentric but explicitly returns barycentric states
   without subtracting the Sun. The new math module has no dependency on this path.

No custom CUDA kernel is present in these three trees. The scalar functions
allocate CuPy arrays, issue many operations, and transfer results back; batch
functions vectorize the approximation. There is no persistent-device-batch API.
No recorded benchmark with hardware, accuracy, and timing methodology was found.
`estimate_time` uses hard-coded assumptions. main's python_test_shell.py only
imports modules. The improvement branch's test_trajectory_methods (1846-1997)
prints comparisons and hard-coded convergence without physics assertions.

## Reproduced audit failures

Read-only probes extract the branch function definitions with Python AST and
execute their actual arithmetic with NumPy as cp, avoiding GUI/ephemeris imports.
This establishes CPU arithmetic/code-path defects; it is not a GPU execution test.

| Probe | Observed |
| --- | --- |
| Active x helper, lambda=0, T=pi/2, start x=0 | Returns 1.570796326794897 instead of known x=0 |
| Returned x checked against paper Eq. (18) | T(x)-T_target = -1.075972173595388 |
| compute_dT_dx | TypeError: compute_psi missing lmbd |
| Enhanced solver with NumPy fallback | AttributeError: numpy has no asnumpy |
| Unit-radius circular quarter orbit, mu=1, dt=pi/2 | Departure/arrival velocity norm errors 0.1758425187995891 and 1.002892978124137 in the test's consistent velocity units |
| Same long-arc problem, r1=(1,0,0), r2=(0,1,0), dt=3*pi/2, mu=1 | h_z scalar=-0.9238795325112867, batch=+0.9238795325112867 |

These demonstrate both false convergence and absent inversion. The original
historical failure could involve other inputs or environment effects; that
attribution remains unconfirmed.

## First implementation: contract and geometry only

Entry point: `src.lambert_geometry.prepare_geometry(LambertInput(...))`.
Python >=3.10, standard library only; binary64 floats. No existing caller is
rerouted yet. There is no root finder, velocity reconstruction, or CUDA code.

- Inputs: two finite 3-component positions in metres, positive elapsed seconds,
  positive mu in m^3/s^2, boolean long_way, integer revolutions=0.
- Both endpoints must use the same central body, right-handed inertial frame,
  axes, and origin. The caller supplies a consistent dynamical interval, such
  as TDB elapsed seconds. Frame names and origins cannot be inferred numerically.
- The core is general two-body math. EMC integration will use Sun-relative
  geometric states and solar mu. SPICE, dates, allocations/transfers, GUI,
  optimizers, perturbed dynamics and correction remain separate host layers.
- long_way=False selects angle in (0,pi), normal along normalized r1 cross r2;
  True selects (pi,2*pi), reverses the normal, and makes lambda negative.
  This is not a prograde/retrograde flag about a fixed axis.
- M=0 has one x solution for a chosen orientation. No low/high or left/right
  multiple-revolution branch is exposed or approximated.
- Status OK means geometry prepared. Other statuses are INVALID_INPUT,
  UNSUPPORTED_REVOLUTIONS, DEGENERATE_GEOMETRY, NUMERICAL_RANGE. Failure returns
  geometry=None. iterations=0 and residual=None accurately indicate no TOF solve.
  Solver nonconvergence and infeasible root-branch statuses will be added with
  those algorithms, not fabricated here.
- Zero radii are invalid. Coincident, same-ray, opposite-ray, and near-collinear
  cases with sin(angle)<=64*binary64_epsilon (1.4210854715202004e-14) are explicitly
  rejected. A supplied-plane treatment of exact pi transfers is deferred.
  Nonrepresentable derived scales, including ratio underflow, return NUMERICAL_RANGE.

The helper implements c=|r2-r1|, s=(r1+r2+c)/2, lambda^2=1-c/s,
T=sqrt(2*mu/s^3)*tof, rho=(r1-r2)/c, and sigma=sqrt(1-rho^2).
No claim is made about all possible extreme binary64 input ranges.

Important choices in the code:

1. math.hypot computes 3-vector norms without naive squared-length overflow.
2. `normal=sign*cross/|cross|`, then `transverse=normal cross radial`, ensures
   an oriented unit basis at both endpoints.
3. The equivalent half-angle identity
   `abs(lambda)=sqrt(r1/s)*sqrt(r2/s)*|radial1+radial2|/2`
   avoids cancellation of 1-c/s near antiparallel geometry.
4. `sigma=sqrt(r1/c)*sqrt(r2/c)*|radial1-radial2|` follows from the chord law
   and avoids cancellation in 1-rho^2 for unequal nearly parallel radii.
5. `time_scale=(s/sqrt(mu))*sqrt(s/2)`, then `T=tof/time_scale`, avoids forming
   s^3. Output vectors are fixed-size tuples, with no problem-sized arrays,
   hidden device state, allocation management, or iteration loop.

## Published-paper checks

Equations were checked against the attached published 15-page PDF, not substituted
from the arXiv manuscript. Source: Izzo, Celest Mech Dyn Astr (2015) 121:1-15,
[DOI](https://doi.org/10.1007/s10569-014-9587-y).

- Section 2.2, pp. 4-5 defines lambda and T.
- Algorithm 1 on p. 12 prints an unnormalized cross product as a hatted normal,
  and self-cross-products for the arrival transverse direction. A self-cross-product
  is zero, so this code instead derives normalized bases geometrically.
- Algorithm 2 on p. 13 prints T<0 as a precondition, contradicting positive t,
  mu, and the definition of T. Positive T is required.
- Eq. (18) loses precision near x=1. The next increment must implement and
  independently validate Eq. (20)'s Battin hypergeometric expression and Eq. (21)'s
  parabolic limit before attempting any iteration.

## Validation results

Run from repository root: `python3 -m unittest discover -s tests -v`.
Seven tests pass on Python 3.12.3. Analytical cases include a 3-4-5 triangle,
quarter and three-quarter circular boundary data at Earth scale, rotations off
the XY plane, endpoint reversal, and L-similar scaled systems. Circular tests
check only geometry, not a solved orbit. Near-collinear cases and explicit
failure paths are included.

500 deterministic random cases (seed 9587) compare against independently evaluated
80-digit Decimal norms, c/s lambda, rho/sigma, and T. Length scales are 1e-3, 1,
1e11 and 1e110 metres, including scales where directly forming s^3 overflows binary64.

| Quantity | Maximum error | Acceptance bound |
| --- | --- | --- |
| Lengths | relative 2.220e-16 | 5e-14 |
| lambda | absolute 2.220e-16 | 5e-14 |
| T | relative 4.441e-16 | 5e-14 |
| rho | absolute 1.665e-15 | 5e-14 |
| sigma | absolute 4.441e-16 | 5e-14 |
| Basis lengths and dot products | absolute 3.331e-16 | 5e-14 |
| L-similarity lambda/T | absolute 5.551e-17 | 4e-16 |

The near-antiparallel lambda=2.5e-10 test passes an absolute 1e-24 bound;
the unequal near-parallel sigma test passes a 1e-23 bound. Individual analytical
checks use tolerances from 1e-16 to 2e-15. No CPU/GPU agreement is used as proof.
Propagation, energy, angular momentum, elliptic/hyperbolic velocity solutions,
and established-solver comparisons await the corresponding CPU solver layer.

## Later repository used only as reference

Per user clarification, work remains in Lambert-Solver. The later repository was
cloned read-only and inspected at main `1ff0e66f8998fc96abcfd214d11d994a8f01ffbc`
and Flyby-Solver `6c58c779a91c8a9a32a514708089583c2f9d7e23`.

[Simple-Lambert-Solver main solver.py:15-16, 96-152](https://github.com/Controller92/Simple-Lambert-Solver/blob/1ff0e66f8998fc96abcfd214d11d994a8f01ffbc/lambertlab/src/lambertlab/solver.py#L96-L152)
delegates Lambert math to poliastro. The
[Flyby-Solver lambert_io.py:17-42, 71-157](https://github.com/Controller92/Simple-Lambert-Solver/blob/6c58c779a91c8a9a32a514708089583c2f9d7e23/src/lambertlab/core/lambert_io.py#L71-L157)
delegates to PyKEP and converts km, km/s and seconds explicitly. Its branch wrapper
ignores requested direction/path flags and requests cw=False only, despite
comments claiming both orientations. Do not adopt its branch labels blindly.
Its propagated-arc workflow and tests are useful later references, not executed
or independently certified here. No code was ported from it.

## Short sequence of next increments

1. Implement scalar T(x,lambda,M=0), including the parabolic limit and stable
   near-parabolic series. Check known x values and high-precision independent
   evaluations across elliptic and hyperbolic domains. Still no iteration.
2. Verify analytic derivatives and initial guesses, then add guarded root solving
   with both step and residual tolerances, actual iteration counts and explicit
   nonconvergence. Add velocity reconstruction only after that layer is validated.
3. Validate circular solutions, two-body propagation, arrival velocity, energy,
   angular momentum, direction, and an established independent implementation.
   An Izzo-based library is useful but shares theory; use Gooding as well where available.
4. Port the verified M=0 math into CUDA C++. Test one solve per thread as a
   hypothesis; retain batches on device. Measure registers/spills/occupancy,
   iteration divergence, layout, kernel-only timing, transfers and full workflow
   separately on documented hardware with an explicit CPU baseline. No speedup
   is promised and no relaxed precision/math is enabled.
5. Add M>0 minima/feasibility and explicitly selected left/right x roots only
   after M=0 validation, with matching independent physics and GPU tests.

## Environment and publication limits

The sandboxed `nvidia-smi` command cannot communicate with the driver, but the
same read-only query outside the sandbox succeeds: NVIDIA RTX PRO 2000 Blackwell
Generation Laptop GPU, compute capability 12.0, driver 615.71.09, 8151 MiB total
memory. nvidia-smi reports CUDA UMD 13.4, which does not establish an installed
compiler toolkit. `nvcc` is not on PATH, and no /usr/local/cuda* or /opt/cuda*
directory was found. CUDA compilation, execution, occupancy and throughput
remain unverified. No CUDA implementation was attempted in this increment.

HTTPS clone/fetch work outside the network sandbox. A Git push dry run fails:
`could not read Username for 'https://github.com': No such device or address`.
No Git credential helper, Git identity or gh CLI was configured. Existing SSH
agent identities also fail GitHub authentication with `Permission denied (publickey)`
when checked against [GitHub's officially published Ed25519 host key](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints) using a
task-local known-hosts file. No user SSH configuration was changed.

The connected app identifies Controller92 and reports repository push permission,
but an actual create-branch call returns HTTP 403, `Resource not accessible by
integration`. Account permission metadata does not establish token write access.
The exact integration restriction is not proven. No remote branch or commit was
created. The verified increment is committed locally with an explicit neutral
Codex identity because the local environment has no configured author identity.
No force-push, merge to main, or modification of another branch is performed.
