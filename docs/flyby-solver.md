# Custom flyby encounter engine (increment 13)

The flyby encounter solver now has a standard-library Python reference, shared
C++17 CPU/CUDA mathematics, persistent GPU batches, independent validation, and
saved CLI runs. It uses no PyKEP/poliastro numerical runtime. Lambert's ABI and
kernels are unchanged; the flyby libraries have their own version/size handshake.

## Run it

Use the same environment/build procedure as the [research CLI](research-cli.md):

```bash
python native/build.py --cuda-root /path/to/cuda --arch 120
python run.py flyby --config examples/flyby-earth.json --backend cuda
python run.py flyby --config examples/flyby-juno-earth-2013.json --backend cpu
```

CPU-only builds use `python native/build.py`. The menu offers **7. Flyby
Encounters**; exit remains option 6. CPU and CUDA must be selected explicitly;
a CUDA failure raises an error. SPICE and Horizons access remain on the host.
No kernels or spacecraft data are fetched by an encounter run.

Each run saves `request.json`, `metadata.json`, and `encounters.npz` in a new
UUID directory. The NPZ contains native `inputs`, complete `results`, and
`vout_inertial_m_s`. Metadata records units, frame, body, TDB epoch, request/data
hashes, the loaded library hash, status counts, precision warnings, and timings. The inertial output is
NaN unless a body velocity was supplied. Failed encounter vectors are NaN.
The CLI loads at most 100,000 input encounters and chunks native work; use the
batch API directly for larger populations. It is not an out-of-core search.

The JSON request supplies a common body, epoch and inertial frame for the batch.
`mu_m3_s2`, `radius_m`, `min_altitude_m`, and `rp_max_m` define its physical bounds.
The radius and GM must be deliberate body/atmosphere model choices; a body label
does not fetch or verify physical constants. All distances are
metres, velocities m/s, GM m³/s², angles radians. `epoch_tdb` is a timezone-free
TDB calendar label; the solver does not infer an ephemeris from that label.

`velocity_origin` is `body_relative` or `inertial`. Inertial input requires
`body_velocity_m_s`, in exactly the same frame at the same epoch. The host
subtracts it from each incoming/target velocity, solves the encounter, and adds
it back to successful outputs. Inputs are **hyperbolic excess/asymptotic**
velocities, not finite-altitude spacecraft velocities. Do not subtract a
planet velocity at a different epoch or mix ecliptic and equatorial components.

## Models and equations

Let S be the unit incoming relative velocity. The B vector points from the body
to the incoming straight-line asymptote. The caller supplies a reference vector:
T is its normalized projection into the plane normal to S, R = S × T, and
Bhat = cos(azimuth) T + sin(azimuth) R. Thus T × R = S. This explicit convention
need not match another library's beta or B-plane axes. A zero reference or one
within 1e-12 in normalized projected length of parallel to S is rejected; no
arbitrary axis is silently substituted.

For incoming speed v and periapsis centre distance rp:

- u = rp v² / GM, e = 1 + u, alpha = asin(1/e), delta = 2 alpha.
- b = rp sqrt(1 + 2/u), vp = sqrt(v² + 2 GM/rp).
- Sout = cos(delta) S − sin(delta) Bhat.
- rhat_peri = sin(alpha) S + cos(alpha) Bhat.
- that_peri = cos(alpha) S − sin(alpha) Bhat.

`forward` takes rp and azimuth and returns the unpowered outgoing velocity,
B vector, periapsis position and incoming/outgoing periapsis velocities. Relative
speed and orbital energy are conserved. A heliocentric speed change follows
from adding the body's inertial velocity; it is not a planet-frame burn.

`match` takes two requested asymptotic velocities. Their magnitudes must agree
within the explicit test `abs(vout − vin) <= speed_atol + speed_rtol * max(vin,vout)`.
Defaults are 1e-6 m/s and 1e-12. A successful result retains the incoming speed;
`target_speed` and signed `speed_error` expose any accepted mismatch. It never
averages incompatible energies or silently pays a powered burn. The inverse is
rp = (GM/v²) cos²(delta/2) / [sin(delta/2)(1 + sin(delta/2))]. This avoids subtracting
nearly equal terms near a reversal.

`powered` is a **tangential impulse at a common periapsis**, with continuous
periapsis position and velocity direction. Incoming and outgoing speeds may
differ. It solves the monotone scalar equation

`delta_required = asin[1/(1+rp vin²/GM)] + asin[1/(1+rp vout²/GM)]`

inside the supplied radius interval, using geometric bisection (maximum 160
iterations). It returns both periapsis velocities and delta-v = abs(vp_out −
vp_in), evaluated with a scaled difference-of-squares expression. An equal-speed
feasible encounter gives zero burn. This model is not a general powered-flyby
optimization allowing arbitrary impulse direction/location or multiple burns.
A rejected turn can still be achievable under a more general model; no minimum
fuel claim is made.

Bounds are inclusive. Inverse feasibility allows 2e-14 relative rounding slack
at a boundary and clamps only that slack back to the exact bound. Powered root
brackets close to 2e-13 relative width and must also satisfy a relative turn or
complementary-turn residual check. Exact zero turn has no finite unpowered
periapsis; exact reversal requires zero radius and fails positive safety limits.

## Stability, statuses and diagnostics

Angles use atan2 of a compensated cross-product norm and dot product, after
exact power-of-two scaling of the original vectors. Python uses Dekker products
and compensated sums; C++/CUDA use explicit error-compensating FMA operations.
Compiler contraction remains disabled elsewhere. Near reversal, calculations
retain pi minus the turn directly rather than subtracting rounded angles.
Small/large u branches preserve sin(alpha) and cos(alpha); scaled speed/impact
parameter expressions avoid unnecessary intermediate overflow. Unsupported
binary64 ranges return a numerical status, never fabricated zero velocities.
There is no fast math, FTZ, or hidden CPU fallback.

Statuses are SUCCESS=0, INVALID_INPUT=1, ZERO_SPEED=2, BAD_BASIS=3,
SPEED_MISMATCH=4, TURN_OUT_OF_RANGE=5, DEGENERATE_TURN=6,
NO_CONVERGENCE=7, NUMERICAL_RANGE=8. `precision_limited` is -1 for failed rows,
0 for ordinary success, and 1 for inverse solutions with conservative input-angle
conditioning `32 eps / min(sin(delta/2),cos(delta/2)) > 1e-8`. It warns that small
input-vector perturbations can substantially change inferred radius; it is not
a measured uncertainty bound. `angle_residual` is model turn minus requested
turn; an angle rounded to pi can hide a tiny complementary-angle residual, so
also inspect the flag and powered radius bracket. `lower`/`upper` are the final
powered bracket (or the solved MATCH radius); forward bounds are NaN.

## Python and native APIs

```python
from src.flyby import Encounter, Mode, solve
from src.flyby_native import FlybyEngine, pack_encounters

p = Encounter(vin=(10000., 0., 0.), mu=3.98600435436096e14,
              rp_min=6.7e6, rp_max=1e9, rp=7e6, azimuth=.4)
reference = solve(p)
rows = pack_encounters([p])
engine = FlybyEngine('cuda')
cpu = engine.cpu(rows)
with engine.cuda_batch(len(rows)) as batch:
    batch.upload(rows)
    milliseconds = batch.run()
    gpu = batch.download()
```

The input dtype is 136 bytes and output dtype 224 bytes. ABI version 1 checks
these before native calls. Buffers are persistent; upload invalidates a previous
result, download requires a completed run, and close/reupload/empty/tail batches
are tested. Context operations serialize access with a lock. `device_info()`
reports resources for the encounter kernel itself. The C ABI is documented in
`native/flyby_api.h`; the host/device numerical routine is `flyby::solve`.

For adjacent Lambert legs at an encounter, use arrival `v2 − body_velocity`
and next-leg departure `v1 − body_velocity`. Both must belong to the same body,
epoch and frame, and preserve their M/left/right/orientation identities. Apply
MATCH or POWERED to every candidate pair before mission-level reduction.
Independent per-leg minimum-C3 winners need not form a feasible flyby chain.
The current porkchop winner arrays do not preserve every successful velocity
candidate, so full chain enumeration remains a separate next increment.

## Independent validation and reproduction

```bash
python -m pip install -r validation/requirements.txt
LAMBERT_TEST_CUDA=1 python -m unittest discover -s flyby_tests -v
python validation/validate_flyby.py --cuda --count 1000
python validation/benchmark_flyby.py
```

The suite checks orbital energy, angular momentum, B-vector sign, independent
Cartesian-to-conic asymptotes, DOP853 integration across periapsis, rotation,
length-unit scaling, time reversal, powered periapsis continuity, speed tolerance,
unsafe turns, inclusive bounds, limiting turns, invalid inputs, CUDA tails,
empty/reused/closed contexts, frame conversion, schema rejection, and saved data.
Multiprecision validation uses 100-digit independent conics/root calculations for
3,022 cases. It compares Python, C++ and actual CUDA velocities, radius,
periapsis states and powered delta-v. Read the measured results and hardware
limits in [the validation report](thirteenth-increment.md).

Cached Horizons snapshots include exact requests, response headers, retrieval
times and content hashes. Ordinary validation is offline; `--refresh` explicitly
requests replacements from the public API. Juno's Earth encounter on 2013-10-09
is used. Incoming/outgoing **osculating asymptotes** are inferred from geometric,
Earth-centred ICRF finite states an hour either side of a five-minute sample near
perigee. A separate five-second table checks closest approach. The Juno example uses
the central-force periapsis epoch, 19:22:32.060483 TDB, as its encounter label;
18:25 TDB remains the source finite-state epoch. Altitudes here
subtract the equatorial reference radius; these are not geodetic heights above
the oblate Earth. Full Horizons dynamics are not exactly a single Keplerian
hyperbola; their discrepancy is reported separately from numerical error.

Primary references:

- [ESA/PyKEP encounter conventions and unpowered constraints](https://esa.github.io/pykep/flyby.html).
  Its beta basis and fb_dv model differ; neither is assumed identical to ours.
- [JPL Horizons API specification](https://ssd-api.jpl.nasa.gov/doc/horizons.html)
  and [time-step/frame manual](https://ssd.jpl.nasa.gov/horizons/manual.html).
- [NASA's Juno Earth flyby account](https://www.nasa.gov/image-article/earth-from-juno/).

## Legacy audit and remaining work

The reference `Simple-Lambert-Solver` Flyby-Solver commit
`6c58c779a91c8a9a32a514708089583c2f9d7e23` provides interface context only.
Its `core/flyby_build.py` treats the heliocentric spacecraft velocity as v-infinity
and adds a perpendicular impulse rather than rotating the body-relative vector.
Its `core/flyby_math.py` averages unequal excess speeds for ballistic feasibility
and uses the entire asymptote angle between periapsis velocities for a powered
burn estimate. These helpers were not copied into the numerical runtime. The
legacy `flows/flyby.py` also wraps PyKEP; this audit does not claim every path in
that branch uses the defective helper formulas.

The encounter engine is operational. Full date-window Earth–Mars–Ceres searches,
Lambert candidate preservation, chained ranking, checkpoint/resume and complete
multi-body mission validation remain subsequent work. Oblateness, atmosphere,
finite sphere-of-influence matching and navigation correction require a richer
dynamical model than this patched-conic encounter.
