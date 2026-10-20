# Complete mission planner

The planner jointly searches Earth–Mars–Ceres–Mars–Earth trajectories with a
Ceres surface stay. It evaluates complete encounter chains with the custom
C++/CUDA Lambert and flyby engines. Python owns SPICE, optimization, branch
selection, persistence and independent validation. The example permits up to
3652.5 days from Earth departure to return and searches ballistic and powered
Mars encounters in separate runs.

## Run

Install `research-requirements.txt`, build the native libraries, and explicitly
fetch the additional pinned Mars centre and orientation kernels:

```bash
python tools/fetch_research_kernels.py --mission
python run.py mission-search --config examples/missions/emc-ten-year.json \
    --model both --backend cuda --outdir results/missions
```

Interactive menu option **8** runs the same workflow. Option 6 still exits.
`mission-search --dry-run` checks the configuration without loading SPICE or
initializing CUDA. Select CPU explicitly when a GPU is unavailable. GPU errors
are reported; they do not trigger a hidden CPU fallback. Population evaluation
uses the selected backend; constrained local refinement explicitly uses C++ CPU.

The example launch window is 2034–2038, with leg durations of 100–450,
200–1600, 200–1600 and 100–500 days and a 30–365 day Ceres stay. These are
configurable research bounds, not consequences of the paper. All dates are TDB,
all states geometric heliocentric ECLIPJ2000, all internal quantities SI. Mars
uses **499, body centre**, with mar099s, rather than the barycentre used in some
older standalone porkchop presets. The archived Ceres kernel coverage ends in
November 2045; dates outside any kernel's coverage fail explicitly.

## Complete-chain constraints and objective

For each set of event times, all configured short/long orientations, revolution
counts and left/right roots are retained on every leg. Default M=0..2 gives ten
policies per leg. An exact dynamic-programming reduction selects a branch path
using the **pair-dependent encounter costs**. Selecting the minimum-C3
Earth–Mars leg first would discard possible complete missions; this planner does
not do that. Angular momentum z policies `prograde` and `retrograde` are also
available and are defined in the selected inertial frame.

The objective sums every modeled propulsive maneuver: Earth parking injection,
Mars impulses (powered model), Ceres landing and launch, and Earth parking
capture. Both endpoint orbits are circular at 400 km altitude in the example,
with **free plane orientation**. There is no launch-site constraint or free
atmospheric Earth return. C3 and arrival v-infinity are saved separately.

Ballistic Mars encounters require the same body state and epoch on both legs,
matching incoming/outgoing v-infinity speed, the full three-dimensional turn,
and an admissible periapsis. The default speed tolerance is 1e-5 m/s plus
1e-12 relative. Powered encounters allow a **tangential impulse at a common
periapsis**; they are not arbitrary vector burns or deep-space maneuvers. Both
models enforce at least 200 km Mars altitude and at most 300,000 km periapsis
radius. Solar osculating perihelion must be at least 0.2 AU in the example.
Optional launch C3 and per-flyby impulse caps are available.

The Ceres visit is an explicit **spherical, rotating surface stop**, not an
unpriced change in heliocentric velocity. Incoming and outgoing hyperbolas have
periapsis at the surface. A common body-fixed site must satisfy both asymptote
constraints after rotation during the stay. The two possible sites are evaluated
and the lower landing-plus-launch cost retained. Full SPICE state transforms
supply surface rotational velocities; actual signed landing and launch burn
vectors are saved. Collinear site-constraint directions are explicitly rejected
rather than assigned an arbitrary surface site. The adopted spherical radius is
482.1 km. Terrain, body oblateness, landing operations and finite burns are not
modeled. These are substantive assumptions when assessing an actual mission.

Vectorized differential evolution searches the continuous event times.
Constraint penalties guide exploration, but **never establish feasibility**.
An archive retains good starts and other time/branch basins. Fixed-branch SLSQP
refinement imposes speed equalities and turn, surface-site, duration, solar and
budget inequalities. Exports require the native encounter successes and all
configured feasibility checks, even if the optimizer reports convergence.
Optimizer termination is recorded separately from physical feasibility.
Finite searches provide neither global optimality nor an impossibility proof.

## Saved results, checkpoints and independent checks

A run saves its resolved configuration, ordered kernel hashes, native library
hashes, Python source snapshots/hashes and runtime versions. Bounded native
batches reuse allocations. Candidate JSON records exact epochs, endpoint body
states, all leg velocities/policies/statuses/precision flags, local encounter
states, surface site/rotations and the complete maneuver budget. `best.json`
is initially **unvalidated**. `best-infeasible.json`, when present instead, is
only a search diagnostic and is never a replacement trajectory.

Validate and plot a candidate explicitly:

```bash
python -m pip install -r validation/requirements.txt
python run.py mission-validate results/missions/RUN/best.json
python run.py mission-plot results/missions/RUN/best.json
python run.py mission-horizons results/missions/RUN/best.json --refresh
# Repeat the Horizons check offline using the saved public snapshots:
python run.py mission-horizons results/missions/RUN/best.json
```

`mission-validate` uses no native solver calls. It checks each leg against
Gooding's independent algorithm and DOP853 propagation, including revolution
counts, complete timeline and fresh SPICE states. It independently reconstructs
planetocentric asymptotes from local orbital elements, checks flyby tangency and
impulses, endpoint circular parking states, the common rotating Ceres site,
signed burns and every budget item. Acceptance requires no precision warnings,
Gooding agreement within 1e-4 m/s and propagation closure within 500 m and
0.001 m/s (actual example errors are much smaller). Validation JSON contains
all residuals/limits plus candidate/config/validator hashes. The original
candidate remains immutable; an adjacent `*.validation.json` with matching hash
is the acceptance record. Validation failure exits with status 2.

Horizons comparison is an additional ephemeris consistency check, with explicit
centre, frame, timescale, units and body/header verification. Fresh snapshots
are fetched only with `--refresh`; cached requests and response hashes are
verified offline. Endpoint state differences and Gooding re-solves with current
Horizons states are recorded. A comparison requires <=10 m in position,
1e-4 m/s body velocity and 0.001 m/s in recomputed leg velocities. It does not
replace the local conic check or certify exact ballistic equalities after an
ephemeris update. See the [official API reference](https://ssd-api.jpl.nasa.gov/doc/horizons.html).

Checkpoints contain population checksum, RNG state, current seed/generation and
the complete archive. A file lock prevents simultaneous writes. To continue an
interrupted run, supply the **original effective config** and its directory:

```bash
python run.py mission-search --config results/missions/RUN/config.json \
    --resume results/missions/RUN
```

Resume rejects changed configuration, code, native libraries, kernels or
recorded runtime versions and corrupt population files. It is a **warm
continuation**: SciPy re-evaluates the initial population, so it is not promised
to reproduce an uninterrupted run bit for bit. A completed run returns its
existing summary. Refinement is restarted if interrupted before completion.

For a fixed-date complete-chain check, supply launch offset from the config's
window start, the four leg durations, then surface-stay durations:

```bash
python run.py mission-evaluate --config examples/missions/emc-ten-year.json \
    --days 546 200 250 450 125 30 --output results/paper-diagnostic.json
```

The command labels an infeasible chain explicitly. A collection of individually
solved legs is not evidence of a feasible mission.

## Research interpretation

A `passed_patched_conic` trajectory is an independently checked **initial
trajectory under the declared impulsive model**. It is not a flight-ready
N-body solution. Finite encounter timing at sphere-of-influence crossings,
planetary/solar perturbations, B-plane targeting with finite encounters,
finite-thrust dynamics, launch/return geometry, mass/propulsion and operational
constraints require subsequent correction and validation. Large maneuver
budgets can make a mathematically feasible trajectory impractical.

The [N-body correction foundation](nbody-correction.md) now supplies a C++
Newtonian point-mass propagator and independently validated local finite Mars
encounter corrector. Full-mission reattachment and higher-order force and
engineering models remain subsequent work.

The existing paper itinerary audit and this replacement search answer different
questions. A new feasible trajectory cannot retroactively validate the paper's
published dates, and an unsuccessful finite replacement search cannot rule out
the architecture. The evidence bundle records both successful and unsuccessful
searches without making either inference.

## Verification

```bash
python -m unittest discover -s mission_tests -v
LAMBERT_TEST_SPICE=1 LAMBERT_TEST_CUDA=1 \
    python -m unittest discover -s mission_tests -v
```

Contracts include branch reduction versus exhaustive enumeration, a case where
minimum launch cost selects the wrong mission, independent local hyperbolas,
rotating-site geometry and signed burns, strict config rejection, native
CPU/CUDA chain parity, SPICE ownership, completed/partial checkpoint continuation,
corruption/config/source rejection, independent export acceptance and a
modified-site rejection. Local-kernel and GPU tests require explicit opt-in.
