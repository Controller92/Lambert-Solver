# Bounded EMC trajectory survey

```sh
python run.py mission-sweep --config examples/missions/emc-ten-year.json \
  --outdir results --seconds 900
```

The default run samples Earth–Mars–Ceres–Mars–Earth missions in the existing
2034–2038 launch window, with a rotating Ceres surface stay and a 10-year
duration cap. It preserves both transfer directions and all configured
multi-revolution branches. CPU ephemerides and Ceres frame transformations
remain on the host; Lambert and flyby batches use the custom CUDA engines.
The mathematical kernels are unchanged by this survey wrapper.

The time budget covers computation, figures and selected independent checks.
Stages check the remaining time between blocks. An individual block or final
validation may finish slightly after the requested budget. Dense independent
grids run first; use a larger `--grid-step` for a shorter preflight. Output is
bounded by one million sampled mission tuples and an 8 GiB chunk-storage limit.

## What is searched and saved

* Four independent departure/flight-time grids, normally at two-day spacing.
  Each grid reports minimum **departure C3** across configured branches. These
  minima are for visualization only; they are not preselected for mission
  joining. The grids do not enforce the full mission's flyby, surface, duration
  and solar-clearance constraints.
* A reproducible scrambled Sobol sample over launch, all four durations and the
  Ceres stay. Tuples over ten years are rejected before solving. Every leg
  branch and both Mars branch-pair matrices are evaluated. Complete chains use
  pair-dependent dynamic programming, including the rotating common surface
  site, Earth parking maneuvers and charged powered-flyby impulses.
* Separate **ballistic** and **powered** Mars encounter sets. Local acceptance
  requires native flyby success and both adjacent successful, solar-safe legs.
  A valid local Earth–Mars–Ceres or Ceres–Mars–Earth connection is not a complete
  mission, and none is automatically an N-body solution.
* Strict ballistic matching is not replaced by a loose speed tolerance. Near
  matches seed additional fixed-branch one-dimensional searches for equal
  incoming/outgoing v-infinity magnitudes. Bracketed roots are re-solved and
  must pass the normal native ballistic and solar-clearance checks. Finite
  bracket sampling is not an exhaustive search and can miss tangencies.
* Previously validated mission tuples, when present in the local workspace,
  are explicitly included as anchors. Counts include these anchors; retaining
  one does not constitute a new trajectory discovery.

Each compressed `chunks/*.npz` file contains the exact sampled decisions,
epochs, geometric heliocentric SPICE states, both velocities for every leg
branch, statuses, root precision diagnostics, and solar-clearance margins.
For each accepted local connection, `powered_1_indices` (or the equivalent
model/node name) contains `[row,incoming_policy,outgoing_policy]` indices into
those arrays. Periapsis radius, burn and flyby metrics are stored alongside
the indices. `policies.json` maps policy indices to direction, M and branch.
Ballistic refinements have individual `ballistic-roots/*/connection.json` and
`data.npz` files with their explicit fixed policies. All locally accepted
connections are saved; the highlight reservoir does not replace the dataset.

## Meaning of the figures

`four-leg-porkchops.png` shows the independent departure-C3 grids. Values
outside each panel's 1st–90th percentile color limits are saturated, not
discarded from the saved grids.

`*-all-trajectory-coverage.png` accumulates both adjacent conic legs of every
accepted sampled connection, drawing each participating leg once per sampled
tuple and branch. Conic loci are sampled at 64 uniform **true anomalies**, not
uniform times; intensity describes geometric coverage, not residence time or
probability. The histogram is saved at ±8 AU in both coordinates. X–Z figures
zoom to ±4 AU vertically, and counts beyond the stored bounds are reported.

`*-plane-change-speed-gain.png` includes a count map of all accepted
connections and a bounded unbiased highlight scatter. Metrics distinguish:

* orbital-plane rotation: the angle between heliocentric angular momenta
  before and after the Mars encounter;
* signed inclination change relative to the ecliptic reference plane;
* planet-relative asymptote turn angle;
* heliocentric speed and specific kinetic-energy changes;
* planet-relative v-infinity speed change (approximately zero for ballistic);
* pre/post-flyby angular misalignment with Ceres's orbital plane;
* periapsis altitude and the explicitly charged powered periapsis burn.

The powered plot's heliocentric gain includes the impulse. It must not be
interpreted as a purely gravitational gain. Plane rotation is a full 3D angle,
and is distinct from an inclination difference or a v-infinity turn.

`*-3d-highlights.png` draws up to 50 sampled local connections per direction.
It is a readable subset; the coverage maps and numerical records cover every
accepted connection. These are patched-conic heliocentric paths joined at a
point encounter, rather than resolved Mars-centred hyperbolas or N-body arcs.

## Resource measurements and limitations

The same actual mission Lambert rows are benchmarked on 1, 2, 4, 8 and all
available CPU threads and on the GPU. Native calls release Python's GIL;
worker threads therefore execute independent C++ loops concurrently. Status
and velocity comparisons check backend agreement. Failed/infeasible branch
attempts are included in rates and explicitly labelled. CPU values are wall
times; standalone resident GPU event times exclude transfers/allocation.

A further bounded stress phase runs all CPU workers and resident GPU launches
concurrently; its throughput is reported separately because CPU contention,
shared laptop power and thermal limits can change the results. Native flyby
CPU/GPU timing is also recorded. `cpu-monitor.csv` records process core usage,
system busy percentage and RSS. `gpu-monitor.csv` records utilization, memory,
power and temperature when NVIDIA monitoring is available. There are no changes
to power limits, clocks or system settings.

`summary.json`, `timings.json`, source copies, kernel/library/source hashes,
and selected independently validated complete-chain exports accompany the
plots. A best sampled candidate can simply reproduce an anchor. This finite
survey supplies neither a global optimum nor an impossibility certificate.
Improved research replacements still require full N-body correction and the
separate acceptance checks used by the mission corrector.
