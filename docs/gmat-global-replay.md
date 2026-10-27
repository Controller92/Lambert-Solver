# Complete-candidate GMAT replay

GMAT R2026a was already available locally at
`../../work/nbody/GMAT/R2026a/bin/GmatConsole`. The saved ballistic and powered
global N-body candidates were replayed in it on 2026-10-27. Both GMAT mission
sequences completed. **Neither passed the external numerical comparison.** Do
not present these candidates as independently GMAT-validated trajectories.

The compact evidence is in [`validation/results/gmat-global`](../validation/results/gmat-global/summary.json):
the generated GMAT scripts, console logs, raw event and midpoint state reports,
comparison JSON, summary, and a [residual plot](../validation/results/gmat-global/gmat-replay-residuals.png).
The full local source candidates and run directories are deliberately outside
Git because they contain large ephemeris and trajectory artifacts.

| Checkpoint | Ballistic position difference | Powered position difference |
| --- | ---: | ---: |
| First outbound cruise cut | 3.17 km | 27.1 km |
| First Mars encounter, before burn | 5.75 km | 126 km |
| Ceres arrival | 335,194 km | 8,748,024 km |
| Second Mars encounter, before burn | 22.4 km | 226 km |
| Earth return | 96,734 km | 1,042,611 km |

These are differences between GMAT's propagated states and the saved candidate
states, not errors relative to the real mission or estimates of orbit uncertainty.
The prescribed Ceres surface stay and launch separate two continuous flight
sequences. The Ceres departure is initialized from its saved state; the zero
departure residual therefore does not validate the arrival-to-departure link.
Within each flight sequence, no target encounter state is inserted. The
powered impulses use the saved velocity jumps at Mars; the ballistic case has
none. The 200 m and 0.0003 m/s event comparison limits are exceeded well before
Ceres. The comparison also records a maximum GMAT-versus-source epoch offset of
0.0017 s for ballistic and 0.0033 s for powered; these are separate from the
large spatial residuals.

## Model and interpretation

The generated GMAT scripts use ICRF Cartesian axes, the candidates' pinned
DE440s, Mars and Ceres SPKs, fourteen point masses and gravitational parameters,
and an explicit degree-2 Mars zonal field. GMAT's force-model integration
origin switches at the saved half-leg shooting cuts: Earth, Mars, Mars, Ceres
outbound and Ceres, Mars, Mars, Earth inbound. Mars J2 is applied on the
Mars-centered halves. It is omitted from distant Earth- and Ceres-centered
cruise halves. This small declared model difference and GMAT's planet
orientation/time/frame and integration-origin conventions need further
reconciliation with the native SSB acceleration and interpolated Mars pole.
The present evidence does **not** identify a single proven cause of the
long-arc divergence.

The original N-body correction still has passing native shooting and
independently coded direct-SPICE/DOP853 checks within its specified force model.
This GMAT replay is a genuinely separate and stricter cross-tool check, and it
currently fails. Neither result establishes flight readiness, a globally
optimal itinerary, or a conclusion about the EMC paper.

To rerun from the repository root with the saved local candidates and pinned
kernels still present:

```bash
../../work/tof-validation-env/bin/python tools/check_gmat_global.py \
  ../Nbody-mission-runs/final/global-ballistic-d75fdc7d0336 \
  --gmat-console ../../work/nbody/GMAT/R2026a/bin/GmatConsole \
  --outdir ../Nbody-mission-runs/gmat-global/ballistic-rerun
../../work/tof-validation-env/bin/python tools/check_gmat_global.py \
  ../Nbody-mission-runs/final/global-powered-56174e576ba4 \
  --gmat-console ../../work/nbody/GMAT/R2026a/bin/GmatConsole \
  --outdir ../Nbody-mission-runs/gmat-global/powered-rerun
```

Exit status 2 is expected while comparison limits fail. Each run checks source
and kernel hashes before writing an output directory. The generated script and
console reports show whether GMAT itself completed. Address the first outbound
cut first: compare single-step accelerations in a common inertial frame and
time scale, then propagate increasingly long no-burn arcs from identical
Cartesian states. Only after force agreement should the full chain be
re-corrected in the reconciled model and replayed. Surface operations, finite
burns, uncertainty and higher-fidelity gravity remain separate later work.
