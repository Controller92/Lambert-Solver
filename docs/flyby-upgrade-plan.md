# Flyby upgrade status

The [custom encounter engine](flyby-solver.md) completes steps 1–4 of the original
plan: explicit SI/frame/B-plane/model contract, independent Python reference,
shared C++/CUDA implementation, actual GPU validation and measured batch timings.
Its CLI supports unpowered propagation, inverse matching and a tangential
common-periapsis powered model. Juno's 2013 Earth encounter provides a cached
Horizons model check. See the [increment report](thirteenth-increment.md).

Next is the full Lambert/flyby chain search:

1. Preserve every relevant M/left/right/orientation velocity candidate at an
   encounter. Per-leg minimum-C3 reduction must not precede flyby feasibility.
2. Enforce identical body, epoch, inertial frame and unit conventions at each
   shared endpoint; compute both relative velocities from one body state.
3. Enumerate/prune encounter pairs using explicit ballistic or powered policy,
   altitude bounds, numerical statuses and conditioning flags.
4. Add tiled mission searches, bounded candidate storage, resume/checkpoint
   output and independently validated complete mission trajectories.

The former Simple-Lambert-Solver menu remains an interface reference; its
numerical helpers and old chain settings are not treated as validated runtime.
