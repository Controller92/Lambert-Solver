# Planned flyby upgrade

The existing Simple-Lambert-Solver Flyby-Solver branch remains the reference for
the desired interface and mission workflows. Its numerical claims and defaults
must be checked independently before adoption. No legacy flyby calculation has
been promoted into the current research runtime by this increment.

The intended sequence follows the Lambert work:

1. Define a precise encounter contract: incoming/outgoing v-infinity vectors,
   body GM and reference radius, periapsis constraints, frame/origin, units,
   B-plane convention, powered/unpowered behavior and explicit failures.
2. Build an isolated Python reference. Validate against independent mathematics,
   conserved quantities, limiting geometries, reversal/rotation/scaling checks,
   and a carefully selected real mission encounter with host-side ephemerides.
3. Port the validated encounter calculation to a shared C++ CPU/CUDA numerical
   core. Preserve diagnostics and physical feasibility checks before tuning.
4. Batch independent encounters/candidates on the GPU and measure accuracy,
   divergence, memory traffic, kernel time and full workflow time.
5. Connect validated flybys to the already working Lambert grid runner, enforcing
   consistent encounter epochs, bodies, frames and velocity matching across legs.
   Preserve all relevant Lambert branches until flyby feasibility is checked;
   a minimum-C3 winner on each individual leg need not form a feasible chain.
6. Add tiled mission searches, resumable output, bounded candidate storage and
   independent full-chain validation before restoring the old three-body menus.

The first operational research CLI currently offers completed porkchop/config/
plot workflows. It does not expose old flyby/chain options as if they were already
validated custom-engine features. The native Lambert engine remains reusable by
both the future encounter kernel and its Python reference.
