# Research CLI: first operational release

The interactive runner adapts the familiar Simple-Lambert-Solver CLI into this
repository and calls the validated custom C++/CUDA engine. It supplies a complete
heliocentric porkchop workflow: explicit SPICE kernels, persistent state caches,
chunked native batches, candidate reduction, numerical output, minimum-C3 tables,
and figures. The [flyby encounter engine](flyby-solver.md) adds a separate saved
batch workflow for unpowered and tangential powered encounters.

## Launch in this workspace

From the Lambert-Solver checkout, the already prepared task-local environment is:

```bash
source ../../work/tof-validation-env/bin/activate
python run.py
```

This opens the menu:

1. Two Body Grid (Porkchop Plot)
2. Run Saved Configuration
3. Run EMC Paper Presets
4. Regenerate Saved Plot
5. View Saved Configurations
6. Exit
7. Flyby Encounters

Choose a backend explicitly. CUDA failures do not fall back to CPU. The CPU
backend is the shared native C++ engine, not PyKEP or the Python reference.
The menu and `--help` load no ephemerides and do not initialize a GPU.

## Setup in a fresh checkout

Use a recent Python (the validated run used Python 3.12), a C++17 compiler, and a
CUDA toolkit if using the GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r research-requirements.txt
python tools/fetch_research_kernels.py
python native/build.py --cuda-root /path/to/cuda --arch 120
python run.py
```

Use `python native/build.py` for the CPU library only, then select `cpu` in the
configuration or command line. Set `--arch` to your GPU's compute capability.
The local tested GPU is an RTX PRO 2000 Blackwell Generation Laptop GPU.
No CUDA toolkit or ephemerides are downloaded implicitly during a run.

The explicit kernel downloader checks a pinned URL, byte count and SHA-256 for
all four files in `configs/kernel-sources.json`. Existing differing kernels are
preserved, with an error asking for another directory. The manifest includes
JPL DE440s and the archived Ceres, GM and time kernels from the earlier repository.
The Ceres snapshot covers 2025-10-07 through 2045-11-06 TDB. Requesting states
outside available coverage fails before any Lambert grid is computed.

The paper presets use **Mars barycenter (4)**, which DE440s supplies. They do not
silently substitute it for Mars body center (499). Add an appropriate Mars
satellite kernel and explicitly choose 499 if body-center states are required.
Both frames J2000 and ECLIPJ2000 are supported; all supplied states are geometric
relative to the Sun with aberration correction NONE.

## Commands and configurations

```bash
python run.py transfer-grid --config configs/smoke-earth-mars.json
python run.py transfer-grid --config configs/smoke-multirev.json
python run.py transfer-grid --config configs/emc-earth-mars.json --dry-run
python run.py transfer-grid --config configs/emc-earth-mars.json --backend cpu
python run.py transfer-grid --config configs/emc-earth-mars.json \
    --orientation both --max-revolutions 2 --outdir results/multi
python run.py plot results/RUN/grid.npz --output results/replotted.png
python run.py configs
```

One command runs the complete four-preset set:

```bash
python run.py batch \
    --config configs/emc-earth-mars.json \
    --config configs/emc-mars-ceres.json \
    --config configs/emc-ceres-mars.json \
    --config configs/emc-mars-earth.json \
    --outdir results/emc
```

The batch prevalidates every configuration, runs sequentially using persistent
cache files, and saves a uniquely named batch summary. Earlier completed runs
are retained if a later run fails. A summary is written only on completion.
Each run lives in its own timestamp/UUID directory and preserves its effective
configuration. The script never replaces an existing configuration implicitly.

New JSON specifications use the fields of `GridConfig` in
`src/research/config.py`. Kernel paths in JSON are relative to the configuration
file. Command-line `--kernel PATH`/`--kernels PATH` options may be repeated;
these override the kernel list and are relative to the shell's working directory.
The legacy `simulation_type: two_body_grid` plus `parameters` JSON format is
accepted, including string-valued numeric settings and `dep_id`/`arr_id`. Supply
kernel paths explicitly when an old configuration does not contain them. Legacy
flyby/chain configurations are rejected. The new `flyby` command has an explicit
versioned encounter schema; it does not interpret old chain-search settings.

Flight time limits and departure/flight steps remain fractional; no conversion
to integer days is applied. All calendar input is explicitly **TDB**, without time
zones. One flight day and departure step day mean 86400 dynamical seconds.
Bounds are inclusive when aligned to the requested step. Otherwise the last point
is the largest grid point below the upper bound. `--dry-run` shows actual counts.

Initial safeguards limit grids to 10 million cells and approximately 2048 MiB of
planned host result storage; `--max-cells` and `--max-memory-mb` explicitly change
those budgets. The host memory estimate is approximate, not a measured peak.
Native candidate batches are bounded by `--batch-size` (default 65536).
This version retains the full final grid and per-candidate status arrays on the
host; it does not provide out-of-core billion-cell grids or resumable execution.

## Orientations, revolution roots and reduction

`orientation` accepts:

- `prograde` or `retrograde`: sign of orbital angular momentum's z component in
  the configured inertial frame. These map per cell to the correct short/long arc.
- `short` or `long`: geometric transfer angle below/above pi.
- `both`: evaluate both short and long orientations, reducing by departure C3.

An exactly vertical orbit plane has no positive/negative z direction. The
fixed-axis prograde/retrograde policies mark that candidate undefined, while an
explicit short/long selection remains available if the geometry is noncollinear.

For every permitted orientation, `max_revolutions=N` searches M=0 and both roots
of each M=1..N. The application limit is N<=100; the lower-level native engine
retains its wider signed-int32 contract. Left/right are relative to the minimum
of T(x), independent of the sign of x, prograde/retrograde or short/long.
M=0 has no left/right multiplicity; its saved branch field is -1.

The reducer selects the successful finite velocity solution having the lowest
**departure C3**. It does not minimize total delta-V or impose an arrival capture
constraint. Arrival v-infinity is saved separately. Ties keep the first candidate
in the documented deterministic order: increasing M, left before right, short
before long for `both`. Precision warnings are preserved on every candidate and
on the winner. A warned solution is not silently discarded or repaired.

Candidate status records contain native top-level, geometry, root and minimum
statuses plus the precision tri-state. Application status 4 means undefined
fixed-axis orientation; status 5 means invalid/nonfinite C3 or v-infinity after
native solving. Both retain their nested native diagnostics. A cell with no
eligible candidate keeps C3/velocities NaN and revolution/branch selections -1.
No zero-energy fallback, interpolation or neighbor-based spike replacement occurs.

## Outputs and provenance

Every successful workflow writes:

- `grid.npz`: complete C3 grid, departure/arrival v-infinity vectors, selected M,
  branch, long-way flag, residual, iteration count, estimated x error, precision
  flag, and status/precision arrays for every attempted candidate. It also retains
  the exact input departure and unique arrival states with their cell mapping.
- `minimum.csv`: the true sampled minimum over the searched flight-time range
  for every departure date, with dates, TOF, C3, M, branch and precision flag.
  Missing minima remain blank, not zero.
- `config.json`: effective settings with resolved kernel paths.
- `run.json`: units, frame, time convention, kernel hashes/order, solar GM,
  native library hash, workflow source hashes, Git revision/dirty state,
  Python/NumPy/SPICE versions, solver tolerances, device information, counts,
  warnings, global minimum, data checksums and separated elapsed timings.
- `porkchop.png`, unless plotting is disabled: C3 color scale, arrival/departure
  dates, labelled flight-time contours and sampled minima. Minima are plotted
  as points so jumps between distinct valleys do not suggest intermediate arcs.

Grid arrays are indexed `[departure, flight_time]`; candidate status arrays use
`[candidate, departure, flight_time]`. Manifest `policies` gives each candidate
as `[M, branch, orientation]`. The compact NPZ `candidates` table encodes
short=0, long=1, prograde=2 and retrograde=3 in its third column.

Native inputs, saved states and v-infinity use **metres and metres/second**.
C3 is converted once to **km^2/s^2** for plotting and reporting. The exact unit
contract is recorded in every manifest. Raw C3 outside the display range remains
in the data; masking changes only the figure. The numerical data is saved before
plotting; if plotting fails the manifest retains the error and the grid is usable.

A saved plot can be regenerated without SPICE or CUDA. The reader verifies the
saved grid's SHA-256 before plotting. Changing native outputs requires a new run.
State caches under `.research-cache` are keyed by exact epochs, body, frame,
ordered kernel content/path hashes and SPICE versions; their own data checksums
are checked before reuse. SPICE requires a clean process, rejects inherited kernel
ordering and unloads only kernels it loaded. It never clears another caller's pool.

## Validated examples and limits

All four daily paper-window presets completed with 3,245,541/3,245,541 valid cells.
Observed batch wall times were about 3.5 seconds with cached states and 3.7
seconds with fresh state-vector queries on this workstation, including
local state preparation, native computation, data/CSV writes and figure rendering.
This used **prograde M=0**, local kernel files, 65536-candidate batches and OS-cached
kernel files. It is a single observed run, not a cold-download or hardware-neutral
benchmark. The batch timer excludes process startup/config parsing and kernel
installation/builds. GPU math itself is a small part of the workflow.

The presets follow the paper's plotted axes/titles where captions disagree, and
use current explicitly pinned ephemerides. They are not asserted to be bitwise or
pixel-identical to the publication's undocumented original data/settings.
Figure 6 appears to reuse Figure 9's grid; Figure 12 is treated as Mars-Earth
according to its chart title. Additional M or orientations increase the workload.

Independent Gooding spot checks include both ends, the global minimum and 48
fixed-seed random cells per paper grid. Selected real M=1 solutions are separately
checked in the multi-revolution smoke run. Scientific checks compare minimum C3
across the entire allowed candidate set, both endpoint velocities and v-infinity
consistency. Existing solver validation remains in force; spot checks do not
certify every cell. Raw verification summaries are in
`validation/results/research-cli-*`.

```bash
LAMBERT_TEST_CUDA=1 LAMBERT_TEST_KERNEL_DIR=data/kernels \
    python -m unittest discover -s research_tests -v
python validation/validate_research.py results/RUN/grid.npz --samples 48
python -m unittest discover -s tests -v
LAMBERT_TEST_CUDA=1 python -m unittest discover -s native/tests -v
```

Gooding/lamberthub is a validation dependency only. There is no external Lambert
solver, mock ephemeris or CPU fallback in the GPU product path.

SPICE API references: [str2et](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/str2et_c.html),
[spkezr](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkezr_c.html).
