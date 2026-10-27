# Lamberts-Problem Documentation

The [research CLI](docs/research-cli.md) now connects the validated native engine
to SPICE ephemerides, interactive prompts, saved JSON configurations, bounded GPU
batches, C3 grids, minimum tables and porkchop figures. Launch `python run.py` or
use `python run.py --help`. The interface adapts Simple-Lambert-Solver's
Flyby-Solver branch; no PyKEP dependency is needed for the runtime. Four EMC
paper-window presets and a real multi-revolution smoke preset are included.
The [bounded EMC survey](docs/emc-sweep.md) adds four-leg porkchops, complete
branch-chain sampling, separate Mars flyby atlases and CPU/GPU stress measurements
through `python run.py mission-sweep`.
The [custom flyby encounter engine](docs/flyby-solver.md) now supplies unpowered
propagation, inverse matching and tangential powered encounters in Python, C++
and CUDA, with saved CLI batches and a Juno Earth-flyby validation. Run
`python run.py flyby --config examples/flyby-earth.json --backend cuda`.
The [complete mission planner](docs/mission-planner.md) now searches consistent
Earth–Mars–Ceres–Mars–Earth chains with a rotating Ceres surface stay, separate
ballistic/powered encounters, all configured Lambert branches, checkpoints and
independent trajectory validation. Run `python run.py mission-search --config
examples/missions/emc-ten-year.json --model both` or choose menu option 8.
The [N-body correction foundation](docs/nbody-correction.md) adds an FP64 C++
propagator, state-transition matrices and finite Mars encounter correction,
independently checked with direct-SPICE DOP853 and NASA GMAT RK89. Run
`python run.py nbody-flyby /path/to/validated-mission/best.json`.
Local corrected encounters still require full-mission reattachment and
higher-order force/engineering models before becoming research replacements.
The [EMC trajectory audit](docs/emc-trajectory-audit.md) checks the manuscript’s
nominal itinerary with body-centred ephemerides, full encounter vectors,
independent Gooding/Horizons comparisons, and a bounded date search.

The zero- and multi-revolution native engine runs in **CUDA C++**, with a shared C++ CPU path and
NumPy batch bindings in `src/lambert_native.py`. See the
[native engine walkthrough](docs/seventh-increment.md) for build instructions,
API examples, independent accuracy checks and measured GPU performance.
The [first optimization pass](docs/eighth-increment.md) reduces representative
search kernel time by 17.9% and includes a reproducible baseline comparison.
The Python mathematical implementation remains the independently validated
reference. The [native multi-revolution port](docs/eleventh-increment.md) adds explicit
left/right branch rows, minimum diagnostics and complete velocities on the GPU.
The optional [terminal-velocity Jacobian](docs/lambert-jacobian.md) adds a
6-by-7 sensitivity matrix on C++ CPU and CUDA, with separate conditioning
statuses and independent multiprecision/variational-equation validation.
Legacy GUI integration remains subsequent work.

The [multi-revolution preparation increment](docs/ninth-increment.md) adds Python
TOF/derivatives, minimum-time detection and explicit feasibility diagnostics.
It includes a cached Horizons Juno case around Jupiter. The
[complete Python multi-revolution engine](docs/tenth-increment.md) now supplies
explicit left/right roots and velocities through `solve_multi_rev` and
`solve_multi_rev_pair` in `src/lambert_multirev_solver.py`.

The zero-revolution CPU entry point is `src.lambert_solver.solve_zero_rev`.
It returns both endpoint velocities, explicit statuses and root diagnostics.
The [sixth increment](docs/sixth-increment.md) fixes tiny-arc precision through
the complete calculation and validates arbitrary 3D planes. The
[Mars 2020 comparison](docs/mars2020-validation.md) exercises cached JPL Horizons
states. The custom mathematical runtime uses only Python's standard library.
The native port is documented in the seventh increment.

The isolated CPU geometry reference and explicit input contract are in
`src/lambert_geometry.py`. See [the first increment and branch audit](docs/first-increment.md)
for conventions, validation, known legacy defects, and the next mathematical step.
Run its standard-library tests with `python3 -m unittest discover -s tests -v`.
This helper does not yet solve Lambert's time-of-flight equation.

The next isolated layer, `src/lambert_tof.py`, evaluates zero-revolution `T(x, lambda)`
for a supplied `x`, including the parabolic limit and stable near-parabolic series.
See [the TOF increment](docs/second-increment.md) for its equations, analytical
tests and reproducible independent high-precision validation. It does not find a root.

`src/lambert_root_preparation.py` adds the first three TOF derivatives and a
zero-revolution initial guess. See [the third increment](docs/third-increment.md)
for the parabolic limits, corrected starter exponent, measured errors, and
limitations. These preparation routines produce no roots or velocities.

`src/lambert_root.py` now provides safeguarded zero-revolution root inversion,
with residuals, iteration counts and conditioning diagnostics. See
[the fourth increment](docs/fourth-increment.md) for validation and limitations.

`src/lambert_solver.py` connects those layers and reconstructs departure and
arrival velocities for zero-revolution transfers. See [the fifth increment](docs/fifth-increment.md)
for the API, analytical/Gooding/propagation validation and precision warnings.
This isolated CPU solver is separate from the legacy GUI. Retained `q=c/s`
preserves tiny-arc shape information; inspect the root conditioning diagnostics.
The dedicated multi-revolution Python entry points are documented above.

## Interplanetary Trajectory Calculator

This project is an application designed to calculate interplanetary trajectories using Lambert's problem. It provides a graphical user interface (GUI) for users to input various parameters related to their trajectory calculations.

### Features

- Input fields for start and end dates, transit times, and orbital elements.
- A dialog box for searching celestial bodies by name or partial name, which automatically fills in the orbital elements for both initial and final orbits.
- Visualization of trajectories and Δv (delta-v) calculations.
- Progress tracking for long calculations.

### File Structure

- `src/gui.py`: Contains the main GUI for the application, including input fields and the celestial body search dialog.
- `src/trajectory.py`: Implements functions for trajectory calculations and estimating transit times.
- `src/spice_interface.py`: Interfaces with the SPICE kernel to retrieve orbital elements for celestial bodies.
- `requirements.txt`: Lists the dependencies required for the project.

### Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Lamberts-Problem
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/gui.py
   ```

### Usage Guidelines

- Input the start and end dates in the format YYYY-MM-DD.
- Specify the minimum and maximum transit times in days.
- Use the search dialog to find celestial bodies and automatically fill in their orbital elements.
- Click on the "Estimate Time" button to get an estimate for the calculations.
- Visualize the trajectory by clicking on the "Show Animation" button.

### Dependencies

- `tkinter`: For GUI development.
- `matplotlib`: For plotting and visualizations.
- `astropy`: For handling time and astronomical calculations.
- `spiceypy`: For interfacing with the SPICE kernel.

### Acknowledgments

This project utilizes the SPICE toolkit for accurate orbital data and trajectory calculations. Special thanks to the developers and maintainers of the SPICE toolkit.

### Joint finite N-body mission correction

`python run.py nbody-mission /path/to/validated/best.json --outdir results/nbody`
corrects the complete EMC chain with shared finite endpoints, fourteen SPICE
point masses and oriented Mars J2. It independently checks continuity, clearances,
surface-site consistency, timelines, impulse accounting and numerical convergence.
See [model assumptions and acceptance limits](docs/nbody-mission.md). These are
feasibility corrections of known basins; spacecraft engineering and independent
whole-mission GMAT validation remain future work.
