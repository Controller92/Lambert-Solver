# Lamberts-Problem Documentation

The [research CLI](docs/research-cli.md) now connects the validated native engine
to SPICE ephemerides, interactive prompts, saved JSON configurations, bounded GPU
batches, C3 grids, minimum tables and porkchop figures. Launch `python run.py` or
use `python run.py --help`. The interface adapts Simple-Lambert-Solver's
Flyby-Solver branch; no PyKEP dependency is needed for the runtime. Four EMC
paper-window presets and a real multi-revolution smoke preset are included.
The [custom flyby encounter engine](docs/flyby-solver.md) now supplies unpowered
propagation, inverse matching and tangential powered encounters in Python, C++
and CUDA, with saved CLI batches and a Juno Earth-flyby validation. Run
`python run.py flyby --config examples/flyby-earth.json --backend cuda`.
Full multi-leg mission search remains a subsequent increment.

The zero- and multi-revolution native engine runs in **CUDA C++**, with a shared C++ CPU path and
NumPy batch bindings in `src/lambert_native.py`. See the
[native engine walkthrough](docs/seventh-increment.md) for build instructions,
API examples, independent accuracy checks and measured GPU performance.
The [first optimization pass](docs/eighth-increment.md) reduces representative
search kernel time by 17.9% and includes a reproducible baseline comparison.
The Python mathematical implementation remains the independently validated
reference. The [native multi-revolution port](docs/eleventh-increment.md) adds explicit
left/right branch rows, minimum diagnostics and complete velocities on the GPU.
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
