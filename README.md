# Lamberts-Problem Documentation

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