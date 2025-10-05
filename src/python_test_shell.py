# python_test_shell.py
import numpy as np
import os
import cupy as cp  # type: ignore
# import spiceypy as spice  # Commented out - now using Skyfield interface
import skyfield_interface as spice  # Use Skyfield interface with same API
import spice_interface
from trajectory import lambert_izzo_gpu
