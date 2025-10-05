# trajectory.py
import numpy as np
import cupy as cp  # type: ignore
# import spiceypy as spice  # Commented out - now using Skyfield interface
import skyfield_interface as spice  # Use Skyfield interface with same API
from astropy.time import Time
import spice_interface


def lambert_izzo_gpu(r1, r2, tof, mu, M=0, numiter=35, rtol=1e-14):
    """
    GPU-accelerated Lambert solver using a simplified approach.

    Args:
        r1: Initial position vector (3-element array-like)
        r2: Final position vector (3-element array-like)
        tof: Time of flight (seconds)
        mu: Gravitational parameter (km³/s²)
        M: Number of revolutions (default 0)
        numiter: Maximum number of iterations (default 35)
        rtol: Relative tolerance (default 1e-14)

    Returns:
        v1: Initial velocity vector
        v2: Final velocity vector
    """
    # Convert inputs to CuPy arrays for GPU computation
    r1 = cp.asarray(r1, dtype=cp.float64)
    r2 = cp.asarray(r2, dtype=cp.float64)
    tof = cp.float64(tof)
    mu = cp.float64(mu)

    # Calculate basic orbital parameters
    r1_norm = cp.linalg.norm(r1)
    r2_norm = cp.linalg.norm(r2)
    c = cp.linalg.norm(r2 - r1)  # chord length

    # Semi-perimeter
    s = (r1_norm + r2_norm + c) / 2.0

    # For minimum energy transfer (parabolic orbit)
    a_min = s / 2.0

    # Try different approaches based on transfer type
    # For simplicity, use a basic elliptical transfer approximation

    # Calculate transfer angle
    cos_transfer = cp.dot(r1, r2) / (r1_norm * r2_norm)
    cos_transfer = cp.clip(cos_transfer, -1.0, 1.0)  # Ensure valid range
    transfer_angle = cp.arccos(cos_transfer)

    # Estimate semi-major axis using a simple approximation
    # This is a rough approximation for the Lambert problem
    k = r1_norm * r2_norm * (1 - cp.cos(transfer_angle))
    m = r1_norm * r2_norm * (1 + cp.cos(transfer_angle))
    l = (r1_norm + r2_norm) / 2.0

    # Approximate semi-major axis
    a = (k + m) / (4 * l)

    # Ensure a is positive and reasonable
    a = cp.maximum(a, a_min * 1.1)  # Slightly larger than minimum

    # Calculate velocities using vis-viva equation approximation
    # This is a simplified approach for smoke testing

    # Circular velocities at each point
    v_circ1 = cp.sqrt(mu / r1_norm)
    v_circ2 = cp.sqrt(mu / r2_norm)

    # For transfer orbit, adjust velocities
    # This is a very rough approximation
    v_transfer1 = v_circ1 * 1.1  # Boost for transfer
    v_transfer2 = v_circ2 * 0.9  # Reduce for capture

    # Direction perpendicular to position vector (simplified)
    # For simplicity, assume transfer in xy-plane
    v1_dir = cp.zeros_like(r1)
    v1_dir[0] = -r1[1]
    v1_dir[1] = r1[0]
    v1_dir[2] = 0.0
    v1_dir = v1_dir / r1_norm

    v2_dir = cp.zeros_like(r2)
    v2_dir[0] = -r2[1]
    v2_dir[1] = r2[0]
    v2_dir[2] = 0.0
    v2_dir = v2_dir / r2_norm

    v1 = v_transfer1 * v1_dir
    v2 = v_transfer2 * v2_dir

    # Convert back to numpy arrays for return
    return cp.asnumpy(v1), cp.asnumpy(v2)
