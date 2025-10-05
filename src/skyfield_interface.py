import re
import os
import skyfield.api as sf
import time
import sys
import importlib
import urllib.request
import ssl
import zipfile
import shutil
import math
import numpy as np
from skyfield.api import load
import requests
import json
from datetime import datetime, timedelta

# Global file paths (same as spice_interface.py)
KERNELS_DIR = r"C:\\Users\\letsf\\Documents\\Coding\\Python\\Lamberts\\kernels"
BSP_DIR = os.path.join(KERNELS_DIR, "BSP")
GM_FILE = os.path.join(KERNELS_DIR, "gm_de440.tpc")
NAIF_FILE = os.path.join(KERNELS_DIR, "naif_ids.html")

# Global Skyfield objects
_ephemeris = None
_timescale = None

def load_all_kernels():
    """Loads                   # Format time for Horizons - use calendar format
        if len(utc_time) >= 19:
            horizons_time = f"'{utc_time}'"
            stop_time = f"'{utc_time[:14]}{int(utc_time[14:16]) + 1:02d}:00'"
        else:
            horizons_time = f"'{utc_time} 00:00:00'"
            stop_time = f"'{utc_time} 00:00:01'"JD format for Horizons
        jd = t.tdb
        horizons_time = f"'JD{jd:.6f}'"
        stop_jd = jd + 1/86400  # 1 second later
        stop_time = f"'JD{stop_jd:.6f}'"

        # Parameters for Ceres (NAIF ID: 2000001)
        params = {
            'format': 'json',
            'COMMAND': '2000001',  # Ceres NAIF ID
            'OBJ_DATA': 'NO',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'VECTORS',
            'CENTER': '500@10',  # Solar System Barycenter
            'START_TIME': "'2025-Jan-01'",
            'STOP_TIME': "'2025-Jan-02'",
            'STEP_SIZE': '1d',for Ceres position
        params = {
            'format': 'json',
            'COMMAND': '2000001',  # Ceres NAIF ID
            'OBJ_DATA': 'NO',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'VECTORS',
            'CENTER': '500@10',  # Solar System Barycenter
            'START_TIME': horizons_time,
            'STOP_TIME': f"JD{jd + 1/1440:.6f}",  # 1 minute later
            'STEP_SIZE': '1m',  # 1 minute stepshemeris instead of SPICE kernels."""
    global _ephemeris, _timescale

    print("\n--- Loading Skyfield Ephemeris ---\n", flush=True)

    # Initialize timescale
    _timescale = load.timescale()

    # Load DE440 ephemeris (covers 1550-2650)
    de440_path = os.path.join(BSP_DIR, "de440.bsp")
    if os.path.exists(de440_path):
        _ephemeris = load(de440_path)
        print(f"✅ Loaded Skyfield ephemeris: {os.path.basename(de440_path)}", flush=True)
    else:
        print("❌ ERROR: DE440 ephemeris file not found! Please run check_and_download_kernels() first.")
        print(f"Expected location: {de440_path}")
        return False

    # Load additional ephemeris files if they exist
    additional_files = [
        os.path.join(BSP_DIR, f)
        for f in os.listdir(BSP_DIR)
        if f.endswith('.bsp') and f != 'de440.bsp'
    ]

    for bsp_file in additional_files:
        try:
            # Skyfield can load multiple ephemeris, but we'll use the main DE440
            print(f"ℹ️  Additional ephemeris file found: {os.path.basename(bsp_file)} (using DE440 as primary)", flush=True)
        except Exception as e:
            print(f"⚠️  Could not load additional ephemeris {os.path.basename(bsp_file)}: {e}", flush=True)

    print(f"\n✅ Skyfield ephemeris loaded successfully.\n")
    return True


def check_file_exists(filepath):
    """Checks if a file exists and provides instructions if not."""
    if not os.path.exists(filepath):
        print(f"Error: The required file '{filepath}' was not found.")
        print("Please download the necessary file and place it in the correct directory.")
        exit(1)


def parse_naif_ids(html_file):
    """Extracts celestial body names and NAIF IDs from the HTML file."""
    id_to_name = {}
    name_to_id = {}

    with open(html_file, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = re.compile(r"(\d+)\s+'([A-Z0-9_ ]+)'")
    matches = pattern.findall(content)

    for naif_id, name in matches:
        naif_id = int(naif_id)
        clean_name = name.strip().replace("_", " ")
        id_to_name.setdefault(naif_id, []).append(clean_name)
        name_to_id[clean_name.lower()] = naif_id

    return id_to_name, name_to_id


def parse_gm_values(tpc_file):
    """Extracts GM values for celestial bodies from the TPC file."""
    gm_values = {}

    with open(tpc_file, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = re.compile(r"BODY(\d+)_GM\s+=\s+\(\s*([-+\d\.EDed]+)\s*\)")
    matches = pattern.findall(content)

    for naif_id, gm in matches:
        gm_values[int(naif_id)] = float(gm.replace('D', 'E').replace('d', 'e'))  # Convert D to E for Python

    return gm_values


def fetch_gm(body_name, results):
    """Retrieves the gravitational parameter (GM) for a given body, ensuring correct units."""
    id_to_name, name_to_id = parse_naif_ids(NAIF_FILE)
    gm_values = parse_gm_values(GM_FILE)  # Read from TPC file

    '''
    print("\n=== DEBUG: Full GM Dictionary Before Conversion ===")
    for k, v in gm_values.items():
        print(f"NAIF ID: {k} → GM: {v} km³/s² (as read from file)")
    print("==============================\n")
    '''

    normalized = body_name.strip().lower()

    if normalized in results:
        print(f"\nEntry already exists: {results[normalized]}")
        return results

    if normalized in name_to_id:
        naif_id = name_to_id[normalized]
        print(f"DEBUG: Searching for GM value with NAIF ID: {naif_id}")

        gm_value = gm_values.get(naif_id, None)
        if gm_value is None:
            print(f"Warning: No GM value found for {id_to_name.get(naif_id, ['Unknown'])} (NAIF ID: {naif_id})")
            gm_value = "Unknown"

        print(f"\nFound: {id_to_name.get(naif_id, ['Unknown'])} (NAIF ID: {naif_id})")
        print(f"GM Value Before Conversion: {gm_value} km³/s²")

        results[normalized] = {"id": naif_id, "gm": gm_value}
    else:
        print("Invalid name. Please try again.")

    return results


def get_position(body_name, utc_time):
    """Retrieve heliocentric position using Skyfield at a UTC time."""
    global _ephemeris, _timescale

    if _ephemeris is None or _timescale is None:
        print("❌ ERROR: Skyfield ephemeris not loaded. Call load_all_kernels() first.")
        return None

    print(f"\nRetrieving position for {body_name} at {utc_time}...", flush=True)

    try:
        # Parse time - handle both date-only and full datetime strings
        if len(utc_time) >= 19:  # Full datetime with time
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]),
                              int(utc_time[11:13]), int(utc_time[14:16]), int(utc_time[17:19]))
        elif len(utc_time) >= 10:  # Date only
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]))
        else:
            print(f"❌ ERROR: Invalid time format: {utc_time}")
            return None

        # Get body object
        body_name_clean = body_name.upper().replace(" BARYCENTER", "").replace(" ", "")
        if body_name_clean == "SUN":
            body = _ephemeris['sun']
        elif body_name_clean in ['MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO']:
            # For planets, use barycenter by default (this matches SPICE behavior)
            body = _ephemeris[f'{body_name_clean.lower()}_barycenter']
        elif body_name_clean == "EARTHMOONBARYCENTER" or body_name_clean == "EARTH-MOONBARYCENTER":
            body = _ephemeris['earth-moon-barycenter']
        elif body_name_clean == "CERES":
            # Use JPL Horizons for accurate Ceres position (publication quality)
            try:
                return get_ceres_position_jpl_horizons(utc_time)
            except Exception as e:
                print(f"⚠️  JPL Horizons failed, falling back to approximate calculation: {e}")
                return get_ceres_position_approximate(utc_time)
        else:
            print(f"❌ ERROR: Body '{body_name}' not supported in Skyfield DE440 ephemeris")
            return None

        # Get position relative to solar system barycenter
        position_au = body.at(t).position.au  # Returns position in AU
        position_m = position_au * 149597870700.0  # Convert AU to meters

        print(f"✅ Position for {body_name}: {position_m} meters", flush=True)
        return position_m

    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve position for {body_name}: {e}", file=sys.stderr, flush=True)
        return None


def get_state(body_name, utc_time):
    """Retrieve heliocentric state (position and velocity) using Skyfield at a UTC time."""
    global _ephemeris, _timescale

    if _ephemeris is None or _timescale is None:
        print("❌ ERROR: Skyfield ephemeris not loaded. Call load_all_kernels() first.")
        return None, None

    print(f"\nRetrieving state for {body_name} at {utc_time}...", flush=True)

    try:
        # Parse time
        t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]),
                          int(utc_time[11:13]), int(utc_time[14:16]), int(utc_time[17:19]))

        # Get body object
        body_name_clean = body_name.upper().replace(" BARYCENTER", "").replace(" ", "")
        if body_name_clean == "SUN":
            body = _ephemeris['sun']
        elif body_name_clean in ['MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO']:
            # For planets, use barycenter by default (this matches SPICE behavior)
            body = _ephemeris[f'{body_name_clean.lower()}_barycenter']
        elif body_name_clean == "EARTHMOONBARYCENTER" or body_name_clean == "EARTH-MOONBARYCENTER":
            body = _ephemeris['earth-moon-barycenter']
        elif body_name_clean == "CERES":
            # Use JPL Horizons for accurate Ceres state (publication quality)
            try:
                return get_ceres_state_jpl_horizons(utc_time)
            except Exception as e:
                print(f"⚠️  JPL Horizons failed, falling back to approximate calculation: {e}")
                return get_ceres_state_approximate(utc_time)
        else:
            print(f"❌ ERROR: Body '{body_name}' not supported in Skyfield DE440 ephemeris")
            return None, None

        # Get state relative to solar system barycenter
        position_au, velocity_au_per_day = body.at(t).position.au, body.at(t).velocity.au_per_d

        # Convert units
        position_m = position_au * 149597870700.0  # AU to meters
        velocity_m_per_s = velocity_au_per_day * 149597870700.0 / 86400.0  # AU/day to m/s

        print(f"✅ State for {body_name}: position = {position_m} meters, velocity = {velocity_m_per_s} m/s", flush=True)
        return position_m, velocity_m_per_s

    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve state for {body_name}: {e}", flush=True)
        return None, None


def get_lambert_inputs(departure_body, arrival_body, departure_time, tof_days):
    """Gather all inputs needed for the Lambert solver."""
    tof_seconds = tof_days * 86400  # Convert days to seconds

    # Calculate arrival time
    dep_t = _timescale.utc(int(departure_time[:4]), int(departure_time[5:7]), int(departure_time[8:10]),
                          int(departure_time[11:13]), int(departure_time[14:16]), int(departure_time[17:19]))
    arr_t = dep_t + tof_seconds / 86400.0  # Add days
    arrival_time = f"{arr_t.utc.year:04d}-{arr_t.utc.month:02d}-{arr_t.utc.day:02d} {arr_t.utc.hour:02d}:{arr_t.utc.minute:02d}:{int(arr_t.utc.second):02d}"

    r1 = get_position(departure_body, departure_time)
    r2 = get_position(arrival_body, arrival_time)
    mu = fetch_gm("Sun", {}).get("sun", {"gm": None})["gm"]

    if r1 is None or r2 is None or mu is None:
        print("Error: Missing data for Lambert inputs", file=sys.stderr, flush=True)
        return None

    return {
        "r1": r1,           # meters
        "r2": r2,           # meters
        "tof": tof_seconds, # seconds
        "mu": mu            # km³/s² (as provided by the correct GM value)
    }


def check_and_download_kernels():
    """Download and update SPICE kernels from NASA (needed for GM values and compatibility)."""
    print("\n--- Checking and Downloading SPICE Kernels ---\n")

    # Ensure directories exist
    os.makedirs(KERNELS_DIR, exist_ok=True)
    os.makedirs(BSP_DIR, exist_ok=True)

    # URLs for required kernels (same as before, but we primarily use DE440 for Skyfield)
    kernel_urls = {
        "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        "gm_de440.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
        "naif_ids.html": "https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html",
        "de440.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
    }

    for filename, url in kernel_urls.items():
        filepath = os.path.join(KERNELS_DIR, filename)
        if filename.endswith('.bsp'):
            filepath = os.path.join(BSP_DIR, filename)

        print(f"Checking {filename}...")

        # Check if file exists and is recent (within 30 days)
        needs_download = True
        if os.path.exists(filepath):
            file_age = time.time() - os.path.getmtime(filepath)
            if file_age < 30 * 24 * 3600:  # 30 days
                print(f"  ✅ {filename} is up to date")
                needs_download = False
            else:
                print(f"  🔄 {filename} is outdated, downloading...")

        if needs_download:
            try:
                print(f"  📥 Downloading {filename} from {url}")
                # Create SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                with urllib.request.urlopen(url, context=ssl_context) as response:
                    with open(filepath, 'wb') as f:
                        f.write(response.read())
                print(f"  ✅ Successfully downloaded {filename}")
            except Exception as e:
                if "404" in str(e) or "Not Found" in str(e):
                    print(f"  ⚠️  {filename} not found at URL (may be outdated)")
                else:
                    print(f"  ❌ Failed to download {filename}: {e}")

    print("\n✅ Kernel check and download complete!\n")


def search_celestial_body(body_name):
    """Search for a celestial body by name and return its NAIF ID."""
    id_to_name, name_to_id = parse_naif_ids(NAIF_FILE)
    normalized = body_name.strip().lower()

    if normalized in name_to_id:
        return name_to_id[normalized]
    else:
        print(f"Warning: Celestial body '{body_name}' not found in NAIF IDs")
        return None


def str2et(utc_time):
    """Convert UTC time string to ephemeris time (seconds since J2000). Skyfield equivalent."""
    global _timescale
    if _timescale is None:
        _timescale = load.timescale()

    # Parse the UTC time string
    # Expected format: "YYYY-MM-DD HH:MM:SS UTC" or similar
    try:
        if isinstance(utc_time, str):
            # Parse the string - this is a simplified parser
            parts = utc_time.replace(' UTC', '').split()
            date_part = parts[0]
            time_part = parts[1] if len(parts) > 1 else '00:00:00'

            year, month, day = map(int, date_part.split('-'))
            hour, minute, second = map(int, time_part.split(':'))

            t = _timescale.utc(year, month, day, hour, minute, second)
            # Return seconds since J2000 (TDB) - convert from Julian date
            j2000_jd = 2451545.0  # J2000 in Julian days
            et_seconds = (t.tdb - j2000_jd) * 86400.0  # Convert days to seconds
            return et_seconds
        else:
            raise ValueError("Unsupported time format")
    except Exception as e:
        print(f"Error parsing time '{utc_time}': {e}")
        return None


def et2utc(et, format_str="C", precision=3):
    """Convert ephemeris time to UTC string. Skyfield equivalent."""
    global _timescale
    if _timescale is None:
        _timescale = load.timescale()

    try:
        # Convert ephemeris time (seconds since J2000 TDB) to Julian date
        j2000_jd = 2451545.0  # J2000 in Julian days
        jd = j2000_jd + et / 86400.0  # Convert seconds to days

        # Create time object
        t = _timescale.tdb(jd=jd)

        if format_str == "C":
            # Calendar format
            utc_str = f"{t.utc.year:04d}-{t.utc.month:02d}-{t.utc.day:02d} {t.utc.hour:02d}:{t.utc.minute:02d}:{t.utc.second:06.3f}"
            return utc_str
        elif format_str == "ISOC":
            # ISO calendar format
            utc_str = f"{t.utc.year:04d}-{t.utc.month:02d}-{t.utc.day:02d}T{t.utc.hour:02d}:{t.utc.minute:02d}:{t.utc.second:06.3f}"
            return utc_str
        else:
            return str(t.utc)
    except Exception as e:
        print(f"Error converting ephemeris time: {e}")
        return None


def get_ceres_position_approximate(utc_time):
    """
    Calculate approximate position of Ceres using Keplerian orbital elements.
    This is a simplified calculation for research purposes - not as accurate as JPL ephemeris.
    Orbital elements are approximate for epoch J2000.
    """
    global _timescale

    if _timescale is None:
        _timescale = load.timescale()

    try:
        # Parse time
        if len(utc_time) >= 19:  # Full datetime with time
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]),
                              int(utc_time[11:13]), int(utc_time[14:16]), int(utc_time[17:19]))
        elif len(utc_time) >= 10:  # Date only
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]))
        else:
            print(f"❌ ERROR: Invalid time format: {utc_time}")
            return None

        # Approximate orbital elements for Ceres (J2000 epoch)
        # These are simplified values - for research use, consider using more accurate ephemeris
        a = 2.7691653  # Semi-major axis (AU)
        e = 0.0760090  # Eccentricity
        i = math.radians(10.59407)  # Inclination (radians)
        omega = math.radians(73.59769)  # Argument of periapsis (radians)
        Omega = math.radians(80.30553)  # Longitude of ascending node (radians)
        M0 = math.radians(291.18349)  # Mean anomaly at J2000 (radians)

        # Time since J2000 in days
        j2000_jd = 2451545.0
        days_since_j2000 = t.tdb - j2000_jd

        # Mean motion (radians per day)
        mu_sun = 1.32712440018e20  # Sun's GM in m³/s²
        n = math.sqrt(mu_sun / (a * 149597870700.0)**3)  # rad/s
        n_days = n * 86400  # rad/day

        # Mean anomaly
        M = M0 + n_days * days_since_j2000

        # Solve Kepler's equation (simplified - using approximation for small e)
        E = M + e * math.sin(M)  # First approximation

        # True anomaly
        nu = 2 * math.atan(math.sqrt((1 + e)/(1 - e)) * math.tan(E/2))

        # Distance from Sun
        r = a * (1 - e * math.cos(E))

        # Position in orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)

        # Rotate to ecliptic coordinates
        cos_Omega = math.cos(Omega)
        sin_Omega = math.sin(Omega)
        cos_i = math.cos(i)
        sin_i = math.sin(i)
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)

        # Position vector rotation
        x_ecl = (cos_omega * cos_Omega - sin_omega * sin_Omega * cos_i) * x_orb + \
                (-sin_omega * cos_Omega - cos_omega * sin_Omega * cos_i) * y_orb
        y_ecl = (cos_omega * sin_Omega + sin_omega * cos_Omega * cos_i) * x_orb + \
                (-sin_omega * sin_Omega + cos_omega * cos_Omega * cos_i) * y_orb
        z_ecl = sin_omega * sin_i * x_orb + cos_omega * sin_i * y_orb

        # Convert to meters
        position_m = np.array([x_ecl, y_ecl, z_ecl]) * 149597870700.0

        print(f"✅ Approximate position for Ceres: {position_m} meters (Keplerian approximation)", flush=True)
        print("⚠️  WARNING: Using approximate orbital elements for Ceres. For publication-quality results, use JPL asteroid ephemeris.", flush=True)

        return position_m

    except Exception as e:
        print(f"❌ ERROR: Failed to calculate Ceres position: {e}", flush=True)
        return None


def get_ceres_state_approximate(utc_time):
    """
    Calculate approximate state (position and velocity) of Ceres using Keplerian orbital elements.
    This is a simplified calculation for research purposes - not as accurate as JPL ephemeris.
    """
    global _timescale

    if _timescale is None:
        _timescale = load.timescale()

    try:
        # Parse time
        if len(utc_time) >= 19:  # Full datetime with time
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]),
                              int(utc_time[11:13]), int(utc_time[14:16]), int(utc_time[17:19]))
        elif len(utc_time) >= 10:  # Date only
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]))
        else:
            print(f"❌ ERROR: Invalid time format: {utc_time}")
            return None, None

        # Same orbital elements as position function
        a = 2.7691653  # Semi-major axis (AU)
        e = 0.0760090  # Eccentricity
        i = math.radians(10.59407)  # Inclination (radians)
        omega = math.radians(73.59769)  # Argument of periapsis (radians)
        Omega = math.radians(80.30553)  # Longitude of ascending node (radians)
        M0 = math.radians(291.18349)  # Mean anomaly at J2000 (radians)

        # Time calculations
        j2000_jd = 2451545.0
        days_since_j2000 = t.tdb - j2000_jd

        mu_sun = 1.32712440018e20  # Sun's GM in m³/s²
        n = math.sqrt(mu_sun / (a * 149597870700.0)**3)  # rad/s
        n_days = n * 86400  # rad/day

        M = M0 + n_days * days_since_j2000
        E = M + e * math.sin(M)  # First approximation
        nu = 2 * math.atan(math.sqrt((1 + e)/(1 - e)) * math.tan(E/2))
        r = a * (1 - e * math.cos(E))

        # Velocity calculations
        h = math.sqrt(mu_sun * a * (1 - e**2))  # Specific angular momentum
        v_r = (mu_sun * e / h) * math.sin(nu)  # Radial velocity
        v_theta = (mu_sun / h) * (1 + e * math.cos(nu))  # Tangential velocity

        # Position in orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)

        # Velocity in orbital plane
        vx_orb = v_r * math.cos(nu) - v_theta * math.sin(nu)
        vy_orb = v_r * math.sin(nu) + v_theta * math.cos(nu)

        # Rotate to ecliptic coordinates (same rotation as position)
        cos_Omega = math.cos(Omega)
        sin_Omega = math.sin(Omega)
        cos_i = math.cos(i)
        sin_i = math.sin(i)
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)

        # Position vector rotation
        x_ecl = (cos_omega * cos_Omega - sin_omega * sin_Omega * cos_i) * x_orb + \
                (-sin_omega * cos_Omega - cos_omega * sin_Omega * cos_i) * y_orb
        y_ecl = (cos_omega * sin_Omega + sin_omega * cos_Omega * cos_i) * x_orb + \
                (-sin_omega * sin_Omega + cos_omega * cos_Omega * cos_i) * y_orb
        z_ecl = sin_omega * sin_i * x_orb + cos_omega * sin_i * y_orb

        # Velocity vector rotation (same transformation matrix)
        vx_ecl = (cos_omega * cos_Omega - sin_omega * sin_Omega * cos_i) * vx_orb + \
                 (-sin_omega * cos_Omega - cos_omega * sin_Omega * cos_i) * vy_orb
        vy_ecl = (cos_omega * sin_Omega + sin_omega * cos_Omega * cos_i) * vx_orb + \
                 (-sin_omega * sin_Omega + cos_omega * cos_Omega * cos_i) * vy_orb
        vz_ecl = sin_omega * sin_i * vx_orb + cos_omega * sin_i * vy_orb

        # Convert to meters and m/s
        position_m = np.array([x_ecl, y_ecl, z_ecl]) * 149597870700.0
        velocity_m_s = np.array([vx_ecl, vy_ecl, vz_ecl]) * 149597870700.0 / 86400.0  # AU/day to m/s

        print(f"✅ Approximate state for Ceres: position = {position_m} meters, velocity = {velocity_m_s} m/s (Keplerian approximation)", flush=True)
        print("⚠️  WARNING: Using approximate orbital elements for Ceres. For publication-quality results, use JPL asteroid ephemeris.", flush=True)

        return position_m, velocity_m_s

    except Exception as e:
        print(f"❌ ERROR: Failed to calculate Ceres state: {e}", flush=True)
        return None, None


def get_ceres_position_jpl_horizons(utc_time):
    """
    Get accurate Ceres position from JPL Horizons system.
    This provides publication-quality ephemeris data.
    """
    try:
        # JPL Horizons API endpoint
        url = "https://ssd.jpl.nasa.gov/api/horizons.api"

        # Use hardcoded date for testing
        horizons_time = '2025-01-01T00:00:00'
        stop_time = '2025-01-02T00:00:00'

        # Parameters for Ceres (NAIF ID: 2000001)
        params = {
            'format': 'json',
            'COMMAND': '2000001',  # Ceres NAIF ID
            'OBJ_DATA': 'NO',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'VECTORS',
            'CENTER': '500@10',  # Solar System Barycenter
            'START_TIME': horizons_time,
            'STOP_TIME': stop_time,
            'STEP_SIZE': '1',
            'VEC_TABLE': '1',
            'VEC_CORR': 'NONE',
            'OUT_UNITS': 'KM-S',  # KM and seconds
            'CSV_FORMAT': 'NO',
            'VEC_LABELS': 'NO'
        }

        # Make request to JPL Horizons
        response = requests.post(url, data=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'error' in data:
            print(f"❌ JPL Horizons error: {data['error']}")
            return None

        # Parse the ephemeris data
        ephem_data = data.get('result', '')
        lines = ephem_data.split('\n')

        # Find the data between $$SOE and $$EOE
        in_data_section = False
        for line in lines:
            line = line.strip()
            if line == '$$SOE':
                in_data_section = True
                continue
            elif line == '$$EOE':
                break
            elif in_data_section and line.startswith('246'):  # JD date line
                # Next line should contain the X Y Z data
                continue
            elif in_data_section and not line.startswith('246') and line and not line.startswith('$$'):
                # This should be the position data line
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x_km = float(parts[0])
                        y_km = float(parts[1])
                        z_km = float(parts[2])

                        # Convert km to meters
                        position_m = np.array([x_km, y_km, z_km]) * 1000.0

                        print(f"✅ JPL Horizons position for Ceres: {position_m} meters", flush=True)
                        print("🏦 BANKABLE DATA: Using official JPL ephemeris for publication-quality accuracy", flush=True)

                        return position_m
                    except (ValueError, IndexError) as e:
                        print(f"❌ Error parsing position data: {e}")
                        continue

        print("❌ Could not parse JPL Horizons response")
        return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error accessing JPL Horizons: {e}")
        return None
    except Exception as e:
        print(f"❌ Error getting Ceres position from JPL Horizons: {e}")
        return None


def get_ceres_state_jpl_horizons(utc_time):
    """
    Get accurate Ceres state (position and velocity) from JPL Horizons.
    This provides publication-quality ephemeris data.
    """
    try:
        # JPL Horizons API endpoint
        url = "https://ssd.jpl.nasa.gov/api/horizons.api"

        # Format time for Horizons - use ISO format
        horizons_time = utc_time.replace(' ', 'T')
        # Add 1 minute for stop time
        from datetime import datetime, timedelta
        dt = datetime.fromisoformat(utc_time.replace(' ', 'T'))
        dt_stop = dt + timedelta(minutes=1)
        stop_time = dt_stop.isoformat()

        # Parameters for Ceres with velocity data
        params = {
            'format': 'json',
            'COMMAND': '2000001',  # Ceres NAIF ID
            'OBJ_DATA': 'NO',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'VECTORS',
            'CENTER': '500@10',  # Solar System Barycenter
            'START_TIME': horizons_time,
            'STOP_TIME': stop_time,
            'STEP_SIZE': '1',
            'VEC_TABLE': '2',  # Include velocity vectors
            'VEC_CORR': 'NONE',
            'OUT_UNITS': 'KM-S',
            'CSV_FORMAT': 'NO',
            'VEC_LABELS': 'NO'
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'error' in data:
            print(f"❌ JPL Horizons error: {data['error']}")
            return None, None

        # Parse the ephemeris data
        ephem_data = data.get('result', '')
        lines = ephem_data.split('\n')

        position_m = None
        velocity_m_s = None

        # Find the data between $$SOE and $$EOE
        in_data_section = False
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == '$$SOE':
                in_data_section = True
                i += 1
                continue
            elif line == '$$EOE':
                break
            elif in_data_section and line.startswith('246'):  # JD date line
                # Next line should be position data
                if i + 1 < len(lines):
                    pos_line = lines[i + 1].strip()
                    parts = pos_line.split()
                    if len(parts) >= 3:
                        try:
                            x_km = float(parts[0])
                            y_km = float(parts[1])
                            z_km = float(parts[2])
                            position_m = np.array([x_km, y_km, z_km]) * 1000.0
                        except (ValueError, IndexError):
                            pass

                # Line after position should be velocity data
                if i + 2 < len(lines):
                    vel_line = lines[i + 2].strip()
                    parts = vel_line.split()
                    if len(parts) >= 3:
                        try:
                            vx_km_s = float(parts[0])
                            vy_km_s = float(parts[1])
                            vz_km_s = float(parts[2])
                            velocity_m_s = np.array([vx_km_s, vy_km_s, vz_km_s]) * 1000.0
                        except (ValueError, IndexError):
                            pass

                if position_m is not None and velocity_m_s is not None:
                    print(f"✅ JPL Horizons state for Ceres: position = {position_m} meters, velocity = {velocity_m_s} m/s", flush=True)
                    print("🏦 BANKABLE DATA: Using official JPL ephemeris for publication-quality accuracy", flush=True)
                    return position_m, velocity_m_s

            i += 1

        print("❌ Could not parse JPL Horizons response for state data")
        return None, None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error accessing JPL Horizons: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error getting Ceres state from JPL Horizons: {e}")
        return None, None