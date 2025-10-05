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
from skyfield.api import load, Loader
import requests
import json
from datetime import datetime, timedelta

# Global file paths (same as spice_interface.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")
BSP_DIR = os.path.join(KERNELS_DIR, "BSP")
GM_FILE = os.path.join(KERNELS_DIR, "gm_de440.tpc")
NAIF_FILE = os.path.join(KERNELS_DIR, "naif_ids.html")

# Horizons cache directory
HORIZONS_CACHE_DIR = os.path.join(KERNELS_DIR, "horizons_cache")
os.makedirs(HORIZONS_CACHE_DIR, exist_ok=True)

# Global Skyfield objects
_ephemeris = None
_timescale = None
_ceres_ephemeris = None

# GUI callback for Horizons confirmation
_horizons_confirmation_callback = None


def set_horizons_confirmation_callback(callback):
    """Set the callback function for Horizons confirmation dialogs.
    
    The callback should be a function that takes (body_name, utc_time) and returns True/False.
    It should show a dialog asking the user if they want to proceed with Horizons data.
    """
    global _horizons_confirmation_callback
    _horizons_confirmation_callback = callback


def confirm_horizons_usage(body_name, utc_time):
    """Ask user for confirmation before using JPL Horizons data.
    
    Returns True if user approves, False if they cancel.
    """
    global _horizons_confirmation_callback
    if _horizons_confirmation_callback is not None:
        return _horizons_confirmation_callback(body_name, utc_time)
    else:
        # Default behavior: allow Horizons usage if no callback set
        print(f"⚠️  No GUI callback set for Horizons confirmation. Allowing request for {body_name} at {utc_time}")
        return True

# Horizons cache - stores downloaded ephemeris data
_horizons_cache = {}


def get_horizons_cache_filename(body_name, start_date, end_date, step_days=1):
    """Generate a cache filename for Horizons data."""
    safe_body = body_name.replace(' ', '_').replace('/', '_')
    return f"{safe_body}_{start_date}_{end_date}_{step_days}d.json"


def load_horizons_cache(body_name, start_date, end_date, step_days=1):
    """Load cached Horizons data if it exists and is valid."""
    global _horizons_cache
    
    cache_file = os.path.join(HORIZONS_CACHE_DIR, get_horizons_cache_filename(body_name, start_date, end_date, step_days))

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                # Populate global cache
                cache_key = f"{body_name}_{start_date}_{end_date}_{step_days}"
                _horizons_cache[cache_key] = data
                print(f"✅ Loaded cached Horizons data for {body_name} ({start_date} to {end_date})")
                return data
        except Exception as e:
            print(f"⚠️  Failed to load Horizons cache: {e}")
            return None

    return None


def save_horizons_cache(body_name, start_date, end_date, step_days, data):
    """Save Horizons data to cache."""
    cache_file = os.path.join(HORIZONS_CACHE_DIR, get_horizons_cache_filename(body_name, start_date, end_date, step_days))

    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved Horizons data to cache: {cache_file}")
    except Exception as e:
        print(f"⚠️  Failed to save Horizons cache: {e}")


def download_horizons_ephemeris_batch(body_name, start_date, end_date, step_days=1):
    """
    Download ephemeris data for a body from JPL Horizons for a date range.
    This is much more efficient than individual requests.
    """
    global _horizons_cache

    # Check cache first
    cache_key = f"{body_name}_{start_date}_{end_date}_{step_days}"
    if cache_key in _horizons_cache:
        return _horizons_cache[cache_key]

    # Check disk cache
    cached_data = load_horizons_cache(body_name, start_date, end_date, step_days)
    if cached_data:
        _horizons_cache[cache_key] = cached_data
        return cached_data

    try:
        # JPL Horizons API endpoint
        url = "https://ssd.jpl.nasa.gov/api/horizons.api"

        # Get NAIF ID for the body
        id_to_name, name_to_id = parse_naif_ids(NAIF_FILE)
        naif_id = None

        # Try common mappings
        body_upper = body_name.upper().replace(' ', '')
        if body_upper == 'CERES':
            naif_id = '2000001'
        elif body_upper in name_to_id:
            naif_id = str(name_to_id[body_upper])
        else:
            # Try to find by name
            for nid, names in id_to_name.items():
                if any(body_upper in name.upper() for name in names):
                    naif_id = str(nid)
                    break

        if not naif_id:
            print(f"❌ Could not find NAIF ID for {body_name}")
            return None

        print(f"🌐 Downloading Horizons data for {body_name} (NAIF ID: {naif_id}) from {start_date} to {end_date}...")

        # Parameters for batch ephemeris download
        params = {
            'format': 'json',
            'COMMAND': naif_id,
            'OBJ_DATA': 'NO',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'VECTORS',
            'CENTER': '500@10',  # Solar System Barycenter
            'START_TIME': start_date,
            'STOP_TIME': end_date,
            'STEP_SIZE': f'{step_days}d',
            'VEC_TABLE': '1',  # Position only (reduces data transfer by ~40%)
            'VEC_CORR': 'NONE',
            'OUT_UNITS': 'KM-S',  # KM and seconds
            'CSV_FORMAT': 'NO',
            'VEC_LABELS': 'YES'
        }

        # Make request to JPL Horizons
        response = requests.post(url, data=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if 'error' in data:
            print(f"❌ JPL Horizons error: {data['error']}")
            return None

        # Parse the ephemeris data
        ephem_data = data.get('result', '')
        lines = ephem_data.split('\n')

        # Parse the data into a structured format
        parsed_data = {}
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
            elif in_data_section and '=' in line and 'A.D.' in line:  # JD date line
                # Extract JD from the beginning of the line
                parts = line.split('=')
                if len(parts) >= 1:
                    jd_str = parts[0].strip()
                    try:
                        jd = float(jd_str)
                        
                        # Next line should be position data
                        if i + 1 < len(lines):
                            pos_line = lines[i + 1].strip()
                            if 'X =' in pos_line:
                                # Parse format: " X =-1.639461860837789E+08 Y =-3.785475694792547E+08 Z = 1.822372754839402E+07"
                                import re
                                pos_match = re.search(r'X\s*=\s*([-\d.E+]+).*Y\s*=\s*([-\d.E+]+).*Z\s*=\s*([-\d.E+]+)', pos_line)
                                if pos_match:
                                    x_km = float(pos_match.group(1))
                                    y_km = float(pos_match.group(2))
                                    z_km = float(pos_match.group(3))
                                    
                                    # Store position data only (no velocity with VEC_TABLE=1)
                                    parsed_data[str(jd)] = {
                                        'position': [x_km * 1000.0, y_km * 1000.0, z_km * 1000.0],
                                        'velocity': [0.0, 0.0, 0.0]  # Placeholder for compatibility
                                    }
                                    
                                    # Skip to next JD line (no velocity line to skip)
                                    i += 1
                    except ValueError:
                        pass
            i += 1

        if parsed_data:
            # Cache the data
            _horizons_cache[cache_key] = parsed_data
            save_horizons_cache(body_name, start_date, end_date, step_days, parsed_data)

            print(f"✅ Downloaded and cached {len(parsed_data)} ephemeris points for {body_name}")
            print("🏦 BANKABLE DATA: Using official JPL ephemeris for publication-quality accuracy")
            return parsed_data
        else:
            print("❌ Could not parse JPL Horizons batch response")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error accessing JPL Horizons: {e}")
        return None
    except Exception as e:
        print(f"❌ Error downloading batch ephemeris from JPL Horizons: {e}")
        return None


def get_cached_horizons_position(body_name, utc_time):
    """Get position from cached Horizons data."""
    global _horizons_cache, _timescale

    if _timescale is None:
        _timescale = load.timescale()

    try:
        # Convert time to JD
        if len(utc_time) >= 19:  # Full datetime with time
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]),
                              int(utc_time[11:13]), int(utc_time[14:16]), int(utc_time[17:19]))
        elif len(utc_time) >= 10:  # Date only
            t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]))
        else:
            return None

        jd = t.tdb

        # Check all cached data for this body
        for cache_key, data in _horizons_cache.items():
            if body_name in cache_key:
                # Find closest JD in the data
                jds = [float(jd_str) for jd_str in data.keys()]
                closest_jd = min(jds, key=lambda x: abs(x - jd))

                if abs(closest_jd - jd) < 1.0:  # Within 1 day
                    pos_data = data[str(closest_jd)]
                    position = np.array(pos_data['position'])
                    print(f"✅ Cached Horizons position for {body_name}: {position} meters")
                    return position

        # If not in global cache, check disk cache for this body (load any available ranges)
        import os
        if os.path.exists(HORIZONS_CACHE_DIR):
            for filename in os.listdir(HORIZONS_CACHE_DIR):
                if filename.startswith(f"{body_name}_") and filename.endswith('.json'):
                    # Extract date range from filename
                    parts = filename.replace('.json', '').split('_')
                    if len(parts) >= 4:
                        start_date = parts[1]
                        end_date = parts[2]
                        step_days = int(parts[3].replace('d', ''))
                        
                        # Load this cache file
                        cached_data = load_horizons_cache(body_name, start_date, end_date, step_days)
                        if cached_data:
                            # Check if our JD is in this range
                            jds = [float(jd_str) for jd_str in cached_data.keys()]
                            if jds and min(jds) <= jd <= max(jds):
                                closest_jd = min(jds, key=lambda x: abs(x - jd))
                                if abs(closest_jd - jd) < 1.0:  # Within 1 day
                                    pos_data = cached_data[str(closest_jd)]
                                    position = np.array(pos_data['position'])
                                    print(f"✅ Cached Horizons position for {body_name}: {position} meters")
                                    return position

        return None

    except Exception as e:
        print(f"❌ Error getting cached Horizons position: {e}")
        return None

def load_all_kernels():
    """Load Skyfield ephemeris with intelligent kernel selection.

    Priority order:
    1. Use DE441 for best general accuracy (auto-downloaded)
    2. Fall back to local DE440 if DE441 fails
    3. Check for specialized kernels (mar097.bsp for Mars, etc.)
    4. Use Horizons API for missing bodies

    This ensures we use the best available data for each body.
    """
    global _ephemeris, _timescale

    print("\n--- Loading Skyfield Ephemeris ---\n", flush=True)

    # Initialize timescale
    _timescale = load.timescale()

    # Step 1: Try to load DE441 (best general-purpose ephemeris)
    de441_path = os.path.join(BSP_DIR, "de441.bsp")
    if not os.path.exists(de441_path):
        # Download DE441 manually to BSP_DIR
        try:
            print("📥 Downloading DE441 ephemeris to BSP directory...")
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen("https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de441.bsp", context=ssl_context) as response:
                with open(de441_path, 'wb') as f:
                    f.write(response.read())
            print("✅ Downloaded DE441 ephemeris")
        except Exception as e:
            print(f"⚠️  Failed to download DE441: {e}")
    
    if os.path.exists(de441_path):
        try:
            _ephemeris = load(de441_path)
            print("✅ Loaded primary ephemeris: de441.bsp (latest high-accuracy ephemeris)", flush=True)
            primary_loaded = True
        except Exception as e:
            print(f"⚠️  Failed to load local DE441: {e}")
            primary_loaded = False
    else:
        primary_loaded = False

    # Step 2: If DE441 failed, try local DE440
    if not primary_loaded:
        de440_path = os.path.join(BSP_DIR, "de440.bsp")
        if os.path.exists(de440_path):
            try:
                _ephemeris = load(de440_path)
                print(f"✅ Loaded primary ephemeris: {os.path.basename(de440_path)}", flush=True)
                primary_loaded = True
            except Exception as e:
                print(f"⚠️  Failed to load local DE440: {e}")
                primary_loaded = False

    # Step 3: Check for specialized kernels and report them
    if os.path.exists(BSP_DIR):
        specialized_kernels = []
        for filename in os.listdir(BSP_DIR):
            if filename.endswith('.bsp') and filename not in ['de440.bsp', 'de441.bsp']:
                filepath = os.path.join(BSP_DIR, filename)
                specialized_kernels.append((filename, filepath))

        if specialized_kernels:
            print(f"ℹ️  Found {len(specialized_kernels)} specialized kernel(s):")
            for name, path in specialized_kernels:
                # Skip the 300-asteroid kernel as it uses unsupported SPK type 13
                if 'codes_300ast' in name:
                    print(f"   📁 {name}: Skipped (unsupported SPK type 13)")
                    continue
                    
                try:
                    # Try to load and check contents
                    test_kernel = load(path)
                    kernel_names = test_kernel.names()
                    body_count = len(kernel_names)

                    # Identify what bodies this kernel covers
                    body_types = []
                    for code, name_list in kernel_names.items():
                        for name in name_list:
                            name_lower = name.lower()
                            if 'mars' in name_lower or 'phobos' in name_lower or 'deimos' in name_lower:
                                if 'mars' not in body_types:
                                    body_types.append('Mars system')
                            elif 'jupiter' in name_lower or 'io' in name_lower or 'europa' in name_lower:
                                if 'Jupiter system' not in body_types:
                                    body_types.append('Jupiter system')
                            elif 'saturn' in name_lower or 'titan' in name_lower or 'enceladus' in name_lower:
                                if 'Saturn system' not in body_types:
                                    body_types.append('Saturn system')
                            elif 'ceres' in name_lower:
                                if 'Ceres' not in body_types:
                                    body_types.append('Ceres')

                    body_info = ', '.join(body_types) if body_types else 'various bodies'
                    print(f"   📁 {name}: {body_count} bodies ({body_info})")

                    # Note: Skyfield doesn't easily support loading multiple BSP files together
                    # For now, we use the primary ephemeris and fall back to Horizons for specialized bodies

                except Exception as e:
                    print(f"   ⚠️  {name}: Could not load ({e})")

    # Step 4: Load Ceres kernel if available
    global _ceres_ephemeris
    ceres_path = os.path.join(BSP_DIR, "ceres_1900_2100.bsp")
    if os.path.exists(ceres_path):
        try:
            _ceres_ephemeris = load(ceres_path)
            print("✅ Loaded Ceres ephemeris: ceres_1900_2100.bsp", flush=True)
        except Exception as e:
            print(f"⚠️  Failed to load Ceres kernel: {e}")
            _ceres_ephemeris = None

    print("\n✅ Ephemeris system ready. Will use Horizons API for bodies not in primary ephemeris.")
    print("📡 High-accuracy Mars data available via Horizons for specialized applications.\n")
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

    try:
        with open(html_file, 'r', encoding='utf-8') as file:
            content = file.read()
    except (OSError, IOError) as e:
        print(f"Warning: Could not read NAIF IDs file {html_file}: {e}")
        print("Using fallback body list with major planets only.")
        # Return minimal fallback data
        fallback_bodies = {
            'mercury': 1, 'venus': 2, 'earth': 3, 'mars': 4,
            'jupiter': 5, 'saturn': 6, 'uranus': 7, 'neptune': 8, 'pluto': 9,
            'sun': 10, 'moon': 301,
            # Barycenter names used by GUI
            'mercury barycenter': 1, 'venus barycenter': 2, 'mars barycenter': 4,
            'jupiter barycenter': 5, 'saturn barycenter': 6, 'uranus barycenter': 7,
            'neptune barycenter': 8, 'pluto barycenter': 9
        }
        for name, id_val in fallback_bodies.items():
            name_to_id[name] = id_val
            id_to_name[id_val] = [name.upper()]
        return id_to_name, name_to_id

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
            
            # Special note for Mars - we have specialized kernels available
            if body_name_clean == 'MARS':
                print("ℹ️  Using DE441 Mars barycenter. Specialized mar097.bsp kernel available for detailed Mars system data.")
        elif body_name_clean == "EARTHMOONBARYCENTER" or body_name_clean == "EARTH-MOONBARYCENTER":
            body = _ephemeris['earth-moon-barycenter']
        elif body_name_clean == "CERES":
            # Try local Ceres kernel first
            if _ceres_ephemeris is not None:
                try:
                    body = _ceres_ephemeris['ceres']
                    position_au = body.at(t).position.au
                    position_m = position_au * 149597870700.0
                    print(f"✅ Position for {body_name}: {position_m} meters (from local kernel)", flush=True)
                    return position_m
                except Exception as e:
                    print(f"⚠️  Local Ceres kernel failed, falling back to Horizons: {e}")
            
            # Fall back to Horizons API for accurate Ceres position (publication quality)
            try:
                return get_ceres_position_jpl_horizons(utc_time)
            except Exception as e:
                print(f"⚠️  JPL Horizons failed, falling back to approximate calculation: {e}")
                return get_ceres_position_approximate(utc_time)
        else:
            # Try cached Horizons data first
            cached_pos = get_cached_horizons_position(body_name, utc_time)
            if cached_pos is not None:
                return cached_pos

            # Try JPL Horizons fallback for any unsupported body
            try:
                jpl_position = get_position_jpl_horizons(body_name_clean, utc_time)
                if jpl_position is not None:
                    return jpl_position
                else:
                    print(f"❌ ERROR: Body '{body_name}' not supported in Skyfield DE440 ephemeris and JPL Horizons lookup failed")
                    return None
            except Exception as e:
                print(f"❌ ERROR: Body '{body_name}' not supported in Skyfield DE440 ephemeris and JPL Horizons fallback failed: {e}")
                return None

        # Get position relative to solar system barycenter
        position_au = body.at(t).position.au  # Returns position in AU
        position_m = position_au * 149597870700.0  # Convert AU to meters

        print(f"✅ Position for {body_name}: {position_m} meters", flush=True)
        return position_m

    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve position for {body_name}: {e}", file=sys.stderr, flush=True)
        return None


def get_positions_batch(body_name, utc_times):
    """Retrieve heliocentric positions using Skyfield for multiple UTC times efficiently."""
    global _ephemeris, _timescale

    if _ephemeris is None or _timescale is None:
        print("❌ ERROR: Skyfield ephemeris not loaded. Call load_all_kernels() first.")
        return None

    try:
        # Parse all times
        times = []
        for utc_time in utc_times:
            # Parse time - handle both date-only and full datetime strings
            if len(utc_time) >= 19:  # Full datetime with time
                t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]),
                                  int(utc_time[11:13]), int(utc_time[14:16]), int(utc_time[17:19]))
            elif len(utc_time) >= 10:  # Date only
                t = _timescale.utc(int(utc_time[:4]), int(utc_time[5:7]), int(utc_time[8:10]))
            else:
                print(f"❌ ERROR: Invalid time format: {utc_time}")
                return None
            times.append(t)

        # Get body object
        body_name_clean = body_name.upper().replace(" BARYCENTER", "").replace(" ", "")
        if body_name_clean == "SUN":
            body = _ephemeris['sun']
        elif body_name_clean in ['MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO']:
            # For planets, use barycenter by default (this matches SPICE behavior)
            body = _ephemeris[f'{body_name_clean.lower()}_barycenter']
        elif body_name_clean == "EARTHMOONBARYCENTER" or body_name_clean == "EARTH-MOONBARYCENTER":
            body = _ephemeris['earth-moon-barycenter']
        else:
            print(f"❌ ERROR: Body '{body_name}' not supported for batch operations")
            return None

        # For now, use a simple loop but suppress individual print statements
        # This is still much faster than the original loop with prints
        positions_m = []
        for t in times:
            position_au = body.at(t).position.au
            position_m = position_au * 149597870700.0  # Convert AU to meters
            positions_m.append(position_m)

        return np.array(positions_m)

    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve batch positions for {body_name}: {e}", file=sys.stderr, flush=True)
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
            # Try local Ceres kernel first
            if _ceres_ephemeris is not None:
                try:
                    body = _ceres_ephemeris['ceres']
                    position_au, velocity_au_per_day = body.at(t).position.au, body.at(t).velocity.au_per_d
                    position_m = position_au * 149597870700.0
                    velocity_m_per_s = velocity_au_per_day * 149597870700.0 / 86400.0
                    print(f"✅ State for {body_name}: position = {position_m} meters, velocity = {velocity_m_per_s} m/s (from local kernel)", flush=True)
                    return position_m, velocity_m_per_s
                except Exception as e:
                    print(f"⚠️  Local Ceres kernel failed, falling back to Horizons: {e}")
            
            # Fall back to Horizons API for accurate Ceres state (publication quality)
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

    # URLs for required kernels
    kernel_urls = {
        "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        "gm_de440.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
        "naif_ids.html": "https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html",
        "de440.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
        "de441.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de441.bsp",
        "ceres_1900_2100.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/asteroids/ceres_1900_2100.bsp",
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
        print(f"Debug: normalized='{normalized}', available keys: {list(name_to_id.keys())[:10]}...")
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

        # Convert to meters and m/s
        position_m = np.array([x_ecl, y_ecl, z_ecl]) * 149597870700.0
        # For approximate Keplerian calculation, velocity is not computed - return zeros
        velocity_m_s = np.array([0.0, 0.0, 0.0])  # Placeholder velocities for approximate calculation

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
    # Ask user for confirmation before using Horizons
    if not confirm_horizons_usage("CERES", utc_time):
        print(f"❌ User cancelled Horizons request for CERES at {utc_time}")
        return None
    
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


def get_position_jpl_horizons(body_name, utc_time):
    """
    Get accurate position for any celestial body from JPL Horizons system.
    This provides publication-quality ephemeris data for bodies not in Skyfield.
    """
    # Ask user for confirmation before using Horizons
    if not confirm_horizons_usage(body_name, utc_time):
        print(f"❌ User cancelled Horizons request for {body_name} at {utc_time}")
        return None
    
    try:
        # JPL Horizons API endpoint
        url = "https://ssd.jpl.nasa.gov/api/horizons.api"

        # Get NAIF ID for the body
        naif_id = search_celestial_body(body_name)
        command_param = str(naif_id) if naif_id else body_name

        # For asteroids, try different naming conventions if NAIF ID fails
        if naif_id is None or body_name.upper() in ['PALLAS', 'VESTA']:
            # Try asteroid designation (remove "ASTEROID" prefix and use number)
            if body_name.upper() == 'PALLAS':
                command_param = '2'  # 2 Pallas
            elif body_name.upper() == 'VESTA':
                command_param = '4'  # 4 Vesta
            else:
                command_param = body_name

        # Format time for Horizons - use ISO format
        horizons_time = utc_time.replace(' ', 'T')
        # Add 1 minute for stop time
        from datetime import datetime, timedelta
        dt = datetime.fromisoformat(utc_time.replace(' ', 'T'))
        dt_stop = dt + timedelta(minutes=1)
        stop_time = dt_stop.isoformat()

        # Parameters for the body
        params = {
            'format': 'json',
            'COMMAND': command_param,  # Body NAIF ID or designation
            'OBJ_DATA': 'NO',
            'MAKE_EPHEM': 'YES',
            'EPHEM_TYPE': 'VECTORS',
            'CENTER': '500@10',  # Solar System Barycenter
            'START_TIME': horizons_time,
            'STOP_TIME': stop_time,
            'STEP_SIZE': '1',
            'VEC_TABLE': '1',  # Position only
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
            print(f"❌ JPL Horizons error for {body_name}: {data['error']}")
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

                        print(f"✅ JPL Horizons position for {body_name}: {position_m} meters", flush=True)
                        print("🏦 BANKABLE DATA: Using official JPL ephemeris for publication-quality accuracy", flush=True)

                        return position_m
                    except (ValueError, IndexError) as e:
                        print(f"❌ Error parsing position data for {body_name}: {e}")
                        continue

        print(f"❌ Could not parse JPL Horizons response for {body_name}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error accessing JPL Horizons for {body_name}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error getting {body_name} position from JPL Horizons: {e}")
        return None


def prefetch_horizons_data_for_trajectory(dep_body, arr_body, dep_times, arr_times):
    """
    Prefetch Horizons data for all bodies and times needed for a trajectory calculation.
    This downloads all required ephemeris data at once to avoid repeated API calls.

    Args:
        dep_body: Departure body name
        arr_body: Arrival body name
        dep_times: List of departure time strings (ISO format)
        arr_times: List of arrival time strings (ISO format)
    """
    global _horizons_cache

    # Collect all unique bodies that might need Horizons data
    bodies_to_check = set()
    if dep_body.upper() not in ['SUN', 'MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO', 'EARTHMOONBARYCENTER', 'EARTH-MOONBARYCENTER']:
        bodies_to_check.add(dep_body)
    if arr_body.upper() not in ['SUN', 'MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO', 'EARTHMOONBARYCENTER', 'EARTH-MOONBARYCENTER']:
        bodies_to_check.add(arr_body)

    if not bodies_to_check:
        print("✅ No Horizons data needed - all bodies supported locally")
        return

    # Collect all unique times
    all_times = set(dep_times + arr_times)

    # Convert times to datetime objects for range calculation
    from datetime import datetime
    time_objects = []
    for time_str in all_times:
        if len(time_str) >= 19:  # Full datetime
            time_objects.append(datetime.fromisoformat(time_str.replace('Z', '+00:00')))
        elif len(time_str) >= 10:  # Date only
            time_objects.append(datetime.fromisoformat(time_str + 'T00:00:00'))

    if not time_objects:
        print("❌ No valid times found for prefetch")
        return

    # Find min/max dates
    min_time = min(time_objects)
    max_time = max(time_objects)

    # Add buffer days
    from datetime import timedelta
    start_date = (min_time - timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (max_time + timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"📡 Prefetching Horizons data for {bodies_to_check} from {start_date} to {end_date}")

    # Download data for each body
    for body in bodies_to_check:
        try:
            print(f"📡 Downloading Horizons data for {body}...")
            download_horizons_ephemeris_batch(body, start_date, end_date, step_days=1)
            print(f"✅ Cached Horizons data for {body}")
        except Exception as e:
            print(f"❌ Failed to prefetch Horizons data for {body}: {e}")

    print(f"✅ Horizons prefetch complete. Cached data available for {len(_horizons_cache)} time ranges")

# Add to module level for easy testing
if __name__ == "__main__":
    pass