import re
import os
import spiceypy as spice
import time
import sys
import importlib
import urllib.request
import ssl
import zipfile
import shutil

# Reload spiceypy to ensure latest version
importlib.reload(spice)

# Global file paths
KERNELS_DIR = r"C:\\Users\\letsf\\Documents\\Coding\\Python\\Lamberts\\kernels"
BSP_DIR = os.path.join(KERNELS_DIR, "BSP")
GM_FILE = os.path.join(KERNELS_DIR, "gm_de440.tpc")
NAIF_FILE = os.path.join(KERNELS_DIR, "naif_ids.html")

def load_all_kernels():
    """Loads all SPICE kernels, including leap seconds kernel."""
    print("\n--- Loading SPICE Kernels ---\n", flush=True)
    spice.kclear()
    
    # Load Leap Seconds Kernel (LSK)
    leap_seconds_file = os.path.join(KERNELS_DIR, "naif0012.tls")
    if os.path.exists(leap_seconds_file):
        spice.furnsh(leap_seconds_file)
        print(f"✅ Loaded Leap Seconds Kernel: {os.path.basename(leap_seconds_file)}", flush=True)
    else:
        print("❌ ERROR: Leap Seconds Kernel (naif0012.tls) not found!", flush=True)
    
    # Load all SPK (Ephemeris) files from BSP directory
    spk_files = [
        os.path.join(BSP_DIR, f)
        for f in os.listdir(BSP_DIR)
        if f.endswith('.bsp')
    ]
    
    if not spk_files:
        print("❌ ERROR: No SPK files found in BSP directory! Position calculations will fail.")
    else:
        print(f"🔄 Found {len(spk_files)} SPK files. Loading them now...")
    
    for spk in spk_files:
        spice.furnsh(spk)
        print(f"✅ Loaded SPK file: {os.path.basename(spk)}", flush=True)
    
    # Load all other necessary files (.tpc, etc.)
    kernel_files = [
        os.path.join(KERNELS_DIR, f)
        for f in os.listdir(KERNELS_DIR)
        if f.endswith('.tpc')
    ]

    if not kernel_files:
        print("❌ ERROR: No TPC kernels found in the directory!")
    else:
        print(f"🔄 Found {len(kernel_files)} TPC kernel files. Loading them now...")
    
    for kernel in kernel_files:
        spice.furnsh(kernel)
        print(f"✅ Loaded kernel: {os.path.basename(kernel)}", flush=True)
    
    # Verify loaded kernels
    total_kernels = spice.ktotal("ALL")
    print(f"\n✅ SPICE Loaded {total_kernels} total kernels.\n")
    
    for i in range(total_kernels):
        try:
            kernel_data = spice.kdata(i, "ALL", 256, 256, 256)
            kernel_name = kernel_data[0]  # Extract only the file name
            print(f" - {kernel_name}", flush=True)
        except Exception as e:
            print(f"❌ ERROR: Failed to retrieve kernel data at index {i}: {e}", file=sys.stderr, flush=True)


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
    """Retrieve heliocentric position from SPICE at a UTC time."""
    print(f"\nRetrieving position for {body_name} at {utc_time}...", flush=True)
    try:
        et = spice.str2et(utc_time)
        state, _ = spice.spkezr(body_name.upper(), et, "J2000", "NONE", "SUN")
        position = state[:3] * 1000  # Convert km to meters
        print(f"✅ Position for {body_name}: {position} meters", flush=True)
        return position
    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve position for {body_name}: {e}", file=sys.stderr, flush=True)
        return None
    
def get_state(body_name, utc_time):
    """Retrieve heliocentric state (position and velocity) from SPICE at a UTC time."""
    import spiceypy as spice
    print(f"\nRetrieving state for {body_name} at {utc_time}...", flush=True)
    try:
        et = spice.str2et(utc_time)
        state, _ = spice.spkezr(body_name.upper(), et, "J2000", "NONE", "SUN")
        position = state[:3] * 1000  # km to meters
        velocity = state[3:] * 1000  # km/s to m/s
        print(f"✅ State for {body_name}: position = {position} meters, velocity = {velocity} m/s", flush=True)
        return position, velocity
    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve state for {body_name}: {e}", flush=True)
        return None, None    
    
def get_lambert_inputs(departure_body, arrival_body, departure_time, tof_days):
    """Gather all inputs needed for the Lambert solver."""
    tof_seconds = tof_days * 86400  # Convert days to seconds
    arrival_time = spice.et2utc(spice.str2et(departure_time) + tof_seconds, "C", 3)
    
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
    """Download and update SPICE kernels from NASA."""
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
        # "mar097.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/mar097.bsp"  # URL not found
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


def check_file_exists(filepath):
    """Checks if a file exists and provides instructions if not."""
    if not os.path.exists(filepath):
        print(f"Error: The required file '{filepath}' was not found.")
        print("Please download the necessary file and place it in the correct directory.")
        exit(1)
