# spice_interface.py - API compatibility layer for Skyfield backend
import skyfield_interface as sf

# Delegate all functions to Skyfield interface
load_all_kernels = sf.load_all_kernels
get_position = sf.get_position
get_positions_batch = sf.get_positions_batch
get_state = sf.get_state
get_lambert_inputs = sf.get_lambert_inputs
check_and_download_kernels = sf.check_and_download_kernels
search_celestial_body = sf.search_celestial_body
fetch_gm = sf.fetch_gm
parse_naif_ids = sf.parse_naif_ids
parse_gm_values = sf.parse_gm_values
check_file_exists = sf.check_file_exists
str2et = sf.str2et
et2utc = sf.et2utc
prefetch_horizons_data_for_trajectory = sf.prefetch_horizons_data_for_trajectory