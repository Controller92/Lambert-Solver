# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
# Ensure the directory containing this script is on sys.path so local imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import trajectory
from trajectory import porkchop_data, porkchop_data_enhanced, porkchop_data_gravity_assist, porkchop_data_gravity_assist_enhanced, extract_optimal_trajectory, plot_trajectory_3d
import spice_interface
from datetime import datetime, timedelta
import skyfield.api as sf
from skyfield.api import load
from astropy.time import Time
import sys
import os
import threading

def jd_to_date(jd):
    """Convert Julian Date to YYYY-MM-DD string"""
    # Using Skyfield for accurate conversion
    ts = sf.load.timescale()
    t = ts.tdb(jd=jd)
    dt = t.utc_datetime()
    return dt.strftime("%Y-%m-%d")

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="yellow", relief="solid", borderwidth=1, wraplength=300, justify="left")
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class DatePicker:
    def __init__(self, parent, default_date="2035-01-01"):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Date entry field
        self.date_var = tk.StringVar(value=default_date)
        self.date_entry = ttk.Entry(self.frame, textvariable=self.date_var, width=12)
        self.date_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.date_entry.bind("<KeyRelease>", lambda e: self.validate_date())
        
        # Calendar button
        self.calendar_button = ttk.Button(self.frame, text="📅", width=3, command=self.show_calendar)
        self.calendar_button.pack(side=tk.LEFT)
        
        self.calendar_window = None
        
    def show_calendar(self):
        if self.calendar_window:
            self.calendar_window.destroy()
        
        self.calendar_window = tk.Toplevel(self.parent)
        self.calendar_window.title("Select Date")
        self.calendar_window.geometry("250x200")
        self.calendar_window.resizable(False, False)
        
        # Simple calendar using spinboxes
        ttk.Label(self.calendar_window, text="Select Date:").pack(pady=10)
        
        cal_frame = ttk.Frame(self.calendar_window)
        cal_frame.pack(pady=5)
        
        # Parse current date
        try:
            year, month, day = map(int, self.date_var.get().split('-'))
        except:
            year, month, day = 2035, 1, 1
        
        # Year spinner
        ttk.Label(cal_frame, text="Year:").grid(row=0, column=0, padx=5)
        self.year_var = tk.IntVar(value=year)
        year_spin = tk.Spinbox(cal_frame, from_=2020, to=2050, textvariable=self.year_var, width=6)
        year_spin.grid(row=0, column=1, padx=5)
        
        # Month spinner
        ttk.Label(cal_frame, text="Month:").grid(row=1, column=0, padx=5)
        self.month_var = tk.IntVar(value=month)
        month_spin = tk.Spinbox(cal_frame, from_=1, to=12, textvariable=self.month_var, width=6)
        month_spin.grid(row=1, column=1, padx=5)
        
        # Day spinner
        ttk.Label(cal_frame, text="Day:").grid(row=2, column=0, padx=5)
        self.day_var = tk.IntVar(value=day)
        day_spin = tk.Spinbox(cal_frame, from_=1, to=31, textvariable=self.day_var, width=6)
        day_spin.grid(row=2, column=1, padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(self.calendar_window)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="OK", command=self.set_date).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.calendar_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Center the window
        self.calendar_window.transient(self.parent)
        self.calendar_window.grab_set()
        
    def set_date(self):
        year = self.year_var.get()
        month = self.month_var.get()
        day = self.day_var.get()
        
        # Basic validation
        if month in [4, 6, 9, 11] and day > 30:
            day = 30
        elif month == 2:
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                if day > 29:
                    day = 29
            else:
                if day > 28:
                    day = 28
        
        # Format date as YYYY-MM-DD
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        self.date_var.set(date_str)
        self.calendar_window.destroy()
        
    def validate_date(self):
        """Validate date format and provide visual feedback"""
        date_str = self.date_var.get()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            self.date_entry.config(foreground="black")  # Valid date
            return True
        except ValueError:
            self.date_entry.config(foreground="red")  # Invalid date
            return False
        
    def get(self):
        return self.date_var.get()
        
    def set(self, date_str):
        self.date_var.set(date_str)

class BodySearchDialog:
    def __init__(self, parent, callback):
        self.parent = parent
        self.callback = callback
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Search Celestial Bodies")
        self.dialog.geometry("500x400")
        self.dialog.resizable(True, True)
        
        # Search frame
        search_frame = ttk.Frame(self.dialog, padding="10")
        search_frame.pack(fill=tk.X)
        
        ttk.Label(search_frame, text="Search for celestial body:").pack(anchor=tk.W)
        
        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.pack(fill=tk.X, pady=(5, 0))
        self.search_entry.bind("<KeyRelease>", self.on_search)
        self.search_entry.focus()
        
        # Results frame
        results_frame = ttk.Frame(self.dialog, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(results_frame, text="Matching bodies:").pack(anchor=tk.W)
        
        # Results listbox with scrollbar
        list_frame = ttk.Frame(results_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_listbox = tk.Listbox(list_frame, height=15, yscrollcommand=scrollbar.set)
        self.results_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_listbox.yview)
        
        # Bind double-click to select
        self.results_listbox.bind("<Double-Button-1>", self.on_select)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog, padding="10")
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Select", command=self.on_select).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
        
        # Load all available bodies
        self.load_all_bodies()
        
        # Initial search
        self.on_search()
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
    
    def load_all_bodies(self):
        """Load all available celestial bodies from NAIF IDs."""
        try:
            import spice_interface
            # Parse NAIF IDs to get all available bodies
            id_to_name, name_to_id = spice_interface.parse_naif_ids(r"C:\Users\letsf\Documents\Coding\Python\Lamberts\kernels\naif_ids.html")
            
            # Create a list of body names, preferring common names
            self.all_bodies = []
            self.body_name_to_id = {}
            
            for naif_id, names in id_to_name.items():
                for name in names:
                    clean_name = name.strip()
                    if clean_name and len(clean_name) > 1:  # Skip single character names
                        self.all_bodies.append(clean_name)
                        self.body_name_to_id[clean_name.lower()] = naif_id
            
            # Sort alphabetically
            self.all_bodies.sort()
            print(f"[DEBUG] Loaded {len(self.all_bodies)} celestial bodies")
            print(f"[DEBUG] First 10: {self.all_bodies[:10]}")
            print(f"[DEBUG] Bodies containing 'CERES': {[b for b in self.all_bodies if 'CERES' in b.upper()]}")
            
        except Exception as e:
            print(f"Error loading body list: {e}")
            self.all_bodies = ["Error loading bodies"]
            self.body_name_to_id = {}
    
    def on_search(self, event=None):
        """Filter bodies based on search text."""
        search_text = self.search_var.get().lower().strip()
        
        if not search_text:
            # Show all bodies (no limit for initial view)
            filtered_bodies = self.all_bodies
        else:
            # Filter bodies containing the search text
            filtered_bodies = [body for body in self.all_bodies if search_text in body.lower()]
            # Limit to 100 results
            filtered_bodies = filtered_bodies[:100]
        
        # Update listbox
        self.results_listbox.delete(0, tk.END)
        for body in filtered_bodies:
            self.results_listbox.insert(tk.END, body)
        
        # Select first item if available
        if filtered_bodies:
            self.results_listbox.selection_set(0)
    
    def on_select(self, event=None):
        """Handle body selection."""
        selection = self.results_listbox.curselection()
        if selection:
            body_name = self.results_listbox.get(selection[0])
            body_id = self.body_name_to_id.get(body_name.lower())
            
            if body_id is not None:
                # Call the callback with the selected body
                self.callback(body_name, body_id)
                self.dialog.destroy()
            else:
                messagebox.showerror("Error", f"Could not find ID for body: {body_name}")

class TrajectoryApp:
    def __init__(self, root):
        print("🚀 [DEBUG] TrajectoryApp.__init__ starting...")
        
        self.root = root
        print("✅ [DEBUG] Root window assigned")
        
        self.root.title("Interplanetary Trajectory Calculator")
        print("✅ [DEBUG] Window title set")
        
        # Add emergency exit keyboard shortcuts
        self.root.bind('<Control-q>', lambda e: self.emergency_exit())
        self.root.bind('<Control-x>', lambda e: self.emergency_exit())
        self.root.bind('<Alt-F4>', lambda e: self.emergency_exit())
        self.root.protocol("WM_DELETE_WINDOW", self.emergency_exit)
        print("✅ [DEBUG] Emergency exit shortcuts added (Ctrl+Q, Ctrl+X, Alt+F4)")
        
        # Check for GPU acceleration
        print("🔍 [DEBUG] Checking GPU acceleration...")
        try:
            import cupy as cp
            self.gpu_available = True
            gpu_status = "GPU acceleration available ✓"
        except ImportError:
            self.gpu_available = False
            gpu_status = "CPU only mode (GPU acceleration not available)"
        
        # Common name to SPICE name mapping
        self.body_name_map = {
            "Mercury": "MERCURY BARYCENTER",
            "Venus": "VENUS BARYCENTER", 
            "Earth": "EARTH",
            "Mars": "MARS BARYCENTER",
            "Jupiter": "JUPITER BARYCENTER",
            "Saturn": "SATURN BARYCENTER",
            "Uranus": "URANUS BARYCENTER",
            "Neptune": "NEPTUNE BARYCENTER",
            "Pluto": "PLUTO BARYCENTER",
            "Sun": "SUN"
        }
        self.spice_to_common = {v: k for k, v in self.body_name_map.items()}
        
        # Create main frames for organization
        self.setup_frame = ttk.LabelFrame(root, text="Mission Setup", padding="10")
        self.setup_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        # Status bar at the top
        self.status_frame = ttk.Frame(root, padding="5")
        self.status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10,0))
        
        self.status_label = ttk.Label(self.status_frame, text=gpu_status, font=("Arial", 9))
        self.status_label.grid(row=0, column=0, sticky="w")
        
        # Kernel status
        self.kernel_status = ttk.Label(self.status_frame, text="SPICE kernels: Loading...", font=("Arial", 9))
        self.kernel_status.grid(row=0, column=1, sticky="e")
        
        # Emergency exit hint
        self.exit_hint = ttk.Label(self.status_frame, text="💡 Emergency exit: Ctrl+Q", font=("Arial", 8), foreground="gray")
        self.exit_hint.grid(row=0, column=2, sticky="e", padx=(10, 0))
        ToolTip(self.exit_hint, "If the GUI freezes, use Ctrl+Q, Ctrl+X, or Alt+F4 to force exit")
        
        self.setup_frame = ttk.LabelFrame(root, text="Mission Setup", padding="10")
        self.setup_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        # Load SPICE kernels
        print("🌌 [DEBUG] Starting kernel loading...")
        try:
            spice_interface.load_all_kernels()
            self.kernels_loaded = True
            self.kernel_status.config(text="SPICE kernels: Loaded ✓", foreground="green")
            print("✅ [DEBUG] Kernels loaded successfully, kernels_loaded = True")
        except Exception as e:
            print(f"❌ [DEBUG] Kernel loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.kernel_status.config(text="SPICE kernels: Failed to load", foreground="red")
            messagebox.showwarning("Kernel Warning", f"SPICE kernels failed to load: {e}. Attempting to update...")
            self.update_kernels()
            self.kernels_loaded = False
            print("❌ [DEBUG] Kernels failed to load, kernels_loaded = False")
        
        # Set up Horizons confirmation callback
        self.setup_horizons_callback()
        
        # Create tabbed interface for main controls
        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        # Mission Setup Tab
        self.setup_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.setup_tab, text="Mission Setup")
        
        # Calculation Parameters Tab
        self.params_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.params_tab, text="Calculation Parameters")
        
        # Research Optimization Tab
        self.research_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.research_tab, text="Research Mode")
        
        # Initialize solve type variable
        self.solve_type_var = tk.StringVar(value="high")
        
        # Initialize body_ids dictionary for caching body lookups
        self.body_ids = {}
        
        # Mission Setup Section (moved to tab)
        tk.Label(self.setup_tab, text="Mission Preset:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.mission_preset = ttk.Combobox(self.setup_tab, values=[
            "Custom Mission", "Earth → Mars", "Earth → Venus", "Mars → Earth", 
            "Earth → Jupiter", "Earth → Saturn", "Earth → Pluto"
        ], state="readonly", width=18)
        self.mission_preset.grid(row=0, column=1, padx=5, pady=2)
        self.mission_preset.set("Earth → Mars")
        self.mission_preset.bind("<<ComboboxSelected>>", self.apply_preset)
        ToolTip(self.mission_preset, "Choose from common mission scenarios or select 'Custom Mission' to set your own parameters")
        
        tk.Label(self.setup_tab, text="Departure Date:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w")
        self.start_date_picker = DatePicker(self.setup_tab, "2035-01-01")
        self.start_date_picker.frame.grid(row=1, column=1, padx=5, pady=2)
        ToolTip(self.start_date_picker.frame, "Select the earliest possible departure date for your mission (YYYY-MM-DD format)")
        
        tk.Label(self.setup_tab, text="Arrival Date:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w")
        self.end_date_picker = DatePicker(self.setup_tab, "2035-12-31")
        self.end_date_picker.frame.grid(row=2, column=1, padx=5, pady=2)
        ToolTip(self.end_date_picker.frame, "Select the latest possible arrival date for your mission (YYYY-MM-DD format)")
        
        tk.Label(self.setup_tab, text="Departure Body:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w")
        
        # Departure body frame with combobox and search button
        dep_body_frame = ttk.Frame(self.setup_tab)
        dep_body_frame.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        
        self.dep_body = ttk.Combobox(dep_body_frame, values=self.get_body_list(), width=15, state="normal")
        self.dep_body.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.dep_body.set("Earth")
        
        dep_search_btn = ttk.Button(dep_body_frame, text="🔍", width=3, command=self.search_departure_body)
        dep_search_btn.pack(side=tk.LEFT, padx=(2, 0))
        ToolTip(dep_search_btn, "Search for other celestial bodies (asteroids, minor planets, etc.)")
        
        ToolTip(self.dep_body, "Choose the celestial body you're departing from (typically Earth for human missions)")
        
        tk.Label(self.setup_tab, text="Arrival Body:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w")
        
        # Arrival body frame with combobox and search button
        arr_body_frame = ttk.Frame(self.setup_tab)
        arr_body_frame.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        
        self.arr_body = ttk.Combobox(arr_body_frame, values=self.get_body_list(), width=15, state="normal")
        self.arr_body.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.arr_body.set("Mars")
        
        arr_search_btn = ttk.Button(arr_body_frame, text="🔍", width=3, command=self.search_arrival_body)
        arr_search_btn.pack(side=tk.LEFT, padx=(2, 0))
        ToolTip(arr_search_btn, "Search for other celestial bodies (asteroids, minor planets, etc.)")
        
        ToolTip(self.arr_body, "Choose the celestial body you're traveling to (Mars is a common destination)")
        
        # Gravity Assist Section
        tk.Label(self.setup_tab, text="Gravity Assist:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky="w")
        self.gravity_assist_var = tk.BooleanVar(value=False)
        self.gravity_assist_check = ttk.Checkbutton(self.setup_tab, text="Enable Gravity Assist", 
                                                   variable=self.gravity_assist_var, 
                                                   command=self.toggle_gravity_assist)
        self.gravity_assist_check.grid(row=5, column=1, padx=5, pady=2, sticky="w")
        ToolTip(self.gravity_assist_check, "Enable multi-body trajectory with planetary gravity assist for more efficient transfers")
        
        tk.Label(self.setup_tab, text="Flyby Body:", font=("Arial", 10, "bold")).grid(row=6, column=0, sticky="w")
        
        # Flyby body frame with combobox and search button
        flyby_body_frame = ttk.Frame(self.setup_tab)
        flyby_body_frame.grid(row=6, column=1, padx=5, pady=2, sticky="ew")
        
        self.flyby_body = ttk.Combobox(flyby_body_frame, values=self.get_body_list(), width=15, state="disabled")
        self.flyby_body.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.flyby_body.set("Mars")
        
        flyby_search_btn = ttk.Button(flyby_body_frame, text="🔍", width=3, command=self.search_flyby_body, state="disabled")
        flyby_search_btn.pack(side=tk.LEFT, padx=(2, 0))
        ToolTip(flyby_search_btn, "Search for other celestial bodies for gravity assist (asteroids, minor planets, etc.)")
        
        ToolTip(self.flyby_body, "Choose the celestial body for gravity assist flyby (typically a planet like Mars or Venus)")
        
        # Calculation Parameters Section (moved to tab)
        tk.Label(self.params_tab, text="Min Transit Time (days):", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.min_time = tk.Entry(self.params_tab, width=20)
        self.min_time.grid(row=0, column=1, padx=5, pady=2)
        self.min_time.insert(0, "100")
        self.min_time.bind("<KeyRelease>", self.on_time_change)
        ToolTip(self.min_time, "Minimum time for the journey (days). Shorter trips require more fuel but are faster.")
        
        tk.Label(self.params_tab, text="Max Transit Time (days):", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w")
        self.max_time = tk.Entry(self.params_tab, width=20)
        self.max_time.grid(row=1, column=1, padx=5, pady=2)
        self.max_time.insert(0, "300")
        self.max_time.bind("<KeyRelease>", self.on_time_change)
        ToolTip(self.max_time, "Maximum time for the journey (days). Longer trips use less fuel but take more time.")
        
        tk.Label(self.params_tab, text="Grid Resolution:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w")
        self.resolution = tk.Entry(self.params_tab, width=20)
        self.resolution.grid(row=2, column=1, padx=5, pady=2)
        self.resolution.insert(0, "5")
        self.resolution.bind("<KeyRelease>", self.on_resolution_change)
        ToolTip(self.resolution, "Number of grid points for calculation (higher = more accurate but slower). Start with 20-50.")
        
        # Solve Type Selection
        tk.Label(self.params_tab, text="Solve Type:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(10, 0))
        
        solve_type_frame = ttk.Frame(self.params_tab)
        solve_type_frame.grid(row=3, column=1, padx=5, pady=(10, 2), sticky="w")
        
        ttk.Radiobutton(solve_type_frame, text="Rough Estimate", variable=self.solve_type_var, 
                       value="rough", command=self.on_solve_type_change).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(solve_type_frame, text="High Confidence", variable=self.solve_type_var, 
                       value="high", command=self.on_solve_type_change).grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        ToolTip(ttk.Radiobutton(solve_type_frame, text="Rough Estimate", variable=self.solve_type_var, 
                               value="rough", command=self.on_solve_type_change), 
               "Fast calculation with lower resolution for quick trajectory assessment")
        ToolTip(ttk.Radiobutton(solve_type_frame, text="High Confidence", variable=self.solve_type_var, 
                               value="high", command=self.on_solve_type_change), 
               "Detailed calculation with higher resolution and enhanced algorithms for mission planning")
        
        # Research Optimization Section (research tab)
        tk.Label(self.research_tab, text="Research Mode:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        tk.Label(self.research_tab, text="Enhanced optimization for publishable results", font=("Arial", 9)).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 15))
        
        tk.Label(self.research_tab, text="Time Splits to Test:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w")
        self.time_splits_entry = tk.Entry(self.research_tab, width=30)
        self.time_splits_entry.grid(row=2, column=1, padx=5, pady=2)
        self.time_splits_entry.insert(0, "0.3,0.4,0.5,0.6,0.7")
        ToolTip(self.time_splits_entry, "Comma-separated fractions of total time in first trajectory arc (0.3 = 30% in first arc)")
        
        tk.Label(self.research_tab, text="Flyby Altitudes (km):", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w")
        self.flyby_altitudes_entry = tk.Entry(self.research_tab, width=30)
        self.flyby_altitudes_entry.grid(row=3, column=1, padx=5, pady=2)
        self.flyby_altitudes_entry.insert(0, "300,500,750,1000,1500,2000")
        ToolTip(self.flyby_altitudes_entry, "Comma-separated flyby altitudes in km to test for optimal ΔV")
        
        tk.Label(self.research_tab, text="Research Resolution:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w")
        self.research_resolution = tk.Entry(self.research_tab, width=20)
        self.research_resolution.grid(row=4, column=1, padx=5, pady=2, sticky="w")
        self.research_resolution.insert(0, "50")
        ToolTip(self.research_resolution, "Grid resolution for research optimization (higher = more accurate but much slower)")
        
        tk.Label(self.research_tab, text="Max Runtime (hours):", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky="w")
        self.max_runtime = tk.Entry(self.research_tab, width=20)
        self.max_runtime.grid(row=5, column=1, padx=5, pady=2, sticky="w")
        self.max_runtime.insert(0, "24")
        ToolTip(self.max_runtime, "Maximum computation time in hours (can resume later if interrupted)")
        
        # Research mode controls
        research_controls_frame = ttk.Frame(self.research_tab)
        research_controls_frame.grid(row=6, column=0, columnspan=2, pady=15)
        
        ttk.Button(research_controls_frame, text="Start Research Optimization", 
                  command=self.start_research_optimization).grid(row=0, column=0, padx=5)
        ToolTip(ttk.Button(research_controls_frame, text="Start Research Optimization", 
                          command=self.start_research_optimization), 
               "Run enhanced optimization for publishable research results (may take days)")
        
        ttk.Button(research_controls_frame, text="Save Results", 
                  command=self.save_research_results).grid(row=0, column=1, padx=5)
        ToolTip(ttk.Button(research_controls_frame, text="Save Results", 
                          command=self.save_research_results), 
               "Save optimization results to file for analysis")
        
        # Research results display
        self.research_results_text = tk.Text(self.research_tab, height=15, width=60, font=("Courier", 9))
        self.research_results_text.grid(row=7, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Progress & Results Section (left side)
        self.progress_frame = ttk.LabelFrame(root, text="Progress & Results", padding="10")
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=(10, 5), pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=5)
        self.progress_label = tk.Label(self.progress_frame, text="Ready to calculate - Click 'Generate Porkchop Plot' to start", font=("Arial", 9))
        self.progress_label.grid(row=1, column=0)
        
        # Controls Section (right side of progress)
        self.controls_frame = ttk.LabelFrame(root, text="Controls", padding="10")
        self.controls_frame.grid(row=2, column=1, sticky="ew", padx=(5, 10), pady=5)
        
        button_frame = ttk.Frame(self.controls_frame)
        button_frame.grid(row=0, column=0, pady=5)
        
        ttk.Button(button_frame, text="Estimate Time", command=self.estimate_time).grid(row=0, column=0, padx=5)
        ToolTip(ttk.Button(button_frame, text="Estimate Time", command=self.estimate_time), "Calculate how long the computation will take based on your resolution setting")
        
        ttk.Button(button_frame, text="Generate Porkchop Plot", command=self.start_porkchop).grid(row=0, column=1, padx=5)
        ToolTip(ttk.Button(button_frame, text="Generate Porkchop Plot", command=self.start_porkchop), "Create the trajectory plot showing fuel requirements for different departure/arrival times")
        
        ttk.Button(button_frame, text="Plot Optimal Trajectory", command=self.plot_optimal_trajectory).grid(row=0, column=2, padx=5)
        ToolTip(ttk.Button(button_frame, text="Plot Optimal Trajectory", command=self.plot_optimal_trajectory), "Display 3D visualization of the optimal trajectory path through space")
        
        ttk.Button(self.controls_frame, text="Update SPICE Kernels", command=self.update_kernels).grid(row=1, column=0, pady=5)
        ToolTip(ttk.Button(self.controls_frame, text="Update SPICE Kernels", command=self.update_kernels), "Download latest astronomical data for accurate calculations")
       
        # Trajectory Plot Section (always visible)
        self.plot_frame = ttk.LabelFrame(root, text="Trajectory Plot", padding="10")
        self.plot_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        
        # Create figure and add navigation toolbar
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Add navigation toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Pack toolbar and canvas
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add zoom recalculate button
        button_frame = ttk.Frame(self.plot_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Zoom & Recalculate", command=self.zoom_recalculate).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Reset View", command=self.reset_plot_view).pack(side=tk.LEFT)
        
        # Add instructions label
        instr_label = ttk.Label(button_frame, text="Use toolbar zoom tools, then click 'Zoom & Recalculate' for higher resolution", 
                               font=("Arial", 8), foreground="gray")
        instr_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Configure grid weights for proper resizing
        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        self.plot_frame.grid_rowconfigure(1, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        self.dates = None
        self.times = None
        self.dv = None
        self.dep_date = tk.Entry(root)  # Hidden for now, used in animation
        self.transit_time = tk.Entry(root)  # Hidden for now, used in animation

        # Dictionary to store SPICE IDs
        self.body_ids = {}
        
        # Store original plot data for zoom operations
        self.original_dates = None
        self.original_times = None
        self.original_dv = None
        
        # Apply initial preset
        self.apply_preset()

    def _fmt_num(self, x, fmt):
        """Safely format a numeric-like value for display.

        If x is a numpy scalar or 0-d array this returns a formatted string.
        If x is an array with multiple elements, return a short str() representation.
        If conversion fails, fall back to str(x).
        """
        try:
            arr = np.asarray(x)
            # 0-d or scalar-like
            if arr.shape == ():
                return format(float(arr.item()), fmt)
            # 1-D or multi-element: try to take first element
            try:
                return format(float(arr.ravel()[0]), fmt)
            except Exception:
                return str(arr)
        except Exception:
            try:
                return format(float(x), fmt)
            except Exception:
                return str(x)

    def toggle_gravity_assist(self):
        """Enable or disable gravity assist options"""
        state = "normal" if self.gravity_assist_var.get() else "disabled"
        self.flyby_body.config(state=state)
        self.flyby_body.set("Mars" if state == "normal" else "")
        
        # Find and enable/disable the flyby search button
        for child in self.setup_tab.winfo_children():
            if isinstance(child, ttk.Frame):  # flyby_body_frame
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Button) and widget.cget("text") == "🔍":
                        widget.config(state=state)
                        break

    def emergency_exit(self):
        """Emergency exit function for frozen GUI"""
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        import sys
        sys.exit(0)

    def setup_horizons_callback(self):
        """Set up the callback for Horizons confirmation dialogs."""
        import skyfield_interface
        skyfield_interface.set_horizons_confirmation_callback(self.confirm_horizons_usage)

    def confirm_horizons_usage(self, body_name, utc_time):
        """Show a confirmation dialog for using JPL Horizons data.
        
        Returns True if user approves, False if they cancel.
        This method is called from background threads, so we need to use after() to show the dialog.
        """
        # Use a threading event to wait for the dialog result
        import threading
        result_event = threading.Event()
        result = [False]  # Use list to allow modification in nested function
        
        def show_dialog():
            try:
                message = f"The system needs to download ephemeris data for {body_name} at {utc_time} from JPL Horizons.\n\n"
                message += "This requires an internet connection and may take a few seconds.\n\n"
                message += "Do you want to proceed?"
                
                user_choice = messagebox.askyesno("JPL Horizons Data Required", message, icon='question')
                result[0] = user_choice
            except Exception as e:
                print(f"Error showing Horizons confirmation dialog: {e}")
                result[0] = False  # Default to cancel on error
            finally:
                result_event.set()
        
        # Show dialog on main thread
        self.root.after(0, show_dialog)
        
        # Wait for the dialog to complete (with timeout)
        result_event.wait(timeout=60)  # Wait up to 60 seconds
        
        return result[0]

    def zoom_recalculate(self):
        """Recalculate with higher resolution based on current zoom level"""
        if self.original_dv is None:
            messagebox.showwarning("No Data", "Please run a calculation first before zooming.")
            return
        
        try:
            # Get current axis limits
            xlims = self.ax.get_xlim()
            ylims = self.ax.get_ylim()
            
            # Convert back to date strings and time values
            start_jd, end_jd = xlims
            min_time_days, max_time_days = ylims
            
            # Convert JD back to date strings
            start_date = jd_to_date(start_jd)
            end_date = jd_to_date(end_jd)
            
            # Calculate new resolution (higher for zoomed view)
            zoomed_width_days = (end_jd - start_jd)
            zoomed_height_days = (max_time_days - min_time_days)
            
            # Adaptive resolution based on zoom level
            base_res = 20
            zoom_factor = max(1, (365 / zoomed_width_days) * (300 / zoomed_height_days))  # Rough heuristic
            new_res = min(100, int(base_res * zoom_factor ** 0.5))  # Square root for gentler scaling
            
            # Update GUI fields with zoomed bounds
            self.start_date_picker.set(start_date)
            self.end_date_picker.set(end_date)
            self.min_time.delete(0, tk.END)
            self.min_time.insert(0, "02d")
            self.max_time.delete(0, tk.END)
            self.max_time.insert(0, "02d")
            self.resolution.delete(0, tk.END)
            self.resolution.insert(0, str(new_res))
            
            # Confirm with user
            message = f"Recalculate with zoomed parameters?\n\n"
            message += f"New date range: {start_date} to {end_date}\n"
            message += f"New time range: {min_time_days:.0f} to {max_time_days:.0f} days\n"
            message += f"New resolution: {new_res}×{new_res} points (higher detail)\n\n"
            message += "This will provide much finer detail in the zoomed region."
            
            if messagebox.askyesno("Zoom & Recalculate", message):
                # Trigger recalculation with new parameters
                self.start_porkchop()
                
        except Exception as e:
            messagebox.showerror("Zoom Error", f"Could not determine zoom bounds: {str(e)}")
    
    def reset_plot_view(self):
        """Reset plot to show full data range"""
        if self.original_dv is not None:
            self.ax.autoscale()
            self.canvas.draw()
        else:
            messagebox.showinfo("No Data", "Please run a calculation first.")
        
        # Store original plot data for zoom operations
        self.original_dates = None
        self.original_times = None
        self.original_dv = None
        
        # Apply initial preset
        self.apply_preset()
        
        # Apply initial preset
        self.apply_preset()

    def apply_preset(self, event=None):
        preset = self.mission_preset.get()
        
        if preset == "Custom Mission":
            # Don't change anything, let user set custom values
            return
        elif preset == "Earth → Mars":
            self.dep_body.set("Earth")
            self.arr_body.set("Mars")
            self.start_date_picker.set("2035-01-01")
            self.end_date_picker.set("2035-12-31")
            self.min_time.delete(0, tk.END)
            self.min_time.insert(0, "100")
            self.max_time.delete(0, tk.END)
            self.max_time.insert(0, "350")
            self.resolution.delete(0, tk.END)
            self.resolution.insert(0, "30")
        elif preset == "Earth → Venus":
            self.dep_body.set("Earth")
            self.arr_body.set("Venus")
            self.start_date_picker.set("2035-01-01")
            self.end_date_picker.set("2035-12-31")
            self.min_time.delete(0, tk.END)
            self.min_time.insert(0, "80")
            self.max_time.delete(0, tk.END)
            self.max_time.insert(0, "200")
            self.resolution.delete(0, tk.END)
            self.resolution.insert(0, "25")
        elif preset == "Mars → Earth":
            self.dep_body.set("Mars")
            self.arr_body.set("Earth")
            self.start_date_picker.set("2035-01-01")
            self.end_date_picker.set("2035-12-31")
            self.min_time.delete(0, tk.END)
            self.min_time.insert(0, "100")
            self.max_time.delete(0, tk.END)
            self.max_time.insert(0, "350")
            self.resolution.delete(0, tk.END)
            self.resolution.insert(0, "30")
        elif preset == "Earth → Jupiter":
            self.dep_body.set("Earth")
            self.arr_body.set("Jupiter")
            self.start_date_picker.set("2035-01-01")
            self.end_date_picker.set("2036-12-31")
            self.min_time.delete(0, tk.END)
            self.min_time.insert(0, "600")
            self.max_time.delete(0, tk.END)
            self.max_time.insert(0, "1200")
            self.resolution.delete(0, tk.END)
            self.resolution.insert(0, "20")
        elif preset == "Earth → Saturn":
            self.dep_body.set("Earth")
            self.arr_body.set("Saturn")
            self.start_date_picker.set("2035-01-01")
            self.end_date_picker.set("2038-12-31")
            self.min_time.delete(0, tk.END)
            self.min_time.insert(0, "1500")
            self.max_time.delete(0, tk.END)
            self.max_time.insert(0, "2500")
            self.resolution.delete(0, tk.END)
            self.resolution.insert(0, "15")
        elif preset == "Earth → Pluto":
            self.dep_body.set("Earth")
            self.arr_body.set("Pluto")
            self.start_date_picker.set("2035-01-01")
            self.end_date_picker.set("2045-12-31")
            self.min_time.delete(0, tk.END)
            self.min_time.insert(0, "3000")
            self.max_time.delete(0, tk.END)
            self.max_time.insert(0, "5000")
            self.resolution.delete(0, tk.END)
            self.resolution.insert(0, "10")

    def validate_date(self, date_str):
        """Validate date string format YYYY-MM-DD"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def validate_positive_number(self, value_str):
        """Validate that string represents a positive number"""
        try:
            val = float(value_str)
            return val > 0
        except ValueError:
            return False
    
    def validate_resolution(self, value_str):
        """Validate resolution is a positive integer"""
        try:
            val = int(value_str)
            return val > 0 and val <= 200
        except ValueError:
            return False
    
    def on_date_change(self, event=None):
        """Validate date inputs and provide feedback"""
        start_date = self.start_date_picker.get()
        end_date = self.end_date_picker.get()
        
        if not self.validate_date(start_date):
            self.progress_label.config(text="Warning: Invalid departure date format (use YYYY-MM-DD)", fg="orange")
            return False
        if not self.validate_date(end_date):
            self.progress_label.config(text="Warning: Invalid arrival date format (use YYYY-MM-DD)", fg="orange")
            return False
        
        # Check date order
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start_dt >= end_dt:
            self.progress_label.config(text="Warning: Departure date must be before arrival date", fg="orange")
            return False
        
        self.progress_label.config(text="Ready to calculate - Click 'Generate Porkchop Plot' to start", fg="black")
        return True
    
    def on_time_change(self, event=None):
        """Validate time range inputs"""
        try:
            min_t = float(self.min_time.get())
            max_t = float(self.max_time.get())
            
            if min_t <= 0 or max_t <= 0:
                self.progress_label.config(text="Warning: Transit times must be positive", fg="orange")
                return False
            if min_t >= max_t:
                self.progress_label.config(text="Warning: Minimum transit time must be less than maximum", fg="orange")
                return False
            
            self.progress_label.config(text="Ready to calculate - Click 'Generate Porkchop Plot' to start", fg="black")
            return True
        except ValueError:
            self.progress_label.config(text="Warning: Transit times must be valid numbers", fg="orange")
            return False
    
    def on_resolution_change(self, event=None):
        """Validate resolution input"""
        res_str = self.resolution.get()
        if not self.validate_resolution(res_str):
            self.progress_label.config(text="Warning: Resolution must be a number between 1-200", fg="orange")
            return False
        
        self.progress_label.config(text="Ready to calculate - Click 'Generate Porkchop Plot' to start", fg="black")
        return True
    
    def on_solve_type_change(self):
        """Update calculation parameters based on solve type selection"""
        solve_type = self.solve_type_var.get()
        
        if solve_type == "rough":
            # Rough estimate: lower resolution, faster calculation
            new_resolution = "20"
            tooltip_text = "Rough Estimate: Fast calculation (20×20 grid) for quick trajectory assessment"
        else:  # high confidence
            # High confidence: higher resolution, more detailed calculation
            new_resolution = "50"
            tooltip_text = "High Confidence: Detailed calculation (50×50 grid) for mission planning"
        
        # Update resolution field
        self.resolution.delete(0, tk.END)
        self.resolution.insert(0, new_resolution)
        
        # Update tooltip for resolution field
        # Note: We can't easily update existing tooltips, but the user will see the change in the field
        
        # Validate the new resolution
        self.on_resolution_change()
        
        # Update progress label to reflect solve type
        solve_name = "Rough Estimate" if solve_type == "rough" else "High Confidence"
        self.progress_label.config(text=f"{solve_name} mode selected - Click 'Generate Porkchop Plot' to start", fg="black")

    def get_body_list(self):
        print(f"[DEBUG] get_body_list called - kernels_loaded: {getattr(self, 'kernels_loaded', 'NOT SET')}")
        print(f"[DEBUG] body_name_map exists: {hasattr(self, 'body_name_map')}")
        if hasattr(self, 'body_name_map'):
            print(f"[DEBUG] body_name_map length: {len(self.body_name_map)}")

        if not getattr(self, 'kernels_loaded', False):
            print("[DEBUG] Kernels not loaded - showing warning and returning empty list")
            messagebox.showwarning("Kernel Warning", "SPICE kernels are not loaded. Please update kernels.")
            return []

        try:
            print("[DEBUG] Attempting to load body list...")
            available_bodies = []
            for common_name, spice_name in self.body_name_map.items():
                print(f"[DEBUG] Looking up {common_name} -> {spice_name}")
                body_id = spice_interface.search_celestial_body(spice_name)
                print(f"[DEBUG] search_celestial_body({spice_name}) = {body_id}")
                if body_id is not None:
                    available_bodies.append(common_name)
                    self.body_ids[spice_name] = body_id
                    print(f"[DEBUG] Added {common_name} to available bodies")
                else:
                    print(f"[DEBUG] Failed to find body_id for {spice_name}")

            result = sorted(available_bodies)
            print(f"[DEBUG] Returning {len(result)} bodies: {result}")
            return result

        except Exception as e:
            print(f"[DEBUG] Exception in get_body_list: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showwarning("Body List Error", f"Failed to load body list: {e}")
            return ["No bodies found"]

    def update_kernels(self):
        try:
            self.kernel_status.config(text="SPICE kernels: Downloading...", foreground="orange")
            self.root.update_idletasks()
            
            spice_interface.check_and_download_kernels()  # Assuming this exists
            spice_interface.load_all_kernels()
            self.kernels_loaded = True
            
            self.kernel_status.config(text="SPICE kernels: Updated ✓", foreground="green")
            messagebox.showinfo("Update Complete", "SPICE kernels have been successfully updated!\n\nThe latest planetary positions and orbital data are now available for accurate trajectory calculations.")
            
            self.dep_body['values'] = self.get_body_list()
            self.arr_body['values'] = self.get_body_list()
            
        except Exception as e:
            self.kernels_loaded = False
            self.kernel_status.config(text="SPICE kernels: Failed", foreground="red")
            messagebox.showerror("Update Failed", f"Could not download the latest astronomical data.\n\nYou can still use the program with existing data, but calculations may be less accurate.\n\nError: {str(e)}")

    def plot_optimal_trajectory(self):
        """Plot 3D visualization of the optimal trajectory."""
        try:
            # Check if porkchop data exists
            if not hasattr(self, 'dates') or self.dates is None:
                messagebox.showerror("No Data", "Please generate a porkchop plot first to identify the optimal trajectory.")
                return

            # Get current body selections
            dep_body_common = self.dep_body.get().strip()
            arr_body_common = self.arr_body.get().strip()
            
            # Convert common names to SPICE names
            dep_body = self.body_name_map.get(dep_body_common, dep_body_common)
            arr_body = self.body_name_map.get(arr_body_common, arr_body_common)
            
            flyby_body = None
            if self.use_gravity_assist:
                flyby_body_common = self.flyby_body.get().strip()
                flyby_body = self.body_name_map.get(flyby_body_common, flyby_body_common)

            # Extract optimal trajectory
            trajectory_data = extract_optimal_trajectory(
                self.dates, self.times, self.dv, dep_body, flyby_body, arr_body
            )

            # Create 3D plot
            fig = plot_trajectory_3d(trajectory_data, dep_body_common, 
                                   flyby_body_common if flyby_body else None, 
                                   arr_body_common)

            if fig is None:
                messagebox.showerror("Plot Error", "Could not create trajectory plot. Please ensure matplotlib is installed.")
                return

            # Clear current plot and show trajectory
            # Clear the entire figure to remove any 2D plot elements and color bars
            self.fig.clear()
            self.ax = self.fig.add_subplot(111, projection='3d')  # Create 3D axes
            
            # Plot trajectory
            if trajectory_data['trajectory_positions'] is not None:
                traj_pos = trajectory_data['trajectory_positions']
                self.ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2],
                           'b-', linewidth=2, alpha=0.8, label='Spacecraft Trajectory')

            # Plot celestial bodies
            body_colors = {
                'Sun': 'yellow',
                'Mercury': 'gray',
                'Venus': 'orange',
                'Earth': 'blue',
                'Mars': 'red',
                'Jupiter': 'orange',
                'Saturn': 'goldenrod',
                'Uranus': 'lightblue',
                'Neptune': 'blue',
                'Ceres': 'gray',
                'Vesta': 'gray',
                'Pallas': 'gray'
            }

            # Plot departure body
            if trajectory_data['dep_pos'] is not None:
                color = body_colors.get(dep_body_common, 'gray')
                self.ax.scatter(trajectory_data['dep_pos'][0], trajectory_data['dep_pos'][1],
                               trajectory_data['dep_pos'][2], c=color, s=100, label=f'{dep_body_common} (Departure)')

            # Plot flyby body
            if flyby_body and trajectory_data['flyby_pos'] is not None:
                color = body_colors.get(flyby_body_common, 'gray')
                self.ax.scatter(trajectory_data['flyby_pos'][0], trajectory_data['flyby_pos'][1],
                               trajectory_data['flyby_pos'][2], c=color, s=100, label=f'{flyby_body_common} (Flyby)')

            # Plot arrival body
            if trajectory_data['arr_pos'] is not None:
                color = body_colors.get(arr_body_common, 'gray')
                self.ax.scatter(trajectory_data['arr_pos'][0], trajectory_data['arr_pos'][1],
                               trajectory_data['arr_pos'][2], c=color, s=100, label=f'{arr_body_common} (Arrival)')

            # Plot Sun at origin
            self.ax.scatter(0, 0, 0, c='yellow', s=200, marker='*', label='Sun')

            # Formatting
            self.ax.set_xlabel('X (km)')
            self.ax.set_ylabel('Y (km)')
            self.ax.set_zlabel('Z (km)')

            # Set equal aspect ratio
            max_range = 0
            for pos in [trajectory_data['dep_pos'], trajectory_data['arr_pos']]:
                if pos is not None:
                    max_range = max(max_range, np.max(np.abs(pos)))
            if flyby_body and trajectory_data['flyby_pos'] is not None:
                max_range = max(max_range, np.max(np.abs(trajectory_data['flyby_pos'])))

            self.ax.set_xlim([-max_range*1.1, max_range*1.1])
            self.ax.set_ylim([-max_range*1.1, max_range*1.1])
            self.ax.set_zlim([-max_range*1.1, max_range*1.1])

            # Title
            title = f"Optimal Trajectory: {dep_body_common}"
            if flyby_body:
                title += f" → {flyby_body_common} → {arr_body_common}"
            else:
                title += f" → {arr_body_common}"
            # Safely format numeric values which might be numpy types
            dv_min_val = trajectory_data.get('dv_min', None)
            tof_val = trajectory_data.get('tof_days', None)
            dv_min_str = self._fmt_num(dv_min_val/1000 if dv_min_val is not None else dv_min_val, '.2f')
            tof_str = self._fmt_num(tof_val, '.1f')
            title += f"\nΔV: {dv_min_str} km/s, TOF: {tof_str} days"
            self.ax.set_title(title)

            self.ax.legend()
            self.ax.grid(True, alpha=0.3)

            # Update the navigation toolbar for 3D plotting
            self.toolbar.update()
            
            # Update canvas
            self.canvas.draw()

            # Show trajectory details
            dep_date = jd_to_date(trajectory_data['dep_jd'])
            arr_date = jd_to_date(trajectory_data['arr_jd'])

            details = f"Optimal Trajectory Details:\n\n"
            details += f"Departure: {dep_date} from {dep_body_common}\n"
            if flyby_body:
                flyby_date = jd_to_date(trajectory_data['flyby_jd'])
                details += f"Flyby: {flyby_date} at {flyby_body_common}\n"
            details += f"Arrival: {arr_date} at {arr_body_common}\n"
            # Use safe formatting for numeric values that might be numpy types
            tof_days_str = self._fmt_num(trajectory_data.get('tof_days', None), '.1f')
            dv_required_str = self._fmt_num(trajectory_data.get('dv_min', None)/1000 if trajectory_data.get('dv_min', None) is not None else None, '.2f')
            details += f"Time of Flight: {tof_days_str} days\n"
            details += f"\u0394V Required: {dv_required_str} km/s\n"

            messagebox.showinfo("Trajectory Plotted", details)

        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not plot trajectory: {str(e)}")

    def search_departure_body(self):
        """Open search dialog for departure body selection."""
        def on_body_selected(body_name, body_id):
            # Update the combobox with the selected body
            current_values = list(self.dep_body['values'])
            if body_name not in current_values:
                current_values.append(body_name)
                self.dep_body['values'] = current_values
            self.dep_body.set(body_name)
            # Store the body ID for later use
            spice_name = body_name.upper().replace(" ", "")
            self.body_ids[spice_name] = body_id
        
        BodySearchDialog(self.root, on_body_selected)

    def search_arrival_body(self):
        """Open search dialog for arrival body selection."""
        def on_body_selected(body_name, body_id):
            # Update the combobox with the selected body
            current_values = list(self.arr_body['values'])
            if body_name not in current_values:
                current_values.append(body_name)
                self.arr_body['values'] = current_values
            self.arr_body.set(body_name)
            # Store the body ID for later use
            spice_name = body_name.upper().replace(" ", "")
            self.body_ids[spice_name] = body_id
        
        BodySearchDialog(self.root, on_body_selected)

    def search_flyby_body(self):
        """Open search dialog for flyby body selection."""
        def on_body_selected(body_name, body_id):
            # Update the combobox with the selected body
            current_values = list(self.flyby_body['values'])
            if body_name not in current_values:
                current_values.append(body_name)
                self.flyby_body['values'] = current_values
            self.flyby_body.set(body_name)
            # Store the body ID for later use
            spice_name = body_name.upper().replace(" ", "")
            self.body_ids[spice_name] = body_id
        
        BodySearchDialog(self.root, on_body_selected)

    def estimate_time(self):
        try:
            res = int(self.resolution.get())
            if res <= 0:
                messagebox.showerror("Invalid Resolution", "Resolution must be a positive number greater than 0.")
                return
            if res > 200:
                messagebox.showwarning("High Resolution Warning", f"Resolution of {res} will take a very long time to compute. Consider using 50-100 for most applications.")
            
            est_time = trajectory.estimate_time(res)
            hours = est_time / 3600
            
            if hours < 1:
                time_str = f"{est_time/60:.1f} minutes"
            else:
                time_str = f"{hours:.2f} hours"
            
            accel_type = "GPU accelerated" if self.gpu_available else "CPU only (no GPU acceleration)"
            msg = f"Calculation Estimate:\n\n"
            msg += f"Grid size: {res} × {res} = {res*res:,} trajectory calculations\n"
            msg += f"Estimated time: {time_str} ({accel_type})\n\n"
            msg += "Tip: Start with resolution 20-50 for quick results, then increase for final analysis."
            
            messagebox.showinfo("Time Estimate", msg)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for resolution (e.g., 20, 50, 100).")

    def _run_porkchop_calculation(self, start, end, min_t, max_t, res, dep_body, arr_body, 
                                  use_gravity_assist, flyby_body, solve_type):
        """Run porkchop calculation in background thread."""
        try:
            # Choose calculation method based on solve type and gravity assist
            if use_gravity_assist:
                if solve_type == "high":
                    # High confidence: use enhanced gravity assist optimization
                    self.dates, self.times, self.dv = porkchop_data_gravity_assist_enhanced(
                        start, end, min_t, max_t, res, dep_body, flyby_body, arr_body, update_callback=self.update_progress
                    )
                else:
                    # Rough estimate: use standard gravity assist
                    self.dates, self.times, self.dv = porkchop_data_gravity_assist(
                        start, end, min_t, max_t, res, dep_body, flyby_body, arr_body, update_callback=self.update_progress
                    )
            else:
                # No gravity assist: use standard porkchop
                self.dates, self.times, self.dv = porkchop_data(
                    start, end, min_t, max_t, res, dep_body, arr_body, update_callback=self.update_progress
                )
            
            # Store original data for zoom operations
            self.original_dates = self.dates.copy()
            self.original_times = self.times.copy()
            self.original_dv = self.dv.copy()
            
            # Schedule final plot update on main thread
            self.root.after(0, self._finalize_plot, use_gravity_assist, flyby_body, solve_type, dep_body, arr_body)
            print("[DEBUG] Calculation completed successfully, scheduled plot update")
            
        except Exception as e:
            # Schedule error display on main thread
            self.root.after(0, lambda: messagebox.showerror("Calculation Error", f"Something went wrong during calculation: {str(e)}"))
            print(f"[DEBUG] Calculation failed with error: {e}")

    def _finalize_plot(self, use_gravity_assist, flyby_body, solve_type, dep_body, arr_body):
        """Finalize the plot display after calculation completes."""
        print("[DEBUG] _finalize_plot called!")
        self.progress_var.set(100)
        self.progress_label.config(text="Plotting complete! Lower Δv values (darker blue) indicate more efficient trajectories.")

        # Debug: Check data before plotting
        print(f"[DEBUG] Plot data shapes: dates={self.dates.shape}, times={self.times.shape}, dv={self.dv.shape}")
        print(f"[DEBUG] DV data range: min={float(np.nanmin(self.dv)):.2f}, max={float(np.nanmax(self.dv)):.2f}")
        print(f"[DEBUG] NaN count: {np.isnan(self.dv).sum()} out of {self.dv.size}")
        print(f"[DEBUG] Sample DV values: {self.dv[0, :5]}")

        # Clear the entire figure to remove any leftover color bars
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)  # Recreate 2D axes

        # Handle NaN values by replacing them with a high value for visualization
        dv_plot = np.where(np.isnan(self.dv), np.nanmax(self.dv) * 1.1, self.dv)

        # Ensure we have valid data range
        dv_min = np.nanmin(dv_plot)
        dv_max = np.nanmax(dv_plot)
        if dv_min == dv_max:
            # All values are the same, create a small range for visualization
            dv_max = dv_min + 1000  # Add 1 km/s range

        # Create contour levels
        levels = np.linspace(dv_min, dv_max, 50)

        # Debug: Print vmin and vmax values for colormap
        print(f"[DEBUG] Colormap vmin={dv_min:.2f}, vmax={dv_max:.2f}")

        # Define vmin and vmax based on dv_min and dv_max
        vmin = dv_min ** 2
        vmax = dv_max ** 2

        # Create final plot
        try:
            pcm = self.ax.contourf(self.dates, self.times, dv_plot, levels=levels, cmap="viridis", extend='both')
            print(f"[DEBUG] Contour plot created successfully with {len(levels)} levels")
        except Exception as e:
            print(f"[DEBUG] Contour plot failed: {e}")
            # Fallback: plot as image
            pcm = self.ax.imshow(dv_plot, extent=[self.dates[0], self.dates[-1], self.times[0], self.times[-1]],
                               origin='lower', cmap="viridis", aspect='auto')

        # Only create color bar for gravity assist plots
        if use_gravity_assist:
            # Remove any existing color bars before creating new one
            for cbar in self.fig.get_children():
                if hasattr(cbar, 'colorbar'):
                    try:
                        cbar.remove()
                    except:
                        pass

            # Create the color bar
            cbar = self.fig.colorbar(pcm, ax=self.ax, label="Fuel Required (Δv in m/s)")
            cbar.ax.tick_params(labelsize=8)  # Make color bar text smaller
        
        # Convert JD to readable dates for x-axis
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter
        
        # Create date ticks - sample every ~30 days
        jd_range = self.dates[-1] - self.dates[0]
        num_ticks = min(8, max(3, int(jd_range / 30)))  # Adaptive number of ticks
        
        tick_jds = []
        tick_labels = []
        for i in range(num_ticks):
            jd_val = self.dates[0] + (jd_range * i / (num_ticks - 1))
            tick_jds.append(jd_val)
            tick_labels.append(jd_to_date(jd_val))
        
        self.ax.set_xticks(tick_jds)
        self.ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        self.ax.set_xlabel("Departure Date")
        self.ax.set_ylabel("Travel Time (days)")
        dep_body_name = [k for k, v in self.body_name_map.items() if v == dep_body][0] if dep_body in self.body_name_map.values() else dep_body
        arr_body_name = [k for k, v in self.body_name_map.items() if v == arr_body][0] if arr_body in self.body_name_map.values() else arr_body
        solve_name = "Rough Estimate" if solve_type == "rough" else "High Confidence"
        title = f"{solve_name}: {dep_body_name} → {arr_body_name}"
        if use_gravity_assist:
            flyby_body_name = [k for k, v in self.body_name_map.items() if v == flyby_body][0] if flyby_body in self.body_name_map.values() else flyby_body
            title += f" (via {flyby_body_name})"
        self.ax.set_title(title)
        self.canvas.draw()
        
        # Update the navigation toolbar for 2D plotting
        self.toolbar.update()

    def update_progress(self, progress, dv, eta, dep_jds, tof_days):
        """Update progress bar and status during calculation."""
        percent = progress * 100
        self.progress_var.set(percent)
        if percent < 100:
            eta_str = f"{eta / 3600:.2f} hours remaining" if eta > 0 else "Calculating time remaining..."
            self.progress_label.config(text=f"Calculating trajectories... {percent:.1f}% complete (ETA: {eta_str})")
        else:
            self.progress_label.config(text="Calculation complete! Plotting results...")
        
        # No longer update plot during progress - wait for final plot
        self.root.update_idletasks()

    def start_porkchop(self, skip_confirmation=False):
        print(f"[DEBUG] start_porkchop called with skip_confirmation={skip_confirmation}")
        try:
            start = self.start_date_picker.get()
            end = self.end_date_picker.get()
            min_t = float(self.min_time.get())
            max_t = float(self.max_time.get())
            res = int(self.resolution.get())
            dep_body_common = self.dep_body.get().strip()
            arr_body_common = self.arr_body.get().strip()
            
            # Convert common names to SPICE names
            dep_body = self.body_name_map.get(dep_body_common, dep_body_common)
            arr_body = self.body_name_map.get(arr_body_common, arr_body_common)
            
            if not dep_body or not arr_body:
                messagebox.showerror("Selection Required", "Please select both a departure planet and destination planet from the dropdown menus.")
                return
            
            if dep_body == arr_body:
                messagebox.showerror("Invalid Selection", "Departure and arrival bodies cannot be the same. Please choose different planets.")
                return
            
            if res <= 0:
                messagebox.showerror("Invalid Resolution", "Resolution must be a positive number (try 20-50 for good results).")
                return
                
            if min_t >= max_t:
                messagebox.showerror("Invalid Time Range", "Minimum transit time must be less than maximum transit time.")
                return
            
            if min_t <= 0 or max_t <= 0:
                messagebox.showerror("Invalid Time Range", "Transit times must be positive numbers.")
                return
            
            est_time = trajectory.estimate_time(res)
            
            message = f"Calculate trajectories from {dep_body_common} to {arr_body_common}?\n\n"
            message += f"Date range: {start} to {end}\n"
            message += f"Travel time: {min_t:.0f} to {max_t:.0f} days\n"
            message += f"Grid resolution: {res}×{res} points\n"
            message += f"Estimated calculation time: {est_time/3600:.2f} hours\n\n"
            accel_msg = "GPU acceleration for fast computation" if self.gpu_available else "CPU computation (GPU not available)"
            message += f"{accel_msg}."
            
            if not skip_confirmation and not messagebox.askyesno("Start Trajectory Calculation", message):
                return
            
            # Get solve type for calculation approach
            solve_type = self.solve_type_var.get()
            solve_name = "Rough Estimate" if solve_type == "rough" else "High Confidence"
            message += f"Solve type: {solve_name}\n"
            
            # Check if gravity assist is enabled
            use_gravity_assist = self.gravity_assist_var.get()
            self.use_gravity_assist = use_gravity_assist  # Store for plotting
            flyby_body = None
            if use_gravity_assist:
                flyby_body_common = self.flyby_body.get().strip()
                flyby_body = self.body_name_map.get(flyby_body_common, flyby_body_common)
                if not flyby_body:
                    messagebox.showerror("Flyby Body Required", "Please select a flyby body for gravity assist.")
                    return
                message += f"Gravity assist via: {flyby_body_common}\n"
            
            accel_init = "Initializing GPU acceleration..." if self.gpu_available else "Initializing calculation..."
            self.progress_var.set(0)
            self.progress_label.config(text=accel_init)
            self.root.update_idletasks()
            
            # Store parameters for use in background thread
            self.use_gravity_assist = use_gravity_assist
            
            # Start calculation in background thread
            calc_thread = threading.Thread(
                target=self._run_porkchop_calculation,
                args=(start, end, min_t, max_t, res, dep_body, arr_body, 
                      use_gravity_assist, flyby_body, solve_type),
                daemon=True
            )
            calc_thread.start()
            
        except ValueError as e:
            if "time data" in str(e).lower():
                messagebox.showerror("Date Format Error", "Please enter dates in YYYY-MM-DD format (e.g., 2035-01-01).")
            else:
                messagebox.showerror("Input Error", f"Please check your inputs: {str(e)}")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Something went wrong during calculation. Please check your inputs and try again.\n\nError: {str(e)}")
            
        except ValueError as e:
            if "time data" in str(e).lower():
                messagebox.showerror("Date Format Error", "Please enter dates in YYYY-MM-DD format (e.g., 2035-01-01).")
            else:
                messagebox.showerror("Input Error", f"Please check your inputs: {str(e)}")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Something went wrong during calculation. Please check your inputs and try again.\n\nError: {str(e)}")

    def _run_porkchop_calculation(self, start, end, min_t, max_t, res, dep_body, arr_body, 
                                 use_gravity_assist, flyby_body, solve_type):
        """Run the porkchop plot calculation in a background thread."""
        try:
            # Choose which calculation method to use based on solve_type and gravity assist
            if use_gravity_assist:
                if solve_type == "high":
                    # Use enhanced gravity assist method
                    self.dates, self.times, self.dv = porkchop_data_gravity_assist_enhanced(
                        start, end, min_t, max_t, res, dep_body, flyby_body, arr_body,
                        update_callback=self.update_progress
                    )
                else:
                    # Use standard gravity assist method
                    self.dates, self.times, self.dv = porkchop_data_gravity_assist(
                        start, end, min_t, max_t, res, dep_body, flyby_body, arr_body,
                        update_callback=self.update_progress
                    )
            else:
                if solve_type == "high":
                    # Use enhanced direct transfer method
                    self.dates, self.times, self.dv = porkchop_data_enhanced(
                        start, end, min_t, max_t, res, dep_body, arr_body,
                        update_callback=self.update_progress
                    )
                else:
                    # Use standard direct transfer method
                    self.dates, self.times, self.dv = porkchop_data(
                        start, end, min_t, max_t, res, dep_body, arr_body,
                        update_callback=self.update_progress
                    )
            
            # Store data for zoom operations
            self.original_dates = self.dates
            self.original_times = self.times  
            self.original_dv = self.dv
            
            # Update GUI on main thread
            self.root.after(0, self._plot_porkchop_results)
            
        except Exception as e:
            # Update GUI on main thread with error
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Calculation Error", f"Calculation failed: {error_msg}"))

    def _plot_porkchop_results(self):
        """Plot the porkchop plot results on the main thread."""
        print("[DEBUG] _finalize_plot called!")
        try:
            # Clear any existing plot
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            
            # Create porkchop plot in Julian date format with C3 colorbar
            if self.dates is not None and self.times is not None and self.dv is not None:
                # Departure Julian dates (1D array)
                dep_jd = np.array(self.dates)
                # Time-of-flight (1D array)
                tof_days = np.array(self.times)
                
                # Create 2D meshgrids for plotting
                # DEP_GRID[i,j] = dep_jd[i], TOF_GRID[i,j] = tof_days[j]
                DEP_GRID, TOF_GRID = np.meshgrid(dep_jd, tof_days, indexing='ij')
                # Arrival JD = Departure JD + TOF
                ARR_GRID = DEP_GRID + TOF_GRID
                
                # For pcolormesh, use the 2D grids
                DEPARTURE = DEP_GRID
                ARRIVAL = ARR_GRID

                # Prepare C3 grid (km^2/s^2) -- ensure dv is in km/s
                dv_plot = np.array(self.dv, dtype=float)
                print(f"[DEBUG] Raw DV stats: min={np.nanmin(dv_plot):.2f}, max={np.nanmax(dv_plot):.2f}, mean={np.nanmean(dv_plot):.2f}")
                
                # If dv is in m/s, convert to km/s
                if np.nanmax(dv_plot) > 100:  # likely m/s
                    print("[DEBUG] Converting DV from m/s to km/s")
                    dv_plot = dv_plot / 1000.0
                else:
                    print("[DEBUG] DV appears to already be in km/s")
                    
                dv_plot[~np.isfinite(dv_plot)] = np.nan
                # Sanity cap: any dv > 50 km/s is likely numerical junk; mark as NaN
                dv_plot = np.where(np.abs(dv_plot) > 50.0, np.nan, dv_plot)
                C3_plot = dv_plot ** 2
                print(f"[DEBUG] C3 stats after conversion: min={np.nanmin(C3_plot):.2f}, max={np.nanmax(C3_plot):.2f}, mean={np.nanmean(C3_plot):.2f}")

                # Ensure orientation matches meshgrid
                expected_shape = (len(dep_jd), len(self.times))
                if C3_plot.shape != expected_shape:
                    if C3_plot.shape == (len(self.times), len(dep_jd)):
                        C3_plot = C3_plot.T
                
                # Debug: Print raw ΔV and C3 values for sample indices
                sample_indices = [(0, 0), (len(dep_jd)//2, len(tof_days)//2), (-1, -1)]
                for idx in sample_indices:
                    try:
                        dep_sample = dep_jd[idx[0]]
                        tof_sample = tof_days[idx[1]]
                        dv_sample = dv_plot[idx]
                        c3_sample = C3_plot[idx]
                        print(f"[DEBUG] Sample at dep_jd={dep_sample:.2f}, tof={tof_sample:.1f}: ΔV={dv_sample:.3f} km/s, C3={c3_sample:.3f} km^2/s^2")
                    except IndexError:
                        print(f"[DEBUG] Sample index {idx} out of bounds")

                # Define vmin and vmax for colormap based on C3 values
                vmin = np.nanmin(C3_plot)
                vmax = np.nanmax(C3_plot)
                
                # Use colormap where low values (good transfers) are green and high values (bad) are red
                # 'RdYlGn' = Green (low C3, good) -> Yellow -> Red (high C3, bad)
                im = self.ax.pcolormesh(DEPARTURE, ARRIVAL, C3_plot, shading='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
                cbar = self.fig.colorbar(im, ax=self.ax, label='C3 (km²/s²)')

                # Overlay time-of-flight lines (100, 200, 300, ... days)
                for tof in range(100, 401, 100):
                    self.ax.plot(dep_jd, dep_jd + tof, color='black', linestyle='--', linewidth=1)
                    self.ax.text(float(dep_jd[0]), float(dep_jd[0] + tof), f'{tof} Days', color='black', fontsize=8, va='bottom')

                self.ax.set_xlabel('Departure Julian Date')
                self.ax.set_ylabel('Arrival Julian Date')
                self.ax.set_title('C3 Plot for Earth-Mars transfer in 2035')
                self.ax.set_xlim(float(dep_jd.min()), float(dep_jd.max()))
                self.ax.set_ylim(float((dep_jd + tof_days.min()).min()), float((dep_jd + tof_days.max()).max()))

                self.toolbar.update()
                self.canvas.draw()
                self.progress_var.set(100)
                self.progress_label.config(text="Calculation complete! Plotting finished.", fg="green")
                self._show_optimal_trajectory_info()
            else:
                self.progress_label.config(text="No data to plot", fg="red")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] Plotting failed with traceback:\n{tb}")
            self.progress_label.config(text=f"Plotting failed: {str(e)}", fg="red")
            messagebox.showerror("Plot Error", f"Could not create porkchop plot: {str(e)}")

    def _show_optimal_trajectory_info(self):
        """Show information about the optimal trajectory."""
        if self.dates is None or self.times is None or self.dv is None:
            return
            
        # Find minimum ΔV
        min_idx = np.unravel_index(np.argmin(self.dv), self.dv.shape)
        min_dv = self.dv[min_idx] / 1000  # Convert to km/s
        dep_date = jd_to_date(self.dates[min_idx[0]])
        tof_days = self.times[min_idx[1]]
        
        # Calculate arrival date
        arr_jd = self.dates[min_idx[0]] + tof_days
        arr_date = jd_to_date(arr_jd)
        
        info = f"Optimal Trajectory Found:\n\n"
        info += f"Departure: {dep_date}\n"
        info += f"Arrival: {arr_date}\n"
        # Safe formatting for numeric values
        tof_days_str = self._fmt_num(tof_days, '.1f')
        min_dv_str = self._fmt_num(min_dv, '.2f')
        info += f"Time of Flight: {tof_days_str} days\n"
        info += f"\u0394V Required: {min_dv_str} km/s\n\n"
        info += "Click 'Plot Optimal Trajectory' for 3D visualization."

        messagebox.showinfo("Optimal Trajectory", info)

    def start_research_optimization(self):
        """Start enhanced research optimization for publishable results."""
        try:
            # Get parameters
            start = self.start_date_picker.get()
            end = self.end_date_picker.get()
            min_t = float(self.min_time.get())
            max_t = float(self.max_time.get())
            res = int(self.research_resolution.get())
            max_hours = float(self.max_runtime.get())
            
            # Parse time splits
            time_splits_str = self.time_splits_entry.get()
            time_splits = [float(x.strip()) for x in time_splits_str.split(',')]
            
            # Parse flyby altitudes
            altitudes_str = self.flyby_altitudes_entry.get()
            flyby_altitudes = [float(x.strip()) * 1000 for x in altitudes_str.split(',')]  # Convert km to m
            
            dep_body_common = self.dep_body.get().strip()
            arr_body_common = self.arr_body.get().strip()
            
            # Convert common names to SPICE names
            dep_body = self.body_name_map.get(dep_body_common, dep_body_common)
            arr_body = self.body_name_map.get(arr_body_common, arr_body_common)
            
            if not dep_body or not arr_body:
                messagebox.showerror("Selection Required", "Please select both a departure planet and destination planet.")
                return
            
            # Check if gravity assist is enabled
            use_gravity_assist = self.gravity_assist_var.get()
            if not use_gravity_assist:
                messagebox.showerror("Gravity Assist Required", "Research optimization requires gravity assist mode to be enabled.")
                return
                
            flyby_body_common = self.flyby_body.get().strip()
            flyby_body = self.body_name_map.get(flyby_body_common, flyby_body_common)
            
            # Calculate total evaluations
            total_evaluations = len(time_splits) * len(flyby_altitudes) * res * res
            
            # Confirm with user
            message = f"🧪 RESEARCH OPTIMIZATION MODE\\n\\n"
            message += f"Mission: {dep_body_common} → {flyby_body_common} → {arr_body_common}\\n"
            message += f"Date range: {start} to {end}\\n"
            message += f"Time splits: {len(time_splits)} variations\\n"
            message += f"Flyby altitudes: {len(flyby_altitudes)} variations\\n"
            message += f"Grid resolution: {res}×{res}\\n"
            message += f"Total evaluations: {total_evaluations:,}\\n"
            message += f"Estimated time: {max_hours} hours maximum\\n\\n"
            message += f"GPU acceleration: {'Available' if self.gpu_available else 'Not available'}\\n\\n"
            message += "This will run for an extended period. Results will be publishable quality!"
            
            if not messagebox.askyesno("Start Research Optimization", message):
                return
            
            # Clear results display
            self.research_results_text.delete(1.0, tk.END)
            self.research_results_text.insert(tk.END, "🚀 Starting research optimization...\\n\\n")
            self.root.update()
            
            # Start optimization
            self.research_results = porkchop_data_gravity_assist_enhanced(
                start, end, min_t, max_t, res, dep_body, flyby_body, arr_body,
                time_splits=time_splits, flyby_altitudes=flyby_altitudes,
                update_callback=self.update_research_progress, max_runtime_hours=max_hours
            )
            
            # Display results
            self.display_research_results()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your inputs: {str(e)}")
        except Exception as e:
            messagebox.showerror("Research Error", f"Error during research optimization: {str(e)}")
    
    def update_research_progress(self, progress, dv_grid, eta, dep_jds, tof_days):
        """Update progress during research optimization."""
        percent = progress * 100
        eta_str = f"{eta/3600:.1f} hours" if eta > 0 else "unknown"
        
        progress_msg = f"Progress: {percent:.1f}% (ETA: {eta_str})\\n"
        self.research_results_text.insert(tk.END, progress_msg)
        self.research_results_text.see(tk.END)
        self.root.update()
    
    def display_research_results(self):
        """Display the research optimization results."""
        if not hasattr(self, 'research_results'):
            return
            
        results = self.research_results
        stats = results['statistics']
        
        self.research_results_text.delete(1.0, tk.END)
        self.research_results_text.insert(tk.END, "🎯 RESEARCH OPTIMIZATION COMPLETE\\n")
        self.research_results_text.insert(tk.END, "=" * 50 + "\\n\\n")
        
        self.research_results_text.insert(tk.END, f"📊 STATISTICS:\\n")
        self.research_results_text.insert(tk.END, f"Total evaluations: {stats['total_evaluations']:,}\\n")
        self.research_results_text.insert(tk.END, f"Best ΔV: {stats['best_dv']/1000:.2f} km/s\\n")
        self.research_results_text.insert(tk.END, f"Mean ΔV: {stats['mean_dv']/1000:.2f} km/s\\n")
        self.research_results_text.insert(tk.END, f"Computation time: {stats['computation_time']/3600:.1f} hours\\n\\n")
        
        self.research_results_text.insert(tk.END, f"🏆 TOP 5 TRAJECTORIES:\\n")
        for i, traj in enumerate(results['best_trajectories'][:5], 1):
            self.research_results_text.insert(tk.END, 
                f"{i}. {traj['departure_date']} → {traj['arrival_date']} "
                f"({traj['total_tof_days']:.0f} days, {traj['dv_km_s']:.2f} km/s ΔV)\\n"
                f"   Time split: {traj['time_split']:.1f}, Altitude: {traj['flyby_altitude']/1000:.0f} km\\n\\n")
        
        self.research_results_text.insert(tk.END, "💾 Use 'Save Results' to export data for publication\\n")
    
    def save_research_results(self):
        """Save research optimization results to file."""
        if not hasattr(self, 'research_results'):
            messagebox.showerror("No Results", "Please run research optimization first.")
            return
            
        try:
            from tkinter import filedialog
            import json
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Research Results"
            )
            
            if filename:
                # Convert numpy arrays to lists for JSON serialization
                results_copy = self.research_results.copy()
                for result in results_copy['all_results']:
                    result['dep_jds'] = result['dep_jds'].tolist()
                    result['tof_days'] = result['tof_days'].tolist()
                    result['dv_grid'] = result['dv_grid'].tolist()
                
                with open(filename, 'w') as f:
                    json.dump(results_copy, f, indent=2)
                
                messagebox.showinfo("Saved", f"Research results saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryApp(root)
    root.mainloop()
