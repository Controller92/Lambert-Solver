# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import trajectory
from trajectory import porkchop_data, porkchop_data_gravity_assist, porkchop_data_gravity_assist_enhanced
import spice_interface
from datetime import datetime, timedelta
import skyfield.api as sf
from skyfield.api import load
from astropy.time import Time
import sys
import os

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
        
        date_str = "04d"
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
        self.resolution.insert(0, "20")
        self.resolution.bind("<KeyRelease>", self.on_resolution_change)
        ToolTip(self.resolution, "Number of grid points for calculation (higher = more accurate but slower). Start with 20-50.")
        
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
        
        ttk.Button(button_frame, text="Show Animation", command=self.show_animation).grid(row=0, column=2, padx=5)
        ToolTip(ttk.Button(button_frame, text="Show Animation", command=self.show_animation), "Display an animated view of the optimal trajectory (coming soon)")
        
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

    def update_progress(self, progress, dv, eta, dep_jds, tof_days):
        percent = progress * 100
        self.progress_var.set(percent)
        if percent < 100:
            eta_str = f"{eta / 3600:.2f} hours remaining" if eta > 0 else "Calculating time remaining..."
            self.progress_label.config(text=f"Calculating trajectories... {percent:.1f}% complete (ETA: {eta_str})")
        else:
            self.progress_label.config(text="Calculation complete! Plotting results...")
        
        if percent % 5 < 0.1 or percent > 99.9:
            self.ax.clear()
            pcm = self.ax.contourf(dep_jds, tof_days, dv, levels=50, cmap="viridis")
            self.fig.colorbar(pcm, ax=self.ax, label="Fuel Required (Δv in m/s)")
            
            # Convert JD to readable dates for x-axis
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter
            
            # Create date ticks - sample every ~30 days
            jd_range = dep_jds[-1] - dep_jds[0]
            num_ticks = min(8, max(3, int(jd_range / 30)))  # Adaptive number of ticks
            
            tick_jds = []
            tick_labels = []
            for i in range(num_ticks):
                jd_val = dep_jds[0] + (jd_range * i / (num_ticks - 1))
                tick_jds.append(jd_val)
                tick_labels.append(jd_to_date(jd_val))
            
            self.ax.set_xticks(tick_jds)
            self.ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            self.ax.set_xlabel("Departure Date")
            self.ax.set_ylabel("Travel Time (days)")
            dep_body_name = self.dep_body.get()
            arr_body_name = self.arr_body.get()
            title = f"Mission Trajectories: {dep_body_name} → {arr_body_name}"
            if hasattr(self, 'use_gravity_assist') and self.use_gravity_assist:
                flyby_body_name = self.flyby_body.get()
                title += f" (via {flyby_body_name})"
            self.ax.set_title(title)
            self.canvas.draw()
        self.root.update_idletasks()

    def start_porkchop(self):
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
            
            if not messagebox.askyesno("Start Trajectory Calculation", message):
                return
            
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
            
            if use_gravity_assist:
                self.dates, self.times, self.dv = porkchop_data_gravity_assist(
                    start, end, min_t, max_t, res, dep_body, flyby_body, arr_body, update_callback=self.update_progress
                )
                title_suffix = f" (via {flyby_body_common})"
            else:
                self.dates, self.times, self.dv = porkchop_data(
                    start, end, min_t, max_t, res, dep_body, arr_body, update_callback=self.update_progress
                )
                title_suffix = ""
            
            # Store original data for zoom operations
            self.original_dates = self.dates.copy()
            self.original_times = self.times.copy()
            self.original_dv = self.dv.copy()
            
            self.progress_label.config(text="Plotting complete! Lower Δv values (darker blue) indicate more efficient trajectories.")
            
        except ValueError as e:
            if "time data" in str(e).lower():
                messagebox.showerror("Date Format Error", "Please enter dates in YYYY-MM-DD format (e.g., 2035-01-01).")
            else:
                messagebox.showerror("Input Error", f"Please check your inputs: {str(e)}")
        except Exception as e:
            messagebox.showerror("Calculation Error", f"Something went wrong during calculation. Please check your inputs and try again.\n\nError: {str(e)}")

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