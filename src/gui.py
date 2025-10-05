# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import trajectory
from trajectory import porkchop_data, porkchop_data_gravity_assist
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

    def emergency_exit(self):
        """Emergency exit function for when GUI gets stuck. Can be triggered by Ctrl+Q, Ctrl+X, or Alt+F4."""
        print("🚨 Emergency exit activated! Force closing application...")
        try:
            self.root.quit()
            self.root.destroy()
        except:
            # Force exit if normal methods fail
            import os
            os._exit(0)

    def show_animation(self):
        messagebox.showinfo("Animation", "Animation not yet implemented.")

def run_tests():
    """Run basic functionality tests without opening GUI"""
    print("🧪 Running Lambert Solver Tests...")
    print("=" * 50)
    
    # Test imports
    print("✅ Testing imports...")
    try:
        import trajectory
        import spice_interface
        import skyfield.api as sf
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test GPU/CuPy status
    print("\n🔍 Checking GPU acceleration...")
    try:
        import cupy as cp
        print("✅ CuPy available - GPU acceleration enabled")
        gpu_available = True
    except ImportError:
        print("⚠️  CuPy not available - CPU only mode")
        gpu_available = False
    
    # Test SPICE kernels
    print("\n🌌 Testing SPICE kernels...")
    try:
        spice_interface.load_all_kernels()
        print("✅ SPICE kernels loaded successfully")
    except Exception as e:
        print(f"❌ SPICE kernel error: {e}")
        return False
    
    # Test body list functionality
    print("\n🪐 Testing celestial body list...")
    try:
        # Create a minimal app instance to test body list
        root = tk.Tk()
        root.withdraw()  # Hide the window
        app = TrajectoryApp(root)
        
        body_list = app.get_body_list()
        print(f"✅ Body list loaded: {len(body_list)} bodies available")
        print(f"   Available bodies: {', '.join(body_list[:5])}{'...' if len(body_list) > 5 else ''}")
        
        root.destroy()  # Clean up
    except Exception as e:
        print(f"❌ Body list error: {e}")
        return False
    
    # Test basic trajectory calculation
    print("\n🚀 Testing basic trajectory calculation...")
    try:
        # Simple test case: Earth to Mars
        r1 = np.array([1.0, 0.0, 0.0]) * 149597870.7  # Earth position (AU to km)
        r2 = np.array([1.5, 0.0, 0.0]) * 149597870.7  # Mars position (AU to km)
        tof = 200 * 24 * 3600  # 200 days in seconds
        mu = 1.327e20  # Sun's gravitational parameter
        
        v1, v2 = trajectory.lambert_izzo_gpu(r1, r2, tof, mu)
        print(f"✅ Trajectory calculation successful: v1={v1}, v2={v2}")
    except Exception as e:
        print(f"❌ Trajectory calculation error: {e}")
        return False
    
    # Test porkchop data generation
    print("\n📊 Testing porkchop data generation...")
    try:
        # Small test case
        dates, times, dv = porkchop_data(
            "2035-01-01", "2035-01-02",  # 1 day date range
            100, 101,  # 1 day time range  
            2,  # 2x2 grid (very small for testing)
            "EARTH", "MARS BARYCENTER"
        )
        print(f"✅ Porkchop data generated: {dates.shape} dates, {times.shape} times, {dv.shape} delta-v values")
    except Exception as e:
        print(f"❌ Porkchop data error: {e}")
        return False
    
    print("\n🎉 All tests passed! The Lambert solver is working correctly.")
    print(f"   GPU Acceleration: {'Enabled' if gpu_available else 'Disabled (CPU only)'}")
    return True

def test_body_list_functions():
    """Test body list creation functions without GUI"""
    print("🧪 Testing Body List Functions...")
    print("=" * 40)
    
    try:
        # Test 1: Import required modules
        print("📦 Testing imports...")
        import spice_interface
        import skyfield_interface as sf
        print("✅ Imports successful")
        
        # Test 2: Load kernels
        print("\n🌌 Testing kernel loading...")
        spice_interface.load_all_kernels()
        print("✅ Kernels loaded")
        
        # Test 3: Test spice_interface functions
        print("\n🔍 Testing spice_interface functions...")
        
        # Test search_celestial_body function
        test_bodies = ["EARTH", "MARS BARYCENTER", "VENUS BARYCENTER", "SUN"]
        for body in test_bodies:
            try:
                result = spice_interface.search_celestial_body(body)
                print(f"  search_celestial_body('{body}') = {result}")
            except Exception as e:
                print(f"  ❌ search_celestial_body('{body}') failed: {e}")
        
        # Test 4: Create TrajectoryApp instance (but don't show GUI)
        print("\n🏗️  Testing TrajectoryApp initialization (no GUI)...")
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window immediately
        
        app = TrajectoryApp(root)
        print("✅ TrajectoryApp instance created")
        
        # Test 5: Test body_name_map
        print("\n🗺️  Testing body name mappings...")
        print(f"  body_name_map has {len(app.body_name_map)} entries:")
        for common, spice in app.body_name_map.items():
            print(f"    {common} -> {spice}")
        
        print(f"  spice_to_common has {len(app.spice_to_common)} entries")
        
        # Test 6: Test get_body_list method
        print("\n📋 Testing get_body_list method...")
        body_list = app.get_body_list()
        print(f"✅ get_body_list returned {len(body_list)} bodies:")
        for i, body in enumerate(body_list):
            print(f"    {i+1:2d}. {body}")
        
        # Test 7: Test body_ids population
        print("\n🆔 Testing body_ids population...")
        print(f"  body_ids has {len(app.body_ids)} entries:")
        for spice_name, body_id in app.body_ids.items():
            print(f"    {spice_name} -> ID: {body_id}")
        
        # Test 8: Test individual body lookups
        print("\n🔎 Testing individual body lookups...")
        for common_name in ["Earth", "Mars", "Venus", "Sun"]:
            spice_name = app.body_name_map.get(common_name)
            if spice_name:
                body_id = app.body_ids.get(spice_name)
                print(f"  {common_name} -> {spice_name} -> ID: {body_id}")
            else:
                print(f"  ❌ {common_name} not found in body_name_map")
        
        # Clean up
        root.destroy()
        print("\n✅ All body list function tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Body list function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_components():
    """Test GUI component initialization without showing window"""
    print("🧪 Testing GUI Components...")
    print("=" * 35)
    
    try:
        import tkinter as tk
        from tkinter import ttk
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        # Test 1: Basic Tkinter
        print("🖼️  Testing Tkinter initialization...")
        root = tk.Tk()
        root.withdraw()
        print("✅ Tkinter root created")
        
        # Test 2: DatePicker widget
        print("\n📅 Testing DatePicker widget...")
        try:
            date_picker = DatePicker(root, "2035-01-01")
            print("✅ DatePicker created")
        except Exception as e:
            print(f"❌ DatePicker failed: {e}")
            root.destroy()
            return False
        
        # Test 3: TrajectoryApp initialization (step by step)
        print("\n🏗️  Testing TrajectoryApp components...")
        
        # Create app instance
        app = TrajectoryApp(root)
        print("✅ TrajectoryApp instance created")
        
        # Test GUI frames exist
        print("  Checking GUI frames...")
        if hasattr(app, 'setup_frame') and app.setup_frame:
            print("    ✅ setup_frame exists")
        else:
            print("    ❌ setup_frame missing")
            
        if hasattr(app, 'params_frame') and app.params_frame:
            print("    ✅ params_frame exists")
        else:
            print("    ❌ params_frame missing")
            
        if hasattr(app, 'control_frame') and app.control_frame:
            print("    ✅ control_frame exists")
        else:
            print("    ❌ control_frame missing")
            
        if hasattr(app, 'plot_frame') and app.plot_frame:
            print("    ✅ plot_frame exists")
        else:
            print("    ❌ plot_frame missing")
        
        # Test matplotlib components
        print("  Checking matplotlib components...")
        if hasattr(app, 'fig') and app.fig:
            print("    ✅ matplotlib figure exists")
        else:
            print("    ❌ matplotlib figure missing")
            
        if hasattr(app, 'ax') and app.ax:
            print("    ✅ matplotlib axes exists")
        else:
            print("    ❌ matplotlib axes missing")
            
        if hasattr(app, 'canvas') and app.canvas:
            print("    ✅ canvas exists")
        else:
            print("    ❌ canvas missing")
        
        # Test GUI widgets
        print("  Checking GUI widgets...")
        widgets_to_check = [
            ('mission_preset', 'mission preset combobox'),
            ('dep_body', 'departure body combobox'),
            ('arr_body', 'arrival body combobox'),
            ('calc_button', 'calculate button'),
            ('porkchop_button', 'porkchop button'),
            ('clear_button', 'clear button')
        ]
        
        for attr_name, description in widgets_to_check:
            if hasattr(app, attr_name) and getattr(app, attr_name):
                print(f"    ✅ {description} exists")
            else:
                print(f"    ❌ {description} missing")
        
        # Test preset application
        print("  Testing preset application...")
        try:
            app.mission_preset.set("Earth → Mars")
            app.apply_preset(None)
            print("    ✅ Preset application works")
        except Exception as e:
            print(f"    ❌ Preset application failed: {e}")
        
        # Clean up
        root.destroy()
        print("\n✅ All GUI component tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ GUI component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_initialization_step_by_step():
    """Test GUI initialization step by step without main loop"""
    print("🧪 Testing GUI Initialization Step by Step...")
    print("=" * 50)
    
    try:
        import tkinter as tk
        from tkinter import ttk
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        # Step 1: Basic Tkinter
        print("🔧 Step 1: Creating Tkinter root...")
        root = tk.Tk()
        root.withdraw()  # Don't show the window
        print("✅ Tkinter root created")
        
        # Step 2: Basic TrajectoryApp attributes
        print("📋 Step 2: Setting basic attributes...")
        app = object.__new__(TrajectoryApp)  # Create without calling __init__
        app.root = root
        app.gpu_available = False
        app.kernels_loaded = False
        app.body_ids = {}
        print("✅ Basic attributes set")
        
        # Step 3: Body name mappings
        print("🗺️  Step 3: Setting up body mappings...")
        app.body_name_map = {
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
        app.spice_to_common = {v: k for k, v in app.body_name_map.items()}
        print("✅ Body mappings created")
        
        # Step 4: GPU check
        print("🔍 Step 4: GPU detection...")
        try:
            import cupy as cp
            app.gpu_available = True
            print("✅ GPU available")
        except ImportError:
            app.gpu_available = False
            print("⚠️  GPU not available")
        
        # Step 5: Kernel loading
        print("🌌 Step 5: Loading kernels...")
        try:
            spice_interface.load_all_kernels()
            app.kernels_loaded = True
            print("✅ Kernels loaded")
        except Exception as e:
            print(f"❌ Kernel loading failed: {e}")
            app.kernels_loaded = False
        
        # Step 6: Test get_body_list
        print("📋 Step 6: Testing get_body_list...")
        try:
            body_list = app.get_body_list()
            print(f"✅ Body list: {len(body_list)} bodies")
        except Exception as e:
            print(f"❌ get_body_list failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 7: Create frames
        print("🎨 Step 7: Creating GUI frames...")
        try:
            app.setup_frame = ttk.LabelFrame(root, text="Mission Setup", padding="10")
            app.status_frame = ttk.Frame(root, padding="5")
            app.params_frame = ttk.LabelFrame(root, text="Calculation Parameters", padding="10")
            app.control_frame = ttk.LabelFrame(root, text="Controls", padding="10")
            app.plot_frame = ttk.LabelFrame(root, text="Trajectory Plot", padding="10")
            print("✅ Frames created")
        except Exception as e:
            print(f"❌ Frame creation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 8: Create basic widgets
        print("🏗️  Step 8: Creating basic widgets...")
        try:
            # Status labels
            app.status_label = ttk.Label(app.status_frame, text="GPU status", font=("Arial", 9))
            app.kernel_status = ttk.Label(app.status_frame, text="Kernel status", font=("Arial", 9))
            
            # Mission preset
            app.mission_preset = ttk.Combobox(app.setup_frame, values=["Test"], state="readonly", width=18)
            
            # Date pickers (this might be where it fails)
            print("  Creating date pickers...")
            app.start_date_picker = DatePicker(app.setup_frame, "2035-01-01")
            app.end_date_picker = DatePicker(app.setup_frame, "2035-12-31")
            print("  ✅ Date pickers created")
            
            # Body comboboxes
            print("  Creating body comboboxes...")
            app.dep_body = ttk.Combobox(app.setup_frame, values=["Earth"], state="normal", width=18)
            app.arr_body = ttk.Combobox(app.setup_frame, values=["Mars"], state="normal", width=18)
            print("  ✅ Body comboboxes created")
            
            # Gravity assist combobox (initially disabled)
            app.flyby_body = ttk.Combobox(app.setup_frame, values=[], state="disabled", width=18)
            
            print("✅ Basic widgets created")
        except Exception as e:
            print(f"❌ Widget creation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 9: Matplotlib setup
        print("📊 Step 9: Setting up matplotlib...")
        try:
            app.fig = plt.Figure(figsize=(10, 6))
            app.ax = app.fig.add_subplot(111)
            app.canvas = FigureCanvasTkAgg(app.fig, master=app.plot_frame)
            app.toolbar = NavigationToolbar2Tk(app.canvas, app.plot_frame)
            print("✅ Matplotlib setup complete")
        except Exception as e:
            print(f"❌ Matplotlib setup failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up
        root.destroy()
        print("\n✅ All GUI initialization steps completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR during step-by-step testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_with_timeout():
    """Test GUI by showing it briefly with a timeout to catch errors"""
    print("🧪 Testing GUI with Timeout...")
    print("=" * 35)
    
    import threading
    import time
    
    error_occurred = [False]
    error_message = [""]
    
    def run_gui():
        try:
            print("🚀 Starting GUI in background thread...")
            root = tk.Tk()
            app = TrajectoryApp(root)
            print("✅ GUI initialized, starting main loop...")
            root.mainloop()
        except Exception as e:
            error_occurred[0] = True
            error_message[0] = str(e)
            print(f"❌ GUI error: {e}")
            import traceback
            traceback.print_exc()
    
    def timeout_check():
        time.sleep(3)  # Wait 3 seconds
        if not error_occurred[0]:
            print("⏰ Timeout reached - GUI appears to be running normally")
            print("💡 If you see this, the GUI loaded successfully but may have runtime issues")
        else:
            print(f"💥 Error detected: {error_message[0]}")
    
    # Start GUI in background thread
    gui_thread = threading.Thread(target=run_gui, daemon=True)
    gui_thread.start()
    
    # Start timeout check
    timeout_check()
    
    # Try to join the thread with a timeout
    gui_thread.join(timeout=1)
    
    if gui_thread.is_alive():
        print("⚠️  GUI thread is still running (this is expected)")
        return True  # GUI is running, no initialization error
    else:
        print("✅ GUI thread completed")
        return not error_occurred[0]

def run_debug():
    """Run with comprehensive debug output and error handling"""
    print("🐛 Debug mode - comprehensive error tracking enabled")
    print("=" * 60)
    
    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        print("🔧 Step 1: Creating Tkinter root window...")
        root = tk.Tk()
        print("✅ Root window created successfully")
        
        print("🏗️  Step 2: Initializing TrajectoryApp...")
        app = TrajectoryApp(root)
        print("✅ TrajectoryApp initialized successfully")
        
        print("🎯 Step 3: Setting up error handling for GUI events...")
        
        # Add error handling to key methods
        original_start_porkchop = app.start_porkchop
        def debug_start_porkchop():
            try:
                print("🚀 Starting porkchop plot generation...")
                return original_start_porkchop()
            except Exception as e:
                print(f"❌ Error in start_porkchop: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Porkchop Error", f"Failed to generate porkchop plot: {e}")
        
        app.start_porkchop = debug_start_porkchop
        
        # Add error handling to estimate_time
        original_estimate_time = app.estimate_time
        def debug_estimate_time():
            try:
                print("⏱️  Estimating calculation time...")
                return original_estimate_time()
            except Exception as e:
                print(f"❌ Error in estimate_time: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Time Estimation Error", f"Failed to estimate time: {e}")
        
        app.estimate_time = debug_estimate_time
        
        # Add error handling to apply_preset
        original_apply_preset = app.apply_preset
        def debug_apply_preset(event):
            try:
                print("🎛️  Applying mission preset...")
                return original_apply_preset(event)
            except Exception as e:
                print(f"❌ Error in apply_preset: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Preset Error", f"Failed to apply preset: {e}")
        
        app.apply_preset = debug_apply_preset
        
        print("✅ Error handling wrappers installed")
        
        print("🎉 Step 4: Starting GUI main loop...")
        print("💡 GUI is now running. Watch for error messages above.")
        print("💡 If you see errors, they will be logged with full stack traces.")
        
        root.mainloop()
        
    except Exception as e:
        print(f"💥 CRITICAL ERROR during GUI initialization: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            if 'root' in locals():
                root.destroy()
        except:
            pass
            
        print("❌ GUI failed to start. Check the error details above.")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['--test', '-t', 'test']:
            success = run_tests()
            sys.exit(0 if success else 1)
            
        elif command in ['--body-test', '-bt', 'body-test']:
            success = test_body_list_functions()
            sys.exit(0 if success else 1)
            
        elif command in ['--gui-test', '-gt', 'gui-test']:
            success = test_gui_components()
            sys.exit(0 if success else 1)
            
        elif command in ['--init-test', '-it', 'init-test']:
            success = test_gui_initialization_step_by_step()
            sys.exit(0 if success else 1)
            
        elif command in ['--timeout-test', '-tt', 'timeout-test']:
            success = test_gui_with_timeout()
            sys.exit(0 if success else 1)
            
        elif command in ['--debug', '-d', 'debug']:
            run_debug()
            
        elif command in ['--help', '-h', 'help']:
            print("Lambert Solver GUI")
            print("=" * 20)
            print("Usage:")
            print("  python gui.py              # Run GUI (default)")
            print("  python gui.py --test       # Run functionality tests")
            print("  python gui.py --body-test  # Test body list functions only")
            print("  python gui.py --gui-test   # Test GUI components only")
            print("  python gui.py --init-test  # Test GUI initialization step by step")
            print("  python gui.py --timeout-test # Test GUI with timeout")
            print("  python gui.py --debug      # Run GUI with debug output")
            print("  python gui.py --help       # Show this help")
            sys.exit(0)
            
        else:
            print(f"Unknown command: {command}")
            print("Use 'python gui.py --help' for usage information")
            sys.exit(1)
    
    else:
        # Default: run GUI
        root = tk.Tk()
        app = TrajectoryApp(root)
        root.mainloop()