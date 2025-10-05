# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import trajectory
from trajectory import porkchop_data
import spice_interface
from astropy.time import Time

class TrajectoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interplanetary Trajectory Calculator")
        
        # Load SPICE kernels
        try:
            spice_interface.load_all_kernels()
            self.kernels_loaded = True
        except Exception as e:
            messagebox.showwarning("Kernel Warning", f"SPICE kernels failed to load: {e}. Attempting to update...")
            self.update_kernels()
            self.kernels_loaded = False
        
        # Input fields
        tk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0)
        self.start_date = tk.Entry(root)
        self.start_date.grid(row=0, column=1)
        self.start_date.insert(0, "2035-01-01")
       
        tk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=1, column=0)
        self.end_date = tk.Entry(root)
        self.end_date.grid(row=1, column=1)
        self.end_date.insert(0, "2035-12-31")
       
        tk.Label(root, text="Min Transit Time (days):").grid(row=2, column=0)
        self.min_time = tk.Entry(root)
        self.min_time.grid(row=2, column=1)
        self.min_time.insert(0, "100")
       
        tk.Label(root, text="Max Transit Time (days):").grid(row=3, column=0)
        self.max_time = tk.Entry(root)
        self.max_time.grid(row=3, column=1)
        self.max_time.insert(0, "300")
       
        tk.Label(root, text="Resolution (grid points):").grid(row=4, column=0)
        self.resolution = tk.Entry(root)
        self.resolution.grid(row=4, column=1)
        self.resolution.insert(0, "10")  # Start small for testing
       
        # Body selection dropdowns
        tk.Label(root, text="Departure Body:").grid(row=5, column=0)
        self.dep_body = ttk.Combobox(root, values=self.get_body_list(), state="readonly")
        self.dep_body.grid(row=5, column=1)
        self.dep_body.set("EARTH")
       
        tk.Label(root, text="Arrival Body:").grid(row=6, column=0)
        self.arr_body = ttk.Combobox(root, values=self.get_body_list(), state="readonly")
        self.arr_body.grid(row=6, column=1)
        self.arr_body.set("MARS BARYCENTER")
       
        tk.Button(root, text="Estimate Time", command=self.estimate_time).grid(row=7, column=0)
        tk.Button(root, text="Generate Porkchop", command=self.start_porkchop).grid(row=7, column=1)
        tk.Button(root, text="Show Animation", command=self.show_animation).grid(row=8, column=0, columnspan=2)
        tk.Button(root, text="Download and Update All Kernels", command=self.update_kernels).grid(row=10, column=0, columnspan=2)
       
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=12, column=0, columnspan=2, pady=5)
        self.progress_label = tk.Label(root, text="Progress: 0% (ETA: N/A)")
        self.progress_label.grid(row=13, column=0, columnspan=2)
       
        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=14, column=0, columnspan=2)
       
        self.dates = None
        self.times = None
        self.dv = None
        self.dep_date = tk.Entry(root)  # Hidden for now, used in animation
        self.transit_time = tk.Entry(root)  # Hidden for now, used in animation

        # Dictionary to store SPICE IDs
        self.body_ids = {}

    def get_body_list(self):
        if not self.kernels_loaded:
            messagebox.showwarning("Kernel Warning", "SPICE kernels are not loaded. Please update kernels.")
            return []
        
        try:
            bodies = [
                "MERCURY BARYCENTER", "VENUS BARYCENTER", "EARTH", "MARS BARYCENTER",
                "JUPITER BARYCENTER", "SATURN BARYCENTER", "URANUS BARYCENTER", "NEPTUNE BARYCENTER",
                "PLUTO BARYCENTER", "SUN"
            ]
            available_bodies = []
            for body in bodies:
                body_id = spice_interface.search_celestial_body(body)  # Assuming this exists
                if body_id is not None:
                    available_bodies.append(body)
                    self.body_ids[body] = body_id
            return sorted(available_bodies)
        except Exception as e:
            messagebox.showwarning("Body List Error", f"Failed to load body list: {e}")
            return ["No bodies found"]

    def update_kernels(self):
        try:
            spice_interface.check_and_download_kernels()  # Assuming this exists
            spice_interface.load_all_kernels()
            self.kernels_loaded = True
            messagebox.showinfo("Success", "SPICE kernels updated successfully.")
            self.dep_body['values'] = self.get_body_list()
            self.arr_body['values'] = self.get_body_list()
        except Exception as e:
            self.kernels_loaded = False
            messagebox.showerror("Error", f"Failed to update SPICE kernels: {e}")

    def estimate_time(self):
        try:
            res = int(self.resolution.get())
            if res <= 0:
                raise ValueError("Resolution must be positive")
            est_time = trajectory.estimate_time(res)
            hours = est_time / 3600
            msg = f"Estimated time for {res}x{res} grid: {hours:.2f} hours (GPU accelerated)"
            messagebox.showinfo("Time Estimate", msg)
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def update_progress(self, progress, dv, eta, dep_jds, tof_days):
        percent = progress * 100
        self.progress_var.set(percent)
        eta_str = f"{eta / 3600:.2f} hours" if eta > 0 else "Calculating..."
        self.progress_label.config(text=f"Progress: {percent:.1f}% (ETA: {eta_str})")
        
        if percent % 5 < 0.1 or percent > 99.9:
            self.ax.clear()
            pcm = self.ax.contourf(dep_jds, tof_days, dv, levels=50, cmap="viridis")
            self.fig.colorbar(pcm, ax=self.ax, label="Δv (m/s)")
            self.ax.set_xlabel("Departure Date (JD)")
            self.ax.set_ylabel("Transit Time (days)")
            self.ax.set_title(f"Porkchop Plot: {self.dep_body.get()} to {self.arr_body.get()}")
            self.canvas.draw()
        self.root.update_idletasks()

    def start_porkchop(self):
        try:
            start = self.start_date.get()
            end = self.end_date.get()
            min_t = float(self.min_time.get())
            max_t = float(self.max_time.get())
            res = int(self.resolution.get())
            dep_body = self.dep_body.get().strip()
            arr_body = self.arr_body.get().strip()
            
            if not dep_body or not arr_body:
                messagebox.showerror("Error", "Please select valid Departure and Arrival Bodies.")
                return
            
            if res <= 0 or min_t >= max_t:
                raise ValueError("Invalid resolution or time range")
            
            est_time = trajectory.estimate_time(res)
            if not messagebox.askyesno("Confirm", f"Estimated time: {est_time/3600:.2f} hours (GPU). Proceed?"):
                return
            
            self.progress_var.set(0)
            self.progress_label.config(text="Progress: 0% (ETA: Calculating...)")
            self.dates, self.times, self.dv = porkchop_data(
                start, end, min_t, max_t, res, dep_body, arr_body, update_callback=self.update_progress
            )
            
            self.ax.clear()
            pcm = self.ax.contourf(self.dates, self.times, self.dv, levels=50, cmap="viridis")
            self.fig.colorbar(pcm, ax=self.ax, label="Δv (m/s)")
            self.ax.set_xlabel("Departure Date (JD)")
            self.ax.set_ylabel("Transit Time (days)")
            self.ax.set_title(f"Porkchop Plot: {dep_body} to {arr_body}")
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {e}")

    def show_animation(self):
        messagebox.showinfo("Animation", "Animation not yet implemented.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryApp(root)
    root.mainloop()