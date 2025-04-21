#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 15:25:22 2025

@author: user

Track Visualizer App with Direction Angles and Compass
This script provides a GUI application for viewing track data with direction information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import FancyArrowPatch, Circle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import math

class TrackVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Track Visualizer")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Data variables
        self.df = None
        self.track_numbers = []
        self.current_track = None

        # Create the main layout
        self.create_layout()

    def create_layout(self):
        """Create the main application layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create top menu bar
        self.create_menu()

        # Create top panel with file selection and track navigation
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, padx=5, pady=5)

        # File selection
        file_frame = ttk.LabelFrame(top_panel, text="File Selection")
        file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=50)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=5, pady=5)

        load_btn = ttk.Button(file_frame, text="Load Data", command=self.load_data)
        load_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Track navigation
        track_frame = ttk.LabelFrame(top_panel, text="Track Navigation")
        track_frame.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        ttk.Label(track_frame, text="Track Number:").pack(side=tk.LEFT, padx=5, pady=5)

        self.track_entry = ttk.Entry(track_frame, width=10)
        self.track_entry.pack(side=tk.LEFT, padx=5, pady=5)

        go_btn = ttk.Button(track_frame, text="Go", command=self.go_to_track)
        go_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Create split view with plot on left and info panel on right
        split_frame = ttk.Frame(main_frame)
        split_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Plot panel
        plot_frame = ttk.LabelFrame(split_frame, text="Track Visualization")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure and canvas
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Compass panel
        compass_frame = ttk.LabelFrame(split_frame, text="Direction Compass", width=200)
        compass_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        compass_frame.pack_propagate(False)  # Prevent the frame from shrinking

        # Create compass figure
        self.compass_fig = plt.Figure(figsize=(2, 2), dpi=100)
        self.compass_ax = self.compass_fig.add_subplot(111, polar=True)

        self.compass_canvas = FigureCanvasTkAgg(self.compass_fig, master=compass_frame)
        self.compass_canvas.draw()
        self.compass_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create the compass
        self.create_compass()

        # Info panel
        info_frame = ttk.LabelFrame(split_frame, text="Track Information", width=300)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        info_frame.pack_propagate(False)  # Prevent the frame from shrinking

        # Track summary
        ttk.Label(info_frame, text="Track Summary", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)

        # Track info display with scrollbar
        info_text_frame = ttk.Frame(info_frame)
        info_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        info_scroll = ttk.Scrollbar(info_text_frame)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_text = tk.Text(info_text_frame, wrap=tk.WORD, width=40, height=20)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_scroll.config(command=self.info_text.yview)
        self.info_text.config(yscrollcommand=info_scroll.set)
        self.info_text.config(state=tk.DISABLED)

        # Track navigation buttons
        nav_frame = ttk.Frame(info_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)

        self.prev_btn = ttk.Button(nav_frame, text="Previous Track", command=self.previous_track)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(nav_frame, text="Next Track", command=self.next_track)
        self.next_btn.pack(side=tk.RIGHT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=2)

        # Disable navigation controls until data is loaded
        self.toggle_controls(False)

    def create_compass(self):
        """Create a directional compass to show angle orientation with inverted Y-axis"""
        # Clear previous compass
        self.compass_ax.clear()

        # Set direction labels (0° at right, 90° at top to match inverted Y-axis)
        self.compass_ax.set_theta_zero_location('E')  # 0 degrees at the right
        self.compass_ax.set_theta_direction(1)  # Counter-clockwise (to match inverted Y)

        # Set the ticks and labels
        angles = np.arange(0, 360, 45)
        labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
        self.compass_ax.set_thetagrids(angles, labels)

        # Remove radial ticks and labels
        self.compass_ax.set_rticks([])

        # Add circle at maximum radius
        circle = plt.Circle((0, 0), 0.9, transform=self.compass_ax.transData._b,
                          fill=False, color='gray')
        self.compass_ax.add_artist(circle)

        # Add directional markers (flipped N and S for inverted Y)
        self.compass_ax.annotate('E', xy=(1.0, 0.0), xytext=(1.1, 0.0),
                              ha='center', va='center', fontweight='bold')
        self.compass_ax.annotate('N', xy=(0.0, -1.0), xytext=(0.0, -1.1),
                              ha='center', va='center', fontweight='bold')
        self.compass_ax.annotate('W', xy=(-1.0, 0.0), xytext=(-1.1, 0.0),
                              ha='center', va='center', fontweight='bold')
        self.compass_ax.annotate('S', xy=(0.0, 1.0), xytext=(0.0, 1.1),
                              ha='center', va='center', fontweight='bold')

        # Add note about Y-axis inversion
        self.compass_ax.text(0, 0, "Y-Axis\nInverted", ha='center', va='center',
                          fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        # Set title
        self.compass_ax.set_title('Direction Convention', fontsize=10)

        # Set figure properties
        self.compass_fig.tight_layout()
        self.compass_canvas.draw()

    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda event: self.browse_file())

    def calculate_direction(self, x1, y1, x2, y2):
        """
        Calculate the direction of travel between two points, with inverted Y-axis.
        Returns angle in degrees (0-360) with 0 being right, 90 being down, etc.
        """
        dx = x2 - x1
        # Invert y-direction to match microscope display (y increases downward)
        dy = -(y2 - y1)  # Negate the difference

        # Handle the case where there's no movement
        if dx == 0 and dy == 0:
            return float('nan')

        # Calculate angle in radians
        angle_rad = math.atan2(dy, dx)

        # Convert to degrees (0-360 range)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def browse_file(self):
        """Open file browser to select a data file"""
        file_types = [("CSV files", "*.csv"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Select a track data file",
            filetypes=file_types
        )
        if filename:
            self.file_path.set(filename)

    def load_data(self):
        """Load track data from the selected file"""
        filename = self.file_path.get().strip()
        if not filename:
            messagebox.showwarning("No File Selected", "Please select a file first.")
            return

        if not os.path.exists(filename):
            messagebox.showerror("File Not Found", f"The file '{filename}' does not exist.")
            return

        try:
            # Load the data
            self.df = pd.read_csv(filename)

            # Check if direction columns exist - we'll recalculate them regardless
            required_cols = ['track_number', 'frame', 'x', 'y']
            missing_cols = [col for col in required_cols if col not in self.df.columns]

            if missing_cols:
                msg = f"Missing required columns: {missing_cols}"
                messagebox.showwarning("Missing Columns", msg)
                return

            # Calculate direction for each track based on inverted Y-axis
            print("Recalculating direction angles with inverted Y-axis...")

            # Create new columns if they don't exist
            if 'direction_degrees' not in self.df.columns:
                self.df['direction_degrees'] = np.nan
            if 'direction_radians' not in self.df.columns:
                self.df['direction_radians'] = np.nan
            if 'direction_x' not in self.df.columns:
                self.df['direction_x'] = np.nan
            if 'direction_y' not in self.df.columns:
                self.df['direction_y'] = np.nan

            # Process each track
            track_numbers = self.df['track_number'].unique()
            for track_num in track_numbers:
                # Get data for this track and sort by frame
                track_data = self.df[self.df['track_number'] == track_num].sort_values('frame')

                if len(track_data) < 2:
                    continue

                # Extract positions
                x_vals = track_data['x'].values
                y_vals = track_data['y'].values
                track_indices = track_data.index.values

                # Calculate direction for each step
                angles = []
                for i in range(len(x_vals) - 1):
                    angle = self.calculate_direction(x_vals[i], y_vals[i], x_vals[i+1], y_vals[i+1])
                    angles.append(angle)

                    # Update the direction for the current point (direction from this point to next)
                    idx = track_indices[i]
                    self.df.at[idx, 'direction_degrees'] = angle
                    self.df.at[idx, 'direction_radians'] = math.radians(angle)
                    self.df.at[idx, 'direction_x'] = math.cos(math.radians(angle))
                    self.df.at[idx, 'direction_y'] = -math.sin(math.radians(angle))  # Negative for display

                # Calculate directional persistence
                if len(angles) >= 2:
                    # Filter out NaN values
                    valid_angles = np.array([a for a in angles if not np.isnan(a)])

                    if len(valid_angles) >= 2:
                        # Calculate angular differences
                        angle_diffs = []
                        for i in range(1, len(valid_angles)):
                            diff = abs(valid_angles[i] - valid_angles[i-1])
                            # Get smallest angle between directions
                            if diff > 180:
                                diff = 360 - diff
                            angle_diffs.append(diff)

                        # Calculate persistence (1 means straight line, 0 means random)
                        persistence = 1 - (np.mean(angle_diffs) / 180)

                        # Update all rows for this track
                        self.df.loc[self.df['track_number'] == track_num, 'directional_persistence'] = persistence

            # Get track numbers
            self.track_numbers = sorted(self.df['track_number'].unique())

            if not self.track_numbers:
                messagebox.showwarning("No Tracks", "No tracks found in the data file.")
                return

            # Enable controls
            self.toggle_controls(True)

            # Display first track
            self.current_track = self.track_numbers[0]
            self.track_entry.delete(0, tk.END)
            self.track_entry.insert(0, str(self.current_track))

            # Update status
            self.status_var.set(f"Loaded {len(self.track_numbers)} tracks from {os.path.basename(filename)}")

            # Display the first track
            self.display_track(self.current_track)

        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {e}")

    def toggle_controls(self, enabled):
        """Enable or disable navigation controls"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.track_entry.config(state=state)
        self.prev_btn.config(state=state)
        self.next_btn.config(state=state)

    def go_to_track(self):
        """Navigate to the specified track number"""
        if not self.df is not None:
            return

        try:
            track_num = int(self.track_entry.get().strip())
            if track_num in self.track_numbers:
                self.current_track = track_num
                self.display_track(track_num)
            else:
                messagebox.showwarning("Invalid Track", f"Track {track_num} not found in data.")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid track number.")

    def previous_track(self):
        """Navigate to the previous track"""
        if self.current_track is None or self.df is None:
            return

        current_idx = self.track_numbers.index(self.current_track)
        if current_idx > 0:
            prev_idx = current_idx - 1
            self.current_track = self.track_numbers[prev_idx]
            self.track_entry.delete(0, tk.END)
            self.track_entry.insert(0, str(self.current_track))
            self.display_track(self.current_track)

    def next_track(self):
        """Navigate to the next track"""
        if self.current_track is None or self.df is None:
            return

        current_idx = self.track_numbers.index(self.current_track)
        if current_idx < len(self.track_numbers) - 1:
            next_idx = current_idx + 1
            self.current_track = self.track_numbers[next_idx]
            self.track_entry.delete(0, tk.END)
            self.track_entry.insert(0, str(self.current_track))
            self.display_track(self.current_track)

    def display_track(self, track_number):
        """Display a single track with direction arrows and angle values"""
        # Clear previous plot
        self.ax.clear()

        # Get data for the selected track
        track_data = self.df[self.df['track_number'] == track_number].sort_values('frame')

        if len(track_data) == 0:
            self.ax.text(0.5, 0.5, f"No data for track {track_number}",
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        # Get positions and directions
        x = track_data['x'].values
        y = track_data['y'].values
        frames = track_data['frame'].values

        # Use a gradient of colors
        colors = plt.cm.cool(np.linspace(0, 1, len(x)))

        # Plot track points
        self.ax.scatter(x, y, c=colors, s=50, alpha=0.8, zorder=2)

        # Plot track line
        self.ax.plot(x, y, 'k-', alpha=0.5, linewidth=1, zorder=1)

        # Plot direction arrows
        has_direction_data = 'direction_degrees' in track_data.columns and not track_data['direction_degrees'].isna().all()

        if has_direction_data:
            # Get direction data
            angles_deg = track_data['direction_degrees'].values
            angles_rad = track_data['direction_radians'].values if 'direction_radians' in track_data.columns else np.radians(angles_deg)

            # Scale arrow length based on track size
            x_range = max(x) - min(x)
            y_range = max(y) - min(y)
            scale = 0.1 * max(x_range, y_range)

            # Plot arrows for each point except the last one
            for i in range(len(x) - 1):
                if not np.isnan(angles_rad[i]):
                    # Calculate arrow components based on the angle
                    dx = scale * np.cos(angles_rad[i])
                    # Invert y component for display (since plot y-axis is inverted)
                    dy = -scale * np.sin(angles_rad[i])

                    arrow = FancyArrowPatch(
                        (x[i], y[i]), (x[i] + dx, y[i] + dy),
                        color=colors[i],
                        arrowstyle='->',
                        mutation_scale=15,
                        linewidth=2,
                        zorder=3
                    )
                    self.ax.add_patch(arrow)

                    # Add the angle value next to the arrow
                    angle_text = f"{angles_deg[i]:.1f}°"
                    text_x = x[i] + dx * 1.1
                    text_y = y[i] + dy * 1.1
                    self.ax.text(text_x, text_y, angle_text, fontsize=8,
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Number the points to show sequence
        for i in range(len(x)):
            self.ax.text(x[i], y[i], str(i), fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7, linewidth=0))

        # Set axis limits with some padding
        x_pad = 0.1 * max(0.1, x_range)
        y_pad = 0.1 * max(0.1, y_range)
        self.ax.set_xlim(min(x) - x_pad, max(x) + x_pad)
        self.ax.set_ylim(max(y) + y_pad, min(y) - y_pad)  # Invert y-axis

        # Set equal aspect ratio for proper display
        self.ax.set_aspect('equal')

        # Update labels and title
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title(f'Track {track_number} Visualization')

        # Draw the updated figure
        self.canvas.draw()

        # Update information panel
        self.update_info_panel(track_number, track_data)



    def update_info_panel(self, track_number, track_data):
        """Update the information panel with track details"""
        # Clear current info
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        # Get track information
        n_points = len(track_data)
        frames = track_data['frame'].values
        duration = frames.max() - frames.min() + 1
        current_idx = self.track_numbers.index(track_number)

        # Calculate total path length
        x = track_data['x'].values
        y = track_data['y'].values
        path_length = 0
        for i in range(1, len(x)):
            path_length += np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)

        # Get directional persistence if it exists
        persistence = track_data['directional_persistence'].iloc[0] if 'directional_persistence' in track_data.columns else "N/A"
        if isinstance(persistence, float) and not np.isnan(persistence):
            persistence = f"{persistence:.3f}"

        # Add basic track info
        self.info_text.insert(tk.END, "Basic Information\n", "section")
        self.info_text.insert(tk.END, f"Track Index: {current_idx} of {len(self.track_numbers)-1}\n")
        self.info_text.insert(tk.END, f"Track Number: {track_number}\n")
        self.info_text.insert(tk.END, f"Points: {n_points}\n")
        self.info_text.insert(tk.END, f"Frames: {int(frames.min())} to {int(frames.max())}\n")
        self.info_text.insert(tk.END, f"Duration: {duration} frames\n")
        self.info_text.insert(tk.END, f"Path Length: {path_length:.2f}\n")
        self.info_text.insert(tk.END, f"Directional Persistence: {persistence}\n\n")

        # Direction information
        self.info_text.insert(tk.END, "Direction Information\n", "section")

        if 'direction_degrees' in track_data.columns:
            angles = track_data['direction_degrees'].values
            valid_angles = angles[~np.isnan(angles)]

            if len(valid_angles) > 0:
                self.info_text.insert(tk.END, f"Mean Direction: {np.mean(valid_angles):.2f}°\n")
                self.info_text.insert(tk.END, f"Direction Range: {np.min(valid_angles):.2f}° to {np.max(valid_angles):.2f}°\n\n")

                # Direction details by point
                self.info_text.insert(tk.END, "Direction by Point:\n")
                for i in range(len(angles)-1):
                    if not np.isnan(angles[i]):
                        self.info_text.insert(tk.END, f"  Point {i}: {angles[i]:.2f}°\n")
            else:
                self.info_text.insert(tk.END, "No valid direction data found.\n")
        else:
            self.info_text.insert(tk.END, "Direction data not available.\n")

        # Position data
        self.info_text.insert(tk.END, "\nPosition Information\n", "section")
        self.info_text.insert(tk.END, f"X Range: {min(x):.2f} to {max(x):.2f}\n")
        self.info_text.insert(tk.END, f"Y Range: {min(y):.2f} to {max(y):.2f}\n")

        # Style the text
        self.info_text.tag_configure("section", font=("Arial", 10, "bold"))

        # Disable editing
        self.info_text.config(state=tk.DISABLED)

        # Update status bar
        self.status_var.set(f"Displaying Track {track_number} ({current_idx+1} of {len(self.track_numbers)})")

    def show_about(self):
        """Show about dialog"""
        about_text = """Track Visualizer App

This application visualizes tracked puncta with direction information.
It displays the movement path and direction of travel for each tracked point.

Features:
- Y-axis inverted to match microscope display
- Direction angles calculated with inverted Y-axis (0° = right, 90° = down)
- Direction compass to help interpret angles
- Direction arrows show travel between points

Created: April 21, 2025
"""
        messagebox.showinfo("About Track Visualizer", about_text)

def main():
    root = tk.Tk()
    app = TrackVisualizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
