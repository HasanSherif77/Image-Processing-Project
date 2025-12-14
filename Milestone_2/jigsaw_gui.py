"""
Milestone 2 GUI: Enhanced interface with visualization capabilities
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import threading


class PuzzleGUI_M2:
    def __init__(self, root):
        self.root = root
        self.root.title("Milestone 2 - Jigsaw Puzzle Solver with Visualization")
        self.root.geometry("1000x800")

        self.tiles_dir = None
        self.solver = None

        # Variables
        self.grid_size = tk.IntVar(value=2)
        self.threshold = tk.DoubleVar(value=0.5)
        self.method_var = tk.StringVar(value="greedy")

        self.create_widgets()

    def create_widgets(self):
        # Configure grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="ew")

        # File selection
        ttk.Button(control_frame, text="📂 Select Tiles Folder",
                   command=self.select_folder, width=20).pack(side=tk.LEFT, padx=5)

        self.folder_label = ttk.Label(control_frame, text="No folder selected", width=30)
        self.folder_label.pack(side=tk.LEFT, padx=10)

        # Grid size
        ttk.Label(control_frame, text="Grid Size:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Combobox(control_frame, textvariable=self.grid_size,
                     values=[2, 3, 4], width=8, state="readonly").pack(side=tk.LEFT, padx=5)

        # Threshold
        ttk.Label(control_frame, text="Match Threshold:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.threshold,
                  orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, textvariable=self.threshold).pack(side=tk.LEFT, padx=5)

        # Method selection
        ttk.Label(control_frame, text="Method:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Combobox(control_frame, textvariable=self.method_var,
                     values=["greedy", "bruteforce"], width=10, state="readonly").pack(side=tk.LEFT, padx=5)

        # Process button
        self.process_btn = ttk.Button(control_frame, text="Process & Visualize",
                                      command=self.process_puzzle, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Configure content grid
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)

        # Left panel - Original tiles
        left_panel = ttk.LabelFrame(content_frame, text="Original Tiles", padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.tiles_canvas = tk.Canvas(left_panel, bg='white')
        self.tiles_canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel - Results
        right_panel = ttk.LabelFrame(content_frame, text="Results", padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Notebook for multiple views
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Reconstructed puzzle
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Reconstruction")

        self.recon_canvas = tk.Canvas(tab1, bg='white')
        self.recon_canvas.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Matches
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Edge Matches")

        self.matches_canvas = tk.Canvas(tab2, bg='white')
        self.matches_canvas.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Statistics
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Statistics")

        self.stats_text = tk.Text(tab3, wrap=tk.WORD, height=15)
        scrollbar = ttk.Scrollbar(tab3, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bottom panel - Log
        log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding="10")
        log_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=6, font=("Consolas", 9), wrap=tk.WORD)
        scrollbar_log = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar_log.set)

        scrollbar_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Save button
        ttk.Button(self.root, text="💾 Save All Results",
                   command=self.save_results).grid(row=3, column=0, pady=10)

    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)

    def select_folder(self):
        """Select tiles folder"""
        folder = filedialog.askdirectory(title="Select Tiles Folder")
        if folder:
            self.tiles_dir = folder
            self.folder_label.config(text=os.path.basename(folder))

            # Count tiles
            tile_count = len([f for f in os.listdir(folder)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            self.status_label.config(text=f"Found {tile_count} tiles")
            self.process_btn.config(state=tk.NORMAL)

            # Display tiles
            self.display_tiles(folder)

    def display_tiles(self, folder):
        """Display loaded tiles"""
        self.tiles_canvas.delete("all")

        tiles = []
        tile_files = []

        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(folder, fname))
                if img is not None:
                    # Resize for display
                    img = cv2.resize(img, (100, 100))
                    tiles.append(img)
                    tile_files.append(fname)

        if not tiles:
            self.tiles_canvas.create_text(200, 200, text="No tiles found",
                                          fill="gray", font=("Arial", 14))
            return

        # Create a grid of tiles
        canvas_width = self.tiles_canvas.winfo_width()
        if canvas_width < 10:  # Canvas not yet sized
            canvas_width = 400

        n_cols = 4
        n_rows = (len(tiles) + n_cols - 1) // n_cols
        tile_size = 100
        spacing = 10

        # Calculate starting position to center
        total_width = n_cols * (tile_size + spacing) - spacing
        start_x = (canvas_width - total_width) // 2 if canvas_width > total_width else 10

        for i, (tile, fname) in enumerate(zip(tiles, tile_files)):
            row = i // n_cols
            col = i % n_cols

            x = start_x + col * (tile_size + spacing)
            y = 10 + row * (tile_size + spacing + 20)  # Extra space for label

            # Convert to PIL Image
            rgb_img = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tk_img = ImageTk.PhotoImage(pil_img)

            # Display image
            img_id = self.tiles_canvas.create_image(x, y, anchor=tk.NW, image=tk_img)

            # Keep reference
            if not hasattr(self, 'tile_images'):
                self.tile_images = []
            self.tile_images.append(tk_img)

            # Add label
            self.tiles_canvas.create_text(x + tile_size // 2, y + tile_size + 5,
                                          text=f"Tile {i + 1}", font=("Arial", 8))

    def process_puzzle(self):
        """Process the puzzle"""
        if not self.tiles_dir:
            messagebox.showerror("Error", "Please select a tiles folder first")
            return

        # Clear previous results
        self.recon_canvas.delete("all")
        self.matches_canvas.delete("all")
        self.stats_text.delete(1.0, tk.END)
        self.clear_log()

        # Update UI
        self.status_label.config(text="Processing...")
        self.process_btn.config(state=tk.DISABLED)

        # Run in thread
        def process_thread():
            try:
                self.log("=" * 60)
                self.log("Starting Milestone 2 Puzzle Solver")
                self.log("=" * 60)

                # Import and create solver
                from Jigsaw_solver import JigsawSolver

                self.solver = JigsawSolver(
                    tiles_dir=self.tiles_dir,
                    grid_size=self.grid_size.get(),
                    tile_size=128
                )

                # Load tiles
                self.log("Loading tiles...")
                self.solver.load_tiles()

                # Find matches
                self.log(f"Finding matches with threshold={self.threshold.get()}...")
                self.solver.find_all_matches(threshold=self.threshold.get())

                # Solve puzzle
                self.log(f"Solving puzzle using {self.method_var.get()} method...")
                self.solver.solve_puzzle(method=self.method_var.get())

                # Update UI with results
                self.root.after(0, self.display_results)

                self.log("\n✓ Processing complete!")
                self.root.after(0, lambda: self.status_label.config(text="Complete"))

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.log(error_msg)
                self.root.after(0, lambda: self.status_label.config(text="Error"))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

            finally:
                self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))

        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

    def display_results(self):
        """Display processing results"""
        if self.solver is None:
            return

        # Display reconstructed puzzle
        if self.solver.reconstructed_img is not None:
            self.display_image_on_canvas(
                self.solver.reconstructed_img,
                self.recon_canvas,
                "Reconstructed Puzzle"
            )

        # Display match visualization
        if len(self.solver.matches) > 0:
            self.display_matches()

        # Display statistics
        self.display_statistics()

    def display_image_on_canvas(self, img, canvas, title):
        """Display an image on a canvas"""
        canvas.delete("all")

        if img is None:
            canvas.create_text(200, 200, text="No image available",
                               fill="gray", font=("Arial", 14))
            return

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 400, 400

        h, w = rgb_img.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.9
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(rgb_img, (new_w, new_h))

        # Convert to PIL Image
        pil_img = Image.fromarray(resized_img)
        tk_img = ImageTk.PhotoImage(pil_img)

        # Display centered
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2

        canvas.create_image(x, y, anchor=tk.NW, image=tk_img)
        canvas.create_text(canvas_width // 2, 20, text=title,
                           font=("Arial", 12, "bold"), fill="black")

        # Keep reference
        if not hasattr(self, 'display_images'):
            self.display_images = []
        self.display_images.append(tk_img)

    def display_matches(self):
        """Display edge matches"""
        canvas = self.matches_canvas
        canvas.delete("all")

        if len(self.solver.matches) == 0:
            canvas.create_text(200, 200, text="No matches found",
                               fill="gray", font=("Arial", 14))
            return

        # Display top 4 matches
        top_matches = self.solver.matches[:4]

        canvas_width = canvas.winfo_width()
        if canvas_width < 10:
            canvas_width = 400

        match_height = 150
        spacing = 20

        for i, match in enumerate(top_matches):
            y = 10 + i * (match_height + spacing)

            # Get tile images
            tile1 = self.solver.tile_features[match['tile1']]['rotations'][match['rotation1']]['image']
            tile2 = self.solver.tile_features[match['tile2']]['rotations'][match['rotation2']]['image']

            # Resize for display
            display_size = 80
            tile1_display = cv2.resize(tile1, (display_size, display_size))
            tile2_display = cv2.resize(tile2, (display_size, display_size))

            # Convert to PIL
            tile1_rgb = cv2.cvtColor(tile1_display, cv2.COLOR_BGR2RGB)
            tile2_rgb = cv2.cvtColor(tile2_display, cv2.COLOR_BGR2RGB)

            pil1 = Image.fromarray(tile1_rgb)
            pil2 = Image.fromarray(tile2_rgb)

            tk_img1 = ImageTk.PhotoImage(pil1)
            tk_img2 = ImageTk.PhotoImage(pil2)

            # Calculate positions
            total_width = 2 * display_size + 100
            start_x = (canvas_width - total_width) // 2 if canvas_width > total_width else 10

            # Display tiles
            canvas.create_image(start_x, y, anchor=tk.NW, image=tk_img1)
            canvas.create_image(start_x + display_size + 100, y, anchor=tk.NW, image=tk_img2)

            # Draw connecting line
            x1 = start_x + display_size
            x2 = start_x + display_size + 100
            mid_y = y + display_size // 2

            canvas.create_line(x1, mid_y, x2, mid_y, fill="red", width=2)
            canvas.create_oval(x1 - 3, mid_y - 3, x1 + 3, mid_y + 3, fill="yellow")
            canvas.create_oval(x2 - 3, mid_y - 3, x2 + 3, mid_y + 3, fill="yellow")

            # Add match info
            info = f"T{match['tile1']} {match['edge1']} ↔ T{match['tile2']} {match['edge2']}"
            canvas.create_text(start_x + total_width // 2, y + display_size + 10,
                               text=info, font=("Arial", 9))

            # Add similarity score
            score_text = f"Similarity: {match['similarity']:.3f}"
            canvas.create_text(start_x + total_width // 2, y + display_size + 25,
                               text=score_text, font=("Arial", 9), fill="blue")

            # Keep references
            if not hasattr(self, 'match_images'):
                self.match_images = []
            self.match_images.extend([tk_img1, tk_img2])

        # Add title
        canvas.create_text(canvas_width // 2, 5, text="Top Edge Matches",
                           font=("Arial", 12, "bold"), anchor=tk.N)

    def display_statistics(self):
        """Display statistics"""
        if self.solver is None:
            return

        self.stats_text.delete(1.0, tk.END)

        # Basic info
        self.stats_text.insert(tk.END, "PUZZLE SOLVING STATISTICS\n")
        self.stats_text.insert(tk.END, "=" * 30 + "\n\n")

        self.stats_text.insert(tk.END, f"Tiles Directory: {self.solver.tiles_dir}\n")
        self.stats_text.insert(tk.END, f"Grid Size: {self.solver.grid_size}x{self.solver.grid_size}\n")
        self.stats_text.insert(tk.END, f"Number of Tiles: {len(self.solver.tiles)}\n")
        self.stats_text.insert(tk.END, f"Match Threshold: {self.threshold.get()}\n")
        self.stats_text.insert(tk.END, f"Solving Method: {self.method_var.get()}\n\n")

        # Match statistics
        if len(self.solver.matches) > 0:
            self.stats_text.insert(tk.END, "MATCH STATISTICS\n")
            self.stats_text.insert(tk.END, "-" * 20 + "\n")
            self.stats_text.insert(tk.END, f"Total Matches Found: {len(self.solver.matches)}\n")

            similarities = [m['similarity'] for m in self.solver.matches]
            self.stats_text.insert(tk.END, f"Best Match Score: {min(similarities):.3f}\n")
            self.stats_text.insert(tk.END, f"Worst Match Score: {max(similarities):.3f}\n")
            self.stats_text.insert(tk.END, f"Average Match Score: {np.mean(similarities):.3f}\n")
            self.stats_text.insert(tk.END, f"Median Match Score: {np.median(similarities):.3f}\n\n")

            # Match type distribution
            edge_counts = {}
            for match in self.solver.matches:
                key = f"{match['edge1']}-{match['edge2']}"
                edge_counts[key] = edge_counts.get(key, 0) + 1

            self.stats_text.insert(tk.END, "MATCH TYPE DISTRIBUTION\n")
            self.stats_text.insert(tk.END, "-" * 20 + "\n")
            for edge_type, count in sorted(edge_counts.items()):
                self.stats_text.insert(tk.END, f"{edge_type}: {count} matches\n")
        else:
            self.stats_text.insert(tk.END, "No matches found.\n")

        self.stats_text.config(state=tk.DISABLED)

    def save_results(self):
        """Save all results"""
        if self.solver is None:
            messagebox.showwarning("Warning", "No results to save. Process a puzzle first.")
            return

        try:
            self.solver.save_results()
            messagebox.showinfo("Success", "All results saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI_M2(root)
    root.mainloop()