"""
Milestone 2 GUI: Enhanced interface with visualization capabilities
Integrated with Beam Search Puzzle Solver using Jigsaw_solver.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import threading
import numpy as np
import sys
from Jigsaw_solver import JigsawSolver

# Add the directory containing the puzzle solver
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class BeamSearchSolver:
    """Wrapper for the JigsawSolver class"""

    def __init__(self, tiles_dir, grid_size=4, R=32, beam_width=20):
        self.tiles_dir = tiles_dir
        self.grid_size = grid_size
        self.R = R
        self.beam_width = beam_width

        # Initialize the solver from Jigsaw_solver.py
        self.solver = JigsawSolver(tiles_dir, grid_size, R, beam_width)

        # Results will be stored here
        self.tiles_original = None
        self.downsampled_tiles = None
        self.tile_edges_rotations = None
        self.solution = None
        self.reconstructed_img = None
        self.total_cost = None
        self.matches = []

    def load_tiles(self):
        """Load images from directory using JigsawSolver"""
        self.tiles_original = self.solver.load_tiles()
        return self.tiles_original

    def solve(self):
        """Solve the puzzle using JigsawSolver"""
        # Solve using JigsawSolver
        solution, total_cost = self.solver.solve()

        # Store results
        self.solution = solution
        self.total_cost = total_cost
        self.tiles_original = self.solver.tiles_original
        self.downsampled_tiles = self.solver.downsampled_tiles
        self.tile_edges_rotations = self.solver.tile_edges_rotations
        self.reconstructed_img = self.solver.reconstructed_img

        # Extract match information for visualization
        self.extract_match_info()

        # Return results in the expected format
        return {
            'solution': self.solution,
            'total_cost': self.total_cost,
            'progress': [],  # JigsawSolver doesn't provide progress tracking
            'grid_size': self.grid_size
        }

    def extract_match_info(self):
        """Extract match information for visualization"""
        self.matches = []

        if self.solution is None or self.tile_edges_rotations is None:
            return

        for idx, (tile_idx, rot) in enumerate(self.solution):
            r = idx // self.grid_size
            c = idx % self.grid_size

            # Check right neighbor
            if c < self.grid_size - 1:
                right_idx = r * self.grid_size + (c + 1)
                right_tile_idx, right_rot = self.solution[right_idx]

                # Compute cost using JigsawSolver's edge_cost method
                e1 = self.tile_edges_rotations[tile_idx][rot]["right"]
                e2 = self.tile_edges_rotations[right_tile_idx][right_rot]["left"]
                cost = np.mean(np.abs(e1.astype(np.float32) - e2.astype(np.float32)))

                self.matches.append({
                    'tile1': tile_idx,
                    'tile2': right_tile_idx,
                    'edge1': 'right',
                    'edge2': 'left',
                    'rotation1': rot,
                    'rotation2': right_rot,
                    'similarity': 1.0 / (1.0 + cost),  # Convert cost to similarity
                    'cost': cost,
                    'position': f"({r},{c})->({r},{c+1})"
                })

            # Check bottom neighbor
            if r < self.grid_size - 1:
                bottom_idx = (r + 1) * self.grid_size + c
                bottom_tile_idx, bottom_rot = self.solution[bottom_idx]

                # Compute cost using JigsawSolver's edge_cost method
                e1 = self.tile_edges_rotations[tile_idx][rot]["bottom"]
                e2 = self.tile_edges_rotations[bottom_tile_idx][bottom_rot]["top"]
                cost = np.mean(np.abs(e1.astype(np.float32) - e2.astype(np.float32)))

                self.matches.append({
                    'tile1': tile_idx,
                    'tile2': bottom_tile_idx,
                    'edge1': 'bottom',
                    'edge2': 'top',
                    'rotation1': rot,
                    'rotation2': bottom_rot,
                    'similarity': 1.0 / (1.0 + cost),  # Convert cost to similarity
                    'cost': cost,
                    'position': f"({r},{c})->({r+1},{c})"
                })

        # Sort matches by similarity (highest first)
        self.matches.sort(key=lambda x: x['similarity'], reverse=True)

        return self.matches

    def rotate_tile(self, tile, angle):
        """Rotate tile by angle (0, 90, 180, 270 degrees)"""
        return self.solver.rotate_tile(tile, angle)

    def reconstruct_image(self):
        """Reconstruct the final image from solution"""
        if self.solver.reconstructed_img is not None:
            self.reconstructed_img = self.solver.reconstructed_img
        return self.reconstructed_img


class PuzzleGUI_M2:
    def __init__(self, root):
        self.root = root
        self.root.title("Milestone 2 - Beam Search Puzzle Solver")
        self.root.geometry("1400x900")

        self.tiles_dir = None
        self.solver = None

        # Variables
        self.grid_size = tk.IntVar(value=4)
        self.downsample_size = tk.IntVar(value=32)
        self.beam_width = tk.IntVar(value=20)

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
        grid_combo = ttk.Combobox(control_frame, textvariable=self.grid_size,
                                  values=[2, 4, 8], width=8, state="readonly")
        grid_combo.pack(side=tk.LEFT, padx=5)

        # Downsample size
        ttk.Label(control_frame, text="Downsample:").pack(side=tk.LEFT, padx=(20, 5))
        downsample_combo = ttk.Combobox(control_frame, textvariable=self.downsample_size,
                                        values=[16, 32, 64, 128], width=8, state="readonly")
        downsample_combo.pack(side=tk.LEFT, padx=5)

        # Beam width
        ttk.Label(control_frame, text="Beam Width:").pack(side=tk.LEFT, padx=(20, 5))
        beam_combo = ttk.Combobox(control_frame, textvariable=self.beam_width,
                                  values=[10, 20, 30, 50, 100], width=8, state="readonly")
        beam_combo.pack(side=tk.LEFT, padx=5)

        # Process button
        self.process_btn = ttk.Button(control_frame, text="🧩 Solve Puzzle",
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
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=2)

        # Left panel - Tiles and Solution
        left_panel = ttk.LabelFrame(content_frame, text="Puzzle", padding="10")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Notebook for left panel
        left_notebook = ttk.Notebook(left_panel)
        left_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Original tiles
        tab1 = ttk.Frame(left_notebook)
        left_notebook.add(tab1, text="Original Tiles")

        # Scrollable canvas for tiles
        self.tiles_frame = tk.Frame(tab1)
        self.tiles_frame.pack(fill=tk.BOTH, expand=True)

        self.tiles_canvas = tk.Canvas(self.tiles_frame, bg='white')
        scrollbar_y = ttk.Scrollbar(self.tiles_frame, orient=tk.VERTICAL, command=self.tiles_canvas.yview)
        scrollbar_x = ttk.Scrollbar(tab1, orient=tk.HORIZONTAL, command=self.tiles_canvas.xview)

        self.tiles_canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tiles_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tab 2: Reconstructed puzzle
        tab2 = ttk.Frame(left_notebook)
        left_notebook.add(tab2, text="Solution")

        self.solution_canvas = tk.Canvas(tab2, bg='white')
        self.solution_canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel - Results and Visualization
        right_panel = ttk.LabelFrame(content_frame, text="Analysis", padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Notebook for right panel
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Statistics
        tab_stats = ttk.Frame(self.notebook)
        self.notebook.add(tab_stats, text="Statistics")

        self.stats_text = tk.Text(tab_stats, wrap=tk.WORD, height=15, font=("Consolas", 9))
        scrollbar_stats = ttk.Scrollbar(tab_stats, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar_stats.set)

        scrollbar_stats.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tab 2: Edge Matches
        tab_matches = ttk.Frame(self.notebook)
        self.notebook.add(tab_matches, text="Edge Matches")

        # Create a frame for canvas and scrollbar
        matches_container = tk.Frame(tab_matches)
        matches_container.pack(fill=tk.BOTH, expand=True)

        # Create canvas with scrollbar
        self.matches_canvas = tk.Canvas(matches_container, bg='white')
        matches_scrollbar = ttk.Scrollbar(matches_container, orient=tk.VERTICAL, command=self.matches_canvas.yview)

        self.matches_canvas.configure(yscrollcommand=matches_scrollbar.set)

        # Pack them properly
        matches_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.matches_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tab 3: Progress
        tab_progress = ttk.Frame(self.notebook)
        self.notebook.add(tab_progress, text="Beam Search Progress")

        self.progress_text = tk.Text(tab_progress, wrap=tk.WORD, height=15, font=("Consolas", 9))
        scrollbar_progress = ttk.Scrollbar(tab_progress, command=self.progress_text.yview)
        self.progress_text.config(yscrollcommand=scrollbar_progress.set)

        scrollbar_progress.pack(side=tk.RIGHT, fill=tk.Y)
        self.progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bottom panel - Log
        log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding="10")
        log_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=6, font=("Consolas", 9), wrap=tk.WORD)
        scrollbar_log = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar_log.set)

        scrollbar_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Save button
        ttk.Button(self.root, text="💾 Save Results",
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
            folder_name = os.path.basename(folder)
            self.folder_label.config(text=folder_name[:30] + "..." if len(folder_name) > 30 else folder_name)

            # Count tiles
            tile_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            tile_count = len(tile_files)

            self.status_label.config(text=f"Found {tile_count} tiles")
            self.process_btn.config(state=tk.NORMAL)

            # Auto-detect grid size
            sqrt = int(np.sqrt(tile_count))
            if sqrt * sqrt == tile_count and sqrt in [2, 3, 4, 5, 6, 7, 8]:
                self.grid_size.set(sqrt)

            # Display tiles
            self.display_tiles(folder)

    def display_tiles(self, folder):
        """Display loaded tiles"""
        self.tiles_canvas.delete("all")

        tiles = []
        tile_files = []

        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = cv2.imread(os.path.join(folder, fname))
                    if img is not None:
                        # Resize for display
                        img = cv2.resize(img, (100, 100))
                        tiles.append(img)
                        tile_files.append(fname)
                except:
                    continue

        if not tiles:
            self.tiles_canvas.create_text(200, 200, text="No tiles found",
                                          fill="gray", font=("Arial", 14))
            return

        # Create a grid of tiles
        n_cols = min(6, len(tiles))
        n_rows = (len(tiles) + n_cols - 1) // n_cols
        tile_size = 100
        spacing = 10
        label_height = 20

        # Calculate canvas size
        canvas_width = n_cols * (tile_size + spacing) - spacing + 20
        canvas_height = n_rows * (tile_size + spacing + label_height) - spacing + 20

        self.tiles_canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))

        if not hasattr(self, 'tile_images'):
            self.tile_images = []
        self.tile_images.clear()

        for i, (tile, fname) in enumerate(zip(tiles, tile_files)):
            row = i // n_cols
            col = i % n_cols

            x = 10 + col * (tile_size + spacing)
            y = 10 + row * (tile_size + spacing + label_height)

            # Convert to PIL Image
            rgb_img = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tk_img = ImageTk.PhotoImage(pil_img)

            # Display image
            self.tiles_canvas.create_image(x, y, anchor=tk.NW, image=tk_img)

            # Keep reference
            self.tile_images.append(tk_img)

            # Add label
            label = f"Tile {i}"
            self.tiles_canvas.create_text(x + tile_size // 2, y + tile_size + 5,
                                          text=label, font=("Arial", 8))

    def process_puzzle(self):
        """Process the puzzle using beam search"""
        if not self.tiles_dir:
            messagebox.showerror("Error", "Please select a tiles folder first")
            return

        # Clear previous results
        self.solution_canvas.delete("all")
        self.matches_canvas.delete("all")
        self.stats_text.delete(1.0, tk.END)
        self.progress_text.delete(1.0, tk.END)
        self.clear_log()

        # Update UI
        self.status_label.config(text="Processing...")
        self.process_btn.config(state=tk.DISABLED)

        # Run in thread
        def process_thread():
            try:
                self.log("=" * 60)
                self.log("Starting Beam Search Puzzle Solver")
                self.log("=" * 60)

                # Get parameters
                grid_size_val = self.grid_size.get()
                R_val = self.downsample_size.get()
                beam_width_val = self.beam_width.get()

                self.log(f"Parameters: Grid={grid_size_val}x{grid_size_val}, "
                        f"Downsample={R_val}, Beam Width={beam_width_val}")

                # Create solver using JigsawSolver
                self.solver = BeamSearchSolver(
                    tiles_dir=self.tiles_dir,
                    grid_size=grid_size_val,
                    R=R_val,
                    beam_width=beam_width_val
                )

                # Solve puzzle
                self.log("Running beam search...")
                result = self.solver.solve()

                self.log("✓ Puzzle solved successfully!")
                self.log(f"Total cost: {self.solver.total_cost:.2f}")

                # Update UI with results
                self.root.after(0, self.display_results)
                self.root.after(0, lambda: self.status_label.config(text="Complete"))

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.log(error_msg)
                import traceback
                self.log(traceback.format_exc())
                self.root.after(0, lambda: self.status_label.config(text="Error"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))

            finally:
                self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))

        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

    def display_results(self):
        """Display processing results"""
        if self.solver is None:
            return

        # Display solution
        if self.solver.reconstructed_img is not None:
            self.display_image_on_canvas(
                self.solver.reconstructed_img,
                self.solution_canvas,
                f"Solved Puzzle (Cost: {self.solver.total_cost:.2f})"
            )

        # Display match visualization
        if hasattr(self.solver, 'matches') and len(self.solver.matches) > 0:
            self.display_matches()

        # Display statistics
        self.display_statistics()

        # Display progress
        self.display_progress()

    def display_image_on_canvas(self, img, canvas, title):
        """Display an image on a canvas"""
        canvas.delete("all")

        if img is None:
            canvas.create_text(200, 200, text="No image available",
                               fill="gray", font=("Arial", 14))
            return

        try:
            # Convert BGR to RGB
            if len(img.shape) == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Get canvas dimensions
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            if canvas_width < 10 or canvas_height < 10:
                canvas_width, canvas_height = 400, 400

            h, w = rgb_img.shape[:2]

            # Calculate scale to fit canvas
            scale = min(canvas_width / w, canvas_height / h) * 0.9
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize image
            if new_w > 0 and new_h > 0:
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
        except Exception as e:
            canvas.create_text(200, 200, text=f"Error displaying image: {str(e)}",
                               fill="red", font=("Arial", 10))

    def display_matches(self):
        """Display edge matches"""
        canvas = self.matches_canvas
        canvas.delete("all")

        if not hasattr(self.solver, 'matches') or len(self.solver.matches) == 0:
            canvas.create_text(200, 200, text="No matches found",
                               fill="gray", font=("Arial", 14))
            return

        # Display top matches
        top_matches = self.solver.matches  # Show ALL matches

        match_height = 80
        spacing = 15

        # Calculate total height needed
        total_height = 30 + len(top_matches) * (match_height + spacing)

        # Get container
        container = canvas.master
        container.update_idletasks()
        container_width = container.winfo_width()

        if container_width < 10:
            container_width = 400

        # Set up scrollable area
        canvas_width = container_width - 20
        canvas.config(scrollregion=(0, 0, canvas_width, total_height))

        canvas.create_text(canvas_width // 2, 10,
                           text=f"Top {len(top_matches)} Edge Matches (by similarity)",
                           font=("Arial", 10, "bold"), anchor=tk.N)

        # Rest of your existing drawing code continues here...
        # (The for loop that creates matches with create_image, create_text, etc.)

        for i, match in enumerate(top_matches):
            y = 30 + i * (match_height + spacing)

            # Get tile indices and info
            tile1_idx = match['tile1']
            tile2_idx = match['tile2']

            # Try to get tile images (use downsized versions)
            if self.solver.downsampled_tiles is not None:
                tile1 = self.solver.rotate_tile(
                    self.solver.downsampled_tiles[tile1_idx],
                    match['rotation1']
                )
                tile2 = self.solver.rotate_tile(
                    self.solver.downsampled_tiles[tile2_idx],
                    match['rotation2']
                )

                # Resize for display
                display_size = 60
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
                start_x = 20

                # Display tiles
                canvas.create_image(start_x, y, anchor=tk.NW, image=tk_img1)
                canvas.create_image(start_x + display_size + 120, y, anchor=tk.NW, image=tk_img2)

                # Add match info
                info = f"T{tile1_idx}({match['rotation1']}°) {match['edge1']} ↔ T{tile2_idx}({match['rotation2']}°) {match['edge2']}"
                canvas.create_text(start_x + display_size + 60, y + display_size // 2,
                                   text=info, font=("Arial", 9), anchor=tk.CENTER)

                # Add similarity score
                score_text = f"Similarity: {match['similarity']:.3f} (Cost: {match['cost']:.3f})"
                canvas.create_text(start_x + display_size + 60, y + display_size // 2 + 20,
                                   text=score_text, font=("Arial", 8), fill="blue", anchor=tk.CENTER)

                # Draw arrow
                canvas.create_line(start_x + display_size, y + display_size // 2,
                                   start_x + display_size + 120, y + display_size // 2,
                                   fill="green", width=2, arrow=tk.LAST)

                # Keep references
                if not hasattr(self, 'match_images'):
                    self.match_images = []
                self.match_images.extend([tk_img1, tk_img2])

    def display_statistics(self):
        """Display statistics"""
        if self.solver is None:
            return

        self.stats_text.delete(1.0, tk.END)

        # Basic info
        self.stats_text.insert(tk.END, "PUZZLE SOLVING STATISTICS\n")
        self.stats_text.insert(tk.END, "=" * 40 + "\n\n")

        self.stats_text.insert(tk.END, f"Tiles Directory: {self.solver.tiles_dir}\n")
        self.stats_text.insert(tk.END, f"Grid Size: {self.solver.grid_size}x{self.solver.grid_size}\n")
        self.stats_text.insert(tk.END, f"Number of Tiles: {len(self.solver.tiles_original)}\n")
        self.stats_text.insert(tk.END, f"Downsample Size: {self.solver.R}\n")
        self.stats_text.insert(tk.END, f"Beam Width: {self.solver.beam_width}\n")
        self.stats_text.insert(tk.END, f"Total Cost: {self.solver.total_cost:.4f}\n\n")

        # Solution info
        self.stats_text.insert(tk.END, "SOLUTION\n")
        self.stats_text.insert(tk.END, "-" * 20 + "\n")

        if self.solver.solution:
            # Display solution as grid
            self.stats_text.insert(tk.END, "Tile Grid (Tile Index, Rotation):\n")
            for r in range(self.solver.grid_size):
                row_text = []
                for c in range(self.solver.grid_size):
                    idx = r * self.solver.grid_size + c
                    tile_idx, rotation = self.solver.solution[idx]
                    row_text.append(f"({tile_idx}, {rotation}°)")
                self.stats_text.insert(tk.END, "  " + " | ".join(row_text) + "\n")

        # Match statistics
        if hasattr(self.solver, 'matches') and len(self.solver.matches) > 0:
            self.stats_text.insert(tk.END, "\nEDGE MATCH STATISTICS\n")
            self.stats_text.insert(tk.END, "-" * 25 + "\n")
            self.stats_text.insert(tk.END, f"Total Edge Matches: {len(self.solver.matches)}\n")

            costs = [m['cost'] for m in self.solver.matches]
            similarities = [m['similarity'] for m in self.solver.matches]

            self.stats_text.insert(tk.END, f"Best Match (Lowest Cost): {min(costs):.4f}\n")
            self.stats_text.insert(tk.END, f"Worst Match (Highest Cost): {max(costs):.4f}\n")
            self.stats_text.insert(tk.END, f"Average Cost: {np.mean(costs):.4f}\n")
            self.stats_text.insert(tk.END, f"Median Cost: {np.median(costs):.4f}\n")
            self.stats_text.insert(tk.END, f"Average Similarity: {np.mean(similarities):.4f}\n\n")

            # Edge type distribution
            edge_counts = {}
            for match in self.solver.matches:
                key = f"{match['edge1']}-{match['edge2']}"
                edge_counts[key] = edge_counts.get(key, 0) + 1

            self.stats_text.insert(tk.END, "EDGE TYPE DISTRIBUTION\n")
            self.stats_text.insert(tk.END, "-" * 25 + "\n")
            for edge_type, count in sorted(edge_counts.items()):
                self.stats_text.insert(tk.END, f"{edge_type}: {count} matches\n")
        else:
            self.stats_text.insert(tk.END, "\nNo edge match data available.\n")

        self.stats_text.config(state=tk.DISABLED)

    def display_progress(self):
        """Display beam search progress"""
        if self.solver is None:
            return

        self.progress_text.delete(1.0, tk.END)

        # Display progress information
        self.progress_text.insert(tk.END, "BEAM SEARCH PROGRESS\n")
        self.progress_text.insert(tk.END, "=" * 25 + "\n\n")

        self.progress_text.insert(tk.END, f"Total positions processed: {self.solver.grid_size * self.solver.grid_size}\n")
        self.progress_text.insert(tk.END, f"Beam width maintained: {self.solver.beam_width}\n")
        self.progress_text.insert(tk.END, f"Final solution cost: {self.solver.total_cost:.4f}\n\n")

        self.progress_text.insert(tk.END, "SOLUTION TILES AND ROTATIONS:\n")
        self.progress_text.insert(tk.END, "-" * 30 + "\n")

        for idx, (tile_idx, rotation) in enumerate(self.solver.solution):
            row = idx // self.solver.grid_size
            col = idx % self.solver.grid_size
            self.progress_text.insert(tk.END, f"Position ({row},{col}): Tile {tile_idx}, Rotation {rotation}°\n")

        self.progress_text.config(state=tk.DISABLED)

    def save_results(self):
        """Save all results"""
        if self.solver is None:
            messagebox.showwarning("Warning", "No results to save. Solve a puzzle first.")
            return

        try:
            # Create output directory
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"puzzle_results_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            # Save reconstructed image
            if self.solver.reconstructed_img is not None:
                recon_path = os.path.join(output_dir, "reconstructed_puzzle.jpg")
                cv2.imwrite(recon_path, self.solver.reconstructed_img)
                self.log(f"✓ Saved reconstructed image to {recon_path}")

            # Save solution data
            import json
            solution_data = {
                'grid_size': self.solver.grid_size,
                'total_cost': float(self.solver.total_cost),
                'solution': [(int(idx), int(rot)) for idx, rot in self.solver.solution],
                'parameters': {
                    'downsample_size': self.solver.R,
                    'beam_width': self.solver.beam_width
                }
            }

            solution_path = os.path.join(output_dir, "solution.json")
            with open(solution_path, 'w') as f:
                json.dump(solution_data, f, indent=2)

            # Save match data
            if hasattr(self.solver, 'matches') and len(self.solver.matches) > 0:
                matches_data = []
                for match in self.solver.matches:
                    matches_data.append({
                        'tile1': int(match['tile1']),
                        'tile2': int(match['tile2']),
                        'rotation1': int(match['rotation1']),
                        'rotation2': int(match['rotation2']),
                        'edge1': match['edge1'],
                        'edge2': match['edge2'],
                        'similarity': float(match['similarity']),
                        'cost': float(match['cost']),
                        'position': match['position']
                    })

                matches_path = os.path.join(output_dir, "matches.json")
                with open(matches_path, 'w') as f:
                    json.dump(matches_data, f, indent=2)
                self.log(f"✓ Saved match data to {matches_path}")

            # Save statistics as text
            stats_path = os.path.join(output_dir, "statistics.txt")
            self.stats_text.config(state=tk.NORMAL)
            stats_content = self.stats_text.get(1.0, tk.END)
            with open(stats_path, 'w') as f:
                f.write(stats_content)
            self.stats_text.config(state=tk.DISABLED)

            self.log(f"✓ Saved statistics to {stats_path}")
            messagebox.showinfo("Success", f"Results saved to:\n{os.path.abspath(output_dir)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI_M2(root)
    root.mainloop()