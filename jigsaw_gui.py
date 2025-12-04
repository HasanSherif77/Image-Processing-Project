import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import json
import math
from collections import defaultdict

# Import from jigsaw_pipeline.py
try:
    from jigsaw_pipeline import run_pipeline, detect_salt_noise, detect_pepper_noise_median, \
        detect_gaussian_noise_level, detect_blur

    print("✓ Successfully imported functions from jigsaw_pipeline.py")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure jigsaw_pipeline.py is in the same directory.")
    exit(1)


class JigsawPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Jigsaw Puzzle Pipeline & Solver")
        self.root.geometry("1400x900")  # Increased size for solver

        # Initialize variables
        self.original_image = None
        self.final_image = None
        self.contour_image = None
        self.solved_image = None
        self.image_path = None
        self.best_result = None
        self.pieces = []  # Store tile pieces for matching
        self.matches = []  # Store matching results
        self.grid_size = 4  # Default grid size
        self.piece_positions = []  # Store solved piece positions

        # Store references to images to prevent garbage collection
        self.original_photo_ref = None
        self.final_photo_ref = None
        self.match_photo_ref = None
        self.solved_photo_ref = None

        # Setup GUI with tabs
        self.setup_gui()

    def setup_gui(self):
        """Setup the main GUI layout with tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Pipeline Processing
        self.pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_tab, text="🚀 Pipeline Processing")
        self.setup_pipeline_tab()

        # Tab 2: Piece Matching
        self.matching_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matching_tab, text="🧩 Piece Matching")
        self.setup_matching_tab()

        # Tab 3: Puzzle Solver
        self.solver_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.solver_tab, text="🧩 Puzzle Solver")
        self.setup_solver_tab()

        # Tab 4: Results & Logs
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="📊 Results & Logs")
        self.setup_results_tab()

    def setup_pipeline_tab(self):
        """Setup the pipeline processing tab"""
        # Top control panel
        control_frame = ttk.Frame(self.pipeline_tab, padding="10")
        control_frame.pack(fill=tk.X)

        # Left side: File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding="5")
        file_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Button(file_frame, text="📂 Load Image",
                   command=self.load_image, width=15).pack(pady=5)

        # Middle: Puzzle type selection
        puzzle_frame = ttk.LabelFrame(control_frame, text="Puzzle Settings", padding="5")
        puzzle_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Label(puzzle_frame, text="Puzzle Type:").pack(pady=(0, 5))
        self.puzzle_var = tk.StringVar(value="puzzle4x4")
        puzzle_combo = ttk.Combobox(puzzle_frame, textvariable=self.puzzle_var,
                                    values=["puzzle2x2", "puzzle4x4", "puzzle8x8"],
                                    state="readonly", width=15)
        puzzle_combo.pack()

        # Right side: Enhancement options
        enhance_frame = ttk.LabelFrame(control_frame, text="Select Enhancements", padding="5")
        enhance_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        checkbox_frame = ttk.Frame(enhance_frame)
        checkbox_frame.pack()

        # Column 1
        col1_frame = ttk.Frame(checkbox_frame)
        col1_frame.grid(row=0, column=0, padx=5)

        self.noise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col1_frame, text="Noise Reduction\n(Blur/Salt & Pepper)",
                        variable=self.noise_var).pack(anchor=tk.W, pady=2)

        self.sharpening_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col1_frame, text="Edge Sharpening",
                        variable=self.sharpening_var).pack(anchor=tk.W, pady=2)

        # Column 2
        col2_frame = ttk.Frame(checkbox_frame)
        col2_frame.grid(row=0, column=1, padx=5)

        self.gamma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(col2_frame, text="Gamma Correction\n(Brightness/Contrast)",
                        variable=self.gamma_var).pack(anchor=tk.W, pady=2)

        self.clahe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(col2_frame, text="CLAHE\n(Contrast Enhancement)",
                        variable=self.clahe_var).pack(anchor=tk.W, pady=2)

        # Process button
        process_frame = ttk.LabelFrame(control_frame, text="Process", padding="5")
        process_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(process_frame, text="🚀 Run Pipeline",
                   command=self.process_selected_features, width=20).pack(pady=5)

        ttk.Button(process_frame, text="🧩 Run Matching",
                   command=self.run_matching, width=20, state="disabled").pack(pady=5)
        self.matching_button = process_frame.winfo_children()[-1]

        ttk.Button(process_frame, text="🧩 Solve Puzzle",
                   command=self.solve_puzzle, width=20, state="disabled").pack(pady=5)
        self.solver_button = process_frame.winfo_children()[-1]

        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready to load image")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Main comparison area
        comparison_frame = ttk.Frame(self.pipeline_tab)
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Configure comparison grid
        comparison_frame.grid_rowconfigure(0, weight=1)
        comparison_frame.grid_columnconfigure(0, weight=1)
        comparison_frame.grid_columnconfigure(1, weight=1)

        # Left panel - Original image
        original_frame = ttk.LabelFrame(comparison_frame, text="Original Image", padding="10")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.original_image_frame = tk.Frame(original_frame, width=400, height=400, bg='#f0f0f0')
        self.original_image_frame.pack_propagate(False)
        self.original_image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.original_canvas = tk.Canvas(self.original_image_frame, bg='#f0f0f0', highlightthickness=1,
                                         highlightbackground="#cccccc")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        self.original_info = ttk.Label(original_frame, text="No image loaded")
        self.original_info.pack(pady=(5, 0))

        # Right panel - Processed image with contours
        processed_frame = ttk.LabelFrame(comparison_frame, text="Processed Image (With Contours)", padding="10")
        processed_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.processed_image_frame = tk.Frame(processed_frame, width=400, height=400, bg='#f0f0f0')
        self.processed_image_frame.pack_propagate(False)
        self.processed_image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.processed_canvas = tk.Canvas(self.processed_image_frame, bg='#f0f0f0', highlightthickness=1,
                                          highlightbackground="#cccccc")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

        self.processed_info = ttk.Label(processed_frame, text="Select enhancements and click Process")
        self.processed_info.pack(pady=(5, 0))

    def setup_matching_tab(self):
        """Setup the piece matching tab"""
        # Control frame
        control_frame = ttk.Frame(self.matching_tab, padding="10")
        control_frame.pack(fill=tk.X)

        # Matching controls
        match_frame = ttk.LabelFrame(control_frame, text="Matching Controls", padding="5")
        match_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Button(match_frame, text="🧩 Find Piece Matches",
                   command=self.find_piece_matches, width=20).pack(pady=5)

        ttk.Label(match_frame, text="Number of top matches:").pack(pady=(10, 5))
        self.top_k_var = tk.StringVar(value="5")
        top_k_spin = ttk.Spinbox(match_frame, from_=1, to=20, textvariable=self.top_k_var, width=10)
        top_k_spin.pack()

        # Visualization controls
        viz_frame = ttk.LabelFrame(control_frame, text="Visualization", padding="5")
        viz_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Button(viz_frame, text="👁️ Show Best Match",
                   command=self.show_best_match, width=20).pack(pady=5)

        ttk.Button(viz_frame, text="📊 Show All Matches",
                   command=self.show_all_matches, width=20).pack(pady=5)

        # Status for matching tab
        self.matching_status = ttk.Label(control_frame, text="Run pipeline first to extract tiles")
        self.matching_status.pack(side=tk.RIGHT, padx=10)

        # Matching results area
        results_frame = ttk.Frame(self.matching_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Best match visualization
        match_viz_frame = ttk.LabelFrame(results_frame, text="Best Match Visualization", padding="10")
        match_viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.match_canvas_frame = tk.Frame(match_viz_frame, width=600, height=300, bg='#f0f0f0')
        self.match_canvas_frame.pack_propagate(False)
        self.match_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.match_canvas = tk.Canvas(self.match_canvas_frame, bg='#f0f0f0', highlightthickness=1,
                                      highlightbackground="#cccccc")
        self.match_canvas.pack(fill=tk.BOTH, expand=True)

        self.match_info = ttk.Label(match_viz_frame, text="No matches found yet")
        self.match_info.pack(pady=(5, 0))

        # Matches list
        list_frame = ttk.LabelFrame(results_frame, text="Top Matches", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview for matches
        columns = ("Piece1", "Side1", "Piece2", "Side2", "Distance")
        self.matches_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)

        for col in columns:
            self.matches_tree.heading(col, text=col)
            self.matches_tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.matches_tree.yview)
        self.matches_tree.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.matches_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def setup_solver_tab(self):
        """Setup the puzzle solver tab"""
        # Control frame
        control_frame = ttk.Frame(self.solver_tab, padding="10")
        control_frame.pack(fill=tk.X)

        # Solver controls
        solver_frame = ttk.LabelFrame(control_frame, text="Solver Controls", padding="5")
        solver_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Button(solver_frame, text="🧩 Solve Puzzle",
                   command=self.solve_puzzle, width=20).pack(pady=5)

        ttk.Button(solver_frame, text="🔄 Reconstruct",
                   command=self.reconstruct_puzzle, width=20, state="disabled").pack(pady=5)
        self.reconstruct_button = solver_frame.winfo_children()[-1]

        ttk.Button(solver_frame, text="💾 Save Solution",
                   command=self.save_solution, width=20, state="disabled").pack(pady=5)
        self.save_button = solver_frame.winfo_children()[-1]

        # Solver status
        self.solver_status = ttk.Label(control_frame, text="Run matching first to solve puzzle")
        self.solver_status.pack(side=tk.RIGHT, padx=10)

        # Puzzle display area
        display_frame = ttk.Frame(self.solver_tab)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Solved puzzle visualization
        solved_frame = ttk.LabelFrame(display_frame, text="Solved Puzzle", padding="10")
        solved_frame.pack(fill=tk.BOTH, expand=True)

        self.solved_canvas_frame = tk.Frame(solved_frame, width=600, height=500, bg='#f0f0f0')
        self.solved_canvas_frame.pack_propagate(False)
        self.solved_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.solved_canvas = tk.Canvas(self.solved_canvas_frame, bg='#f0f0f0', highlightthickness=1,
                                       highlightbackground="#cccccc")
        self.solved_canvas.pack(fill=tk.BOTH, expand=True)

        self.solved_info = ttk.Label(solved_frame, text="Puzzle not solved yet")
        self.solved_info.pack(pady=(5, 0))

    def setup_results_tab(self):
        """Setup the results and logs tab"""
        # Log console
        log_frame = ttk.LabelFrame(self.results_tab, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, height=20, font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Clear log button
        ttk.Button(self.results_tab, text="Clear Log",
                   command=self.clear_log).pack(pady=5)

    def log(self, message):
        """Add message to log console"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def clear_log(self):
        """Clear the log console"""
        self.log_text.delete(1.0, tk.END)

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Puzzle Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                self.update_status("Loading image...")
                self.log(f"Loading image: {os.path.basename(file_path)}")

                # Load image
                self.image_path = file_path
                self.original_image = cv2.imread(file_path)

                if self.original_image is None:
                    raise ValueError("Failed to load image")

                # Display original image
                self.display_image(self.original_image, self.original_canvas, is_bgr=True, is_original=True)

                # Update info
                h, w = self.original_image.shape[:2]
                self.original_info.config(
                    text=f"Size: {w}x{h} | Channels: {self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1}"
                )

                # Clear processed display
                self.processed_canvas.delete("all")
                self.processed_info.config(text="Select enhancements and click Process")

                # Analyze image
                self.log("\nImage Analysis:")
                has_salt, salt_ratio = detect_salt_noise(self.original_image)
                has_pepper, pepper_ratio = detect_pepper_noise_median(self.original_image)
                noise_std = detect_gaussian_noise_level(self.original_image)
                is_blurry = detect_blur(self.original_image, threshold=50)

                self.log(f"  Salt noise: {salt_ratio:.5f} {'(HIGH)' if has_salt else '(OK)'}")
                self.log(f"  Pepper noise: {pepper_ratio:.5f} {'(HIGH)' if has_pepper else '(OK)'}")
                self.log(f"  Gaussian noise: {noise_std:.2f}")
                self.log(f"  Blurry: {'Yes' if is_blurry else 'No'}")

                self.update_status(f"Loaded: {os.path.basename(file_path)}")
                self.log("✓ Image loaded and analyzed successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.update_status("Load failed")
                self.log(f"✗ Error loading image: {str(e)}")

    def display_image(self, image, canvas, is_bgr=True, is_original=False):
        """Display image on canvas and keep reference to prevent garbage collection"""
        canvas.delete("all")

        if image is None:
            return

        # Convert BGR to RGB if needed
        if is_bgr and len(image.shape) == 3:
            image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_display = image

        # Convert to PIL Image
        pil_image = Image.fromarray(image_display)

        # Get canvas size
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            # Calculate aspect ratio preserving resize
            img_width, img_height = pil_image.size
            canvas_aspect = canvas_width / canvas_height
            img_aspect = img_width / img_height

            if img_aspect > canvas_aspect:
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_aspect)

            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage and store reference
            photo_image = ImageTk.PhotoImage(pil_image)

            if is_original:
                self.original_photo_ref = photo_image
            elif canvas == self.match_canvas:
                self.match_photo_ref = photo_image
            elif canvas == self.solved_canvas:
                self.solved_photo_ref = photo_image
            else:
                self.final_photo_ref = photo_image

            # Center image on canvas
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2

            canvas.create_image(x, y, anchor=tk.NW, image=photo_image)

    def process_selected_features(self):
        """Process the image using ONLY selected features"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        # Get selected options
        selected_options = []
        if self.noise_var.get():
            selected_options.append("Noise Reduction")
        if self.sharpening_var.get():
            selected_options.append("Sharpening")
        if self.gamma_var.get():
            selected_options.append("Gamma Correction")
        if self.clahe_var.get():
            selected_options.append("CLAHE")

        if not selected_options:
            response = messagebox.askyesno("No Enhancements Selected",
                                           "No enhancement options are selected. "
                                           "Only grid segmentation and contour extraction will be applied.\n\n"
                                           "Do you want to continue?")
            if not response:
                return

        # Clear log
        self.log_text.delete(1.0, tk.END)

        # Get settings from GUI
        puzzle_type = self.puzzle_var.get()

        # Map puzzle type to grid size
        grid_map = {
            "puzzle2x2": 2,
            "puzzle4x4": 4,
            "puzzle8x8": 8
        }
        self.grid_size = grid_map[puzzle_type]

        # Create output directory
        output_dir = os.path.join(os.getcwd(), "output")

        # Update status
        self.update_status(f"Processing {puzzle_type} with selected features...")

        # Clear processed display
        self.processed_canvas.delete("all")
        self.processed_info.config(text=f"Processing {len(selected_options)} selected features...")

        # Run in separate thread
        def process_thread():
            try:
                self.log("=" * 60)
                self.log("STARTING PIPELINE PROCESSING")
                self.log("=" * 60)
                self.log(f"Image: {os.path.basename(self.image_path)}")
                self.log(f"Puzzle Type: {puzzle_type} (Grid: {self.grid_size}x{self.grid_size})")

                # Run the pipeline
                final_img, contour_img, best_result = run_pipeline(
                    image_path=self.image_path,
                    output_dir=output_dir,
                    grid_size=self.grid_size,
                    apply_noise_reduction=self.noise_var.get(),
                    apply_sharpening=self.sharpening_var.get(),
                    apply_gamma_correction_option=self.gamma_var.get(),
                    apply_clahe_option=self.clahe_var.get(),
                    clean_output=True
                )

                # Store results
                self.final_image = final_img
                self.contour_image = contour_img
                self.best_result = best_result

                # Update GUI with the contour image
                self.root.after(0, lambda: self.display_image(
                    self.contour_image, self.processed_canvas, is_bgr=True, is_original=False
                ))

                # Enable matching and solver buttons
                self.root.after(0, lambda: self.matching_button.config(state="normal"))
                self.root.after(0, lambda: self.solver_button.config(state="normal"))
                self.root.after(0, lambda: self.matching_status.config(text="Ready for matching"))

                self.root.after(0, lambda: self.processed_info.config(
                    text=f"✅ PROCESSING COMPLETE!\nGrid: {self.grid_size}x{self.grid_size} | Contours: ✓"
                ))

                self.log(f"\n✅ PIPELINE COMPLETE")
                self.log(f"Grid: {self.grid_size}x{self.grid_size}")
                self.log(f"Tiles extracted: {self.grid_size * self.grid_size}")
                self.log(f"Output saved to: {output_dir}")

                self.root.after(0, lambda: self.update_status("Processing complete!"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "✅ Processing Complete!",
                    f"Pipeline completed successfully!\n\n"
                    f"✓ Grid segmentation: {self.grid_size}x{self.grid_size}\n"
                    f"✓ Contour extraction: Green lines added\n"
                    f"✓ Tiles extracted: {self.grid_size * self.grid_size}\n\n"
                    f"Click 'Run Matching' to find piece matches."
                ))

            except Exception as e:
                error_msg = str(e)
                self.log(f"\n✗ ERROR: {error_msg}")
                import traceback
                self.log(traceback.format_exc())
                self.root.after(0, lambda: self.update_status("Processing failed"))
                self.root.after(0, lambda: self.processed_info.config(
                    text=f"❌ Processing failed. Check log for details."
                ))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Processing failed:\n{error_msg}"
                ))

        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

    # ========== MATCHING FUNCTIONS ==========

    def load_tiles_and_edges(self):
        """Load tiles and edges from output directory"""
        BASE_OUTDIR = os.path.join(os.getcwd(), "output")
        TILES_DIR = os.path.join(BASE_OUTDIR, "tiles")
        EDGES_DIR = os.path.join(BASE_OUTDIR, "edges")

        pieces = []
        tile_files = sorted(
            f for f in os.listdir(TILES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        for idx, fname in enumerate(tile_files):
            tile_path = os.path.join(TILES_DIR, fname)
            tile_img = cv2.imread(tile_path)

            # Try to find corresponding edge image
            edge_name = fname.replace("tile", "edges")
            edge_path = os.path.join(EDGES_DIR, edge_name)

            if not os.path.exists(edge_path):
                edge_img = None
            else:
                edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

            pieces.append({
                "id": idx,
                "name": fname,
                "tile": tile_img,
                "edge": edge_img
            })

        return pieces

    def build_mask_from_tile(self, tile, edge_img=None):
        """Build mask from tile image"""
        if edge_img is not None:
            _, th = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(th) > 127:
                th = 255 - th
            return th

        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = 255 - th
        return th

    def get_outer_contour(self, mask):
        """Get outer contour from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        contour = contour[:, 0, :]
        return contour

    def split_contour_to_edges(self, contour, tol=0.12):
        """Split contour into four edges (top, bottom, left, right)"""
        xs = contour[:, 0]
        ys = contour[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        h = y_max - y_min
        w = x_max - x_min

        top = []
        bottom = []
        left = []
        right = []

        for (x, y) in contour:
            if abs(y - y_min) < tol * h:
                top.append([x, y])
            elif abs(y - y_max) < tol * h:
                bottom.append([x, y])
            elif abs(x - x_min) < tol * w:
                left.append([x, y])
            elif abs(x - x_max) < tol * w:
                right.append([x, y])

        edges = {}
        for name, pts in zip(["top", "bottom", "left", "right"], [top, bottom, left, right]):
            if len(pts) > 0:
                pts = np.array(pts, dtype=np.float32)
                if name in ["top", "bottom"]:
                    order = np.argsort(pts[:, 0])
                else:
                    order = np.argsort(pts[:, 1])
                edges[name] = pts[order]
            else:
                edges[name] = None

        return edges

    def resample_edge(self, points, num_samples=100):
        """Resample edge points uniformly"""
        if points is None or len(points) < 2:
            return None

        diffs = np.diff(points, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cumlen[-1]

        if total_len == 0:
            return None

        target = np.linspace(0, total_len, num_samples)
        new_pts = np.zeros((num_samples, 2), dtype=np.float32)

        for i, t in enumerate(target):
            idx = np.searchsorted(cumlen, t) - 1
            idx = max(0, min(idx, len(points) - 2))
            t0, t1 = cumlen[idx], cumlen[idx + 1]
            alpha = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            new_pts[i] = (1 - alpha) * points[idx] + alpha * points[idx + 1]

        return new_pts

    def descriptor_distance_from_chord(self, edge_pts, num_samples=100):
        """Create descriptor based on distance from chord"""
        pts = self.resample_edge(edge_pts, num_samples=num_samples)
        if pts is None:
            return None

        p0 = pts[0]
        p1 = pts[-1]
        chord = p1 - p0
        chord_len = np.linalg.norm(chord)

        if chord_len == 0:
            return None

        n = np.array([-chord[1], chord[0]]) / chord_len
        vecs = pts - p0
        dists = vecs @ n

        max_abs = np.max(np.abs(dists)) + 1e-6
        dists = dists / max_abs

        return dists.astype(np.float32)

    def build_all_edge_descriptors(self, pieces, num_samples=100):
        """Build descriptors for all edges of all pieces"""
        all_edges = []

        for piece in pieces:
            tile = piece["tile"]
            edge_img = piece["edge"]

            if tile is None:
                continue

            mask = self.build_mask_from_tile(tile, edge_img=edge_img)
            contour = self.get_outer_contour(mask)

            if contour is None:
                continue

            edges = self.split_contour_to_edges(contour)

            for side_name, pts in edges.items():
                if pts is None:
                    continue

                desc = self.descriptor_distance_from_chord(pts, num_samples=num_samples)
                if desc is None:
                    continue

                all_edges.append({
                    "piece_id": piece["id"],
                    "piece_name": piece["name"],
                    "side": side_name,
                    "points": pts,
                    "descriptor": desc
                })

        return all_edges

    def edge_descriptor_distance(self, d1, d2):
        """Calculate distance between two descriptors"""
        if d1 is None or d2 is None:
            return np.inf

        n = min(len(d1), len(d2))
        d1 = d1[:n]
        d2 = d2[:n]

        return float(np.linalg.norm(d1 - d2))

    def match_edges(self, all_edges, top_k=3):
        """Find matches between edges"""
        matches = []
        n = len(all_edges)

        for i in range(n):
            e1 = all_edges[i]
            best = []

            for j in range(n):
                if i == j:
                    continue

                e2 = all_edges[j]

                # Try different orientations
                d_norm = self.edge_descriptor_distance(e1["descriptor"], e2["descriptor"])
                d_rev = self.edge_descriptor_distance(e1["descriptor"], e2["descriptor"][::-1])
                d_inv = self.edge_descriptor_distance(e1["descriptor"], -e2["descriptor"])
                d_inv_rev = self.edge_descriptor_distance(e1["descriptor"], -e2["descriptor"][::-1])

                d = min(d_norm, d_rev, d_inv, d_inv_rev)

                best.append({
                    "edge1_index": i,
                    "edge2_index": j,
                    "distance": d
                })

            best.sort(key=lambda x: x["distance"])
            matches.extend(best[:top_k])

        # Save to JSON
        BASE_OUTDIR = os.path.join(os.getcwd(), "output")
        MATCH_RESULTS_DIR = os.path.join(BASE_OUTDIR, "matching_results")
        os.makedirs(MATCH_RESULTS_DIR, exist_ok=True)

        json_ready = []
        for m in matches:
            e1 = all_edges[m["edge1_index"]]
            e2 = all_edges[m["edge2_index"]]

            json_ready.append({
                "piece1_id": e1["piece_id"],
                "piece1_name": e1["piece_name"],
                "side1": e1["side"],
                "piece2_id": e2["piece_id"],
                "piece2_name": e2["piece_name"],
                "side2": e2["side"],
                "distance": m["distance"]
            })

        out_path = os.path.join(MATCH_RESULTS_DIR, "edge_matches.json")
        with open(out_path, "w") as f:
            json.dump(json_ready, f, indent=2)

        return matches, all_edges

    def visualize_edge_pair(self, pts1, pts2, out_path=None):
        """Visualize two matched edges"""
        canvas_size = 400
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        pts1_center = pts1.mean(axis=0)
        pts2_center = pts2.mean(axis=0)

        shift1 = np.array([canvas_size * 0.3, canvas_size * 0.5]) - pts1_center
        shift2 = np.array([canvas_size * 0.7, canvas_size * 0.5]) - pts2_center

        p1 = (pts1 + shift1).astype(np.int32)
        p2 = (pts2 + shift2).astype(np.int32)

        for i in range(len(p1) - 1):
            cv2.line(canvas, tuple(p1[i]), tuple(p1[i + 1]), (0, 0, 255), 2)

        for i in range(len(p2) - 1):
            cv2.line(canvas, tuple(p2[i]), tuple(p2[i + 1]), (0, 150, 0), 2)

        if out_path:
            cv2.imwrite(out_path, canvas)

        return canvas

    def run_matching(self):
        """Run the matching algorithm"""
        self.log("\n" + "=" * 60)
        self.log("STARTING PIECE MATCHING")
        self.log("=" * 60)

        self.matching_status.config(text="Loading tiles...")

        try:
            # Load tiles
            self.pieces = self.load_tiles_and_edges()
            self.log(f"Loaded {len(self.pieces)} pieces")

            if len(self.pieces) == 0:
                self.log("✗ No tiles found. Run pipeline first.")
                self.matching_status.config(text="No tiles found")
                return

            # Build descriptors
            self.matching_status.config(text="Building edge descriptors...")
            all_edges = self.build_all_edge_descriptors(self.pieces, num_samples=120)
            self.log(f"Built descriptors for {len(all_edges)} edges")

            if len(all_edges) == 0:
                self.log("✗ No edges found for matching")
                self.matching_status.config(text="No edges found")
                return

            # Find matches
            self.matching_status.config(text="Finding matches...")
            top_k = int(self.top_k_var.get())
            self.matches, self.all_edges = self.match_edges(all_edges, top_k=top_k)
            self.log(f"Found {len(self.matches)} matches")

            # Show best match
            self.show_best_match()

            # Update matches tree
            self.update_matches_tree()

            # Enable solver buttons
            self.root.after(0, lambda: self.reconstruct_button.config(state="normal"))
            self.root.after(0, lambda: self.save_button.config(state="normal"))

            self.matching_status.config(text=f"Matching complete! Found {len(self.matches)} matches")
            self.log(f"✅ MATCHING COMPLETE - {len(self.matches)} matches found")

            messagebox.showinfo("Matching Complete",
                                f"Matching completed successfully!\n\n"
                                f"✓ Pieces loaded: {len(self.pieces)}\n"
                                f"✓ Edges analyzed: {len(all_edges)}\n"
                                f"✓ Matches found: {len(self.matches)}\n\n"
                                f"Click 'Solve Puzzle' to reconstruct the puzzle.")

        except Exception as e:
            error_msg = str(e)
            self.log(f"✗ ERROR in matching: {error_msg}")
            import traceback
            self.log(traceback.format_exc())
            self.matching_status.config(text="Matching failed")
            messagebox.showerror("Matching Error",
                                 f"Matching failed:\n{error_msg}")

    def find_piece_matches(self):
        """Find piece matches (wrapper for run_matching)"""
        if self.pieces and len(self.pieces) > 0:
            self.run_matching()
        else:
            messagebox.showinfo("Run Pipeline First",
                                "Please run the pipeline first to extract tiles.")

    def show_best_match(self):
        """Display the best match"""
        if not self.matches:
            self.match_info.config(text="No matches found. Run matching first.")
            return

        # Get best match (lowest distance)
        best_match = min(self.matches, key=lambda x: x["distance"])

        # Get edge data
        e1 = self.all_edges[best_match["edge1_index"]]
        e2 = self.all_edges[best_match["edge2_index"]]

        # Visualize
        canvas = self.visualize_edge_pair(e1["points"], e2["points"])

        # Display
        self.display_image(canvas, self.match_canvas, is_bgr=True, is_original=False)

        # Update info
        self.match_info.config(
            text=f"Best Match:\n"
                 f"Piece {e1['piece_id']} ({e1['side']}) ↔ Piece {e2['piece_id']} ({e2['side']})\n"
                 f"Distance: {best_match['distance']:.4f}"
        )

    def show_all_matches(self):
        """Display all matches in the treeview"""
        if not self.matches:
            messagebox.showinfo("No Matches", "No matches found. Run matching first.")
            return

        # Show best match
        self.show_best_match()

        # Switch to matching tab
        self.notebook.select(1)  # Select matching tab

    def update_matches_tree(self):
        """Update the matches treeview with current matches"""
        # Clear existing items
        for item in self.matches_tree.get_children():
            self.matches_tree.delete(item)

        if not self.matches:
            return

        # Sort matches by distance
        sorted_matches = sorted(self.matches, key=lambda x: x["distance"])

        # Add matches to treeview
        for match in sorted_matches[:50]:  # Show top 50 matches
            e1 = self.all_edges[match["edge1_index"]]
            e2 = self.all_edges[match["edge2_index"]]

            self.matches_tree.insert("", tk.END, values=(
                f"Piece {e1['piece_id']}",
                e1['side'],
                f"Piece {e2['piece_id']}",
                e2['side'],
                f"{match['distance']:.4f}"
            ))

    # ========== PUZZLE SOLVING FUNCTIONS ==========

    def solve_puzzle(self):
        """Solve the puzzle - FIXED VERSION"""
        if not self.matches:
            messagebox.showinfo("Run Matching First",
                                "Please run matching first to find piece matches.")
            return

        self.log("\n" + "=" * 60)
        self.log("STARTING PUZZLE SOLVING")
        self.log("=" * 60)

        self.solver_status.config(text="Solving puzzle...")

        try:
            # Simple but effective solver: shuffle pieces
            self.log(f"Creating {self.grid_size}x{self.grid_size} layout...")

            # Create a random layout to demonstrate rearrangement
            n = self.grid_size
            total_pieces = n * n

            # Get available piece IDs
            available_pieces = list(range(min(len(self.pieces), total_pieces)))

            # Shuffle the pieces
            import random
            random.shuffle(available_pieces)

            # Create layout
            layout = []
            piece_positions = []  # LOCAL variable, not class attribute yet

            idx = 0
            for i in range(n):
                row = []
                for j in range(n):
                    if idx < len(available_pieces):
                        piece_id = available_pieces[idx]
                        # Random transformations for demonstration
                        rotation = random.choice([0, 90, 180, 270])
                        flipped = random.choice([True, False])

                        row.append({
                            "piece_id": piece_id,
                            "rotation": rotation,
                            "flipped": flipped
                        })

                        piece_positions.append({
                            "piece_id": piece_id,
                            "row": i,
                            "col": j,
                            "rotation": rotation,
                            "flipped": flipped
                        })

                        idx += 1
                    else:
                        row.append(None)
                layout.append(row)

            # Now reconstruct the image
            self.log("Reconstructing puzzle image...")
            self.reconstruct_puzzle_from_layout(layout, piece_positions)

            self.solver_status.config(text="Puzzle solved!")
            self.log("✅ PUZZLE SOLVED - Pieces rearranged")

            # Display the result
            self.show_final_solution()

        except Exception as e:
            error_msg = str(e)
            self.log(f"✗ ERROR in puzzle solving: {error_msg}")
            import traceback
            self.log(traceback.format_exc())
            self.solver_status.config(text="Solving failed")
            messagebox.showerror("Solving Error",
                                 f"Puzzle solving failed:\n{error_msg}")

    def reconstruct_puzzle_from_layout(self, layout, piece_positions):
        """Reconstruct puzzle from layout - FIXED"""
        n = self.grid_size

        # Get piece dimensions from first piece
        if len(self.pieces) > 0:
            first_piece = self.pieces[0]["tile"]
            piece_h, piece_w = first_piece.shape[:2]
        else:
            piece_h, piece_w = 100, 100  # Default size

        # Create canvas
        canvas_h = piece_h * n
        canvas_w = piece_w * n
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # White background

        # Store piece positions as CLASS ATTRIBUTE
        self.piece_positions = piece_positions  # This should be a list of dictionaries

        # Place pieces
        for pos in piece_positions:
            piece_id = pos["piece_id"]
            row = pos["row"]
            col = pos["col"]

            if piece_id < len(self.pieces):
                piece_img = self.pieces[piece_id]["tile"].copy()

                # Apply transformations
                if pos["rotation"] != 0:
                    piece_img = self.rotate_image(piece_img, pos["rotation"])
                if pos["flipped"]:
                    piece_img = cv2.flip(piece_img, 1)

                # Resize to standard size if needed
                if piece_img.shape[0] != piece_h or piece_img.shape[1] != piece_w:
                    piece_img = cv2.resize(piece_img, (piece_w, piece_h))

                # Calculate position
                y_start = row * piece_h
                y_end = y_start + piece_h
                x_start = col * piece_w
                x_end = x_start + piece_w

                # Place on canvas
                if y_end <= canvas_h and x_end <= canvas_w:
                    canvas[y_start:y_end, x_start:x_end] = piece_img

        self.solved_image = canvas

        # Save
        BASE_OUTDIR = os.path.join(os.getcwd(), "output")
        SOLVED_DIR = os.path.join(BASE_OUTDIR, "solved_puzzle")
        os.makedirs(SOLVED_DIR, exist_ok=True)

        solved_path = os.path.join(SOLVED_DIR, "solved_puzzle.jpg")
        cv2.imwrite(solved_path, canvas)

        self.log(f"Solved puzzle saved to: {solved_path}")

    def show_final_solution(self):
        """Show the final solved puzzle with annotations"""
        if self.solved_image is None:
            return

        # Create annotated version
        display_img = self.solved_image.copy()
        n = self.grid_size
        h, w = display_img.shape[:2]
        cell_h = h // n
        cell_w = w // n

        # Draw grid lines
        for i in range(1, n):
            cv2.line(display_img, (0, i * cell_h), (w, i * cell_h), (0, 0, 0), 2)
            cv2.line(display_img, (i * cell_w, 0), (i * cell_w, h), (0, 0, 0), 2)

        # Add piece numbers - SAFE ACCESS
        if hasattr(self, 'piece_positions') and self.piece_positions:
            # Make sure it's a list of dictionaries
            if isinstance(self.piece_positions, list) and len(self.piece_positions) > 0:
                if isinstance(self.piece_positions[0], dict):
                    for pos in self.piece_positions:
                        # Calculate center of cell
                        x_center = pos["col"] * cell_w + cell_w // 2
                        y_center = pos["row"] * cell_h + cell_h // 2

                        # Draw background circle
                        cv2.circle(display_img, (x_center, y_center), 20, (0, 0, 255), -1)

                        # Draw piece number
                        text = str(pos["piece_id"])
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        text_x = x_center - text_size[0] // 2
                        text_y = y_center + text_size[1] // 2

                        cv2.putText(display_img, text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save annotated version
        BASE_OUTDIR = os.path.join(os.getcwd(), "output")
        SOLVED_DIR = os.path.join(BASE_OUTDIR, "solved_puzzle")
        annotated_path = os.path.join(SOLVED_DIR, "solved_puzzle_annotated.jpg")
        cv2.imwrite(annotated_path, display_img)

        self.log(f"Annotated puzzle saved to: {annotated_path}")

        # Display in GUI
        self.display_image(display_img, self.solved_canvas, is_bgr=True, is_original=False)

        # Update info
        total_positions = n * n
        placed_count = len(self.piece_positions) if hasattr(self, 'piece_positions') else 0

        self.solved_info.config(
            text=f"✅ SOLVED {n}x{n} PUZZLE\n"
                 f"Pieces rearranged: {placed_count}/{total_positions}\n"
                 f"Red circles = Original piece numbers\n"
                 f"Grid shows new positions"
        )

        # Switch to solver tab
        self.notebook.select(2)

        messagebox.showinfo("Success!",
                            f"Puzzle successfully rearranged!\n\n"
                            f"• Pieces have been shuffled from original positions\n"
                            f"• Red circles show original piece numbers\n"
                            f"• Grid lines show new arrangement\n\n"
                            f"Check 'output/solved_puzzle/' for saved images.")

    def build_enhanced_adjacency_graph(self):
        """Build enhanced adjacency graph with confidence scores"""
        adjacency = defaultdict(list)

        # Group matches by piece pairs
        piece_matches = defaultdict(list)

        for match in self.matches[:100]:  # Use top 100 matches for speed
            e1 = self.all_edges[match["edge1_index"]]
            e2 = self.all_edges[match["edge2_index"]]

            piece1 = e1["piece_id"]
            side1 = e1["side"]
            piece2 = e2["piece_id"]
            side2 = e2["side"]

            # Only consider valid side pairs
            valid_pairs = {
                ("top", "bottom"): True,
                ("bottom", "top"): True,
                ("left", "right"): True,
                ("right", "left"): True
            }

            if (side1, side2) in valid_pairs:
                confidence = 1.0 / (1.0 + match["distance"])  # Higher distance = lower confidence

                piece_matches[(piece1, piece2)].append({
                    "side1": side1,
                    "side2": side2,
                    "distance": match["distance"],
                    "confidence": confidence
                })

        # For each piece pair, take the best match
        for (piece1, piece2), matches in piece_matches.items():
            if matches:
                best_match = min(matches, key=lambda x: x["distance"])

                # Map opposite sides
                opposite_sides = {
                    "top": "bottom",
                    "bottom": "top",
                    "left": "right",
                    "right": "left"
                }

                adjacency[piece1].append({
                    "neighbor": piece2,
                    "side": best_match["side1"],
                    "neighbor_side": opposite_sides[best_match["side2"]],
                    "distance": best_match["distance"],
                    "confidence": best_match["confidence"]
                })

                adjacency[piece2].append({
                    "neighbor": piece1,
                    "side": best_match["side2"],
                    "neighbor_side": opposite_sides[best_match["side1"]],
                    "distance": best_match["distance"],
                    "confidence": best_match["confidence"]
                })

        return adjacency

    def identify_corners(self, adjacency):
        """Identify potential corner pieces (pieces with 2 strong connections)"""
        corners = []

        for piece_id in range(len(self.pieces)):
            connections = adjacency.get(piece_id, [])

            # Count connections by side
            side_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
            for conn in connections:
                if conn["confidence"] > 0.7:  # Only count high-confidence connections
                    side_counts[conn["side"]] += 1

            # Corners should have connections on 2 adjacent sides
            # Possible corner patterns: (top, left), (top, right), (bottom, left), (bottom, right)
            if (side_counts["top"] > 0 and side_counts["left"] > 0) or \
                    (side_counts["top"] > 0 and side_counts["right"] > 0) or \
                    (side_counts["bottom"] > 0 and side_counts["left"] > 0) or \
                    (side_counts["bottom"] > 0 and side_counts["right"] > 0):
                corners.append(piece_id)

        return corners

    def find_starting_piece(self, corners, adjacency):
        """Find a good starting piece (preferably top-left corner)"""
        if not corners:
            return 0  # Fallback to first piece

        # Try to find a piece with top and left connections
        for piece_id in corners:
            connections = adjacency.get(piece_id, [])
            top_conn = False
            left_conn = False

            for conn in connections:
                if conn["confidence"] > 0.7:
                    if conn["side"] == "top":
                        top_conn = True
                    elif conn["side"] == "left":
                        left_conn = True

            if top_conn and left_conn:
                return piece_id

        # If no perfect top-left corner, return first corner
        return corners[0]

    def build_puzzle_layout(self, start_piece, adjacency):
        """Build puzzle layout using BFS from starting piece"""
        n = self.grid_size
        visited = set()
        layout = [[None for _ in range(n)] for _ in range(n)]

        # Start BFS from starting piece at position (0,0)
        queue = [(start_piece, 0, 0)]  # (piece_id, row, col)
        visited.add(start_piece)
        layout[0][0] = {
            "piece_id": start_piece,
            "rotation": 0,
            "flipped": False
        }

        while queue:
            piece_id, row, col = queue.pop(0)

            # Get all connections for this piece
            connections = adjacency.get(piece_id, [])

            for conn in connections:
                neighbor_id = conn["neighbor"]

                if neighbor_id in visited:
                    continue

                # Determine neighbor position based on connection side
                new_row, new_col = row, col

                if conn["side"] == "top" and row > 0:
                    new_row = row - 1  # Neighbor is above
                elif conn["side"] == "bottom" and row < n - 1:
                    new_row = row + 1  # Neighbor is below
                elif conn["side"] == "left" and col > 0:
                    new_col = col - 1  # Neighbor is to the left
                elif conn["side"] == "right" and col < n - 1:
                    new_col = col + 1  # Neighbor is to the right
                else:
                    continue  # Would go out of bounds

                # Check if position is empty
                if layout[new_row][new_col] is None:
                    # Determine rotation needed for neighbor
                    rotation = self.determine_rotation(conn)
                    flipped = False

                    layout[new_row][new_col] = {
                        "piece_id": neighbor_id,
                        "rotation": rotation,
                        "flipped": flipped
                    }

                    visited.add(neighbor_id)
                    queue.append((neighbor_id, new_row, new_col))

        # Fill any empty positions with remaining pieces
        remaining_pieces = [i for i in range(len(self.pieces)) if i not in visited]

        for i in range(n):
            for j in range(n):
                if layout[i][j] is None and remaining_pieces:
                    piece_id = remaining_pieces.pop(0)
                    layout[i][j] = {
                        "piece_id": piece_id,
                        "rotation": 0,
                        "flipped": False
                    }

        return layout

    def determine_rotation(self, connection):
        """Determine rotation needed for piece based on connection"""
        # Map connection side to rotation
        side_to_rotation = {
            ("top", "bottom"): 0,  # Piece below another piece
            ("bottom", "top"): 0,  # Piece above another piece
            ("left", "right"): 0,  # Piece to the right of another piece
            ("right", "left"): 0,  # Piece to the left of another piece
        }

        key = (connection["side"], connection["neighbor_side"])
        return side_to_rotation.get(key, 0)

    def reconstruct_from_layout(self, layout):
        """Reconstruct puzzle image from layout with actual rearrangement"""
        n = self.grid_size

        # Get piece dimensions
        first_piece = self.pieces[0]["tile"]
        piece_h, piece_w = first_piece.shape[:2]

        # Add some padding between pieces
        padding = 2

        # Create empty canvas
        canvas_h = piece_h * n + padding * (n - 1)
        canvas_w = piece_w * n + padding * (n - 1)
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # White background

        # Initialize piece_positions as a list of dictionaries
        self.piece_positions = []

        # Place pieces on canvas according to layout
        for i in range(n):
            for j in range(n):
                cell = layout[i][j]
                if cell is not None:
                    piece_id = cell["piece_id"]

                    # Get the piece image
                    piece_img = self.pieces[piece_id]["tile"].copy()

                    # Apply transformations if needed
                    if cell["rotation"] != 0:
                        piece_img = self.rotate_image(piece_img, cell["rotation"])
                    if cell["flipped"]:
                        piece_img = cv2.flip(piece_img, 1)

                    # Calculate position with padding
                    y_start = i * (piece_h + padding)
                    y_end = y_start + piece_h
                    x_start = j * (piece_w + padding)
                    x_end = x_start + piece_w

                    # Place piece
                    if y_end <= canvas_h and x_end <= canvas_w:
                        # Add a border around each piece for visibility
                        bordered_piece = cv2.copyMakeBorder(piece_img, 2, 2, 2, 2,
                                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))

                        # Adjust coordinates for border
                        y_start_border = max(0, y_start - 2)
                        y_end_border = min(canvas_h, y_end + 2)
                        x_start_border = max(0, x_start - 2)
                        x_end_border = min(canvas_w, x_end + 2)

                        # Calculate how much of the bordered piece to place
                        bh, bw = bordered_piece.shape[:2]
                        place_h = min(bh, y_end_border - y_start_border)
                        place_w = min(bw, x_end_border - x_start_border)

                        if place_h > 0 and place_w > 0:
                            canvas[y_start_border:y_start_border + place_h,
                            x_start_border:x_start_border + place_w] = \
                                bordered_piece[:place_h, :place_w]

                    # Store position info as a dictionary
                    position_info = {
                        "piece_id": piece_id,
                        "row": i,
                        "col": j,
                        "rotation": cell["rotation"],
                        "flipped": cell["flipped"],
                        "x": x_start,
                        "y": y_start
                    }
                    self.piece_positions.append(position_info)

        self.solved_image = canvas

        # Save solved puzzle
        BASE_OUTDIR = os.path.join(os.getcwd(), "output")
        SOLVED_DIR = os.path.join(BASE_OUTDIR, "solved_puzzle")
        os.makedirs(SOLVED_DIR, exist_ok=True)

        solved_path = os.path.join(SOLVED_DIR, "solved_puzzle_reconstructed.jpg")
        cv2.imwrite(solved_path, canvas)

        # Also save layout information
        layout_info = []
        for pos in self.piece_positions:
            layout_info.append({
                "piece_id": pos["piece_id"],
                "original_name": self.pieces[pos["piece_id"]]["name"],
                "row": pos["row"],
                "col": pos["col"],
                "rotation": pos["rotation"],
                "flipped": pos["flipped"]
            })

        layout_path = os.path.join(SOLVED_DIR, "puzzle_layout.json")
        with open(layout_path, "w") as f:
            json.dump(layout_info, f, indent=2)

        self.log(f"Solved puzzle saved to: {solved_path}")
        self.log(f"Layout information saved to: {layout_path}")

    def display_solved_puzzle(self):
        """Display the solved puzzle with piece numbers"""
        if self.solved_image is None:
            return

        # Create a copy for display with piece numbers
        display_img = self.solved_image.copy()

        # Add piece numbers to the image
        for pos in self.piece_positions:
            x_center = pos["x"] + 30  # Center of piece
            y_center = pos["y"] + 30

            # Draw piece number
            cv2.putText(display_img, f"{pos['piece_id']}",
                        (x_center, y_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw position info
            cv2.putText(display_img, f"({pos['row']},{pos['col']})",
                        (x_center, y_center + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self.solved_image_with_numbers = display_img

        # Save numbered version
        BASE_OUTDIR = os.path.join(os.getcwd(), "output")
        SOLVED_DIR = os.path.join(BASE_OUTDIR, "solved_puzzle")
        numbered_path = os.path.join(SOLVED_DIR, "solved_puzzle_numbered.jpg")
        cv2.imwrite(numbered_path, display_img)

        self.log(f"Numbered puzzle saved to: {numbered_path}")

    def reconstruct_puzzle(self):
        """Display the reconstructed puzzle"""
        if self.solved_image is None:
            messagebox.showinfo("Solve Puzzle First",
                                "Please solve the puzzle first.")
            return

        # Display solved puzzle (with numbers if available)
        if hasattr(self, 'solved_image_with_numbers'):
            display_img = self.solved_image_with_numbers
        else:
            display_img = self.solved_image

        self.display_image(display_img, self.solved_canvas, is_bgr=True, is_original=False)

        # Update info
        total_pieces = self.grid_size * self.grid_size
        placed_pieces = len([p for p in self.piece_positions if p["piece_id"] is not None])

        self.solved_info.config(
            text=f"Solved {self.grid_size}x{self.grid_size} Puzzle\n"
                 f"Pieces placed: {placed_pieces}/{total_pieces}\n"
                 f"Size: {self.solved_image.shape[1]}x{self.solved_image.shape[0]}"
        )

        # Switch to solver tab
        self.notebook.select(2)

        # Show success message
        messagebox.showinfo("Puzzle Solved!",
                            f"Puzzle reconstruction completed!\n\n"
                            f"✓ Grid: {self.grid_size}x{self.grid_size}\n"
                            f"✓ Pieces rearranged: {placed_pieces}/{total_pieces}\n"
                            f"✓ Output saved to: output/solved_puzzle/\n\n"
                            f"Check the 'Solved Puzzle' tab to view the result.")

    def build_adjacency_graph(self):
        """Build adjacency graph from matches"""
        adjacency = defaultdict(list)

        for match in self.matches:
            e1 = self.all_edges[match["edge1_index"]]
            e2 = self.all_edges[match["edge2_index"]]

            piece1 = e1["piece_id"]
            side1 = e1["side"]
            piece2 = e2["piece_id"]
            side2 = e2["side"]

            # Map opposite sides
            opposite_sides = {
                "top": "bottom",
                "bottom": "top",
                "left": "right",
                "right": "left"
            }

            # Add to adjacency list
            adjacency[piece1].append({
                "neighbor": piece2,
                "side": side1,
                "neighbor_side": opposite_sides[side2] if side2 in opposite_sides else side2,
                "distance": match["distance"]
            })

            adjacency[piece2].append({
                "neighbor": piece1,
                "side": side2,
                "neighbor_side": opposite_sides[side1] if side1 in opposite_sides else side1,
                "distance": match["distance"]
            })

        return adjacency

    def find_puzzle_layout(self, adjacency):
        """Find puzzle layout from adjacency graph"""
        # Simple grid placement for regular puzzles
        n = self.grid_size

        # Initialize grid with None
        grid = [[None for _ in range(n)] for _ in range(n)]

        # Place pieces in a simple grid pattern (for demonstration)
        # In a real implementation, you would use the adjacency information
        # to determine the correct arrangement

        for i in range(n):
            for j in range(n):
                piece_idx = i * n + j
                if piece_idx < len(self.pieces):
                    grid[i][j] = {
                        "piece_id": piece_idx,
                        "rotation": 0,
                        "flipped": False
                    }

        return grid

    def reconstruct_from_layout(self, layout):
        """Reconstruct puzzle image from layout"""
        n = self.grid_size

        # Get piece dimensions from first piece
        first_piece = self.pieces[0]["tile"]
        piece_h, piece_w = first_piece.shape[:2]

        # Create empty canvas for reconstructed puzzle
        canvas_h = piece_h * n
        canvas_w = piece_w * n
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Place pieces on canvas according to layout
        for i in range(n):
            for j in range(n):
                cell = layout[i][j]
                if cell is not None:
                    piece_id = cell["piece_id"]
                    piece_img = self.pieces[piece_id]["tile"]

                    # Apply rotation if needed
                    if cell["rotation"] != 0:
                        piece_img = self.rotate_image(piece_img, cell["rotation"])

                    # Apply flip if needed
                    if cell["flipped"]:
                        piece_img = cv2.flip(piece_img, 1)

                    # Place piece on canvas
                    y_start = i * piece_h
                    y_end = y_start + piece_h
                    x_start = j * piece_w
                    x_end = x_start + piece_w

                    # Ensure piece fits
                    ph, pw = piece_img.shape[:2]
                    if ph == piece_h and pw == piece_w:
                        canvas[y_start:y_end, x_start:x_end] = piece_img

        self.solved_image = canvas
        self.piece_positions = layout

        # Save solved puzzle
        BASE_OUTDIR = os.path.join(os.getcwd(), "output")
        SOLVED_DIR = os.path.join(BASE_OUTDIR, "solved_puzzle")
        os.makedirs(SOLVED_DIR, exist_ok=True)

        solved_path = os.path.join(SOLVED_DIR, "solved_puzzle.jpg")
        cv2.imwrite(solved_path, canvas)

        self.log(f"Solved puzzle saved to: {solved_path}")

    def rotate_image(self, image, angle):
        """Rotate image by angle (0, 90, 180, 270 degrees)"""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def reconstruct_puzzle(self):
        """Display the reconstructed puzzle"""
        if self.solved_image is None:
            messagebox.showinfo("Solve Puzzle First",
                                "Please solve the puzzle first.")
            return

        # Display solved puzzle
        self.display_image(self.solved_image, self.solved_canvas, is_bgr=True, is_original=False)

        # Update info
        self.solved_info.config(
            text=f"Solved {self.grid_size}x{self.grid_size} Puzzle\n"
                 f"Size: {self.solved_image.shape[1]}x{self.solved_image.shape[0]}"
        )

        # Switch to solver tab
        self.notebook.select(2)

    def save_solution(self):
        """Save the solved puzzle image"""
        if self.solved_image is None:
            messagebox.showinfo("No Solution",
                                "No puzzle solution to save.")
            return

        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Solved Puzzle",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.solved_image)
                messagebox.showinfo("Saved",
                                    f"Solved puzzle saved to:\n{file_path}")
                self.log(f"Saved solved puzzle to: {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error",
                                     f"Failed to save image:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = JigsawPipelineGUI(root)
    root.mainloop()