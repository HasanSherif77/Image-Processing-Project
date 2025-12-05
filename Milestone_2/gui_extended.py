"""
Extended GUI with Milestone 2 features
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np

# Import Milestone 1 functions
try:
    from jigsaw_pipeline import run_pipeline, detect_salt_noise, detect_pepper_noise_median, \
        detect_gaussian_noise_level, detect_blur

    print("✓ Successfully imported Milestone 1 functions")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure jigsaw_pipeline.py is in the same directory.")
    exit(1)

# Import Milestone 2 modules
try:
    from contour_matcher import ContourMatcher
    from rotation_invariant import RotationInvariant
    from puzzle_assembler import PuzzleAssembler
    from visualization import MatchVisualizer

    print("✓ Successfully imported Milestone 2 modules")
except ImportError as e:
    print(f"Note: Some Milestone 2 modules not found: {e}")
    print("Some features may be disabled")


class ExtendedJigsawGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Jigsaw Puzzle Solver - Milestone 2")
        self.root.geometry("1400x800")

        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.contour_image = None
        self.image_path = None
        self.grid_size = 4
        self.pieces = []  # Will store extracted pieces
        self.contours = []  # Will store piece contours
        self.matches = []  # Will store match results

        # Store image references
        self.original_photo_ref = None
        self.processed_photo_ref = None
        self.match_photo_ref = None

        # Setup GUI with tabs
        self.setup_gui()

    def setup_gui(self):
        """Setup the main GUI layout with tabs"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Milestone 1 - Pipeline Processing
        self.pipeline_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_tab, text="🚀 Pipeline Processing")
        self.setup_pipeline_tab()

        # Tab 2: Milestone 2 - Contour Matching
        self.matching_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matching_tab, text="🔍 Contour Matching")
        self.setup_matching_tab()

        # Tab 3: Milestone 2 - Puzzle Assembly
        self.assembly_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.assembly_tab, text="🧩 Puzzle Assembly")
        self.setup_assembly_tab()

        # Tab 4: Results & Logs
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="📊 Results")
        self.setup_results_tab()

    def setup_pipeline_tab(self):
        """Setup the pipeline processing tab (from your existing code)"""
        # This would contain your existing GUI code for Milestone 1
        # Control panel
        control_frame = ttk.Frame(self.pipeline_tab, padding="10")
        control_frame.pack(fill=tk.X)

        # ... (Your existing control panel code)

        # Image display area
        display_frame = ttk.Frame(self.pipeline_tab)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Original image
        original_frame = ttk.LabelFrame(display_frame, text="Original Image", padding="10")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.original_canvas = tk.Canvas(original_frame, width=400, height=400, bg='#f0f0f0')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Processed image
        processed_frame = ttk.LabelFrame(display_frame, text="Processed Image", padding="10")
        processed_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.processed_canvas = tk.Canvas(processed_frame, width=400, height=400, bg='#f0f0f0')
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

    def setup_matching_tab(self):
        """Setup the contour matching tab"""
        control_frame = ttk.Frame(self.matching_tab, padding="10")
        control_frame.pack(fill=tk.X)

        # Matching controls
        match_frame = ttk.LabelFrame(control_frame, text="Contour Matching", padding="5")
        match_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Button(match_frame, text="📐 Extract Contours",
                   command=self.extract_contours, width=20).pack(pady=5)

        ttk.Button(match_frame, text="🔄 Match with Rotation",
                   command=self.match_with_rotation, width=20).pack(pady=5)

        ttk.Button(match_frame, text="🎯 Find Best Matches",
                   command=self.find_best_matches, width=20).pack(pady=5)

        # Parameters
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="5")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Label(param_frame, text="Matching Method:").pack()
        self.match_method_var = tk.StringVar(value="hu_moments")
        ttk.Combobox(param_frame, textvariable=self.match_method_var,
                     values=["hu_moments", "fourier", "shape_context"],
                     state="readonly", width=15).pack(pady=5)

        ttk.Label(param_frame, text="Threshold:").pack()
        self.threshold_var = tk.DoubleVar(value=0.8)
        ttk.Scale(param_frame, from_=0.1, to=2.0, variable=self.threshold_var,
                  orient=tk.HORIZONTAL, length=150).pack(pady=5)

        # Results display
        self.matching_status = ttk.Label(control_frame, text="Ready for matching")
        self.matching_status.pack(side=tk.RIGHT, padx=10)

        # Results area
        results_frame = ttk.Frame(self.matching_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Match visualization
        match_viz_frame = ttk.LabelFrame(results_frame, text="Match Visualization", padding="10")
        match_viz_frame.pack(fill=tk.BOTH, expand=True)

        self.match_canvas = tk.Canvas(match_viz_frame, width=600, height=300, bg='#f0f0f0')
        self.match_canvas.pack(fill=tk.BOTH, expand=True)

        # Matches list
        list_frame = ttk.LabelFrame(results_frame, text="Top Matches", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        columns = ("Piece 1", "Piece 2", "Score", "Rotation", "Method")
        self.matches_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8)

        for col in columns:
            self.matches_tree.heading(col, text=col)
            self.matches_tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.matches_tree.yview)
        self.matches_tree.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.matches_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def setup_assembly_tab(self):
        """Setup the puzzle assembly tab"""
        control_frame = ttk.Frame(self.assembly_tab, padding="10")
        control_frame.pack(fill=tk.X)

        # Assembly controls
        assembly_frame = ttk.LabelFrame(control_frame, text="Puzzle Assembly", padding="5")
        assembly_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Button(assembly_frame, text="🧩 Assemble Puzzle",
                   command=self.assemble_puzzle, width=20).pack(pady=5)

        ttk.Button(assembly_frame, text="💾 Save Assembly",
                   command=self.save_assembly, width=20).pack(pady=5)

        # Assembly parameters
        param_frame = ttk.LabelFrame(control_frame, text="Assembly Parameters", padding="5")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Label(param_frame, text="Grid Size:").pack()
        self.assembly_grid_var = tk.IntVar(value=4)
        ttk.Spinbox(param_frame, from_=2, to=8, textvariable=self.assembly_grid_var,
                    width=10).pack(pady=5)

        ttk.Label(param_frame, text="Start Piece:").pack()
        self.start_piece_var = tk.IntVar(value=0)
        ttk.Spinbox(param_frame, from_=0, to=63, textvariable=self.start_piece_var,
                    width=10).pack(pady=5)

        # Status
        self.assembly_status = ttk.Label(control_frame, text="Ready for assembly")
        self.assembly_status.pack(side=tk.RIGHT, padx=10)

        # Assembly display
        display_frame = ttk.Frame(self.assembly_tab)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.assembly_canvas = tk.Canvas(display_frame, width=600, height=400, bg='#f0f0f0')
        self.assembly_canvas.pack(fill=tk.BOTH, expand=True)

        self.assembly_info = ttk.Label(display_frame, text="Assembly will appear here")
        self.assembly_info.pack(pady=5)

    def setup_results_tab(self):
        """Setup the results tab"""
        log_frame = ttk.LabelFrame(self.results_tab, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, height=20, font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Clear button
        ttk.Button(self.results_tab, text="Clear Log",
                   command=self.clear_log).pack(pady=5)

    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def clear_log(self):
        """Clear log"""
        self.log_text.delete(1.0, tk.END)

    def extract_contours(self):
        """Extract contours from pieces"""
        if not self.pieces:
            messagebox.showinfo("No Pieces", "Please run pipeline first to extract pieces")
            return

        self.log("\n" + "=" * 60)
        self.log("EXTRACTING CONTOURS FROM PIECES")
        self.log("=" * 60)

        try:
            self.contours = []
            contours_dir = os.path.join("output", "contours")

            for i, piece in enumerate(self.pieces):
                # Extract contours from piece
                gray = cv2.cvtColor(piece["image"], cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Get largest contour
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Simplify contour
                    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                    self.contours.append({
                        "id": i,
                        "contour": simplified_contour,
                        "area": cv2.contourArea(simplified_contour),
                        "perimeter": cv2.arcLength(simplified_contour, True)
                    })

                    self.log(f"  Piece {i}: Area={cv2.contourArea(simplified_contour):.1f}")

            self.log(f"✓ Extracted contours from {len(self.contours)} pieces")
            self.matching_status.config(text=f"Extracted {len(self.contours)} contours")

        except Exception as e:
            self.log(f"✗ Error extracting contours: {str(e)}")
            messagebox.showerror("Error", f"Failed to extract contours: {str(e)}")

    def match_with_rotation(self):
        """Match contours with rotation invariance"""
        if not self.contours:
            messagebox.showinfo("No Contours", "Please extract contours first")
            return

        self.log("\n" + "=" * 60)
        self.log("ROTATION-INVARIANT MATCHING")
        self.log("=" * 60)

        try:
            method = self.match_method_var.get()
            threshold = self.threshold_var.get()

            # Initialize matchers
            matcher = ContourMatcher(method=method)
            rotator = RotationInvariant(rotation_steps=8)

            # Find matches
            matches = []

            for i in range(len(self.contours)):
                for j in range(i + 1, len(self.contours)):
                    # Find best rotation match
                    best_angle, best_score = rotator.find_best_rotation_match(
                        self.contours[i]["contour"],
                        self.contours[j]["contour"],
                        matcher
                    )

                    if best_score < threshold:
                        matches.append({
                            "piece1": self.contours[i]["id"],
                            "piece2": self.contours[j]["id"],
                            "score": best_score,
                            "rotation": best_angle,
                            "method": method
                        })

            # Sort matches
            matches.sort(key=lambda x: x["score"])
            self.matches = matches

            # Update treeview
            for item in self.matches_tree.get_children():
                self.matches_tree.delete(item)

            for match in matches[:20]:  # Show top 20
                self.matches_tree.insert("", tk.END, values=(
                    f"Piece {match['piece1']}",
                    f"Piece {match['piece2']}",
                    f"{match['score']:.4f}",
                    f"{match['rotation']:.1f}°",
                    match["method"]
                ))

            self.log(f"✓ Found {len(matches)} matches")
            self.matching_status.config(text=f"Found {len(matches)} matches")

            # Visualize best match
            if matches:
                self.visualize_best_match(matches[0])

        except Exception as e:
            self.log(f"✗ Error in matching: {str(e)}")
            messagebox.showerror("Error", f"Failed to match contours: {str(e)}")

    def find_best_matches(self):
        """Find best matches using simple contour matching"""
        if not self.contours:
            messagebox.showinfo("No Contours", "Please extract contours first")
            return

        self.log("\n" + "=" * 60)
        self.log("FINDING BEST MATCHES")
        self.log("=" * 60)

        try:
            method = self.match_method_var.get()
            threshold = self.threshold_var.get()

            matcher = ContourMatcher(method=method)
            matches = matcher.find_best_matches(self.contours, threshold=threshold)

            self.matches = matches

            # Update treeview
            for item in self.matches_tree.get_children():
                self.matches_tree.delete(item)

            for match in matches[:20]:  # Show top 20
                self.matches_tree.insert("", tk.END, values=(
                    f"Piece {match['piece1']}",
                    f"Piece {match['piece2']}",
                    f"{match['score']:.4f}",
                    "N/A",
                    match["method"]
                ))

            self.log(f"✓ Found {len(matches)} matches")
            self.matching_status.config(text=f"Found {len(matches)} matches")

        except Exception as e:
            self.log(f"✗ Error finding matches: {str(e)}")
            messagebox.showerror("Error", f"Failed to find matches: {str(e)}")

    def assemble_puzzle(self):
        """Assemble puzzle using matches"""
        if not self.matches:
            messagebox.showinfo("No Matches", "Please find matches first")
            return

        self.log("\n" + "=" * 60)
        self.log("ASSEMBLING PUZZLE")
        self.log("=" * 60)

        try:
            grid_size = self.assembly_grid_var.get()
            start_piece = self.start_piece_var.get()

            assembler = PuzzleAssembler(grid_size=grid_size)

            # Build adjacency graph
            adjacency = assembler.build_adjacency_graph(self.matches)

            # Greedy assembly
            positions = assembler.greedy_assemble(adjacency, start_piece=start_piece)

            # Create assembly image
            if self.pieces:
                # Get tile size from first piece
                tile_h, tile_w = self.pieces[0]["image"].shape[:2]
                assembly_img = assembler.create_assembly_image(self.pieces, (tile_h, tile_w))

                # Display assembly
                self.display_image(assembly_img, self.assembly_canvas)

                self.assembly_info.config(
                    text=f"Assembled {len(positions)}/{grid_size * grid_size} pieces"
                )

                self.log(f"✓ Assembled {len(positions)} pieces")
                self.assembly_status.config(text=f"Assembled {len(positions)} pieces")

        except Exception as e:
            self.log(f"✗ Error assembling puzzle: {str(e)}")
            messagebox.showerror("Error", f"Failed to assemble puzzle: {str(e)}")

    def visualize_best_match(self, match):
        """Visualize the best match"""
        try:
            # Find the two pieces
            piece1 = None
            piece2 = None

            for piece in self.pieces:
                if piece["id"] == match["piece1"]:
                    piece1 = piece["image"]
                elif piece["id"] == match["piece2"]:
                    piece2 = piece["image"]

            if piece1 is not None and piece2 is not None:
                # Create visualization
                h1, w1 = piece1.shape[:2]
                h2, w2 = piece2.shape[:2]

                # Create combined image
                total_width = w1 + w2 + 20
                max_height = max(h1, h2)

                combined = np.ones((max_height, total_width, 3), dtype=np.uint8) * 240

                # Place pieces
                combined[:h1, :w1] = piece1
                combined[:h2, w1 + 20:] = piece2

                # Add labels
                cv2.putText(combined, f"Piece {match['piece1']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(combined, f"Piece {match['piece2']}", (w1 + 30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, f"Score: {match['score']:.4f}", (total_width // 2 - 100, max_height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                if "rotation" in match:
                    cv2.putText(combined, f"Rotation: {match['rotation']:.1f}°",
                                (total_width // 2 - 100, max_height - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Display
                self.display_image(combined, self.match_canvas)

        except Exception as e:
            self.log(f"✗ Error visualizing match: {str(e)}")

    def save_assembly(self):
        """Save assembly result"""
        # Implementation for saving assembly
        pass

    def display_image(self, image, canvas):
        """Display image on canvas"""
        canvas.delete("all")

        if image is None:
            return

        # Convert BGR to RGB
        if len(image.shape) == 3:
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

            # Resize
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo_image = ImageTk.PhotoImage(pil_image)

            # Keep reference
            if canvas == self.match_canvas:
                self.match_photo_ref = photo_image
            elif canvas == self.assembly_canvas:
                self.assembly_photo_ref = photo_image

            # Center image
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2

            canvas.create_image(x, y, anchor=tk.NW, image=photo_image)


def main():
    root = tk.Tk()
    app = ExtendedJigsawGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()