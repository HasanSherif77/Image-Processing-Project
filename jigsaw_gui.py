import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import threading

# Import from jigsaw_pipeline.py
try:
    from jigsaw_pipeline import run_pipeline, detect_salt_noise, detect_pepper_noise_median, detect_gaussian_noise_level, detect_blur
    print("✓ Successfully imported functions from jigsaw_pipeline.py")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure jigsaw_pipeline.py is in the same directory.")
    exit(1)

import cv2
import numpy as np

class JigsawPipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Jigsaw Puzzle Pipeline GUI")
        self.root.geometry("1300x750")  # Increased width for more checkboxes
        
        # Initialize variables
        self.original_image = None
        self.final_image = None
        self.contour_image = None
        self.assembled_image = None
        self.image_path = None
        self.best_result = None

        # Milestone 2 variables
        self.matcher = None
        self.assembly_suggestions = None
        self.current_matches = None

        # Store references to images to prevent garbage collection
        self.original_photo_ref = None
        self.final_photo_ref = None
        self.match_photo_ref = None
        self.assembled_photo_ref = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Configure main grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="ew")
        
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
        
        # Right side: Enhancement options (4 checkboxes in 2 columns)
        enhance_frame = ttk.LabelFrame(control_frame, text="Select Enhancements", padding="5")
        enhance_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Create 2x2 grid for checkboxes
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

        ttk.Button(process_frame, text="🚀 Process Selected Features",
                  command=self.process_selected_features, width=25).pack(pady=5)

        ttk.Button(process_frame, text="🔍 Run Edge Matching",
                  command=self.run_edge_matching_gui, width=25).pack(pady=5)

        ttk.Button(process_frame, text="🧩 Assemble Puzzle",
                  command=self.assemble_puzzle_gui, width=25).pack(pady=5)

        # Status label on the far right
        self.status_label = ttk.Label(control_frame, text="Ready to load image")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Main comparison area
        comparison_frame = ttk.Frame(self.root)
        comparison_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        
        # Configure comparison grid (now 3 columns)
        comparison_frame.grid_rowconfigure(0, weight=1)
        comparison_frame.grid_columnconfigure(0, weight=1)
        comparison_frame.grid_columnconfigure(1, weight=1)
        comparison_frame.grid_columnconfigure(2, weight=1)

        # Left panel - Original image
        original_frame = ttk.LabelFrame(comparison_frame, text="Original Image", padding="10")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))

        # Use a Frame with fixed size for the original image
        self.original_image_frame = tk.Frame(original_frame, width=300, height=300, bg='#f0f0f0')
        self.original_image_frame.pack_propagate(False)
        self.original_image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.original_canvas = tk.Canvas(self.original_image_frame, bg='#f0f0f0', highlightthickness=1,
                                        highlightbackground="#cccccc")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        self.original_info = ttk.Label(original_frame, text="No image loaded")
        self.original_info.pack(pady=(5,0))

        # Middle panel - FINAL Processed image (WITH CONTOURS)
        processed_frame = ttk.LabelFrame(comparison_frame, text="Processed Image (With Contours)", padding="10")
        processed_frame.grid(row=0, column=1, sticky="nsew", padx=(0,5))

        # Use a Frame with fixed size for the processed image
        self.processed_image_frame = tk.Frame(processed_frame, width=300, height=300, bg='#f0f0f0')
        self.processed_image_frame.pack_propagate(False)
        self.processed_image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.processed_canvas = tk.Canvas(self.processed_image_frame, bg='#f0f0f0', highlightthickness=1,
                                         highlightbackground="#cccccc")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

        self.processed_info = ttk.Label(processed_frame, text="Select enhancements and click Process")
        self.processed_info.pack(pady=(5,0))

        # Right panel - ASSEMBLED PUZZLE
        assembled_frame = ttk.LabelFrame(comparison_frame, text="Assembled Puzzle", padding="10")
        assembled_frame.grid(row=0, column=2, sticky="nsew", padx=(5,0))

        # Use a Frame with fixed size for the assembled image
        self.assembled_image_frame = tk.Frame(assembled_frame, width=300, height=300, bg='#f0f0f0')
        self.assembled_image_frame.pack_propagate(False)
        self.assembled_image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.assembled_canvas = tk.Canvas(self.assembled_image_frame, bg='#f0f0f0', highlightthickness=1,
                                         highlightbackground="#cccccc")
        self.assembled_canvas.pack(fill=tk.BOTH, expand=True)

        self.assembled_info = ttk.Label(assembled_frame, text="Run assembly after processing")
        self.assembled_info.pack(pady=(5,0))
        
        # Log console at the bottom
        log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding="10")
        log_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0,10))
        
        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def log(self, message):
        """Add message to log console"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
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
                
                # Analyze image for noise and blur
                self.log("\nImage Analysis:")
                has_salt, salt_ratio = detect_salt_noise(self.original_image)
                has_pepper, pepper_ratio = detect_pepper_noise_median(self.original_image)
                noise_std = detect_gaussian_noise_level(self.original_image)
                is_blurry = detect_blur(self.original_image, threshold=50)
                
                self.log(f"  Salt noise: {salt_ratio:.5f} {'(HIGH)' if has_salt else '(OK)'}")
                self.log(f"  Pepper noise: {pepper_ratio:.5f} {'(HIGH)' if has_pepper else '(OK)'}")
                self.log(f"  Gaussian noise: {noise_std:.2f}")
                self.log(f"  Blurry: {'Yes' if is_blurry else 'No'}")
                
                # Suggest enhancements based on analysis
                self.log("\nSuggested enhancements (based on analysis):")
                if has_salt or has_pepper or noise_std > 40:
                    self.log("  ✓ Consider enabling 'Noise Reduction'")
                if is_blurry:
                    self.log("  ✓ Consider enabling 'Edge Sharpening'")
                if salt_ratio < 0.001 and pepper_ratio < 0.001 and not is_blurry:
                    self.log("  ✓ Image is clean. Consider 'Gamma Correction' or 'CLAHE' for contrast")
                
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
        canvas.update()  # Ensure canvas has proper dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate aspect ratio preserving resize
            img_width, img_height = pil_image.size
            canvas_aspect = canvas_width / canvas_height
            img_aspect = img_width / img_height
            
            if img_aspect > canvas_aspect:
                # Image is wider than canvas
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect)
            else:
                # Image is taller than canvas
                new_height = canvas_height
                new_width = int(canvas_height * img_aspect)
            
            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and store reference
            photo_image = ImageTk.PhotoImage(pil_image)
            
            if is_original:
                self.original_photo_ref = photo_image  # Keep reference for original
            else:
                self.final_photo_ref = photo_image  # Keep reference for final
            
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
        grid_size = grid_map[puzzle_type]
        
        # Create output directory in current folder
        output_dir = os.path.join(os.getcwd(), "output")
        final_image_dir = os.path.join(output_dir, "final_image")
        
        # Update status
        self.update_status(f"Processing {puzzle_type} with selected features...")
        
        # Clear processed display while processing
        self.processed_canvas.delete("all")
        self.processed_info.config(text=f"Processing {len(selected_options)} selected features...")
        
        # Run in separate thread
        def process_thread():
            try:
                self.log("=" * 60)
                self.log("STARTING PIPELINE WITH USER-SELECTED FEATURES")
                self.log("=" * 60)
                self.log(f"Image: {os.path.basename(self.image_path)}")
                self.log(f"Puzzle Type: {puzzle_type} (Grid: {grid_size}x{grid_size})")
                self.log(f"\nSelected features:")
                self.log(f"  - Noise reduction: {'✓' if self.noise_var.get() else '✗'}")
                self.log(f"  - Sharpening: {'✓' if self.sharpening_var.get() else '✗'}")
                self.log(f"  - Gamma correction: {'✓' if self.gamma_var.get() else '✗'}")
                self.log(f"  - CLAHE: {'✓' if self.clahe_var.get() else '✗'}")
                self.log(f"\nMandatory steps (always applied):")
                self.log(f"  - Grid segmentation: ✓ ({grid_size}x{grid_size})")
                self.log(f"  - Contour extraction: ✓")
                self.log(f"\nOutput Directory: {output_dir}")
                self.log("=" * 60)
                
                # Run the pipeline with ONLY selected features
                self.log("\nStarting pipeline with selected features...")
                
                final_img, contour_img, best_result = run_pipeline(
                    image_path=self.image_path,
                    output_dir=output_dir,
                    grid_size=grid_size,
                    apply_noise_reduction=self.noise_var.get(),
                    apply_sharpening=self.sharpening_var.get(),
                    apply_gamma_correction_option=self.gamma_var.get(),
                    apply_clahe_option=self.clahe_var.get(),
                    clean_output=True  # Clean output directory to override files
                )
                
                # Store results - we want to show the CONTOUR image in GUI
                self.final_image = final_img
                self.contour_image = contour_img  # This has the green contour lines
                self.best_result = best_result
                
                # Update GUI with the FINAL image WITH CONTOURS
                self.root.after(0, lambda: self.display_image(
                    self.contour_image, self.processed_canvas, is_bgr=True, is_original=False
                ))
                
                # Create applied features string
                applied_features = []
                if self.noise_var.get():
                    applied_features.append("NoiseRed")
                if self.sharpening_var.get():
                    applied_features.append("Sharp")
                if self.gamma_var.get() or self.clahe_var.get():
                    applied_features.append(f"BC({best_result['name']})")
                
                features_str = "+".join(applied_features) if applied_features else "NoEnhance"
                
                self.root.after(0, lambda: self.processed_info.config(
                    text=f"✅ PROCESSING COMPLETE!\n"
                         f"Features: {features_str}\n"
                         f"Grid: {grid_size}x{grid_size} | Contours: ✓"
                ))
                
                # Log completion
                self.log(f"\n" + "="*60)
                self.log("✅ PIPELINE COMPLETE - SELECTED FEATURES APPLIED")
                self.log("="*60)
                self.log(f"Applied features:")
                self.log(f"  Noise reduction: {'✓ Applied' if self.noise_var.get() else '✗ Skipped'}")
                self.log(f"  Sharpening: {'✓ Applied' if self.sharpening_var.get() else '✗ Skipped'}")
                
                if self.gamma_var.get() or self.clahe_var.get():
                    self.log(f"  Brightness/contrast: ✓ ({best_result['name']})")
                    self.log(f"    - Brightness: {best_result['brightness']:.1f}")
                    self.log(f"    - Contrast: {best_result['contrast']:.1f}")
                else:
                    self.log(f"  Brightness/contrast: ✗ Skipped")
                
                self.log(f"  Grid segmentation: ✓ ({grid_size}x{grid_size} grid)")
                self.log(f"  Contour extraction: ✓ (green contour lines added)")
                self.log(f"  Tiles extracted: {grid_size * grid_size}")
                self.log(f"\nOutput saved to: {output_dir}")
                self.log(f"Final image with contours saved in: {final_image_dir}")
                
                # List output files
                if os.path.exists(final_image_dir):
                    self.log(f"\nFiles created in 'final_image' folder:")
                    for file in os.listdir(final_image_dir):
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(final_image_dir, file)
                            file_size = os.path.getsize(file_path) / 1024  # Size in KB
                            self.log(f"  - {file} ({file_size:.1f} KB)")
                
                # List other output folders
                self.log(f"\nOther output created:")
                folders = ["tiles", "edges", "contours", "visualizations", "data_enhanced"]
                for folder in folders:
                    folder_path = os.path.join(output_dir, folder)
                    if os.path.exists(folder_path):
                        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                        self.log(f"  - {folder}/: {file_count} files")
                
                self.log("\n" + "="*60)
                self.log("FINAL IMAGE WITH CONTOURS IS NOW DISPLAYED")
                self.log("="*60)
                
                self.root.after(0, lambda: self.update_status("Processing complete!"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "✅ Processing Complete!", 
                    f"Selected features applied:\n\n"
                    f"✓ Noise reduction: {'Yes' if self.noise_var.get() else 'No'}\n"
                    f"✓ Sharpening: {'Yes' if self.sharpening_var.get() else 'No'}\n"
                    f"✓ Gamma/CLAHE: {'Yes' if (self.gamma_var.get() or self.clahe_var.get()) else 'No'}\n"
                    f"✓ Grid segmentation: {grid_size}x{grid_size}\n"
                    f"✓ Contour extraction: Green lines added\n\n"
                    f"Output saved to:\n{output_dir}"
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
        
        # Start processing thread
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

    def run_edge_matching_gui(self):
        """Run edge matching from GUI"""
        if self.final_image is None:
            messagebox.showwarning("Warning", "Please process an image first using 'Process Selected Features'")
            return

        # Get grid size from puzzle type
        puzzle_type = self.puzzle_var.get()
        grid_map = {"puzzle2x2": 2, "puzzle4x4": 4, "puzzle8x8": 8}
        grid_size = grid_map[puzzle_type]

        self.update_status("Running edge matching...")

        # Clear log
        self.log_text.delete(1.0, tk.END)

        def matching_thread():
            try:
                self.log("=" * 60)
                self.log("STARTING EDGE MATCHING")
                self.log("=" * 60)

                # Get output directory
                output_dir = os.path.join(os.getcwd(), "output")

                # Import here to avoid issues if not available
                from jigsaw_matcher import run_milestone2_pipeline

                # Run the matching pipeline
                self.log("Running edge matching pipeline...")
                self.matcher, self.assembly_suggestions = run_milestone2_pipeline(output_dir, grid_size)

                if self.matcher is None:
                    self.root.after(0, lambda: self.update_status("Edge matching failed"))
                    return

                self.log("Edge matching completed successfully!")
                self.root.after(0, lambda: self.update_status("Edge matching complete! Ready to assemble."))

                self.root.after(0, lambda: messagebox.showinfo(
                    "✅ Edge Matching Complete!",
                    f"Successfully analyzed {len(self.matcher.edge_features)} puzzle pieces.\n\n"
                    f"Ready to assemble the puzzle!"
                ))

            except Exception as e:
                error_msg = str(e)
                self.log(f"✗ ERROR: {error_msg}")
                self.root.after(0, lambda: self.update_status("Edge matching failed"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Edge matching failed:\n{error_msg}"))

        # Start matching thread
        thread = threading.Thread(target=matching_thread)
        thread.daemon = True
        thread.start()

    def assemble_puzzle_gui(self):
        """Assemble the puzzle and display result in GUI"""
        if self.matcher is None:
            messagebox.showwarning("Warning", "Please run edge matching first using 'Run Edge Matching'")
            return

        self.update_status("Assembling puzzle...")

        # Get grid size
        puzzle_type = self.puzzle_var.get()
        grid_map = {"puzzle2x2": 2, "puzzle4x4": 4, "puzzle8x8": 8}
        grid_size = grid_map[puzzle_type]

        try:
            # Assemble the puzzle
            assembled_image = self.matcher.assemble_puzzle_from_matches(grid_size)

            if assembled_image is not None:
                # Store and display the assembled image
                self.assembled_image = assembled_image

                # Display in the assembled canvas
                self.root.after(0, lambda: self.display_image(
                    self.assembled_image, self.assembled_canvas, is_bgr=True, is_original=False
                ))

                # Update info
                self.root.after(0, lambda: self.assembled_info.config(
                    text=f"✅ PUZZLE ASSEMBLED!\n{grid_size}x{grid_size} grid\n{len(self.matcher.edge_features)} pieces placed"
                ))

                self.update_status("Puzzle assembled successfully!")

                messagebox.showinfo("🎉 Puzzle Assembled!",
                                  f"Successfully assembled {grid_size}x{grid_size} puzzle!\n\n"
                                  f"The solved puzzle is now displayed in the right panel.")

            else:
                self.update_status("Assembly failed")
                messagebox.showerror("Error", "Failed to assemble the puzzle")

        except Exception as e:
            error_msg = str(e)
            self.log(f"✗ Assembly ERROR: {error_msg}")
            self.update_status("Assembly failed")
            messagebox.showerror("Error", f"Puzzle assembly failed:\n{error_msg}")


if __name__ == "__main__":
    root = tk.Tk()
    app = JigsawPipelineGUI(root)
    root.mainloop()