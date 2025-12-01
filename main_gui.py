# image_processing_gui.py (UPDATED - FIXES ORIGINAL IMAGE DISPLAY)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
import datetime

# Import the feature modules
try:
    from adjust_Brightness import apply as apply_brightness
    from edge_Sharpening import apply as apply_sharpening
    from blur import apply as apply_blur
    from segmentation import apply as apply_segmentation
except ImportError:
    # Fallback to placeholder functions if modules not found
    print("Warning: Some feature modules not found. Using placeholders.")
    # Define placeholder functions
    def apply_brightness(img, output_dir="outputs/brightness"):
        if img is None:
            return None, {}
        enhanced = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"brightness_{timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced)
        return enhanced, {'output_path': output_path, 'filename': filename, 'feature': 'brightness'}
    
    def apply_sharpening(img, output_dir="outputs/sharpening"):
        if img is None:
            return None, {}
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(img, -1, kernel)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"sharpening_{timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced)
        return enhanced, {'output_path': output_path, 'filename': filename, 'feature': 'sharpening'}
    
    def apply_blur(img, output_dir="outputs/blur"):
        if img is None:
            return None, {}
        enhanced = cv2.GaussianBlur(img, (5, 5), 0)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"blur_{timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, enhanced)
        return enhanced, {'output_path': output_path, 'filename': filename, 'feature': 'blur'}
    
    def apply_segmentation(img, output_dir="outputs/segmentation"):
        if img is None:
            return None, {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"segmentation_{timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, segmented)
        return segmented, {'output_path': output_path, 'filename': filename, 'feature': 'segmentation'}

class ImageProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Suite - 4 Features")
        self.root.geometry("1400x850")
        
        # Variables
        self.original_image = None
        self.current_image = None
        self.final_image = None  # NEW: Store final processed image separately
        self.output_images = {}
        self.output_info = {}
        self.processing = False
        self.processing_order = []
        self.original_photo = None  # Store reference to prevent garbage collection
        self.final_photo = None  # Store reference to prevent garbage collection
        
        # Style configuration
        self.setup_styles()
        
        # Create output directories
        self.create_output_directories()
        
        # Setup GUI
        self.setup_ui()
        
    def setup_styles(self):
        """Configure color schemes and styles"""
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'text': '#2c3e50',
            'bg': '#ecf0f1'
        }
        
    def create_output_directories(self):
        """Create the output folder structure"""
        output_dirs = [
            "outputs",
            "outputs/brightness",
            "outputs/sharpening", 
            "outputs/blur",
            "outputs/segmentation",
            "outputs/final"
        ]
        
        for dir_path in output_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
    
    def setup_ui(self):
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Top frame for controls
        control_frame = tk.Frame(self.root, height=120, bg=self.colors['primary'])
        control_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=0, pady=0)
        control_frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(control_frame, text="📷 Image Processing Suite", 
                              font=('Arial', 24, 'bold'), 
                              bg=self.colors['primary'], fg='white')
        title_label.pack(pady=(15, 5))
        
        subtitle_label = tk.Label(control_frame, 
                                 text="Blur • Brightness • Sharpening • Segmentation",
                                 font=('Arial', 11), 
                                 bg=self.colors['primary'], fg=self.colors['light'])
        subtitle_label.pack()
        
        # Button frame
        button_frame = tk.Frame(control_frame, bg=self.colors['primary'])
        button_frame.pack(pady=15)
        
        # Load Image button
        load_btn = tk.Button(button_frame, text="📂 Load Image", command=self.load_image,
                           font=('Arial', 12, 'bold'), 
                           bg=self.colors['success'], fg='white',
                           padx=25, pady=10, relief='raised', cursor='hand2')
        load_btn.pack(side='left', padx=10)
        
        # Process All Features button
        self.process_btn = tk.Button(button_frame, text="⚡ Process All Features", 
                                    command=self.process_all_features,
                                    font=('Arial', 12, 'bold'), 
                                    bg=self.colors['accent'], fg='white',
                                    padx=25, pady=10, relief='raised', 
                                    cursor='hand2', state='disabled')
        self.process_btn.pack(side='left', padx=10)
        
        # Separator
        sep = tk.Frame(button_frame, height=40, width=2, bg=self.colors['light'])
        sep.pack(side='left', padx=20)
        
        # Clear button
        clear_btn = tk.Button(button_frame, text="🗑️ Clear", command=self.clear_all,
                            font=('Arial', 11), 
                            bg=self.colors['warning'], fg='white',
                            padx=20, pady=8, relief='raised', cursor='hand2')
        clear_btn.pack(side='left', padx=10)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="👋 Ready to load an image", 
                                    font=('Arial', 10), 
                                    bg=self.colors['primary'], fg='white')
        self.status_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress_frame = tk.Frame(control_frame, bg=self.colors['primary'])
        self.progress_frame.pack(pady=(0, 10))
        
        self.progress_label = tk.Label(self.progress_frame, text="", 
                                      font=('Arial', 9), 
                                      bg=self.colors['primary'], fg=self.colors['light'])
        self.progress_label.pack()
        
        self.progress = ttk.Progressbar(self.progress_frame, mode='determinate', length=400)
        self.progress.pack()
        self.progress_frame.pack_forget()
        
        # Main content area
        content_frame = tk.Frame(self.root, bg=self.colors['bg'])
        content_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Left panel - Original Image (ALWAYS SHOWN)
        left_frame = tk.Frame(content_frame, relief='groove', bd=2, bg='white')
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        original_label = tk.Label(left_frame, text="📤 Original Image (Before Processing)", 
                                 font=('Arial', 14, 'bold'), bg='white')
        original_label.pack(pady=10)
        
        self.original_canvas = tk.Canvas(left_frame, width=600, height=450, bg='#f8f9fa')
        self.original_canvas.pack(padx=10, pady=10)
        
        # Original image info
        self.original_info = tk.Label(left_frame, text="No image loaded", 
                                     font=('Arial', 9), fg='gray', bg='white')
        self.original_info.pack(pady=(0, 10))
        
        # Right panel - Final Output Image (AFTER PROCESSING)
        right_frame = tk.Frame(content_frame, relief='groove', bd=2, bg='white')
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        output_label = tk.Label(right_frame, text="📥 Final Output Image (After Processing)", 
                               font=('Arial', 14, 'bold'), bg='white')
        output_label.pack(pady=10)
        
        self.output_canvas = tk.Canvas(right_frame, width=600, height=450, bg='#f8f9fa')
        self.output_canvas.pack(padx=10, pady=10)
        
        # Output image info
        self.output_info_label = tk.Label(right_frame, text="Process image to see output", 
                                         font=('Arial', 9), fg='gray', bg='white')
        self.output_info_label.pack(pady=(0, 10))
        
        # Bottom frame for feature outputs
        bottom_frame = tk.Frame(self.root, height=180, relief='groove', bd=2, bg='white')
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=(0, 10))
        bottom_frame.grid_propagate(False)
        
        features_label = tk.Label(bottom_frame, text="🔧 Feature Processing Status", 
                                 font=('Arial', 13, 'bold'), bg='white')
        features_label.pack(pady=10)
        
        # Create frames for each feature output
        features_frame = tk.Frame(bottom_frame, bg='white')
        features_frame.pack(pady=10)
        
        self.feature_frames = {}
        self.feature_labels = {}
        self.feature_icons = {}
        
        features = [
            ('brightness', '☀️ Brightness'),
            ('sharpening', '🔍 Sharpening'),
            ('blur', '🌀 Blur'),
            ('segmentation', '🎯 Segmentation')
        ]
        
        for i, (feature_key, feature_name) in enumerate(features):
            frame = tk.Frame(features_frame, relief='ridge', bd=1, bg='white')
            frame.grid(row=0, column=i, padx=15, pady=5)
            
            # Icon and name
            icon_label = tk.Label(frame, text=feature_name.split()[0], 
                                 font=('Arial', 16), bg='white')
            icon_label.pack(pady=(5, 0))
            
            name_label = tk.Label(frame, text=feature_name.split()[1], 
                                 font=('Arial', 10, 'bold'), bg='white')
            name_label.pack()
            
            # Status
            status = tk.Label(frame, text="⏳ Waiting", 
                            fg='gray', font=('Arial', 9), bg='white')
            status.pack(pady=5)
            
            # File info
            file_info = tk.Label(frame, text="", 
                               fg='darkgray', font=('Arial', 8), bg='white', wraplength=120)
            file_info.pack(pady=(0, 5))
            
            self.feature_frames[feature_key] = frame
            self.feature_labels[feature_key] = status
            self.feature_icons[feature_key] = (icon_label, file_info)
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Read image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("Error", "Could not load image file")
                    return
                
                # Store a working copy
                self.current_image = self.original_image.copy()
                self.final_image = None  # Reset final image
                
                # Display original image (left panel)
                self.display_original_image(self.original_image)
                
                # Clear output (right panel)
                self.clear_output_display()
                
                # Reset feature status
                self.reset_feature_status()
                
                # Enable process button
                self.process_btn.config(state='normal')
                
                # Update status and info
                filename = os.path.basename(file_path)
                img_shape = self.original_image.shape
                self.original_info.config(
                    text=f"{filename} | {img_shape[1]}×{img_shape[0]} | {img_shape[2]} channels" 
                    if len(img_shape) == 3 else f"{filename} | {img_shape[1]}×{img_shape[0]}")
                
                self.status_label.config(text=f"✅ Loaded: {filename}")
                
                # Reset output info
                self.output_images = {}
                self.output_info = {}
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_original_image(self, image):
        """Display the original image on the left canvas"""
        self.original_photo = self.convert_cv2_to_tk(image, self.original_canvas)
        if self.original_photo:
            self.original_canvas.delete("all")
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 600, 450
            
            self.original_canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                anchor='center', 
                image=self.original_photo
            )
    
    def display_final_image(self, image):
        """Display the final processed image on the right canvas"""
        self.final_photo = self.convert_cv2_to_tk(image, self.output_canvas)
        if self.final_photo:
            self.output_canvas.delete("all")
            canvas_width = self.output_canvas.winfo_width()
            canvas_height = self.output_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 600, 450
            
            self.output_canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                anchor='center', 
                image=self.final_photo
            )
    
    def convert_cv2_to_tk(self, image, canvas):
        """Convert CV2 image to Tkinter PhotoImage"""
        if image is None:
            return None
            
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Get canvas dimensions
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 600, 450
        
        # Calculate resize ratio
        img_width, img_height = pil_image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        return ImageTk.PhotoImage(pil_image)
    
    def clear_output_display(self):
        """Clear the output display"""
        self.output_canvas.delete("all")
        self.output_canvas.create_text(300, 225, 
            text="Click 'Process All Features' to see result", 
            font=('Arial', 14), fill='gray', tags="placeholder")
        self.output_info_label.config(text="Waiting for processing...")
        self.final_photo = None  # Clear reference
    
    def reset_feature_status(self):
        """Reset all feature status indicators"""
        for feature_key in self.feature_labels:
            self.feature_labels[feature_key].config(text="⏳ Waiting", fg='gray')
            icon_label, file_info = self.feature_icons[feature_key]
            file_info.config(text="")
            self.feature_frames[feature_key].config(bg='white')
    
    def clear_all(self):
        """Clear all images and reset the interface"""
        self.original_image = None
        self.current_image = None
        self.final_image = None
        self.output_images = {}
        self.output_info = {}
        
        # Clear canvases
        self.original_canvas.delete("all")
        self.output_canvas.delete("all")
        
        # Reset labels
        self.original_info.config(text="No image loaded")
        self.output_info_label.config(text="Process image to see output")
        self.status_label.config(text="👋 Ready to load an image")
        
        # Reset feature status
        self.reset_feature_status()
        
        # Disable process button
        self.process_btn.config(state='disabled')
        
        # Clear photo references
        self.original_photo = None
        self.final_photo = None
        
        # Add placeholder text
        self.original_canvas.create_text(300, 225, 
            text="Load an image to begin", 
            font=('Arial', 14), fill='gray')
        self.output_canvas.create_text(300, 225, 
            text="Process image to see output", 
            font=('Arial', 14), fill='gray')
    
    def process_all_features(self):
        """Process all 4 features in sequence"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.processing:
            return
        
        # Start processing in a separate thread
        self.processing = True
        self.process_btn.config(state='disabled')
        self.show_progress_bar("Starting image processing...")
        
        # Define processing order
        self.processing_order = ['brightness', 'sharpening', 'blur', 'segmentation']
        
        threading.Thread(target=self.process_features_thread, daemon=True).start()
    
    def show_progress_bar(self, message):
        """Show and configure the progress bar"""
        self.progress_frame.pack()
        self.progress_label.config(text=message)
        self.progress['value'] = 0
        self.root.update()
    
    def update_progress(self, value, message):
        """Update progress bar value and message"""
        self.root.after(0, lambda: self.progress_label.config(text=message))
        self.root.after(0, lambda: self.progress.config(value=value))
        self.root.update()
    
    def hide_progress_bar(self):
        """Hide the progress bar"""
        self.root.after(0, self.progress_frame.pack_forget)
    
    def process_features_thread(self):
        """Thread function to process all features"""
        try:
            # Reset outputs (but keep original image!)
            self.output_images = {}
            self.output_info = {}
            current_image = self.original_image.copy()  # Start with original
            
            total_steps = len(self.processing_order)
            
            for i, feature_key in enumerate(self.processing_order):
                # Calculate progress
                progress_value = (i / total_steps) * 100
                feature_name = feature_key.capitalize()
                self.update_progress(progress_value, f"Processing {feature_name}...")
                
                # Update feature status
                self.root.after(0, lambda fk=feature_key: 
                    self.feature_labels[fk].config(text="🔄 Processing...", fg='blue'))
                
                # Apply the feature
                if feature_key == 'brightness':
                    result, info = apply_brightness(
                        current_image, 
                        output_dir="outputs/brightness"
                    )
                elif feature_key == 'sharpening':
                    result, info = apply_sharpening(
                        current_image,
                        strength=1.0,
                        kernel_size=5,
                        output_dir="outputs/sharpening"
                    )
                elif feature_key == 'blur':
                    result, info = apply_blur(
                        current_image,
                        kernel_size=5,  # Fixed: integer instead of tuple
                        output_dir="outputs/blur"
                    )
                elif feature_key == 'segmentation':
                    result, info = apply_segmentation(
                        current_image,
                        output_dir="outputs/segmentation"
                    )
                else:
                    continue
                
                if result is not None:
                    # Store results
                    self.output_images[feature_key] = result
                    self.output_info[feature_key] = info
                    current_image = result.copy()  # Pass result to next feature
                    
                    # Update feature status with success
                    filename = os.path.basename(info.get('output_path', ''))
                    self.root.after(0, lambda fk=feature_key, fn=filename: 
                        self.update_feature_success(fk, fn))
                else:
                    # Update feature status with error
                    self.root.after(0, lambda fk=feature_key: 
                        self.update_feature_error(fk))
            
            # Store final image
            self.final_image = current_image.copy()
            
            # Final progress
            self.update_progress(100, "Saving final output...")
            
            # Save final output
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            final_path = os.path.join("outputs/final", f"final_output_{timestamp}.jpg")
            cv2.imwrite(final_path, self.final_image)
            
            # Update GUI with final image (ORIGINAL STILL SHOWS ON LEFT!)
            self.root.after(0, self.update_final_display, self.final_image, final_path)
            
            # Show completion message
            self.root.after(0, self.show_completion_message, final_path)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Processing Error", f"An error occurred: {str(e)}"))
            self.update_status(f"Error: {str(e)}")
        
        finally:
            # Clean up
            self.processing = False
            self.root.after(0, self.hide_progress_bar)
            self.root.after(0, lambda: self.process_btn.config(state='normal'))
    
    def update_feature_success(self, feature_key, filename):
        """Update feature status to success"""
        self.feature_labels[feature_key].config(text="✅ Complete", fg='green')
        self.feature_frames[feature_key].config(bg='#f0fff0')  # Light green background
        icon_label, file_info = self.feature_icons[feature_key]
        short_name = filename[:20] + "..." if len(filename) > 20 else filename
        file_info.config(text=short_name)
    
    def update_feature_error(self, feature_key):
        """Update feature status to error"""
        self.feature_labels[feature_key].config(text="❌ Failed", fg='red')
        self.feature_frames[feature_key].config(bg='#fff0f0')  # Light red background
    
    def update_final_display(self, final_image, final_path):
        """Update the final output display (right panel only)"""
        # Display final image on RIGHT canvas
        self.display_final_image(final_image)
        
        # Update output info
        filename = os.path.basename(final_path)
        self.output_info_label.config(
            text=f"✅ {filename}\nSaved in: outputs/final/")
        
        # Update status (keep original image displayed on left!)
        self.status_label.config(
            text=f"✅ All features processed! Compare original (left) with result (right)")
    
    def show_completion_message(self, final_path):
        """Show completion message box"""
        messagebox.showinfo(
            "Success", 
            f"All 4 features processed successfully!\n\n"
            f"Outputs saved in:\n"
            f"- outputs/brightness/\n"
            f"- outputs/sharpening/\n"
            f"- outputs/blur/\n"
            f"- outputs/segmentation/\n"
            f"- outputs/final/\n\n"
            f"Compare original (left) with final result (right)."
        )
    
    def update_status(self, message):
        """Update status label from thread"""
        self.root.after(0, lambda: self.status_label.config(text=message))

def main():
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()