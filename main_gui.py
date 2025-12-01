# image_processing_gui.py
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time

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
        return img.copy(), {'output_path': '', 'feature': 'brightness'}
    
    def apply_sharpening(img, output_dir="outputs/sharpening"):
        return img.copy(), {'output_path': '', 'feature': 'sharpening'}
    
    def apply_blur(img, output_dir="outputs/blur"):
        return img.copy(), {'output_path': '', 'feature': 'blur'}
    
    def apply_segmentation(img, output_dir="outputs/segmentation"):
        return img.copy(), {'output_path': '', 'feature': 'segmentation'}

class ImageProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Suite")
        self.root.geometry("1400x800")
        
        # Variables
        self.original_image = None
        self.current_image = None
        self.output_images = {}
        self.output_info = {}
        self.processing = False
        
        # Create output directories
        self.create_output_directories()
        
        # Setup GUI
        self.setup_ui()
        
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
            print(f"Created directory: {dir_path}")
    
    def setup_ui(self):
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Top frame for controls
        control_frame = tk.Frame(self.root, height=100, bg='#f0f0f0')
        control_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        control_frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(control_frame, text="Image Processing Suite", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Button frame
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Load Image button
        load_btn = tk.Button(button_frame, text="Load Image", command=self.load_image,
                           font=('Arial', 12), bg='#4CAF50', fg='white',
                           padx=20, pady=10, relief='raised')
        load_btn.pack(side='left', padx=10)
        
        # Process All Features button
        process_btn = tk.Button(button_frame, text="Process All Features", command=self.process_all_features,
                              font=('Arial', 12), bg='#2196F3', fg='white',
                              padx=20, pady=10, relief='raised', state='disabled')
        process_btn.pack(side='left', padx=10)
        self.process_btn = process_btn
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Ready to load image", 
                                    font=('Arial', 10), bg='#f0f0f0', fg='#666')
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate', length=300)
        
        # Main content area
        content_frame = tk.Frame(self.root)
        content_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Left panel - Original Image
        left_frame = tk.Frame(content_frame, relief='groove', bd=2)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        original_label = tk.Label(left_frame, text="Original Image", font=('Arial', 14, 'bold'))
        original_label.pack(pady=10)
        
        self.original_canvas = tk.Canvas(left_frame, width=600, height=500, bg='#e0e0e0')
        self.original_canvas.pack(padx=10, pady=10)
        
        # Right panel - Final Output Image
        right_frame = tk.Frame(content_frame, relief='groove', bd=2)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        output_label = tk.Label(right_frame, text="Final Output Image", font=('Arial', 14, 'bold'))
        output_label.pack(pady=10)
        
        self.output_canvas = tk.Canvas(right_frame, width=600, height=500, bg='#e0e0e0')
        self.output_canvas.pack(padx=10, pady=10)
        
        # Bottom frame for feature outputs
        bottom_frame = tk.Frame(self.root, height=200, relief='groove', bd=2)
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        bottom_frame.grid_propagate(False)
        
        features_label = tk.Label(bottom_frame, text="Feature Outputs", font=('Arial', 12, 'bold'))
        features_label.pack(pady=10)
        
        # Create frames for each feature output
        features_frame = tk.Frame(bottom_frame)
        features_frame.pack(pady=10)
        
        self.feature_labels = {}
        features = ['Brightness', 'Sharpening', 'Blur', 'Segmentation']
        
        for i, feature in enumerate(features):
            frame = tk.Frame(features_frame, relief='ridge', bd=1)
            frame.grid(row=0, column=i, padx=20, pady=5)
            
            label = tk.Label(frame, text=feature, font=('Arial', 10))
            label.pack(pady=5)
            
            status = tk.Label(frame, text="Not processed", fg='gray', font=('Arial', 9))
            status.pack(pady=5)
            
            self.feature_labels[feature.lower()] = status
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
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
                
                # Display original image
                self.display_image(self.original_image, self.original_canvas)
                
                # Clear output
                self.output_canvas.delete("all")
                self.output_canvas.create_text(300, 250, 
                    text="Process image to see output", 
                    font=('Arial', 14), fill='gray')
                
                # Reset feature status
                for feature in self.feature_labels:
                    self.feature_labels[feature].config(text="Not processed", fg='gray')
                
                # Enable process button
                self.process_btn.config(state='normal')
                
                # Update status
                filename = os.path.basename(file_path)
                self.status_label.config(text=f"Loaded: {filename} ({self.original_image.shape[1]}x{self.original_image.shape[0]})")
                
                # Reset output info
                self.output_images = {}
                self.output_info = {}
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image, canvas):
        """Display an image on a canvas"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = canvas.winfo_width() - 20
        canvas_height = canvas.winfo_height() - 20
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 600, 500
        
        img_width, img_height = pil_image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, 
                          anchor='center', image=self.tk_image)
    
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
        self.progress.pack(pady=5)
        self.progress.start()
        
        threading.Thread(target=self.process_features_thread, daemon=True).start()
    
    def process_features_thread(self):
        """Thread function to process all features"""
        try:
            # Reset output
            self.output_images = {}
            self.output_info = {}
            
            # Start with original image
            current_image = self.original_image.copy()
            
            # Update status
            self.update_status("Processing brightness adjustment...")
            self.root.after(0, lambda: self.feature_labels['brightness'].config(
                text="Processing...", fg='blue'))
            
            # 1. Brightness adjustment
            bright_img, bright_info = apply_brightness(
                current_image, 
                output_dir="outputs/brightness"
            )
            self.output_images['brightness'] = bright_img
            self.output_info['brightness'] = bright_info
            current_image = bright_img.copy()
            self.root.after(0, lambda: self.feature_labels['brightness'].config(
                text=f"Saved: {bright_info.get('filename', 'N/A')}", fg='green'))
            
            # 2. Sharpening
            self.update_status("Processing sharpening...")
            self.root.after(0, lambda: self.feature_labels['sharpening'].config(
                text="Processing...", fg='blue'))
            
            sharp_img, sharp_info = apply_sharpening(
                current_image,
                strength=1.0,
                kernel_size=5,
                output_dir="outputs/sharpening"
            )
            self.output_images['sharpening'] = sharp_img
            self.output_info['sharpening'] = sharp_info
            current_image = sharp_img.copy()
            self.root.after(0, lambda: self.feature_labels['sharpening'].config(
                text=f"Saved: {sharp_info.get('filename', 'N/A')}", fg='green'))
            
            # 3. Blur
            self.update_status("Processing blur...")
            self.root.after(0, lambda: self.feature_labels['blur'].config(
                text="Processing...", fg='blue'))
            
            blur_img, blur_info = apply_blur(
                current_image,
                kernel_size=5,
                output_dir="outputs/blur"
            )
            self.output_images['blur'] = blur_img
            self.output_info['blur'] = blur_info
            current_image = blur_img.copy()
            self.root.after(0, lambda: self.feature_labels['blur'].config(
                text=f"Saved: {blur_info.get('filename', 'N/A')}", fg='green'))
            
            # 4. Segmentation
            self.update_status("Processing segmentation...")
            self.root.after(0, lambda: self.feature_labels['segmentation'].config(
                text="Processing...", fg='blue'))
            
            seg_img, seg_info = apply_segmentation(
                current_image,
                output_dir="outputs/segmentation"
            )
            self.output_images['segmentation'] = seg_img
            self.output_info['segmentation'] = seg_info
            final_image = seg_img.copy()
            
            self.root.after(0, lambda: self.feature_labels['segmentation'].config(
                text=f"Saved: {seg_info.get('filename', 'N/A')}", fg='green'))
            
            # Save final output
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            final_path = os.path.join("outputs/final", f"final_output_{timestamp}.jpg")
            cv2.imwrite(final_path, final_image)
            
            # Update GUI with final image
            self.root.after(0, self.update_final_display, final_image)
            
            # Update status
            self.update_status(f"All features processed! Final output saved.")
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", 
                f"All 4 features processed successfully!\n\n"
                f"Outputs saved in:\n"
                f"- outputs/brightness/\n"
                f"- outputs/sharpening/\n"
                f"- outputs/blur/\n"
                f"- outputs/segmentation/\n"
                f"- outputs/final/\n\n"
                f"Check the folders for all processed images."
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Processing Error", f"An error occurred: {str(e)}"))
            self.update_status(f"Error: {str(e)}")
        
        finally:
            # Clean up
            self.processing = False
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)
            self.root.after(0, lambda: self.process_btn.config(state='normal'))
    
    def update_status(self, message):
        """Update status label from thread"""
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def update_final_display(self, final_image):
        """Update the final output display"""
        self.display_image(final_image, self.output_canvas)
        
        # Store the final image
        self.current_image = final_image.copy()
        
        # Show file paths in status
        feature_count = len(self.output_info)
        self.status_label.config(
            text=f"Processed {feature_count} features. Check output folders for results."
        )

def main():
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()