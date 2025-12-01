import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

import adjust_Brightness
import blur
import sharpening
import segmentation


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1b1b1d")

        self.original = None
        self.final = None
        self.feature_images = [None]*4
        self.current_image_path = None  # Store the path of loaded image

        # Use the 'apply' function from each module
        self.features = [
            self.apply_brightness_adjustment,  # Custom function for brightness
            blur.apply,
            sharpening.apply,
            segmentation.apply
        ]
        self.feature_names = [
            "Adjust Brightness",
            "Apply Blur",
            "Sharpen Image",
            "Segment Image"
        ]
        
        # Output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.build_ui()

    def build_ui(self):
        # Scrollable frame setup
        main_canvas = tk.Canvas(self.root, bg="#1b1b1d")
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=main_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        main_canvas.configure(yscrollcommand=scrollbar.set)

        scrollable_frame = tk.Frame(main_canvas, bg="#1b1b1d")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Title
        tk.Label(
            scrollable_frame,
            text="Image Processor",
            bg="#1b1b1d",
            fg="white",
            font=("Segoe UI", 28, "bold")
        ).pack(pady=15)

        # Load Image Button
        load_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        load_frame.pack(pady=10)
        
        tk.Button(
            load_frame,
            text="📁 Load Image",
            command=self.load_image,
            bg="#190c28",
            fg="white",
            font=("Segoe UI", 13, "bold"),
            bd=0,
            height=2,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Save Final Image Button
        tk.Button(
            load_frame,
            text="💾 Save Final Image",
            command=self.save_final_image,
            bg="#006400",
            fg="white",
            font=("Segoe UI", 13, "bold"),
            bd=0,
            height=2,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Open Output Folder Button
        tk.Button(
            load_frame,
            text="📂 Open Output Folder",
            command=self.open_output_folder,
            bg="#8B4513",
            fg="white",
            font=("Segoe UI", 13, "bold"),
            bd=0,
            height=2,
            width=18
        ).pack(side=tk.LEFT, padx=5)

        # Status Label
        self.status_label = tk.Label(
            scrollable_frame,
            text="No image loaded",
            bg="#1b1b1d",
            fg="#888888",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(pady=5)

        # Top frame: Before / After final
        top_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        top_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=10)

        tk.Label(top_frame, text="Before / After Final",
                 bg="#1b1b1d", fg="white", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.top_canvas = tk.Canvas(top_frame, bg="#0e0e0f", height=400)
        self.top_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom frame: 4 feature previews
        bottom_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        bottom_frame.pack(fill=tk.X, expand=False, padx=20, pady=10)

        tk.Label(bottom_frame, text="Feature Previews",
                 bg="#1b1b1d", fg="white", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.feature_frame = tk.Frame(bottom_frame, bg="#1b1b1d")
        self.feature_frame.pack(fill=tk.X, expand=True)

        self.feature_canvases = []
        self.feature_labels = []
        for i in range(4):
            # Container for each feature
            feature_container = tk.Frame(self.feature_frame, bg="#1b1b1d")
            feature_container.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
            
            # Feature label
            label = tk.Label(
                feature_container,
                text=self.feature_names[i],
                bg="#1b1b1d",
                fg="white",
                font=("Segoe UI", 11, "bold")
            )
            label.pack()
            
            # Canvas for image
            c = tk.Canvas(feature_container, bg="#0e0e0f", width=300, height=200)
            c.pack()
            
            self.feature_canvases.append(c)
            self.feature_labels.append(label)

        # Buttons for applying features
        btn_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        btn_frame.pack(pady=10)

        # Feature buttons with colors
        button_colors = ["#3a3aff", "#ff6b6b", "#4ecdc4", "#ffe66d"]
        
        for i in range(4):
            tk.Button(
                btn_frame,
                text=self.feature_names[i],
                command=lambda idx=i: self.apply_feature(idx),
                bg=button_colors[i],
                fg="white",
                font=("Segoe UI", 12, "bold"),
                bd=0,
                height=2,
                width=15,
                relief=tk.RAISED
            ).pack(side=tk.LEFT, padx=10)

        # Special buttons
        special_btn_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        special_btn_frame.pack(pady=10)

        tk.Button(
            special_btn_frame,
            text="✨ Apply All Features",
            command=self.apply_all,
            bg="#ff9900",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            bd=0,
            height=2,
            width=18
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            special_btn_frame,
            text="🔄 Reset All",
            command=self.reset_all,
            bg="#dc3545",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            bd=0,
            height=2,
            width=15
        ).pack(side=tk.LEFT, padx=10)

        # Console output area
        console_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(console_frame, text="Processing Log",
                 bg="#1b1b1d", fg="white", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        
        # Text widget for console output
        self.console_text = tk.Text(
            console_frame,
            bg="#0e0e0f",
            fg="#00ff00",
            font=("Consolas", 10),
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        
        # Add scrollbar to console
        console_scrollbar = tk.Scrollbar(console_frame, command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=console_scrollbar.set)
        
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log_message(self, message, color="#00ff00"):
        """Add a message to the console log"""
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, message + "\n", color)
        self.console_text.see(tk.END)  # Auto-scroll to bottom
        self.console_text.config(state=tk.DISABLED)
        self.root.update()  # Update GUI immediately

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if not path:
            return
            
        self.original = cv2.imread(path)
        if self.original is None:
            messagebox.showerror("Error", "Could not load image. Please try another file.")
            return
            
        self.current_image_path = path
        self.final = self.original.copy()
        self.feature_images = [None]*4
        
        # Update status
        filename = os.path.basename(path)
        self.status_label.config(
            text=f"Loaded: {filename} ({self.original.shape[1]}x{self.original.shape[0]})",
            fg="#4ecdc4"
        )
        
        # Log message
        self.log_message(f"✅ Image loaded: {filename}")
        self.log_message(f"   Size: {self.original.shape[1]}x{self.original.shape[0]}")
        self.log_message(f"   Original brightness: {self.get_brightness(self.original):.1f}")
        
        self.update_top_canvas()
        self.update_feature_canvases()

    def get_brightness(self, image):
        """Calculate brightness of an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def apply_brightness_adjustment(self, image):
        """
        Custom brightness adjustment function that saves to output folder
        and returns the enhanced image for GUI display
        """
        try:
            self.log_message("⚡ Applying brightness adjustment...")
            
            # Use the apply_and_save function from adjust_Brightness
            enhanced_img, output_info = adjust_Brightness.apply_and_save(image.copy())
            
            # Log the results
            brightness_change = output_info['enhanced_brightness'] - output_info['original_brightness']
            self.log_message(f"   Selected: {output_info['enhancement_info']['name']}")
            self.log_message(f"   Brightness change: {brightness_change:+.1f}")
            self.log_message(f"   Saved to: {os.path.basename(output_info['enhanced_image_path'])}")
            self.log_message(f"   Comparison saved to: comparisons/{os.path.basename(output_info['comparison_image_path'])}")
            
            return enhanced_img
            
        except Exception as e:
            self.log_message(f"❌ Error in brightness adjustment: {str(e)}", "#ff6b6b")
            return image  # Return original if error

    def apply_feature(self, idx):
        if self.original is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        try:
            self.log_message(f"🔧 Applying {self.feature_names[idx]}...")
            
            # Apply the feature
            if idx == 0:  # Brightness adjustment (special handling)
                enhanced_img = self.features[idx](self.original.copy())
            else:
                enhanced_img = self.features[idx](self.original.copy())
            
            # Update feature preview with enhanced image
            self.feature_images[idx] = enhanced_img
            
            # Update final image cumulatively
            if idx == 0:
                # For brightness, replace final with enhanced
                self.final = enhanced_img.copy()
            else:
                # For other features, apply to final cumulatively
                self.final = self.features[idx](self.final.copy())
            
            # Highlight the applied feature label
            self.feature_labels[idx].config(fg="#ffcc00")
            
            # Update display
            self.update_top_canvas()
            self.update_feature_canvases()
            
            # Update status
            if idx == 0:
                current_brightness = self.get_brightness(self.final)
                self.status_label.config(
                    text=f"Brightness adjusted: {current_brightness:.1f}",
                    fg="#ffcc00"
                )
            
            self.log_message(f"✅ {self.feature_names[idx]} applied successfully")
            
        except Exception as e:
            self.log_message(f"❌ Error applying {self.feature_names[idx]}: {str(e)}", "#ff6b6b")

    def apply_all(self):
        if self.original is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        try:
            self.log_message("✨ Starting to apply all features...")
            
            # Reset final image
            self.final = self.original.copy()
            
            # Apply all features sequentially
            for i in range(4):
                self.log_message(f"   Applying {self.feature_names[i]}...")
                
                if i == 0:  # Brightness adjustment
                    enhanced_img = self.features[i](self.original.copy())
                else:
                    enhanced_img = self.features[i](self.original.copy())
                
                # Update feature preview
                self.feature_images[i] = enhanced_img
                
                # Update final image
                if i == 0:
                    self.final = enhanced_img.copy()
                else:
                    self.final = self.features[i](self.final.copy())
                
                # Highlight the feature label
                self.feature_labels[i].config(fg="#ffcc00")
                
                self.log_message(f"   ✓ {self.feature_names[i]} done")
            
            # Update display
            self.update_top_canvas()
            self.update_feature_canvases()
            
            # Update status
            final_brightness = self.get_brightness(self.final)
            self.status_label.config(
                text=f"All features applied! Final brightness: {final_brightness:.1f}",
                fg="#ffcc00"
            )
            
            self.log_message("✅ All features applied successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Error applying all features: {str(e)}", "#ff6b6b")

    def reset_all(self):
        if self.original is None:
            return
        
        # Reset to original
        self.final = self.original.copy()
        self.feature_images = [None]*4
        
        # Reset feature labels
        for label in self.feature_labels:
            label.config(fg="white")
        
        # Reset status
        if self.current_image_path:
            filename = os.path.basename(self.current_image_path)
            self.status_label.config(
                text=f"Reset: {filename}",
                fg="#888888"
            )
        
        # Log reset
        self.log_message("🔄 All features reset to original")
        
        # Update display
        self.update_top_canvas()
        self.update_feature_canvases()

    def save_final_image(self):
        if self.final is None:
            messagebox.showwarning("No Image", "No image to save!")
            return
        
        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Final Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if save_path:
            try:
                cv2.imwrite(save_path, self.final)
                self.log_message(f"💾 Final image saved to: {save_path}")
                messagebox.showinfo("Success", f"Image saved successfully!\n{save_path}")
            except Exception as e:
                self.log_message(f"❌ Error saving image: {str(e)}", "#ff6b6b")
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

    def open_output_folder(self):
        """Open the output folder in file explorer"""
        try:
            if os.path.exists(self.output_dir):
                os.startfile(self.output_dir) if os.name == 'nt' else os.system(f'open "{self.output_dir}"')
                self.log_message(f"📂 Opened output folder: {self.output_dir}")
            else:
                self.log_message("ℹ️ Output folder doesn't exist yet", "#ffcc00")
        except Exception as e:
            self.log_message(f"❌ Error opening output folder: {str(e)}", "#ff6b6b")

    def update_top_canvas(self):
        if self.original is None or self.final is None:
            return
        
        try:
            before = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
            after = cv2.cvtColor(self.final, cv2.COLOR_BGR2RGB)
            
            # Calculate target size for canvas
            canvas_width = self.top_canvas.winfo_width() or 800
            canvas_height = self.top_canvas.winfo_height() or 400
            
            # Calculate sizes for side-by-side display
            each_width = canvas_width // 2 - 20
            
            # Resize images proportionally
            scale = min(each_width / before.shape[1], canvas_height / before.shape[0])
            new_width = int(before.shape[1] * scale)
            new_height = int(before.shape[0] * scale)
            
            before_resized = cv2.resize(before, (new_width, new_height))
            after_resized = cv2.resize(after, (new_width, new_height))
            
            # Create combined image with separator
            separator = np.zeros((new_height, 10, 3), dtype=np.uint8)
            separator[:, :, :] = [64, 64, 64]  # Gray separator
            
            combined = cv2.hconcat([before_resized, separator, after_resized])
            
            self.display_image(combined, self.top_canvas, center=True)
            
        except Exception as e:
            self.log_message(f"❌ Error updating top canvas: {str(e)}", "#ff6b6b")

    def update_feature_canvases(self):
        for i, c in enumerate(self.feature_canvases):
            img = self.feature_images[i]
            if img is None:
                c.delete("all")
                c.create_rectangle(0, 0, 300, 200, fill="#0e0e0f")
                # Add placeholder text
                c.create_text(150, 100, text=f"{self.feature_names[i]}\n(Not applied)",
                            fill="#888888", font=("Arial", 11), justify="center")
            else:
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.display_image(img_rgb, c, center=True)
                except Exception as e:
                    c.delete("all")
                    c.create_rectangle(0, 0, 300, 200, fill="#0e0e0f")
                    c.create_text(150, 100, text="Error", fill="red", font=("Arial", 12))

    def display_image(self, img, canvas, center=False):
        try:
            pil = Image.fromarray(img)
            
            # Get canvas dimensions
            w = canvas.winfo_width() or 300
            h = canvas.winfo_height() or 200
            
            # Calculate scaling
            img_width, img_height = pil.size
            scale = min(w / img_width, h / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            pil = pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            tk_img = ImageTk.PhotoImage(pil)
            
            # Clear canvas and display image
            canvas.delete("all")
            
            if center:
                # Center the image
                x_offset = (w - new_width) // 2
                y_offset = (h - new_height) // 2
                canvas.create_image(x_offset, y_offset, image=tk_img, anchor="nw")
            else:
                canvas.create_image(0, 0, image=tk_img, anchor="nw")
            
            # Keep reference to prevent garbage collection
            canvas.image = tk_img
            
        except Exception as e:
            print(f"Display error: {e}")


def main():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    # Add numpy import for get_brightness method
    import numpy as np
    main()