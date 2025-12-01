#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import datetime

def enhance_image(img, kernel_size=(3, 3)):
    """
    Enhanced image processing with noise reduction and sharpening.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image in BGR format
    kernel_size : tuple, optional
        Gaussian kernel size (default: (3, 3))
        
    Returns:
    --------
    numpy.ndarray
        Enhanced image in BGR format
    """
    # Noise reduction with Gaussian blur
    blurred = cv2.GaussianBlur(img, kernel_size, 0)

    # Simple unsharp mask (deblurring)
    enhanced = cv2.addWeighted(img, 1.2, blurred, -0.2, 0)

    return enhanced

def apply(image, kernel_size=(3, 3), output_dir="outputs/blur"):
    """
    Main function for GUI integration.
    Applies blur enhancement and saves the result.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    kernel_size : tuple, optional
        Gaussian kernel size (default: (3, 3))
    output_dir : str
        Directory to save output
        
    Returns:
    --------
    tuple: (enhanced_image, output_info)
        enhanced_image: Enhanced image in BGR format
        output_info: Dictionary with output paths and analysis info
    """
    if image is None:
        return None, {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image (already loaded in GUI, but keep for consistency)
    img = image.copy()
    
    # Enhance the image
    enhanced = enhance_image(img, kernel_size)
    
    # Calculate mean pixel difference
    mean_diff = np.mean(cv2.absdiff(img, enhanced))
    
    # Create timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save the enhanced image
    filename = f"blur_enhanced_{timestamp}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, enhanced)
    
    # Prepare output info
    output_info = {
        'output_path': output_path,
        'filename': filename,
        'feature': 'blur',
        'kernel_size': kernel_size,
        'mean_difference': mean_diff,
        'original_shape': img.shape,
        'enhanced_shape': enhanced.shape,
        'timestamp': timestamp
    }
    
    # Print processing info (for debugging)
    print(f"Blur Enhancement Complete:")
    print(f"  Original shape: {img.shape}")
    print(f"  Enhanced shape: {enhanced.shape}")
    print(f"  Mean pixel difference: {mean_diff:.2f}")
    print(f"  Saved to: {output_path}")
    
    return enhanced, output_info

# For backward compatibility and testing
def quick_apply(image):
    """
    Simple blur enhancement without saving.
    Good for quick testing.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
        
    Returns:
    --------
    numpy.ndarray
        Enhanced image in BGR format
    """
    return enhance_image(image.copy())

# For testing the module independently
if __name__ == "__main__":
    # Test with a sample image
    import tkinter as tk
    from tkinter import filedialog
    
    print("=" * 60)
    print("BLUR ENHANCEMENT MODULE TEST")
    print("=" * 60)
    
    # Ask user to select an image
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
    )
    
    if file_path:
        # Load image
        img = cv2.imread(file_path)
        
        if img is not None:
            print(f"\nProcessing image: {os.path.basename(file_path)}")
            
            # Apply blur enhancement with saving
            enhanced_img, output_info = apply(img)
            
            # Display results
            cv2.imshow("Original", img)
            cv2.imshow("Enhanced (Blur)", enhanced_img)
            
            # Show difference
            diff = cv2.absdiff(img, enhanced_img)
            cv2.imshow("Difference", diff)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print("\n✅ Processing complete!")
            print(f"📁 Check the 'outputs/blur' folder for saved files")
        else:
            print("❌ Failed to load image")
    else:
        print("❌ No image selected")