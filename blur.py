# blur.py (placeholder)
import cv2
import numpy as np
import os
import datetime

def apply_gaussian_blur(image, kernel_size=5):
    """Apply Gaussian blur to image"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply(image, kernel_size=5, output_dir="outputs/blur"):
    """
    Main function for GUI integration.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    kernel_size : int, optional
        Gaussian kernel size (default: 5)
    output_dir : str
        Directory to save output
        
    Returns:
    --------
    tuple: (blurred_image, output_info)
        blurred_image: Blurred image
        output_info: Dictionary with output paths
    """
    if image is None:
        return None, {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply blur
    blurred_img = apply_gaussian_blur(image.copy(), kernel_size)
    
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save the blurred image
    filename = f"blur_{timestamp}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, blurred_img)
    
    # Prepare output info
    output_info = {
        'output_path': output_path,
        'filename': filename,
        'feature': 'blur',
        'kernel_size': kernel_size
    }
    
    return blurred_img, output_info