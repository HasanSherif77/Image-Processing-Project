# sharpening.py
import cv2
import numpy as np
import os
import datetime

def edge_sharpening(image, strength=1.0, kernel_size=5):
    """
    Apply edge sharpening using unsharp masking.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    strength : float, optional
        Sharpening strength (default: 1.0)
    kernel_size : int, optional
        Gaussian kernel size for blurring (default: 5)
        
    Returns:
    --------
    numpy.ndarray
        Sharpened image in BGR format
    """
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 0)
    
    # Apply unsharp masking
    sharpened = cv2.addWeighted(image_float, 1.0 + strength, blurred, -strength, 0)
    
    # Clip and convert back to uint8
    sharpened = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
    
    return sharpened

def apply(image, strength=1.0, kernel_size=5, output_dir="outputs/sharpening"):
    """
    Main function for GUI integration.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    strength : float, optional
        Sharpening strength (default: 1.0)
    kernel_size : int, optional
        Gaussian kernel size (default: 5)
    output_dir : str
        Directory to save output
        
    Returns:
    --------
    tuple: (sharpened_image, output_info)
        sharpened_image: Sharpened image
        output_info: Dictionary with output paths
    """
    if image is None:
        return None, {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply sharpening
    sharpened_img = edge_sharpening(image.copy(), strength, kernel_size)
    
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save the sharpened image
    filename = f"sharpening_{timestamp}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, sharpened_img)
    
    # Prepare output info
    output_info = {
        'output_path': output_path,
        'filename': filename,
        'feature': 'sharpening',
        'strength': strength,
        'kernel_size': kernel_size
    }
    
    return sharpened_img, output_info