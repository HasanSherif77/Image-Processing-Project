#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import datetime

def apply_gamma_correction(image, gamma):
    """Apply gamma correction with correct formula: output = input^(1/gamma)"""
    gamma_exp = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_exp) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def apply_clahe(image):
    """Apply CLAHE for contrast enhancement"""
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2BGR)

def analyze_image(image):
    """Analyze image brightness and contrast"""
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return brightness, contrast

def select_best_enhancement(image):
    """
    Select the best enhancement for an image.
    Returns the enhanced image and enhancement details.
    """
    original_brightness, original_contrast = analyze_image(image)
    
    # Apply different enhancement methods
    gamma_bright_img = apply_gamma_correction(image, 2.0)    # Darken
    gamma_dark_img = apply_gamma_correction(image, 0.5)      # Brighten
    clahe_img = apply_clahe(image)                           # CLAHE
    
    # Calculate metrics for all images
    results = []
    
    results.append({
        'name': 'Original',
        'image': image,
        'brightness': original_brightness,
        'contrast': original_contrast
    })
    
    # Gamma Brighten
    bright_brightness, bright_contrast = analyze_image(gamma_bright_img)
    results.append({
        'name': 'Gamma Brighten (γ=0.5)',
        'image': gamma_bright_img,
        'brightness': bright_brightness,
        'contrast': bright_contrast
    })
    
    # Gamma Darken
    dark_brightness, dark_contrast = analyze_image(gamma_dark_img)
    results.append({
        'name': 'Gamma Darken (γ=2.0)',
        'image': gamma_dark_img,
        'brightness': dark_brightness,
        'contrast': dark_contrast
    })
    
    # CLAHE
    clahe_brightness, clahe_contrast = analyze_image(clahe_img)
    results.append({
        'name': 'CLAHE',
        'image': clahe_img,
        'brightness': clahe_brightness,
        'contrast': clahe_contrast
    })
    
    # Select best result for puzzle processing
    IDEAL_BRIGHTNESS_MIN, IDEAL_BRIGHTNESS_MAX = 80, 180
    IDEAL_CONTRAST_MIN = 30
    
    best_result = None
    best_score = -9999
    
    # Include ALL results (including original) in the selection process
    for result in results:
        score = 0
        
        # Brightness scoring (closer to middle is better)
        mid_point = (IDEAL_BRIGHTNESS_MIN + IDEAL_BRIGHTNESS_MAX) / 2
        brightness_distance = abs(result['brightness'] - mid_point)
        brightness_score = 100 - (brightness_distance / mid_point * 100)
        score += brightness_score
        
        # Contrast scoring (higher is better)
        if result['contrast'] >= IDEAL_CONTRAST_MIN:
            contrast_score = 50
        else:
            contrast_score = (result['contrast'] / IDEAL_CONTRAST_MIN) * 50
        score += contrast_score
        
        # Bonus points for original image if it's already good
        if result['name'] == 'Original':
            # Extra bonus if original is already in ideal range
            if IDEAL_BRIGHTNESS_MIN <= result['brightness'] <= IDEAL_BRIGHTNESS_MAX and result['contrast'] >= IDEAL_CONTRAST_MIN:
                score += 30  # Significant bonus for already-perfect image
            elif (70 <= result['brightness'] <= 190 and result['contrast'] >= 25):
                score += 15  # Moderate bonus for good original image
        
        # Bonus for darkening if original was too bright (only for enhanced images)
        elif original_brightness > 170 and 'Darken' in result['name']:
            score += 20
        
        # Store the score
        result['calculated_score'] = score
        
        if score > best_score:
            best_score = score
            best_result = result
    
    print(f"Brightness Adjustment: Selected {best_result['name']}")
    print(f"  Original brightness: {original_brightness:.1f}")
    print(f"  Enhanced brightness: {best_result['brightness']:.1f}")
    
    return best_result['image'], best_result

def save_enhanced_image(enhanced_img, enhancement_info):
    """
    Save the enhanced image to output folder.
    
    Parameters:
    -----------
    enhanced_img : numpy.ndarray
        Enhanced image
    enhancement_info : dict
        Information about the enhancement applied
    
    Returns:
    --------
    str: Path to saved image
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the enhanced image
    filename = f"adjusted_brightness_{timestamp}.jpg"
    output_path = os.path.join(output_dir, filename)
    
    # Save the image
    cv.imwrite(output_path, enhanced_img)
    
    # Save enhancement info as text file
    info_filename = f"enhancement_info_{timestamp}.txt"
    info_path = os.path.join(output_dir, info_filename)
    
    with open(info_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("BRIGHTNESS ENHANCEMENT INFO\n")
        f.write("=" * 50 + "\n")
        f.write(f"Enhancement Applied: {enhancement_info['name']}\n")
        f.write(f"Brightness: {enhancement_info['brightness']:.1f}\n")
        f.write(f"Contrast: {enhancement_info['contrast']:.1f}\n")
        f.write(f"Score: {enhancement_info.get('calculated_score', 'N/A')}\n")
        f.write(f"Saved at: {timestamp}\n")
        f.write(f"Image saved as: {filename}\n")
    
    print(f"Enhanced image saved to: {output_path}")
    print(f"Enhancement info saved to: {info_path}")
    
    return output_path

def save_comparison_image(original_img, enhanced_img, enhancement_info):
    """
    Create and save a comparison image showing before/after.
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        Original image
    enhanced_img : numpy.ndarray
        Enhanced image
    enhancement_info : dict
        Information about the enhancement
    
    Returns:
    --------
    str: Path to saved comparison image
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison image
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    original_rgb = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
    axes[0].imshow(original_rgb)
    original_brightness, original_contrast = analyze_image(original_img)
    axes[0].set_title(f'ORIGINAL\nBrightness: {original_brightness:.1f}',
                     fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # Enhanced image
    enhanced_rgb = cv.cvtColor(enhanced_img, cv.COLOR_BGR2RGB)
    axes[1].imshow(enhanced_rgb)
    axes[1].set_title(f'ENHANCED: {enhancement_info["name"]}\nBrightness: {enhancement_info["brightness"]:.1f}',
                     fontsize=12, weight='bold', color='green')
    axes[1].axis('off')
    
    # Add improvement text
    brightness_change = enhancement_info['brightness'] - original_brightness
    if brightness_change > 0:
        change_text = f"Brightened by {brightness_change:+.1f}"
    elif brightness_change < 0:
        change_text = f"Darkened by {abs(brightness_change):.1f}"
    else:
        change_text = "No brightness change"
    
    plt.figtext(0.5, 0.01, f"RESULT: {change_text}",
               fontsize=10, weight='bold', ha='center',
               bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow"))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the comparison
    comparison_filename = f"comparison_{timestamp}.jpg"
    comparison_path = os.path.join(output_dir, "comparisons", comparison_filename)
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison image saved to: {comparison_path}")
    return comparison_path

# Main function for GUI integration
def apply(image, save_to_output=True, return_output_info=False):
    """
    Main function to apply brightness adjustment.
    This is the function called by your GUI.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    save_to_output : bool, optional
        Whether to save the result to output folder (default: True)
    return_output_info : bool, optional
        Whether to return output information (default: False)
        
    Returns:
    --------
    numpy.ndarray or tuple
        If return_output_info=False: Enhanced image in BGR format
        If return_output_info=True: (enhanced_image, output_info)
    """
    if image is None:
        return None if not return_output_info else (None, {})
    
    # Make a copy to avoid modifying original
    img = image.copy()
    
    # Apply brightness adjustment
    enhanced_img, enhancement_info = select_best_enhancement(img)
    
    output_info = {}
    
    # Save to output folder if requested
    if save_to_output:
        image_path = save_enhanced_image(enhanced_img, enhancement_info)
        comparison_path = save_comparison_image(img, enhanced_img, enhancement_info)
        
        output_info = {
            'enhanced_image_path': image_path,
            'comparison_image_path': comparison_path,
            'enhancement_info': enhancement_info,
            'original_brightness': analyze_image(img)[0],
            'enhanced_brightness': enhancement_info['brightness']
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("BRIGHTNESS ADJUSTMENT COMPLETE")
        print("=" * 50)
        print(f"Selected enhancement: {enhancement_info['name']}")
        print(f"Brightness change: {output_info['enhanced_brightness'] - output_info['original_brightness']:+.1f}")
        print(f"Enhanced image saved: {os.path.basename(image_path)}")
        print(f"Comparison saved: comparisons/{os.path.basename(comparison_path)}")
        print("All outputs saved in 'output' folder")
    
    if return_output_info:
        return enhanced_img, output_info
    else:
        return enhanced_img

# Alternative function that always saves and returns info
def apply_and_save(image):
    """
    Apply brightness adjustment and always save to output folder.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
        
    Returns:
    --------
    tuple: (enhanced_image, output_info)
        enhanced_image: Enhanced image for GUI display
        output_info: Dictionary with output paths and info
    """
    return apply(image, save_to_output=True, return_output_info=True)

# Simple version for quick testing
def quick_apply(image):
    """
    Simple brightness adjustment without saving.
    Good for quick testing in GUI.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
        
    Returns:
    --------
    numpy.ndarray
        Enhanced image in BGR format
    """
    return apply(image, save_to_output=False, return_output_info=False)

# For backward compatibility
def notebook_apply(image):
    """
    Legacy function for notebook compatibility
    """
    img = image.copy()
    
    # Apply gamma correction and CLAHE
    gamma_bright_img = apply_gamma_correction(img, 2.0)
    gamma_dark_img = apply_gamma_correction(img, 0.5)
    clahe_img = apply_clahe(img)
    
    # Calculate metrics
    original_brightness = np.mean(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    original_contrast = np.std(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    
    # For now, return CLAHE - you can add the full selection logic
    return clahe_img

# For testing purposes
if __name__ == "__main__":
    # Test the function
    import tkinter as tk
    from tkinter import filedialog
    
    print("=" * 60)
    print("BRIGHTNESS ADJUSTMENT TEST")
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
        img = cv.imread(file_path)
        
        if img is not None:
            print(f"\nProcessing image: {os.path.basename(file_path)}")
            print(f"Image size: {img.shape}")
            
            # Apply brightness adjustment with saving
            enhanced_img, output_info = apply_and_save(img)
            
            # Display results
            cv.imshow("Original", img)
            cv.imshow("Enhanced (Shown in GUI)", enhanced_img)
            
            # Also show the saved image from output folder
            if os.path.exists(output_info['enhanced_image_path']):
                saved_img = cv.imread(output_info['enhanced_image_path'])
                cv.imshow("Saved in Output Folder", saved_img)
            
            cv.waitKey(0)
            cv.destroyAllWindows()
            
            print("\n✅ Processing complete!")
            print(f"📁 Check the 'output' folder for saved files:")
            print(f"   - {os.path.basename(output_info['enhanced_image_path'])}")
            print(f"   - comparisons/{os.path.basename(output_info['comparison_image_path'])}")
        else:
            print("❌ Failed to load image")
    else:
        print("❌ No image selected")