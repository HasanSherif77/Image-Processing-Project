# brightness.py
import cv2 as cv
import numpy as np
import os
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
    
    # Select best result
    IDEAL_BRIGHTNESS_MIN, IDEAL_BRIGHTNESS_MAX = 80, 180
    IDEAL_CONTRAST_MIN = 30
    
    best_result = None
    best_score = -9999
    
    for result in results:
        score = 0
        
        # Brightness scoring
        mid_point = (IDEAL_BRIGHTNESS_MIN + IDEAL_BRIGHTNESS_MAX) / 2
        brightness_distance = abs(result['brightness'] - mid_point)
        brightness_score = 100 - (brightness_distance / mid_point * 100)
        score += brightness_score
        
        # Contrast scoring
        if result['contrast'] >= IDEAL_CONTRAST_MIN:
            contrast_score = 50
        else:
            contrast_score = (result['contrast'] / IDEAL_CONTRAST_MIN) * 50
        score += contrast_score
        
        if result['name'] == 'Original':
            if IDEAL_BRIGHTNESS_MIN <= result['brightness'] <= IDEAL_BRIGHTNESS_MAX and result['contrast'] >= IDEAL_CONTRAST_MIN:
                score += 30
            elif (70 <= result['brightness'] <= 190 and result['contrast'] >= 25):
                score += 15
        
        # Bonus for darkening if original was too bright
        elif original_brightness > 170 and 'Darken' in result['name']:
            score += 20
        
        result['calculated_score'] = score
        
        if score > best_score:
            best_score = score
            best_result = result
    
    return best_result['image'], best_result

def apply(image, output_dir="outputs/brightness"):
    """
    Main function for GUI integration.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    output_dir : str
        Directory to save output
        
    Returns:
    --------
    tuple: (enhanced_image, output_info)
        enhanced_image: Enhanced image
        output_info: Dictionary with output paths
    """
    if image is None:
        return None, {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply brightness adjustment
    enhanced_img, enhancement_info = select_best_enhancement(image.copy())
    
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save the enhanced image
    filename = f"brightness_{timestamp}.jpg"
    output_path = os.path.join(output_dir, filename)
    cv.imwrite(output_path, enhanced_img)
    
    # Prepare output info
    output_info = {
        'output_path': output_path,
        'filename': filename,
        'feature': 'brightness',
        'enhancement_info': enhancement_info
    }
    
    return enhanced_img, output_info