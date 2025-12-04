# Jigsaw Puzzle — Milestone 1 Pipeline

import cv2
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import sys
import shutil

if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

IMAGE_PATH = None  # Will be set by the user or GUI
OUT_DIR = None  # Will be set by the user or GUI
FALLBACK_GRID = None
PADDING = 4
RESIZE_TO = None
SAVE_CONTOUR_NPY = True


def detect_salt_noise(img, bright_thresh=220, ratio=0.001):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    salt_ratio = np.sum(gray > bright_thresh) / gray.size
    return salt_ratio > ratio, salt_ratio


def detect_gaussian_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_std = np.std(gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0))
    return noise_std


def detect_gaussian_noise_level(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()


def detect_pepper_noise_median(img, med_kernel=3, diff_threshold=25, ratio=0.002):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, med_kernel)

    diff = median.astype(np.int32) - gray.astype(np.int32)

    pepper_pixels = np.sum(diff > diff_threshold)
    pepper_ratio = pepper_pixels / gray.size

    return pepper_ratio > ratio, pepper_ratio


def detect_blur(img, threshold=120):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


def enhance_image(img):
    print("\n===== IMAGE ANALYSIS =====")

    has_salt, salt_ratio = detect_salt_noise(img)
    has_pepper, pepper_ratio = detect_pepper_noise_median(img)
    noise_std = detect_gaussian_noise_level(img)
    is_blurry = detect_blur(img, threshold=50)

    print(f"Salt ratio:   {salt_ratio:.5f}")
    print(f"Pepper ratio: {pepper_ratio:.5f}")
    print(f"Gaussian STD: {noise_std:.2f}")
    print(f"Blurry:       {is_blurry}")
    print("==========================\n")

    if (
            salt_ratio < 0.009 and
            pepper_ratio < 0.009 and
            noise_std < 70 and
            not is_blurry
    ):
        print("Image is CLEAN → No enhancement applied.\n")
        return img.copy()

    enhanced = img.copy()

    significant_sp_noise = (salt_ratio > 0.01 or pepper_ratio > 0.01) and (noise_std > 40 and noise_std < 60)

    if significant_sp_noise:
        print("Significant salt/pepper noise → median 5x5 applied")
        enhanced = cv2.medianBlur(enhanced, 5)
    else:
        print("Salt/Pepper noise negligible → skip median filter")

    if noise_std < 30:
        print("Very low Gaussian noise → bilateral 3x3")
        enhanced = cv2.bilateralFilter(enhanced, 3, 50, 50)

    elif noise_std < 50:
        print("Light Gaussian noise → bilateral 5x5")
        enhanced = cv2.bilateralFilter(enhanced, 9, 50, 50)

    if is_blurry:
        print("Image is blurry → sharpening applied")
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        enhanced = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)

    else:
        print("Image is sharp → no sharpening")

    return enhanced


def edge_sharpening(image, strength=1.8, kernel_size=5):
    """Apply edge sharpening"""
    image_float = image.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(image_float, 1.0 + strength, blurred, -strength, 0)
    sharpened = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
    return sharpened


def apply_gamma_correction(image, gamma):
    gamma_exp = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_exp) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def create_final_comparison(original_img, best_result, enhanced_img, output_dir=None):

    if output_dir is None:
        output_dir = OUT_DIR

    gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    original_brightness = np.mean(gray_original)
    enhanced_brightness = np.mean(gray_enhanced)
    brightness_change = enhanced_brightness - original_brightness

    is_original_best = (np.array_equal(original_img, enhanced_img) or
                        best_result['name'] == 'Original')

    if is_original_best:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original Image
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'ORIGINAL IMAGE (OPTIMAL)\nBrightness: {original_brightness:.1f}',
                          fontsize=14, weight='bold', color='green')
        axes[0].axis('off')

        # Original Histogram
        hist_original = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
        axes[1].plot(hist_original, color='green')
        axes[1].axvline(original_brightness, color='red', linestyle='--',
                        label=f'Mean: {original_brightness:.1f}')
        axes[1].set_title('Optimal Histogram')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.figtext(0.5, 0.01, "NO ENHANCEMENT NEEDED - Image is already optimal for puzzle processing",
                    fontsize=12, weight='bold', ha='center', color='green',
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))

    else:
        # Side-by-side comparison when enhancement was applied
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original Image
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'ORIGINAL\nBrightness: {original_brightness:.1f}',
                             fontsize=14, weight='bold')
        axes[0, 0].axis('off')

        # Original Histogram
        hist_original = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
        axes[0, 1].plot(hist_original, color='blue')
        axes[0, 1].axvline(original_brightness, color='red', linestyle='--',
                           label=f'Mean: {original_brightness:.1f}')
        axes[0, 1].set_title('Original Histogram')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Enhanced Image (FINAL image with all processing)
        axes[1, 0].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))

        # Create enhancement description
        if 'BrightnessContrast' in best_result['name']:
            enhancement_desc = best_result['name']
        else:
            enhancement_desc = "Enhanced"

        axes[1, 0].set_title(f'ENHANCED: {enhancement_desc}\nBrightness: {enhanced_brightness:.1f}',
                             fontsize=14, weight='bold', color='green')
        axes[1, 0].axis('off')

        # Enhanced Histogram
        hist_enhanced = cv2.calcHist([gray_enhanced], [0], None, [256], [0, 256])
        axes[1, 1].plot(hist_enhanced, color='green')
        axes[1, 1].axvline(enhanced_brightness, color='red', linestyle='--',
                           label=f'Mean: {enhanced_brightness:.1f}')
        axes[1, 1].set_title('Enhanced Histogram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add improvement text
        if brightness_change > 0:
            change_text = f"Brightened by {brightness_change:+.1f}"
        elif brightness_change < 0:
            change_text = f"Darkened by {abs(brightness_change):.1f}"
        else:
            change_text = "No brightness change"

        plt.figtext(0.5, 0.01, f"RESULT: {change_text}",
                    fontsize=12, weight='bold', ha='center',
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow"))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    if is_original_best:
        comparison_path = os.path.join(visualizations_dir, "ORIGINAL_OPTIMAL_COMPARISON.jpg")
    else:
        comparison_path = os.path.join(visualizations_dir, "ENHANCEMENT_COMPARISON.jpg")

    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

    print(f"Comparison saved: {comparison_path}")
    return fig


def cleanup_output_directory(output_dir):
    print(f"\nCleaning up output directory: {output_dir}")

    # List of subdirectories to clean
    subdirs = ["tiles", "edges", "contours", "visualizations", "data_enhanced", "final_image"]

    for subdir in subdirs:
        dir_path = os.path.join(output_dir, subdir)
        if os.path.exists(dir_path):
            try:
                # Remove all files in the directory
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                print(f"  Cleaned: {subdir}/")
            except Exception as e:
                print(f"  Warning: Could not clean {subdir}/: {e}")

    # Also clean any loose files in the main output directory
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")


def run_pipeline(image_path=None, output_dir=None, grid_size=None,
                 apply_noise_reduction=True, apply_sharpening=True,
                 apply_gamma_correction_option=False, apply_clahe_option=False,
                 clean_output=True):
    """Main pipeline execution with OPTIMAL processing order
    Returns: (final_processed_image, final_image_with_contours, best_result)

    OPTIMAL ORDER:
    1. Noise reduction → 2. Brightness/Contrast → 3. Edge sharpening → 4. Grid/Contours

    Parameters:
    - apply_noise_reduction: Apply blur/salt & pepper noise reduction
    - apply_sharpening: Apply edge sharpening
    - apply_gamma_correction_option: Apply gamma correction (if True, uses optimal gamma)
    - apply_clahe_option: Apply CLAHE contrast enhancement
    """

    global IMAGE_PATH, OUT_DIR
    if image_path is None:
        raise ValueError("Image path must be provided")

    IMAGE_PATH = image_path

    if output_dir is None:
        # Use current directory + 'output' as default
        output_dir = os.path.join(os.getcwd(), "output")

    OUT_DIR = output_dir

    if clean_output and os.path.exists(output_dir):
        cleanup_output_directory(output_dir)

    # Create output directories
    DATA_ENHANCED_DIR = os.path.join(OUT_DIR, "data_enhanced")
    FINAL_IMAGE_DIR = os.path.join(OUT_DIR, "final_image")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "tiles"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "visualizations"), exist_ok=True)
    os.makedirs(DATA_ENHANCED_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "contours"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "edges"), exist_ok=True)
    os.makedirs(FINAL_IMAGE_DIR, exist_ok=True)

    print(f"Output directory: {OUT_DIR}")
    print(f"Final image will be saved in: {FINAL_IMAGE_DIR}")

    print("\n" + "=" * 60)
    print("STARTING PIPELINE WITH OPTIMAL PROCESSING ORDER")
    print("=" * 60)
    print(f"User selected options:")
    print(f"  - Noise reduction: {'✓' if apply_noise_reduction else '✗'}")
    print(f"  - Gamma correction: {'✓' if apply_gamma_correction_option else '✗'}")
    print(f"  - CLAHE: {'✓' if apply_clahe_option else '✗'}")
    print(f"  - Edge sharpening: {'✓' if apply_sharpening else '✗'}")
    print("=" * 60)

    # Load and preserve original image
    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        raise ValueError(f"Could not load image from {IMAGE_PATH}")

    print(f"\n1. Original image loaded: {original_img.shape}")
    print(f"   Original brightness: {np.mean(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)):.1f}")

    # Start with original image for processing
    current_img = original_img.copy()

    # Track what enhancements were applied
    applied_enhancements = []

    # ========== OPTIMAL PROCESSING ORDER ==========

    # STEP 1: Noise reduction
    if apply_noise_reduction:
        print(f"\n2. APPLYING NOISE REDUCTION (Step 1/3 - Optimal Order)...")
        current_img = enhance_image(current_img)
        applied_enhancements.append("NoiseReduction")
        print(f"   ✓ Noise reduction applied (cleaned image for better processing)")
    else:
        print(f"\n2. Skipping NOISE REDUCTION (user request)")

    # STEP 2: Brightness/Contrast enhancement (if selected) - SECOND (OPTIMAL)
    if apply_gamma_correction_option or apply_clahe_option:
        print(f"\n3. APPLYING BRIGHTNESS/CONTRAST ENHANCEMENT (Step 2/3 - Optimal Order)...")

        # Create results list for comparison
        results = []
        original_brightness = np.mean(cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY))
        original_contrast = np.std(cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY))

        # Add original to results
        results.append({
            'name': 'Original',
            'image': current_img.copy(),
            'brightness': original_brightness,
            'contrast': original_contrast
        })

        # Apply gamma correction if selected
        if apply_gamma_correction_option:
            print(f"   Testing Gamma correction methods...")

            # Test different gamma values
            gamma_bright_img = apply_gamma_correction(current_img, 0.5)  # Brighten
            gamma_dark_img = apply_gamma_correction(current_img, 2.0)  # Darken

            # Gamma Brighten
            gray_bright = cv2.cvtColor(gamma_bright_img, cv2.COLOR_BGR2GRAY)
            results.append({
                'name': 'Gamma',
                'image': gamma_bright_img,
                'brightness': np.mean(gray_bright),
                'contrast': np.std(gray_bright)
            })

            # Gamma Darken
            gray_dark = cv2.cvtColor(gamma_dark_img, cv2.COLOR_BGR2GRAY)
            results.append({
                'name': 'Gamma',
                'image': gamma_dark_img,
                'brightness': np.mean(gray_dark),
                'contrast': np.std(gray_dark)
            })

        # Apply CLAHE if selected
        if apply_clahe_option:
            print(f"   Testing CLAHE enhancement...")
            clahe_img = apply_clahe(current_img)
            gray_clahe = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
            results.append({
                'name': 'CLAHE',
                'image': clahe_img,
                'brightness': np.mean(gray_clahe),
                'contrast': np.std(gray_clahe)
            })

        # Select best result based on scoring
        IDEAL_BRIGHTNESS_MIN, IDEAL_BRIGHTNESS_MAX = 80, 180
        IDEAL_CONTRAST_MIN = 30

        best_result = None
        best_score = -9999

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

            # Bonus for original if already good
            if result['name'] == 'Original':
                if IDEAL_BRIGHTNESS_MIN <= result['brightness'] <= IDEAL_BRIGHTNESS_MAX and result[
                    'contrast'] >= IDEAL_CONTRAST_MIN:
                    score += 30

            if score > best_score:
                best_score = score
                best_result = result

        current_img = best_result['image']
        applied_enhancements.append(f"BrightnessContrast({best_result['name']})")

        print(f"   Selected: {best_result['name']}")
        print(f"   Brightness: {best_result['brightness']:.1f}")
        print(f"   Contrast: {best_result['contrast']:.1f}")
        print(f"   ✓ Brightness/contrast optimized (better for edge detection)")

    else:
        print(f"\n3. Skipping BRIGHTNESS/CONTRAST ENHANCEMENT (user request)")
        original_brightness = np.mean(cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY))
        original_contrast = np.std(cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY))
        best_result = {
            'name': 'Original',
            'image': current_img,
            'brightness': original_brightness,
            'contrast': original_contrast,
            'calculated_score': 0
        }

    # STEP 3: Edge sharpening
    if apply_sharpening:
        print(f"\n4. APPLYING EDGE SHARPENING (Step 3/3 - Optimal Order)...")
        current_img = edge_sharpening(current_img, strength=1.0, kernel_size=5)
        applied_enhancements.append("Sharpening")
        print(f"   ✓ Edge sharpening applied (works best after contrast adjustment)")
    else:
        print(f"\n4. Skipping EDGE SHARPENING (user request)")

    input_filename = os.path.basename(IMAGE_PATH)
    name, ext = os.path.splitext(input_filename)

    # Create enhancement name for filename
    if applied_enhancements:
        enhancement_name = "_".join(applied_enhancements)
    else:
        enhancement_name = "NoEnhancement"

    enhanced_filename = f"{name}_enhanced_{enhancement_name}{ext}"
    enhanced_path = os.path.join(DATA_ENHANCED_DIR, enhanced_filename)
    cv2.imwrite(enhanced_path, current_img)
    print(f"\n   Saved enhanced image: {enhanced_filename}")

    # STEP 4: GRID SEGMENTATION + TILE EXTRACTION + CONTOURS (ALWAYS APPLIED)
    # If grid_size not provided, infer from folder name
    if grid_size is None:
        # Try to guess from folder name: look for 2x2, 4x4, 4x4, 8x8
        folder_name = os.path.basename(os.path.dirname(IMAGE_PATH))
        if "2x2" in folder_name:
            grid_size = 2
        elif "4x4" in folder_name:
            grid_size = 4
        elif "8x8" in folder_name:
            grid_size = 8
        else:
            # Fallback: assume smallest 2x2
            grid_size = 2

    print(f"\n5. APPLYING GRID SEGMENTATION & CONTOUR EXTRACTION (Mandatory Steps)...")
    print(f"   Grid Size: {grid_size}x{grid_size}")

    # OUTPUT FOLDERS
    TILES_DIR = os.path.join(OUT_DIR, "tiles")
    CONTOURS_DIR = os.path.join(OUT_DIR, "contours")
    EDGES_DIR = os.path.join(OUT_DIR, "edges")
    VIZ_DIR = os.path.join(OUT_DIR, "visualizations")

    # Process the FINAL enhanced image
    img = current_img
    h, w, _ = img.shape

    tile_h = h // grid_size
    tile_w = w // grid_size

    tile_count = 0

    # Create a copy of the final image to draw contours on
    final_image_with_contours = img.copy()

    # Also create image with grid lines
    final_image_with_grid = img.copy()

    # GRID SEGMENTATION + TILE EXTRACTION + CONTOUR DETECTION
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w

            tile = img[y1:y2, x1:x2]

            # Save tile
            tile_filename = f"tile_{tile_count}.png"
            cv2.imwrite(os.path.join(TILES_DIR, tile_filename), tile)

            # Edge extraction
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            edge_filename = f"edges_{tile_count}.png"
            cv2.imwrite(os.path.join(EDGES_DIR, edge_filename), edges)

            # Contour extraction
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_img = tile.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

            contour_filename = f"contour_{tile_count}.png"
            cv2.imwrite(os.path.join(CONTOURS_DIR, contour_filename), contour_img)

            # Draw contours on the final image
            for contour in contours:
                # Adjust contour coordinates to original image
                adjusted_contour = contour + np.array([x1, y1])
                cv2.drawContours(final_image_with_contours, [adjusted_contour], -1, (0, 255, 0), 2)

            tile_count += 1

    # Add grid lines to the grid image
    for i in range(1, grid_size):
        cv2.line(final_image_with_grid, (0, i * tile_h), (w, i * tile_h), (0, 0, 255), 2)
        cv2.line(final_image_with_grid, (i * tile_w, 0), (i * tile_w, h), (0, 0, 255), 2)

    # Save visualization images
    cv2.imwrite(os.path.join(VIZ_DIR, "grid_visualization.png"), final_image_with_grid)
    cv2.imwrite(os.path.join(VIZ_DIR, "final_with_contours.png"), final_image_with_contours)

    print(f"   ✓ Extracted {tile_count} tiles")
    print(f"   ✓ Detected contours on all tiles")
    print(f"   ✓ Added contour lines to final image")

    # STEP 5: Save final results
    print(f"\n6. SAVING FINAL RESULTS...")

    # Create descriptive filename based on applied enhancements
    if applied_enhancements:
        enhancement_suffix = "_".join(applied_enhancements)
    else:
        enhancement_suffix = "NoEnhancement"

    # Save final enhanced image (without contours)
    final_image_filename = f"{name}_final_{enhancement_suffix}{ext}"
    final_image_path = os.path.join(FINAL_IMAGE_DIR, final_image_filename)
    cv2.imwrite(final_image_path, current_img)

    # Save final image with contours
    final_contour_filename = f"{name}_final_{enhancement_suffix}_with_contours{ext}"
    final_contour_path = os.path.join(FINAL_IMAGE_DIR, final_contour_filename)
    cv2.imwrite(final_contour_path, final_image_with_contours)

    # Also save with generic names
    cv2.imwrite(os.path.join(FINAL_IMAGE_DIR, "PROCESSED_PUZZLE_IMAGE.jpg"), current_img)
    cv2.imwrite(os.path.join(FINAL_IMAGE_DIR, "PROCESSED_WITH_CONTOURS.jpg"), final_image_with_contours)

    print(f"   ✓ Final enhanced image: {final_image_filename}")
    print(f"   ✓ Final image with contours: {final_contour_filename}")
    print(f"   ✓ Generic copies: PROCESSED_PUZZLE_IMAGE.jpg, PROCESSED_WITH_CONTOURS.jpg")

    # STEP 6: Create comparison visualization (only if enhancements were applied)
    if applied_enhancements:
        print(f"\n7. CREATING COMPARISON VISUALIZATION...")
        # Pass: original, best_result, FINAL enhanced image
        create_final_comparison(original_img, best_result, current_img, OUT_DIR)
        print(f"   ✓ Comparison visualization created")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - OPTIMAL PROCESSING ORDER USED")
    print("=" * 60)
    print(f"Processing order followed:")
    print(f"  1. Noise reduction: {'✓' if apply_noise_reduction else '✗'}")
    print(f"  2. Brightness/Contrast: {'✓' if (apply_gamma_correction_option or apply_clahe_option) else '✗'}")
    print(f"  3. Edge sharpening: {'✓' if apply_sharpening else '✗'}")
    print(f"  4. Grid segmentation: ✓ ({grid_size}x{grid_size})")
    print(f"  5. Contour extraction: ✓")
    print("=" * 60)

    # Return BOTH images: enhanced image AND image with contours
    return current_img, final_image_with_contours, best_result


def main():
    """Main pipeline execution - for command line use with hardcoded values"""
    # For backward compatibility - using hardcoded values
    image_path = "D:/Image-Processing-Project/Gravity_Falls/puzzle_4x4/3.jpg"
    output_dir = "D:/Image-Processing-Project/output"

    print("Running pipeline with hardcoded values (for testing)")
    print(f"Image path: {image_path}")
    print(f"Output directory: {output_dir}")

    try:
        final_img, contour_img, best_result = run_pipeline(
            image_path=image_path,
            output_dir=output_dir,
            grid_size=4,
            apply_noise_reduction=True,
            apply_sharpening=True,
            apply_gamma_correction_option=True,
            apply_clahe_option=True,
            clean_output=True
        )
        print(f"\nFinal image shape: {final_img.shape}")
        print(f"Contour image shape: {contour_img.shape}")
        print(f"Best method: {best_result['name']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: You might need to update the hardcoded paths in the main() function")


if __name__ == "__main__":
    main()