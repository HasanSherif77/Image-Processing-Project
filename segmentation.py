#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import numpy as np
import datetime

def infer_grid_size_from_path(image_path):
    """
    Infer grid size from folder name.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    int: Grid size (2, 4, or 8)
    """
    # Try to guess from folder name: look for 2x2, 4x4, 8x8
    folder_name = os.path.basename(os.path.dirname(image_path))
    if "2x2" in folder_name:
        return 2
    elif "4x4" in folder_name:
        return 4
    elif "8x8" in folder_name:
        return 8
    else:
        # Fallback: assume smallest 2x2
        return 2

def segment_image(image, grid_size=None, padding=4):
    """
    Main segmentation function: Extract tiles, edges, and contours.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    grid_size : int, optional
        Number of grid divisions (default: infer from image path or use 2)
    padding : int, optional
        Pixels to include around tiles (default: 4)
        
    Returns:
    --------
    dict: Dictionary containing segmented results and visualization images
    """
    if image is None:
        return {}
    
    h, w, _ = image.shape
    
    # Determine grid size
    if grid_size is None:
        grid_size = 2  # Default
    
    tile_h = h // grid_size
    tile_w = w // grid_size
    
    tile_count = 0
    full_contour_img = image.copy()
    
    # Prepare results dictionary
    results = {
        'grid_size': grid_size,
        'tile_height': tile_h,
        'tile_width': tile_w,
        'tiles': [],
        'edges': [],
        'contours': [],
        'grid_visualization': None,
        'contour_visualization': None
    }
    
    # GRID SEGMENTATION + TILE EXTRACTION
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w

            tile = image[y1:y2, x1:x2].copy()
            
            # Edge extraction
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            
            # Contour extraction
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = tile.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            
            # Store results
            results['tiles'].append(tile)
            results['edges'].append(edges)
            results['contours'].append(contours)
            
            # Draw contours on full image
            for cnt in contours:
                cnt_offset = cnt + [x1, y1]
                cv2.drawContours(full_contour_img, [cnt_offset], -1, (0, 255, 0), 2)

            tile_count += 1
    
    # GRID VISUALIZATION IMAGE
    grid_viz = image.copy()
    for i in range(1, grid_size):
        cv2.line(grid_viz, (0, i * tile_h), (w, i * tile_h), (0, 0, 255), 2)
        cv2.line(grid_viz, (i * tile_w, 0), (i * tile_w, h), (0, 0, 255), 2)
    
    results['grid_visualization'] = grid_viz
    results['contour_visualization'] = full_contour_img
    
    return results

def apply(image, grid_size=None, padding=4, output_dir="outputs/segmentation"):
    """
    Main function for GUI integration.
    Performs segmentation and saves all results.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    grid_size : int, optional
        Number of grid divisions (default: infer or use 2)
    padding : int, optional
        Pixels to include around tiles (default: 4)
    output_dir : str
        Directory to save output
        
    Returns:
    --------
    tuple: (primary_output_image, output_info)
        primary_output_image: Main visualization image (contours on original)
        output_info: Dictionary with all output paths and results
    """
    if image is None:
        return None, {}
    
    # Create output directories
    base_dir = output_dir
    tiles_dir = os.path.join(base_dir, "tiles")
    contours_dir = os.path.join(base_dir, "contours")
    edges_dir = os.path.join(base_dir, "edges")
    viz_dir = os.path.join(base_dir, "visualizations")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(contours_dir, exist_ok=True)
    os.makedirs(edges_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Perform segmentation
    results = segment_image(image.copy(), grid_size, padding)
    
    if not results:
        return None, {}
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save individual tiles, edges, and contours
    saved_files = {
        'tiles': [],
        'edges': [],
        'contours': []
    }
    
    for i, (tile, edges, contours) in enumerate(zip(
        results['tiles'], 
        results['edges'], 
        results['contours']
    )):
        # Save tile
        tile_filename = f"tile_{i}_{timestamp}.png"
        tile_path = os.path.join(tiles_dir, tile_filename)
        cv2.imwrite(tile_path, tile)
        saved_files['tiles'].append(tile_path)
        
        # Save edges
        edge_filename = f"edges_{i}_{timestamp}.png"
        edge_path = os.path.join(edges_dir, edge_filename)
        cv2.imwrite(edge_path, edges)
        saved_files['edges'].append(edge_path)
        
        # Save contour image
        contour_img = results['tiles'][i].copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        contour_filename = f"contour_{i}_{timestamp}.png"
        contour_path = os.path.join(contours_dir, contour_filename)
        cv2.imwrite(contour_path, contour_img)
        saved_files['contours'].append(contour_path)
    
    # Save visualization images
    grid_viz_path = os.path.join(viz_dir, f"grid_visualization_{timestamp}.png")
    contour_viz_path = os.path.join(viz_dir, f"original_with_contours_{timestamp}.png")
    
    cv2.imwrite(grid_viz_path, results['grid_visualization'])
    cv2.imwrite(contour_viz_path, results['contour_visualization'])
    
    # Prepare output info
    output_info = {
        'output_path': contour_viz_path,  # Primary output for GUI display
        'filename': os.path.basename(contour_viz_path),
        'feature': 'segmentation',
        'grid_size': results['grid_size'],
        'tile_count': len(results['tiles']),
        'tile_height': results['tile_height'],
        'tile_width': results['tile_width'],
        'saved_files': saved_files,
        'visualizations': {
            'grid': grid_viz_path,
            'contours': contour_viz_path
        },
        'timestamp': timestamp
    }
    
    # Print summary
    print(f"Segmentation Complete:")
    print(f"  Grid size: {results['grid_size']}x{results['grid_size']}")
    print(f"  Tiles extracted: {len(results['tiles'])}")
    print(f"  Main output: {os.path.basename(contour_viz_path)}")
    print(f"  All outputs saved in: {output_dir}")
    
    # Return the contour visualization as the primary output image
    return results['contour_visualization'], output_info

# Simple version for quick testing without saving
def quick_segment(image, grid_size=None):
    """
    Simple segmentation without saving files.
    Good for quick testing.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
    grid_size : int, optional
        Number of grid divisions
        
    Returns:
    --------
    numpy.ndarray
        Contour visualization image
    """
    results = segment_image(image.copy(), grid_size)
    if results:
        return results['contour_visualization']
    return image

# For testing the module independently
if __name__ == "__main__":
    # Test with a sample image
    import tkinter as tk
    from tkinter import filedialog
    
    print("=" * 60)
    print("SEGMENTATION MODULE TEST")
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
            print(f"Image size: {img.shape}")
            
            # Apply segmentation with saving
            segmented_img, output_info = apply(img)
            
            if segmented_img is not None:
                # Display results
                cv2.imshow("Original", img)
                cv2.imshow("Segmentation (Contours)", segmented_img)
                
                # Also show grid visualization
                grid_img = cv2.imread(output_info['visualizations']['grid'])
                if grid_img is not None:
                    cv2.imshow("Grid Visualization", grid_img)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                print("\n✅ Segmentation complete!")
                print(f"📁 Check the 'outputs/segmentation' folder for saved files:")
                print(f"   - {output_info['tile_count']} tiles in 'tiles/' folder")
                print(f"   - Edge images in 'edges/' folder")
                print(f"   - Contour images in 'contours/' folder")
                print(f"   - Visualizations in 'visualizations/' folder")
            else:
                print("❌ Segmentation failed")
        else:
            print("❌ Failed to load image")
    else:
        print("❌ No image selected")