# Image Processing Project: Jigsaw Puzzle Edge Matching

Automatically analyze jigsaw puzzle pieces from images, segment individual pieces, and suggest possible matches between puzzle edges based on their contour shapes using classical computer vision techniques.

## Project Overview

This project demonstrates practical computer vision applications through a two-milestone approach to jigsaw puzzle analysis:

### 🎯 Milestone 1: Image Preprocessing & Segmentation
**Objective**: Prepare jigsaw puzzle images for analysis by applying robust preprocessing techniques.

**Key Features**:
- **Noise Reduction**: Adaptive filtering for salt & pepper and Gaussian noise
- **Edge Enhancement**: Sharpening filters to improve puzzle piece boundaries
- **Brightness/Contrast Adjustment**: Gamma correction and CLAHE for optimal visibility
- **Grid-based Segmentation**: Automatic division into individual puzzle pieces
- **Contour Extraction**: Precise boundary detection for each piece

### 🧩 Milestone 2: Edge Matching & Assembly Suggestions
**Objective**: Analyze processed puzzle pieces and suggest potential matches using shape descriptors.

**Key Features**:
- **Shape Descriptors**: Fourier descriptors and Hu moments for rotation-invariant shape representation
- **Edge Classification**: Automatic categorization of straight vs. curved edges
- **Multi-feature Matching**: Weighted combination of multiple similarity metrics
- **Match Visualization**: Side-by-side comparison of potential matching pieces
- **Assembly Suggestions**: Ranked list of neighboring piece candidates

## Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy
- Matplotlib
- Tkinter (for GUI)
- SciPy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd image-processing-project

# Install dependencies
pip install opencv-python numpy matplotlib scipy
```

## Project Structure

```
image-processing-project/
├── jigsaw_pipeline.py      # Milestone 1: Image preprocessing pipeline
├── jigsaw_gui.py           # GUI interface for Milestone 1
├── jigsaw_matcher.py       # Milestone 2: Edge matching system
├── milestone2_demo.py      # Comprehensive demo script
├── README.md              # This documentation
└── output/                # Generated outputs (created automatically)
    ├── tiles/             # Individual puzzle pieces
    ├── edges/             # Edge detection results
    ├── contours/          # Contour visualizations
    ├── final_image/       # Processed images
    └── match_visualizations/  # Milestone 2 results
```

## Milestone 1: Image Preprocessing Implementation

### Step-by-Step Implementation Guide

1. **Load and Analyze Image**:
```python
import cv2
import numpy as np

# Load puzzle image
image = cv2.imread('puzzle_image.jpg')

# Analyze for noise and quality issues
from jigsaw_pipeline import detect_salt_noise, detect_pepper_noise_median, detect_blur

has_salt, salt_ratio = detect_salt_noise(image)
has_pepper, pepper_ratio = detect_pepper_noise_median(image)
is_blurry = detect_blur(image)
```

2. **Apply Optimal Processing Order**:
```python
# 1. Noise reduction (first)
processed = enhance_image(image)  # Adaptive noise reduction

# 2. Brightness/Contrast adjustment (second)
processed = apply_gamma_correction(processed)  # or apply_clahe()

# 3. Edge sharpening (third)
processed = edge_sharpening(processed)

# 4. Grid segmentation and contour extraction (final)
# This creates individual tiles and extracts contours
```

3. **Grid Segmentation**:
```python
# Define grid size (2x2, 4x4, or 8x8)
grid_size = 4
h, w = processed.shape[:2]
tile_h, tile_w = h // grid_size, w // grid_size

# Extract individual pieces
for row in range(grid_size):
    for col in range(grid_size):
        tile = processed[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w]
        # Save tile and extract contours
```

### Milestone 1 Outputs

After running Milestone 1, you'll have:
- `output/tiles/tile_0.png` through `tile_N.png`: Individual puzzle pieces
- `output/edges/edges_0.png` through `edges_N.png`: Edge detection results
- `output/contours/contour_0.png` through `contour_N.png`: Contour visualizations
- `output/final_image/`: Processed images with contour overlays

## Milestone 2: Edge Matching Implementation

### Core Concept: Shape Descriptors

Milestone 2 uses **rotation-invariant shape descriptors** to compare puzzle edges:

1. **Fourier Descriptors**: Capture shape frequency content
2. **Hu Moments**: 7 invariant moments for global shape characteristics
3. **Curvature Features**: Local shape variations (straight vs. curved)

### Implementation Steps

#### 1. Shape Feature Extraction

```python
from jigsaw_matcher import JigsawMatcher

# Initialize matcher with Milestone 1 outputs
matcher = JigsawMatcher(output_dir="output")

# Process all puzzle pieces
num_pieces = matcher.load_and_process_all_pieces()

# Extract features for each piece
for piece_id, features in matcher.edge_features.items():
    print(f"Piece {piece_id}: {features['edge_type']} edge")
    print(f"  - Fourier descriptors: {len(features['fourier_descriptors'])}")
    print(f"  - Hu moments: {len(features['hu_moments'])}")
    print(f"  - Curvature features: {len(features['curvature_features'])}")
```

#### 2. Edge Classification

```python
# Classify edges as straight or curved
edge_type = matcher.classify_edge_type(contour)
# Returns: 'straight' or 'curved'
```

#### 3. Shape Matching Algorithm

```python
# Find best matches for a target piece
target_piece = 0
matches = matcher.find_best_matches(target_piece, num_matches=5)

for rank, (match_id, distance, details) in enumerate(matches, 1):
    print(f"Rank {rank}: Piece {match_id}, Distance: {distance:.4f}")
```

#### 4. Multi-Feature Distance Calculation

```python
# Calculate similarity using weighted combination of features
distance, details = matcher.calculate_feature_distance(features1, features2)

# Weights can be adjusted based on experimental results
weights = {
    'fourier': 0.4,     # Shape frequency content
    'hu_moments': 0.3,  # Global shape characteristics
    'curvature': 0.2,   # Local shape variations
    'geometric': 0.1    # Size and compactness
}
```

#### 5. Match Visualization

```python
# Create visual comparison of matching pieces
visualization = matcher.create_match_visualization(piece1_id, piece2_id)
cv2.imshow('Match Visualization', visualization)
```

### Running Milestone 2

#### Method 1: Using the Demo Script

```bash
# Run interactive demo
python milestone2_demo.py

# Specify custom output directory and grid size
python milestone2_demo.py -d output -g 4

# Include detailed comparison report
python milestone2_demo.py --compare
```

#### Method 2: Programmatic Usage

```python
from jigsaw_matcher import run_milestone2_pipeline

# Run complete pipeline
matcher, suggestions = run_milestone2_pipeline(output_dir="output", grid_size=4)

# Access results
for piece_id, data in suggestions.items():
    print(f"Piece {piece_id} ({data['edge_type']}): {len(data['matches'])} potential matches")
```

### Understanding the Results

#### Edge Type Analysis
```
Edge Type Analysis:
Total pieces: 16
- straight: 8 pieces (50.0%)
- curved: 8 pieces (50.0%)
```

#### Match Quality Metrics
Each match includes:
- **Overall Distance**: Combined similarity score (lower = better match)
- **Feature Breakdown**: Individual contributions from each descriptor
- **Edge Type Compatibility**: Whether edges are similar types

#### Visualization Output
Match visualizations show:
- Original puzzle pieces side-by-side
- Contour overlays in different colors
- Match quality score
- Edge type information

## Technical Details

### Shape Descriptors Used

#### Fourier Descriptors
- **Purpose**: Capture shape in frequency domain
- **Properties**: Rotation, scale, translation invariant (after normalization)
- **Usage**: Excellent for overall shape similarity

#### Hu Moments
- **Purpose**: Global shape characteristics
- **Properties**: 7 invariant moments
- **Usage**: Complements Fourier descriptors for detailed shape analysis

#### Curvature Features
- **Purpose**: Local shape variations
- **Properties**: Measures how much edges deviate from straight lines
- **Usage**: Distinguishes straight edges from curved edges

### Distance Metrics

The system uses multiple distance metrics combined with weights:

```python
# Weighted combination for robust matching
total_distance = (
    0.4 * fourier_distance +      # Shape frequency similarity
    0.3 * hu_moments_distance +   # Global shape similarity
    0.2 * curvature_distance +    # Local shape similarity
    0.1 * geometric_distance      # Size similarity
)
```

### Edge Classification Thresholds

```python
# Curvature thresholds for edge classification
CURVATURE_THRESHOLD = 0.1

if mean_curvature < threshold and std_curvature < threshold:
    edge_type = 'straight'
else:
    edge_type = 'curved'
```

## Output Files

### Milestone 1 Outputs
- `output/tiles/tile_*.png`: Individual puzzle pieces
- `output/edges/edges_*.png`: Edge detection results
- `output/contours/contour_*.png`: Contour visualizations
- `output/final_image/*.jpg`: Processed images

### Milestone 2 Outputs
- `output/edge_features.json`: Extracted shape features
- `output/match_visualizations/*.png`: Match comparison images
- Console output with detailed analysis

## Demo and Examples

### Quick Start Demo

```bash
# 1. Run Milestone 1 (if not done already)
python jigsaw_gui.py  # Use GUI to process image

# 2. Run Milestone 2 demo
python milestone2_demo.py

# 3. View results
# - Check output/match_visualizations/ for visual results
# - Review output/edge_features.json for detailed data
```

### Advanced Usage

```python
# Load and analyze specific pieces
matcher = JigsawMatcher("output")
matcher.load_and_process_all_pieces()

# Compare two specific pieces
distance, details = matcher.calculate_feature_distance(
    matcher.edge_features[0],
    matcher.edge_features[1]
)

# Create custom visualization
viz = matcher.create_match_visualization(0, 1, "custom_match.png")
```

## Performance Considerations

### Computational Complexity
- **Feature Extraction**: O(N) per piece, where N is contour length
- **Matching**: O(M × P) for M pieces and P features
- **Visualization**: O(1) per match pair

### Memory Usage
- Stores contour data for all pieces
- Feature vectors are compact (typically < 100 floats per piece)

### Optimization Tips
- Process pieces in batches for large puzzles
- Cache distance calculations for repeated comparisons
- Use edge type filtering to reduce comparison space

## Troubleshooting

### Common Issues

1. **"No pieces processed"**
   - Ensure Milestone 1 completed successfully
   - Check that output/tiles/ contains PNG files

2. **Poor matching quality**
   - Verify Milestone 1 preprocessing was effective
   - Try adjusting feature weights in calculate_feature_distance()
   - Check edge detection quality in Milestone 1

3. **Memory errors**
   - Reduce grid size for large images
   - Process pieces in smaller batches

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Academic Context

This implementation demonstrates:
- **Classical Computer Vision**: No machine learning or deep learning used
- **Shape Analysis**: Fourier descriptors, moment invariants
- **Feature Engineering**: Multi-modal feature combination
- **Distance Metrics**: Euclidean, cosine similarity
- **Rotation Invariance**: Proper normalization techniques

## Future Enhancements

Potential improvements for advanced implementations:
- **Graph-based Assembly**: Use matching results to construct assembly graph
- **Geometric Constraints**: Incorporate piece position relationships
- **Advanced Descriptors**: Zernike moments, shape context
- **Similarity Learning**: Optimize feature weights automatically

## References

### Shape Descriptors
- [Fourier Descriptors for Shape Analysis](https://en.wikipedia.org/wiki/Fourier_descriptor)
- [Hu Moment Invariants](https://en.wikipedia.org/wiki/Image_moment)
- [Curvature Scale Space](https://en.wikipedia.org/wiki/Curvature_scale_space)

### Computer Vision Libraries
- [OpenCV Documentation](https://docs.opencv.org/)
- [SciPy Spatial Distance](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

---

**Note**: This implementation strictly follows the project requirements of using only classical computer vision techniques without machine learning or deep learning methods.
