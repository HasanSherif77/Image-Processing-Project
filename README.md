# Jigsaw Puzzle Solver - Image Processing Project

A comprehensive system for automatically analyzing jigsaw puzzle images, segmenting individual pieces, extracting edges and contours, and solving puzzles using computer vision and optimization algorithms.

## Overview

This project is organized into two main milestones:

- **Milestone 1**: Image processing pipeline with noise detection, enhancement, grid segmentation, and contour extraction
- **Milestone 2**: Puzzle solving using beam search algorithm with batch processing capabilities

## Features

### Image Processing (Milestone 1)
- **Noise Detection**: Automatic detection of salt & pepper noise, Gaussian noise, and blur
- **Image Enhancement**: 
  - Noise reduction (median/bilateral filtering)
  - Edge sharpening
  - Gamma correction for brightness adjustment
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Grid Segmentation**: Automatic extraction of puzzle tiles from images (supports 2x2, 4x4, 8x8 grids)
- **Edge & Contour Extraction**: Canny edge detection and contour extraction for each tile
- **GUI Interface**: Interactive graphical interface for image processing

### Puzzle Solving (Milestone 2)
- **Beam Search Algorithm**: Efficient puzzle reconstruction using beam search
- **Edge Matching**: Compares tile edges (top, bottom, left, right) using L1 distance
- **Rotation Handling**: Tests all 4 rotations (0°, 90°, 180°, 270°) for optimal tile placement
- **Batch Processing**: Process entire folders of puzzles automatically
- **Cost Analysis**: Computes compatibility costs between tile pairs

## Project Structure

```
Image-Processing-Project/
├── Milestone_1/
│   ├── jigsaw_pipeline.py    # Main image processing pipeline
│   └── jigsaw_gui.py          # GUI application for image processing
├── Milestone_2/
│   ├── Jigsaw_solver.py       # Beam search puzzle solver class
│   ├── jigsaw_gui.py          # GUI with solver integration
│   ├── batch_process_puzzles.py  # Batch processing script
│   └── batched_gui.py         # Batch processing GUI
├── Gravity_Falls/            # Test puzzle datasets
│   ├── puzzle_2x2/           # 2x2 puzzle images
│   ├── puzzle_4x4/           # 4x4 puzzle images
│   ├── puzzle_8x8/           # 8x8 puzzle images
│   └── correct/              # Correctly solved reference images
├── output/                    # Output directory
│   ├── tiles/                # Extracted puzzle tiles
│   ├── edges/                # Edge detection results
│   ├── contours/             # Contour extraction results
│   ├── final_image/          # Processed images
│   └── batch_solved/         # Batch processing results
└── README.md                 # This file
```

## Installation

### Requirements
```bash
pip install opencv-python numpy matplotlib pillow
```

### Dependencies
- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow (PIL)
- Tkinter (usually included with Python)

## Usage

### Milestone 1: Image Processing

#### GUI Application
```bash
cd Milestone_1
python jigsaw_gui.py
```

Features:
- Load puzzle images
- Select enhancement options
- Choose puzzle type (2x2, 4x4, 8x8)
- Process images with real-time feedback
- View original vs processed comparison

#### Command Line Pipeline
```python
from Milestone_1.jigsaw_pipeline import run_pipeline

final_img, contour_img, best_result = run_pipeline(
    image_path="path/to/puzzle.jpg",
    output_dir="output",
    grid_size=4,
    apply_noise_reduction=True,
    apply_sharpening=True,
    apply_gamma_correction_option=True,
    apply_clahe_option=True
)
```

### Milestone 2: Puzzle Solving

#### Standalone Solver
```python
from Milestone_2.Jigsaw_solver import JigsawSolver

solver = JigsawSolver(
    tiles_dir="output/tiles",
    grid_size=4,
    R=32,              # Downsample size for edge matching
    beam_width=20      # Beam search width
)

solver.load_tiles()
solver.solve()
solver.save_result("output/solved_puzzle.png")
```

#### Batch Processing
```bash
cd Milestone_2
python batch_process_puzzles.py "path/to/puzzle/folder" --grid-size 2
```

Options:
- `--grid-size`: Puzzle grid size (2, 4, or 8) - default: 2
- `--output-dir`: Output directory - default: `output/batch_solved`
- `--beam-width`: Beam search width - default: 80
- `--downsample-size`: Edge matching resolution - default: 48

#### GUI with Solver
```bash
cd Milestone_2
python jigsaw_gui.py
```

## Algorithms

### Image Enhancement Pipeline
1. **Noise Detection**: Analyzes salt/pepper noise, Gaussian noise, and blur levels
2. **Adaptive Filtering**: Applies median/bilateral filters based on noise type
3. **Contrast Enhancement**: Selects optimal brightness/contrast method (Gamma or CLAHE)
4. **Edge Sharpening**: Enhances edges for better contour detection

### Puzzle Solving Algorithm
1. **Edge Extraction**: Extracts top, bottom, left, right edges from each tile
2. **Downsampling**: Reduces edge resolution for efficient matching
3. **Cost Computation**: Calculates L1 distance between edge pairs
4. **Beam Search**: Explores best tile placements using beam search
5. **Rotation Testing**: Tests all 4 rotations for each tile placement

## Output Structure

### Image Processing Output
```
output/
├── tiles/
│   ├── tile_0.png
│   ├── tile_1.png
│   └── ...
├── edges/
│   ├── edges_0.png
│   └── ...
├── contours/
│   ├── contour_0.png
│   └── ...
└── final_image/
    ├── PROCESSED_PUZZLE_IMAGE.jpg
    └── PROCESSED_WITH_CONTOURS.jpg
```

### Solver Output
```
output/
└── solved_puzzle_fullres.png
```

### Batch Processing Output
```
output/batch_solved/
├── 0_solved.png
├── 1_solved.png
└── ...
```

## Configuration

### Solver Parameters
- **R (Downsample Size)**: Edge matching resolution (default: 32-48)
  - Higher = more accurate but slower
  - Lower = faster but less accurate
- **BEAM_WIDTH**: Number of states kept in beam search (default: 20-80)
  - Higher = better solutions but slower
  - Lower = faster but may miss optimal solutions

## Examples

### Complete Workflow
1. Process image and extract tiles:
```python
from Milestone_1.jigsaw_pipeline import run_pipeline
run_pipeline("puzzle.jpg", grid_size=4)
```

2. Solve the puzzle:
```python
from Milestone_2.Jigsaw_solver import JigsawSolver
solver = JigsawSolver("output/tiles", grid_size=4)
solver.load_tiles()
solver.solve()
solver.save_result("output/solved.png")
```

## License

This project is for educational and research purposes.
