"""
Batch Process Puzzles Script

This script processes all images in a folder of 2x2 puzzles:
1. Extracts tiles using grid segmentation from jigsaw_pipeline.py
2. Solves each puzzle using the solver from simulated_annealing_solver.py
3. Saves solved puzzles to output directory

Usage:
    python batch_process_puzzles.py <puzzle_folder_path>

Example:
    python batch_process_puzzles.py "D:/Image-Processing-Project/Gravity_Falls/correct"
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path


# =========================
# TILE EXTRACTION (from jigsaw_pipeline.py)
# =========================
def extract_tiles_from_image(image_path, grid_size=2, output_dir=None):
    """
    Extract tiles from an image using grid segmentation.

    Args:
        image_path: Path to the puzzle image
        grid_size: Size of the grid (2 for 2x2, 4 for 4x4, etc.)
        output_dir: Directory to save tiles. If None, uses a temp directory.

    Returns:
        tuple: (tiles_list, tiles_directory_path)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    h, w, _ = img.shape

    # Calculate tile dimensions
    tile_h = h // grid_size
    tile_w = w // grid_size

    # Create output directory for tiles
    if output_dir is None:
        # Create a temporary directory based on image name
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join("output", "batch_tiles", image_name)

    os.makedirs(output_dir, exist_ok=True)

    # Extract tiles
    tiles = []
    tile_count = 0

    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * tile_h, (row + 1) * tile_h
            x1, x2 = col * tile_w, (col + 1) * tile_w

            tile = img[y1:y2, x1:x2]
            tiles.append(tile)

            # Save tile
            tile_filename = f"tile_{tile_count}.png"
            cv2.imwrite(os.path.join(output_dir, tile_filename), tile)

            tile_count += 1

    print(f"  Extracted {tile_count} tiles from {os.path.basename(image_path)}")
    return tiles, output_dir


# =========================
# SOLVER FUNCTIONS (from simulated_annealing_solver.py)
# =========================
def extract_edges(tile):
    """Extract top, bottom, left, right edges from a tile."""
    h, w, c = tile.shape
    top = tile[0, :, :]
    bottom = tile[h - 1, :, :]
    left = tile[:, 0, :]
    right = tile[:, w - 1, :]
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def compute_edge_loss(edge1, edge2):
    """Compute L1 loss between two edges."""
    return np.mean(np.abs(edge1.astype(int) - edge2.astype(int)))


def rotate_tile(tile, angle):
    """Rotate a tile by the given angle."""
    if angle == 0:
        return tile
    elif angle == 90:
        return cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(tile, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270")


def solve_puzzle(tiles_dir, grid_size=2, R=48, beam_width=80):
    """
    Solve a puzzle using beam search algorithm.

    Args:
        tiles_dir: Directory containing tile images (tile_0.png, tile_1.png, ...)
        grid_size: Size of the grid (2 for 2x2)
        R: Downsample size for edge matching
        beam_width: Beam width for beam search

    Returns:
        tuple: (solved_image, final_cost)
    """
    # Load original tiles
    tiles_original = []
    files = sorted([f for f in os.listdir(tiles_dir) if f.startswith("tile_") and f.endswith(".png")])
    for f in files:
        path = os.path.join(tiles_dir, f)
        img = cv2.imread(path)
        if img is not None:
            tiles_original.append(img)

    if len(tiles_original) != grid_size * grid_size:
        raise ValueError(f"Expected {grid_size * grid_size} tiles, found {len(tiles_original)}")

    N = len(tiles_original)

    # Downsample tiles for edge matching
    downsampled_tiles = []
    for tile in tiles_original:
        resized = cv2.resize(tile, (R, R), interpolation=cv2.INTER_AREA)
        downsampled_tiles.append(resized)

    # Precompute edges for all rotations
    tile_edges_rotations = {}
    for idx, tile in enumerate(downsampled_tiles):
        tile_edges_rotations[idx] = {}
        for angle in [0, 90, 180, 270]:
            tile_edges_rotations[idx][angle] = extract_edges(rotate_tile(tile, angle))

    def edge_pair_cost(tile1_idx, angle1, edge1_name, tile2_idx, angle2, edge2_name):
        e1 = tile_edges_rotations[tile1_idx][angle1][edge1_name]
        e2 = tile_edges_rotations[tile2_idx][angle2][edge2_name]
        return compute_edge_loss(e1, e2)

    # Beam search
    class State:
        def __init__(self, grid, used, cost):
            self.grid = grid
            self.used = used
            self.cost = cost

        def __lt__(self, other):
            return self.cost < other.cost

    beam = [State([], set(), 0.0)]

    for pos in range(grid_size * grid_size):
        r = pos // grid_size
        c = pos % grid_size
        new_beam = []

        for state in beam:
            for t in range(N):
                if t in state.used:
                    continue
                for rot in [0, 90, 180, 270]:
                    cost = 0.0
                    if c > 0:
                        left_idx, left_rot = state.grid[-1]
                        cost += edge_pair_cost(left_idx, left_rot, 'right', t, rot, 'left')
                    if r > 0:
                        top_idx, top_rot = state.grid[(r - 1) * grid_size + c]
                        cost += edge_pair_cost(top_idx, top_rot, 'bottom', t, rot, 'top')
                    new_grid = state.grid + [(t, rot)]
                    new_used = state.used | {t}
                    new_beam.append(State(new_grid, new_used, state.cost + cost))

        new_beam.sort(key=lambda s: s.cost)
        beam = new_beam[:beam_width]
        print(f"    Position {pos + 1}/{grid_size * grid_size} — Beam size: {len(beam)}")

    # Get best state
    best_state = min(beam, key=lambda s: s.cost)
    print(f"    Final cost: {best_state.cost:.4f}")

    # Reconstruct final image using original tiles
    tile_h, tile_w, _ = tiles_original[0].shape
    final_image = np.zeros((grid_size * tile_h, grid_size * tile_w, 3), dtype=np.uint8)

    for idx, (tile_idx, rot) in enumerate(best_state.grid):
        r = idx // grid_size
        c = idx % grid_size
        final_image[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = rotate_tile(tiles_original[tile_idx],
                                                                                            rot)

    return final_image, best_state.cost


# =========================
# MAIN BATCH PROCESSING
# =========================
def process_puzzle_folder(folder_path, grid_size=2, output_base_dir="output/batch_solved",
                          beam_width=80, downsample_size=48):
    """
    Process all puzzle images in a folder.

    Args:
        folder_path: Path to folder containing puzzle images
        grid_size: Grid size (default 2 for 2x2 puzzles)
        output_base_dir: Base directory for output
        beam_width: Beam width for solver
        downsample_size: Downsample size R for edge matching
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in folder_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"BATCH PROCESSING PUZZLES")
    print(f"{'=' * 60}")
    print(f"Folder: {folder_path}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Found {len(image_files)} image(s)")
    print(f"{'=' * 60}\n")

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Process each image
    successful = 0
    failed = 0

    for idx, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 60)

        try:
            # Step 1: Extract tiles
            print("Step 1: Extracting tiles...")
            tiles, tiles_dir = extract_tiles_from_image(
                str(image_path),
                grid_size=grid_size,
                output_dir=os.path.join(output_base_dir, "tiles", image_path.stem)
            )

            # Step 2: Solve puzzle
            print("Step 2: Solving puzzle...")
            solved_image, final_cost = solve_puzzle(
                tiles_dir,
                grid_size=grid_size,
                R=downsample_size,
                beam_width=beam_width
            )

            # Step 3: Save solved puzzle
            print("Step 3: Saving solved puzzle...")
            output_filename = f"{image_path.stem}_solved.png"
            output_path = os.path.join(output_base_dir, output_filename)
            cv2.imwrite(output_path, solved_image)

            print(f"✓ Successfully processed {image_path.name}")
            print(f"  Solved puzzle saved to: {output_path}")
            print(f"  Final cost: {final_cost:.4f}")

            successful += 1

        except Exception as e:
            print(f"✗ Failed to process {image_path.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successful: {successful}/{len(image_files)}")
    print(f"Failed: {failed}/{len(image_files)}")
    print(f"Output directory: {output_base_dir}")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Batch process 2x2 puzzles: extract tiles and solve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_puzzles.py "D:/Image-Processing-Project/Gravity_Falls/correct"
  python batch_process_puzzles.py "puzzles/2x2" --grid-size 2
  python batch_process_puzzles.py "puzzles" --output-dir "my_results"
        """
    )

    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to folder containing puzzle images"
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=2,
        help="Grid size (default: 2 for 2x2 puzzles)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/batch_solved",
        help="Output directory for solved puzzles (default: output/batch_solved)"
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=80,
        help="Beam width for solver (default: 80)"
    )

    parser.add_argument(
        "--downsample-size",
        type=int,
        default=48,
        help="Downsample size R for edge matching (default: 48)"
    )

    args = parser.parse_args()

    # Process the folder
    try:
        process_puzzle_folder(
            args.folder_path,
            grid_size=args.grid_size,
            output_base_dir=args.output_dir,
            beam_width=args.beam_width,
            downsample_size=args.downsample_size
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()