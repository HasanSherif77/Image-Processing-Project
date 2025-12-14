"""
Milestone 2: Jigsaw Puzzle Solver with Edge Matching Visualization
Author: [Your Name]
Date: [Current Date]

This module implements a complete jigsaw puzzle solver that:
1. Loads puzzle pieces from Phase 1 output
2. Extracts and analyzes edge features
3. Finds matching edges between pieces
4. Visualizes matches with connecting lines
5. Reconstructs the puzzle using optimized matching
"""

import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.spatial.distance import cdist
import itertools


class JigsawSolver:
    def __init__(self, tiles_dir, grid_size=2, tile_size=128):
        """
        Initialize the puzzle solver

        Args:
            tiles_dir: Directory containing puzzle pieces from Phase 1
            grid_size: Size of the puzzle grid (e.g., 2 for 2x2)
            tile_size: Size of each tile in pixels
        """
        self.tiles_dir = tiles_dir
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.tiles = []
        self.tile_names = []
        self.rotated_tiles = []
        self.edge_features = {}
        self.matches = []
        self.reconstructed_img = None

        # Visualization settings
        self.visualization_dir = "milestone2_visualizations"
        os.makedirs(self.visualization_dir, exist_ok=True)

    def load_tiles(self):
        """Load all puzzle pieces from the directory"""
        print(f"Loading tiles from: {self.tiles_dir}")

        self.tiles = []
        self.tile_names = []

        for fname in sorted(os.listdir(self.tiles_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(self.tiles_dir, fname)
                img = cv2.imread(img_path)

                if img is not None:
                    # Resize to consistent size
                    img = cv2.resize(img, (self.tile_size, self.tile_size))
                    self.tiles.append(img)
                    self.tile_names.append(fname)

                    print(f"  ✓ Loaded: {fname} - {img.shape}")

        if len(self.tiles) == 0:
            raise ValueError(f"No tiles found in {self.tiles_dir}")

        print(f"Total tiles loaded: {len(self.tiles)}")
        return self.tiles

    def generate_rotations(self, tile):
        """Generate 4 rotations of a tile (0°, 90°, 180°, 270°)"""
        rotations = []
        current = tile.copy()

        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = current.copy()
            elif angle == 90:
                rotated = cv2.rotate(current, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(current, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(current, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rotations.append({
                'angle': angle,
                'image': rotated
            })

        return rotations

    def extract_edge_features(self, tile):
        """
        Extract comprehensive edge features from a tile

        Features extracted:
        1. Color histogram (RGB and LAB space)
        2. Gradient magnitude (edge strength)
        3. Texture features using GLCM
        4. Mean and standard deviation of pixel values
        """
        h, w = tile.shape[:2]

        # Define edge regions (take 5 pixels from each edge for robustness)
        edge_width = 5

        # Extract edge pixels
        top_edge = tile[0:edge_width, :, :]
        bottom_edge = tile[h - edge_width:h, :, :]
        left_edge = tile[:, 0:edge_width, :]
        right_edge = tile[:, w - edge_width:w, :]

        features = {}

        for edge_name, edge_pixels in [
            ('top', top_edge),
            ('bottom', bottom_edge),
            ('left', left_edge),
            ('right', right_edge)
        ]:
            # Convert to different color spaces
            edge_rgb = edge_pixels
            edge_gray = cv2.cvtColor(edge_pixels, cv2.COLOR_BGR2GRAY)
            edge_lab = cv2.cvtColor(edge_pixels, cv2.COLOR_BGR2LAB)

            # Feature 1: Color histograms (RGB)
            hist_r = cv2.calcHist([edge_rgb[:, :, 0]], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([edge_rgb[:, :, 1]], [0], None, [16], [0, 256])
            hist_b = cv2.calcHist([edge_rgb[:, :, 2]], [0], None, [16], [0, 256])

            # Feature 2: LAB color space statistics
            lab_mean = np.mean(edge_lab, axis=(0, 1))
            lab_std = np.std(edge_lab, axis=(0, 1))

            # Feature 3: Gradient magnitude (edge strength)
            sobel_x = cv2.Sobel(edge_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(edge_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # Feature 4: Texture features (simplified)
            _, binary_edge = cv2.threshold(edge_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_pixels = np.sum(binary_edge > 127)
            total_pixels = binary_edge.size
            texture_density = white_pixels / total_pixels if total_pixels > 0 else 0

            # Combine all features into a single vector
            edge_features = np.concatenate([
                hist_r.flatten(),
                hist_g.flatten(),
                hist_b.flatten(),
                lab_mean,
                lab_std,
                [np.mean(gradient_mag), np.std(gradient_mag)],
                [texture_density]
            ])

            features[edge_name] = edge_features

        return features

    def compute_edge_similarity(self, features1, features2):
        """
        Compute similarity between two edge feature vectors

        Uses multiple distance metrics and combines them:
        1. Euclidean distance
        2. Cosine similarity
        3. Manhattan distance
        """
        # Normalize features
        f1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
        f2_norm = features2 / (np.linalg.norm(features2) + 1e-8)

        # Euclidean distance
        euclidean_dist = np.linalg.norm(features1 - features2)

        # Cosine similarity (1 - cosine distance)
        cosine_sim = np.dot(f1_norm, f2_norm)
        cosine_dist = 1 - cosine_sim

        # Manhattan distance
        manhattan_dist = np.sum(np.abs(features1 - features2))

        # Weighted combination
        total_dist = 0.4 * euclidean_dist + 0.4 * cosine_dist + 0.2 * manhattan_dist

        return total_dist

    def find_all_matches(self, threshold=0.3):
        """
        Find all potential matches between tile edges

        Args:
            threshold: Maximum distance for a match to be considered

        Returns:
            List of matches with similarity scores
        """
        print("\nFinding edge matches...")

        self.matches = []
        n_tiles = len(self.tiles)

        # Build feature database for all tiles and rotations
        self.tile_features = []

        for tile_idx, tile in enumerate(self.tiles):
            tile_data = {
                'tile_id': tile_idx,
                'name': self.tile_names[tile_idx],
                'original': tile,
                'rotations': [],
                'edge_features': {}
            }

            # Generate rotations
            rotations = self.generate_rotations(tile)

            for rot_idx, rotation in enumerate(rotations):
                rot_data = {
                    'rotation_id': rot_idx,
                    'angle': rotation['angle'],
                    'image': rotation['image'],
                    'features': self.extract_edge_features(rotation['image'])
                }

                tile_data['rotations'].append(rot_data)

                # Store features for quick access
                key = f"t{tile_idx}_r{rot_idx}"
                self.edge_features[key] = rot_data['features']

            self.tile_features.append(tile_data)

        # Find matches between all tile edges
        match_id = 0

        for i in range(n_tiles):
            for j in range(i + 1, n_tiles):  # Avoid comparing tile with itself
                for ri in range(4):  # All rotations
                    for rj in range(4):  # All rotations

                        # Get feature dictionaries
                        features_i = self.edge_features[f"t{i}_r{ri}"]
                        features_j = self.edge_features[f"t{j}_r{rj}"]

                        # Check all edge combinations
                        edge_pairs = [
                            ('right', 'left'),  # i's right matches j's left
                            ('left', 'right'),  # i's left matches j's right
                            ('bottom', 'top'),  # i's bottom matches j's top
                            ('top', 'bottom')  # i's top matches j's bottom
                        ]

                        for edge_i, edge_j in edge_pairs:
                            # Calculate similarity
                            similarity = self.compute_edge_similarity(
                                features_i[edge_i],
                                features_j[edge_j]
                            )

                            # Check if it's a good match
                            if similarity < threshold:
                                match = {
                                    'id': match_id,
                                    'tile1': i,
                                    'tile2': j,
                                    'rotation1': ri,
                                    'rotation2': rj,
                                    'edge1': edge_i,
                                    'edge2': edge_j,
                                    'similarity': similarity,
                                    'angle1': ri * 90,
                                    'angle2': rj * 90
                                }

                                self.matches.append(match)
                                match_id += 1

        print(f"Found {len(self.matches)} potential matches")

        # Sort matches by similarity (best matches first)
        self.matches.sort(key=lambda x: x['similarity'])

        return self.matches

    def visualize_matches(self, top_n=10):
        """
        Visualize the top N matches with connecting lines

        Args:
            top_n: Number of top matches to visualize
        """
        print(f"\nVisualizing top {top_n} matches...")

        if len(self.matches) == 0:
            print("No matches found to visualize")
            return

        # Create a figure with subplots
        n_cols = min(4, top_n)
        n_rows = (top_n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Visualize each match
        for match_idx, match in enumerate(self.matches[:top_n]):
            row = match_idx // n_cols
            col = match_idx % n_cols
            ax = axes[row, col]

            # Get tile images with rotations
            tile1_img = self.tile_features[match['tile1']]['rotations'][match['rotation1']]['image']
            tile2_img = self.tile_features[match['tile2']]['rotations'][match['rotation2']]['image']

            # Convert BGR to RGB for display
            tile1_rgb = cv2.cvtColor(tile1_img, cv2.COLOR_BGR2RGB)
            tile2_rgb = cv2.cvtColor(tile2_img, cv2.COLOR_BGR2RGB)

            # Create a composite image
            composite = np.hstack([tile1_rgb, tile2_rgb])

            # Display the composite
            ax.imshow(composite)
            ax.axis('off')

            # Add title with match information
            title = f"Match {match_idx + 1}\n"
            title += f"T{match['tile1']} {match['edge1']} ↔ T{match['tile2']} {match['edge2']}\n"
            title += f"Sim: {match['similarity']:.3f}"

            ax.set_title(title, fontsize=10)

            # Draw a line connecting the matching edges
            h, w = tile1_rgb.shape[:2]

            # Determine line coordinates based on which edges match
            if match['edge1'] == 'right' and match['edge2'] == 'left':
                # Line from right edge of tile1 to left edge of tile2
                x1, y1 = w, h // 2  # Right middle of tile1
                x2, y2 = w, h // 2  # Left middle of tile2 (in composite coordinates)
            elif match['edge1'] == 'left' and match['edge2'] == 'right':
                x1, y1 = 0, h // 2
                x2, y2 = w, h // 2
            elif match ['edge1'] == 'bottom' and match['edge2'] == 'top':
                x1, y1 = w // 2, h  # bottom middle of tile1
                x2, y2 = w + w // 2, 0  # top middle of tile2 (in composite)
            elif match ['edge1'] == 'top' and match['edge2'] == 'bottom':
                x1, y1 = w // 2, 0
                x2, y2 = w + w // 2, h

            # Adjust coordinates for composite image
            x2 += w  # Move to second tile

            # Draw connection line
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
            ax.plot([x1, x2], [y1, y2], 'yo', markersize=8, alpha=0.7)

        # Hide empty subplots
        for idx in range(top_n, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle(f"Top {min(top_n, len(self.matches))} Edge Matches", fontsize=16, y=1.02)
        plt.tight_layout()

        # Save the visualization
        save_path = os.path.join(self.visualization_dir, "edge_matches.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved match visualization to: {save_path}")

        # Also create a summary visualization
        self.create_match_summary()

        plt.show()

    def create_match_summary(self):
        """Create a summary visualization of all matches"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Similarity distribution
        # After computing all similarities
        similarities = [self.compute_edge_similarity(f1, f2)
                        for f1, f2 in itertools.combinations(self.edge_features.values(), 2)]
        threshold = np.percentile(similarities, 25)  # best 25% matches

        ax1.hist(similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Similarity Score (lower = better)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Match Similarities')
        ax1.grid(True, alpha=0.3)

        # Add vertical line for median
        median_sim = np.median(similarities)
        ax1.axvline(median_sim, color='red', linestyle='--',
                    label=f'Median: {median_sim:.3f}')
        ax1.legend()

        # Plot 2: Match type distribution
        edge_types = {}
        for match in self.matches:
            edge_pair = f"{match['edge1']}-{match['edge2']}"
            edge_types[edge_pair] = edge_types.get(edge_pair, 0) + 1

        edge_pairs = list(edge_types.keys())
        counts = list(edge_types.values())

        colors = plt.cm.Set3(np.linspace(0, 1, len(edge_pairs)))
        bars = ax2.bar(edge_pairs, counts, color=colors, edgecolor='black')
        ax2.set_xlabel('Edge Pair Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Match Types')
        ax2.tick_params(axis='x', rotation=45)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{count}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Edge Matching Analysis Summary', fontsize=16, y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.visualization_dir, "match_summary.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved match summary to: {save_path}")

        plt.show()

    def solve_puzzle(self, method='greedy'):
        """
        Solve the puzzle using the found matches

        Args:
            method: Solving method ('greedy' or 'bruteforce')

        Returns:
            Reconstructed puzzle image
        """
        print(f"\nSolving puzzle using {method} method...")

        if len(self.matches) == 0:
            print("No matches found. Finding matches first...")
            self.find_all_matches()

        if method == 'greedy':
            self.reconstructed_img = self._solve_greedy()
        elif method == 'bruteforce':
            self.reconstructed_img = self._solve_bruteforce()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Save the reconstructed image
        if self.reconstructed_img is not None:
            save_path = os.path.join(self.visualization_dir, "reconstructed_puzzle.png")
            cv2.imwrite(save_path, self.reconstructed_img)
            print(f"✓ Saved reconstructed puzzle to: {save_path}")

        return self.reconstructed_img

    def _solve_greedy(self):
        """Improved greedy solver for 2x2 puzzles with rotation handling"""

        n_tiles = len(self.tiles)
        if n_tiles != 4:
            print(f"Warning: This solver is optimized for 2x2 puzzles. Found {n_tiles} tiles.")

        # Step 1: Precompute all edge similarities
        edge_pairs = ['top', 'bottom', 'left', 'right']
        self.sim_matrix = {}  # key: (tile1, rot1, edge1, tile2, rot2, edge2) -> similarity

        for i in range(n_tiles):
            for j in range(n_tiles):
                if i == j:
                    continue
                for ri in range(4):
                    for rj in range(4):
                        f1 = self.edge_features[f"t{i}_r{ri}"]
                        f2 = self.edge_features[f"t{j}_r{rj}"]
                        for e1, e2 in [('right', 'left'), ('left', 'right'), ('bottom', 'top'), ('top', 'bottom')]:
                            key = (i, ri, e1, j, rj, e2)
                            self.sim_matrix[key] = self.compute_edge_similarity(f1[e1], f2[e2])

        # Step 2: Generate all possible placements with rotations
        best_grid = None
        best_score = float('inf')

        tiles_indices = [0, 1, 2, 3]

        # Try all permutations of tiles
        for perm in itertools.permutations(tiles_indices):
            # Try all rotation combinations
            for rot_combo in itertools.product(range(4), repeat=4):
                grid = [
                    [(perm[0], rot_combo[0]), (perm[1], rot_combo[1])],
                    [(perm[2], rot_combo[2]), (perm[3], rot_combo[3])]
                ]

                # Compute total score
                score = 0
                # Row 0
                score += self.sim_matrix[(grid[0][0][0], grid[0][0][1], 'right',
                                          grid[0][1][0], grid[0][1][1], 'left')]
                # Row 1
                score += self.sim_matrix[(grid[1][0][0], grid[1][0][1], 'right',
                                          grid[1][1][0], grid[1][1][1], 'left')]
                # Column 0
                score += self.sim_matrix[(grid[0][0][0], grid[0][0][1], 'bottom',
                                          grid[1][0][0], grid[1][0][1], 'top')]
                # Column 1
                score += self.sim_matrix[(grid[0][1][0], grid[0][1][1], 'bottom',
                                          grid[1][1][0], grid[1][1][1], 'top')]

                if score < best_score:
                    best_score = score
                    best_grid = grid

        print(f"Best score (lower is better): {best_score:.3f}")
        return self._assemble_from_grid(best_grid)

    def _get_match_score(self, tile1, rot1, edge1, tile2, rot2, edge2):
        """Get match score between two tile edges"""
        for match in self.matches:
            if (match['tile1'] == tile1 and match['rotation1'] == rot1 and
                    match['tile2'] == tile2 and match['rotation2'] == rot2 and
                    match['edge1'] == edge1 and match['edge2'] == edge2):
                return match['similarity']
        # Return large penalty if no match exists
        return 1e3

    def _solve_bruteforce(self):
        """Brute force algorithm for small puzzles (2x2, 3x3)"""
        if self.grid_size > 3:
            print("Brute force too slow for large puzzles. Switching to greedy.")
            return self._solve_greedy()

        n_tiles = len(self.tiles)
        all_tile_indices = list(range(n_tiles))

        best_grid = None
        best_score = float('inf')

        # Generate all permutations
        total_perms = 1
        for i in range(1, n_tiles + 1):
            total_perms *= i

        print(f"Testing {total_perms} permutations...")

        # For each permutation
        for perm_idx, perm in enumerate(itertools.permutations(all_tile_indices)):
            if perm_idx % 1000 == 0:
                print(f"  Tested {perm_idx}/{total_perms} permutations...")

            # Test all rotation combinations for this permutation
            for rot_combo in itertools.product(range(4), repeat=n_tiles):
                # Create grid
                grid = []
                idx = 0
                for r in range(self.grid_size):
                    row = []
                    for c in range(self.grid_size):
                        row.append((perm[idx], rot_combo[idx]))
                        idx += 1
                    grid.append(row)

                # Calculate total score
                score = self._calculate_grid_score(grid)

                if score < best_score:
                    best_score = score
                    best_grid = grid

        print(f"Best score found: {best_score}")
        return self._assemble_from_grid(best_grid)

    def _calculate_grid_score(self, grid):
        """Calculate total compatibility score for a grid"""
        score = 0
        grid_size = len(grid)

        for r in range(grid_size):
            for c in range(grid_size):
                tile1, rot1 = grid[r][c]

                # Check right neighbor
                if c < grid_size - 1:
                    tile2, rot2 = grid[r][c + 1]
                    match_score = self._get_match_score(
                        tile1, rot1, 'right',
                        tile2, rot2, 'left'
                    )
                    if match_score is not None:
                        score += match_score

                # Check bottom neighbor
                if r < grid_size - 1:
                    tile2, rot2 = grid[r + 1][c]
                    match_score = self._get_match_score(
                        tile1, rot1, 'bottom',
                        tile2, rot2, 'top'
                    )
                    if match_score is not None:
                        score += match_score

        return score

    def _assemble_from_grid(self, grid):
        """Assemble final image from grid"""
        rows = []

        for r in range(len(grid)):
            row_images = []
            for c in range(len(grid[r])):
                tile_idx, rot_idx = grid[r][c]
                tile_img = self.tile_features[tile_idx]['rotations'][rot_idx]['image']
                row_images.append(tile_img)

            row = np.hstack(row_images)
            rows.append(row)

        final_img = np.vstack(rows)
        return final_img

    def _arrange_simple(self):
        """Simple linear arrangement (fallback)"""
        rows = []

        for r in range(self.grid_size):
            row_images = []
            for c in range(self.grid_size):
                idx = r * self.grid_size + c
                if idx < len(self.tiles):
                    row_images.append(self.tiles[idx])
                else:
                    # Black tile if missing
                    row_images.append(np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8))

            row = np.hstack(row_images)
            rows.append(row)

        return np.vstack(rows)

    def visualize_reconstruction(self):
        """Visualize the reconstruction process"""
        if self.reconstructed_img is None:
            print("No reconstructed image available. Run solve_puzzle() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Original tiles
        axes[0, 0].set_title("Original Puzzle Pieces", fontsize=12, weight='bold')
        tile_display = self._create_tile_grid(self.tiles, max_tiles=min(16, len(self.tiles)))
        axes[0, 0].imshow(cv2.cvtColor(tile_display, cv2.COLOR_BGR2RGB))
        axes[0, 0].axis('off')
        axes[0, 0].text(0.5, -0.1, f"{len(self.tiles)} pieces loaded",
                        transform=axes[0, 0].transAxes, ha='center')

        # 2. Best matches
        axes[0, 1].set_title("Best Edge Matches", fontsize=12, weight='bold')
        if len(self.matches) > 0:
            # Create a montage of best matches
            best_matches_display = self._create_match_montage(num_matches=4)
            axes[0, 1].imshow(best_matches_display)
        else:
            axes[0, 1].text(0.5, 0.5, "No matches found",
                            transform=axes[0, 1].transAxes, ha='center', va='center')
        axes[0, 1].axis('off')

        # 3. Reconstructed puzzle
        axes[1, 0].set_title("Reconstructed Puzzle", fontsize=12, weight='bold')
        axes[1, 0].imshow(cv2.cvtColor(self.reconstructed_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].axis('off')

        # 4. Match statistics
        axes[1, 1].set_title("Matching Statistics", fontsize=12, weight='bold')
        if len(self.matches) > 0:
            stats_text = f"Total Matches: {len(self.matches)}\n"
            stats_text += f"Best Similarity: {self.matches[0]['similarity']:.3f}\n"
            stats_text += f"Worst Similarity: {self.matches[-1]['similarity']:.3f}\n"
            stats_text += f"Median Similarity: {np.median([m['similarity'] for m in self.matches]):.3f}\n\n"

            # Count matches by type
            edge_counts = {}
            for match in self.matches:
                key = f"{match['edge1']}-{match['edge2']}"
                edge_counts[key] = edge_counts.get(key, 0) + 1

            stats_text += "Matches by type:\n"
            for edge_type, count in edge_counts.items():
                stats_text += f"  {edge_type}: {count}\n"
        else:
            stats_text = "No matches found"

        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top')
        axes[1, 1].axis('off')

        plt.suptitle("Jigsaw Puzzle Reconstruction Pipeline", fontsize=16, weight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(self.visualization_dir, "reconstruction_pipeline.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved reconstruction pipeline visualization to: {save_path}")

        plt.show()

    def _create_tile_grid(self, tiles, max_tiles=16):
        """Create a grid display of tiles"""
        n_tiles = min(len(tiles), max_tiles)
        n_cols = int(np.ceil(np.sqrt(n_tiles)))
        n_rows = int(np.ceil(n_tiles / n_cols))

        tile_h, tile_w = tiles[0].shape[:2]
        grid_h = n_rows * tile_h
        grid_w = n_cols * tile_w

        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i in range(n_tiles):
            row = i // n_cols
            col = i % n_cols
            y_start = row * tile_h
            x_start = col * tile_w
            grid[y_start:y_start + tile_h, x_start:x_start + tile_w] = tiles[i]

        return grid

    def _create_match_montage(self, num_matches=4):
        """Create a montage of the best matches"""
        num_matches = min(num_matches, len(self.matches))

        montage_width = 2 * self.tile_size
        montage_height = num_matches * self.tile_size

        montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

        for i in range(num_matches):
            match = self.matches[i]

            # Get tile images
            tile1 = self.tile_features[match['tile1']]['rotations'][match['rotation1']]['image']
            tile2 = self.tile_features[match['tile2']]['rotations'][match['rotation2']]['image']

            # Place in montage
            y_start = i * self.tile_size
            montage[y_start:y_start + self.tile_size, 0:self.tile_size] = tile1
            montage[y_start:y_start + self.tile_size, self.tile_size:2 * self.tile_size] = tile2

            # Add similarity label
            cv2.putText(montage, f"{match['similarity']:.3f}",
                        (10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        return montage

    def save_results(self, output_dir="milestone2_results"):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save reconstructed image
        if self.reconstructed_img is not None:
            recon_path = os.path.join(output_dir, "reconstructed_puzzle.jpg")
            cv2.imwrite(recon_path, self.reconstructed_img)
            print(f"✓ Saved reconstructed puzzle to: {recon_path}")

        # Save match data
        if len(self.matches) > 0:
            matches_data = []
            for match in self.matches:
                matches_data.append({
                    'tile1': int(match['tile1']),
                    'tile2': int(match['tile2']),
                    'rotation1': int(match['rotation1']),
                    'rotation2': int(match['rotation2']),
                    'edge1': match['edge1'],
                    'edge2': match['edge2'],
                    'similarity': float(match['similarity'])
                })

            matches_path = os.path.join(output_dir, "matches.json")
            with open(matches_path, 'w') as f:
                json.dump(matches_data, f, indent=2)
            print(f"✓ Saved match data to: {matches_path}")

        # Save configuration
        config = {
            'tiles_dir': self.tiles_dir,
            'grid_size': self.grid_size,
            'tile_size': self.tile_size,
            'num_tiles': len(self.tiles),
            'num_matches': len(self.matches)
        }

        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Saved configuration to: {config_path}")
        print(f"✓ All results saved in: {output_dir}")


# -------------------------------------------------
# Demonstration function
# -------------------------------------------------
def demonstrate_solver():
    """Demonstrate the solver on sample puzzles"""
    print("=" * 70)
    print("MILESTONE 2: JIGSAW PUZZLE SOLVER DEMONSTRATION")
    print("=" * 70)

    # Example 1: Clean 2x2 puzzle
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Clean 2x2 Puzzle")
    print("=" * 70)

    # Update this path to your actual tiles directory
    tiles_dir_clean = "C:/Users/mmmsa/PycharmProjects/Image-Processing-Project/Milestone_1/output/tiles"

    if os.path.exists(tiles_dir_clean):
        # Create solver instance
        solver_clean = JigsawSolver(tiles_dir_clean, grid_size=2, tile_size=128)

        # Load tiles
        solver_clean.load_tiles()

        # Find matches
        solver_clean.find_all_matches(threshold=0.5)

        # Visualize matches
        solver_clean.visualize_matches(top_n=8)

        # Solve puzzle
        solver_clean.solve_puzzle(method='greedy')

        # Visualize reconstruction
        solver_clean.visualize_reconstruction()

        # Save results
        solver_clean.save_results("results_clean_2x2")
    else:
        print(f"Tiles directory not found: {tiles_dir_clean}")
        print("Please update the path to your actual tiles directory")

    # Example 2: Noisy/rotated puzzle (simulated)
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Challenging Case (Simulated)")
    print("=" * 70)

    # Create a simulated challenging case
    simulate_challenging_case()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


def simulate_challenging_case():
    """Simulate a challenging case with noise and rotations"""
    print("Simulating a challenging puzzle case...")

    # Create a simple test pattern
    test_tiles = []
    tile_size = 128

    for i in range(4):
        # Create a tile with gradient
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

        # Different colors for different tiles
        if i == 0:  # Top-left
            color = (255, 100, 100)  # Reddish
        elif i == 1:  # Top-right
            color = (100, 255, 100)  # Greenish
        elif i == 2:  # Bottom-left
            color = (100, 100, 255)  # Blueish
        else:  # Bottom-right
            color = (255, 255, 100)  # Yellowish

        # Create gradient
        for y in range(tile_size):
            for x in range(tile_size):
                intensity = (x + y) / (2 * tile_size)
                tile[y, x] = [int(c * intensity) for c in color]

        # Add some noise
        noise = np.random.normal(0, 30, tile.shape).astype(np.uint8)
        tile = cv2.add(tile, noise)

        # Rotate some tiles
        if i % 2 == 0:
            tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)

        test_tiles.append(tile)

    # Save test tiles to a temporary directory
    temp_dir = "temp_challenging_tiles"
    os.makedirs(temp_dir, exist_ok=True)

    for i, tile in enumerate(test_tiles):
        cv2.imwrite(os.path.join(temp_dir, f"tile_{i}.png"), tile)

    # Create solver for challenging case
    solver_challenging = JigsawSolver(temp_dir, grid_size=2, tile_size=128)

    # Load tiles
    solver_challenging.load_tiles()

    # Find matches with higher threshold (more permissive)
    solver_challenging.find_all_matches(threshold=0.7)

    # Visualize matches
    solver_challenging.visualize_matches(top_n=6)

    # Solve puzzle
    solver_challenging.solve_puzzle(method='greedy')

    # Visualize reconstruction
    solver_challenging.visualize_reconstruction()

    # Save results
    solver_challenging.save_results("results_challenging")

    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)

    print("Challenging case simulation complete!")


# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == "__main__":
    # Run demonstration
    demonstrate_solver()