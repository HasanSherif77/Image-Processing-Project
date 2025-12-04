#!/usr/bin/env python3
"""
Jigsaw Puzzle Edge Matching — Milestone 2
Computer Vision Course Project

This module implements classical computer vision techniques for matching jigsaw puzzle edges.
It uses rotation-invariant shape descriptors and contour comparison to suggest piece matches.

Key Features:
- Fourier descriptors for shape representation
- Hu moments for global shape characteristics
- Curvature analysis for edge classification
- Multi-feature distance metrics for matching
- Visualization of potential matches

Author: Computer Vision Course Project Team
"""

import cv2
import numpy as np
import os
import json
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class JigsawMatcher:
    """
    Main class for jigsaw puzzle edge matching using classical computer vision.

    This class processes puzzle pieces that have been segmented by Milestone 1
    and finds potential matches based on edge shape similarity.
    """

    def __init__(self, output_dir="output"):
        """
        Initialize the matcher with output directory from Milestone 1.

        Args:
            output_dir (str): Directory containing Milestone 1 outputs (tiles, edges, contours)
        """
        self.output_dir = output_dir
        self.tiles_dir = os.path.join(output_dir, "tiles")
        self.edges_dir = os.path.join(output_dir, "edges")
        self.contours_dir = os.path.join(output_dir, "contours")

        # Validate that required directories exist
        for dir_path in [self.tiles_dir, self.edges_dir, self.contours_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory not found: {dir_path}. "
                                      "Please run Milestone 1 first.")

        # Storage for extracted features
        self.edge_features = {}  # piece_id -> feature_dict
        self.match_cache = {}    # Cache for computed matches

    def extract_fourier_descriptors(self, contour, num_descriptors=16):
        """
        Extract Fourier descriptors from contour for rotation-invariant shape description.

        Fourier descriptors capture the frequency components of the shape boundary,
        making them robust to rotation, scaling, and translation when properly normalized.

        Args:
            contour: OpenCV contour (numpy array of shape (N, 1, 2))
            num_descriptors: Number of Fourier coefficients to keep

        Returns:
            fourier_descriptors: Normalized Fourier descriptors (always length num_descriptors)
        """
        if len(contour) < 4:
            return np.zeros(num_descriptors)

        # Convert contour to complex numbers for FFT
        contour_points = contour.reshape(-1, 2)
        complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]

        # Center the contour (translation invariance)
        centroid = np.mean(complex_contour)
        centered_contour = complex_contour - centroid

        # Compute Discrete Fourier Transform
        fft_coeffs = np.fft.fft(centered_contour)

        # Extract first N descriptors (excluding DC component)
        # DC component represents the centroid, which we already removed
        descriptors = fft_coeffs[1:num_descriptors+1]

        # Handle case where contour has fewer points than requested descriptors
        if len(descriptors) < num_descriptors:
            # Pad with zeros to ensure consistent length
            descriptors = np.pad(descriptors, (0, num_descriptors - len(descriptors)), 'constant')

        # Convert to magnitude (rotation invariance)
        fourier_descriptors = np.abs(descriptors)

        # Normalize by first descriptor (scale invariance)
        if len(fourier_descriptors) > 0 and fourier_descriptors[0] > 0:
            fourier_descriptors = fourier_descriptors / fourier_descriptors[0]

        return fourier_descriptors

    def extract_hu_moments(self, contour):
        """
        Extract Hu invariant moments from contour.

        Hu moments are 7 values that are invariant to:
        - Translation
        - Scale
        - Rotation (to some degree)

        These moments capture global shape characteristics and are commonly used
        in shape recognition and classification tasks.

        Args:
            contour: OpenCV contour

        Returns:
            hu_moments: Array of 7 Hu invariant moments
        """
        if len(contour) < 4:
            return np.zeros(7)

        # Create binary mask from contour
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Shift contour to fit in mask
        shifted_contour = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted_contour], -1, 255, -1)

        # Calculate moments
        moments = cv2.moments(mask)

        # Compute Hu invariant moments
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log transform to make moments more comparable
        # Hu moments span many orders of magnitude
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return hu_moments

    def extract_curvature_features(self, contour, window_size=5):
        """
        Extract curvature-based features from contour.

        Curvature measures how much the contour deviates from being straight.
        This helps distinguish between straight edges and curved edges.

        Args:
            contour: OpenCV contour
            window_size: Size of sliding window for curvature calculation

        Returns:
            curvature_stats: Statistical features of curvature values
        """
        if len(contour) < window_size * 2:
            return np.zeros(6)

        contour_points = contour.reshape(-1, 2)
        curvatures = []

        # Calculate curvature using sliding window approach
        for i in range(window_size, len(contour_points) - window_size):
            # Get three points: p1, p2, p3
            p1 = contour_points[i - window_size]
            p2 = contour_points[i]
            p3 = contour_points[i + window_size]

            # Calculate vectors
            v1 = p2 - p1  # Vector from p1 to p2
            v2 = p3 - p2  # Vector from p2 to p3

            # Calculate norms
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 > 0 and norm_v2 > 0:
                # Calculate cross product for curvature approximation
                cross = v1[0] * v2[1] - v1[1] * v2[0]

                # Curvature approximation: κ ≈ 2 * sin(θ) / ||v1||
                # where θ is angle between vectors
                curvature = 2 * cross / (norm_v1 * norm_v2 * (norm_v1 + norm_v2))
                curvatures.append(abs(curvature))

        if not curvatures:
            return np.zeros(6)

        curvatures = np.array(curvatures)

        # Extract statistical features
        features = [
            np.mean(curvatures),      # Average curvature
            np.std(curvatures),       # Curvature variation
            np.max(curvatures),       # Maximum curvature
            np.min(curvatures),       # Minimum curvature
            np.median(curvatures),    # Median curvature
            len(curvatures) / len(contour_points)  # Ratio of high-curvature points
        ]

        return np.array(features)

    def classify_edge_type(self, contour, curvature_threshold=0.1):
        """
        Classify edge as straight or curved based on curvature analysis.

        This classification helps in matching strategy - straight edges might
        match better with other straight edges, curved with curved.

        Args:
            contour: OpenCV contour
            curvature_threshold: Threshold for classification

        Returns:
            edge_type: 'straight' or 'curved'
        """
        curvature_features = self.extract_curvature_features(contour)

        mean_curvature = curvature_features[0]
        std_curvature = curvature_features[1]

        # Classify as straight if both mean and std are low
        if mean_curvature < curvature_threshold and std_curvature < curvature_threshold:
            return 'straight'
        else:
            return 'curved'

    def extract_edge_features(self, edge_image_path, contour_image_path, piece_id):
        """
        Extract comprehensive shape features from a single puzzle piece edge.

        This method combines multiple shape descriptors to create a robust
        representation that captures different aspects of edge shape.

        Args:
            edge_image_path: Path to edge image from Milestone 1
            contour_image_path: Path to contour image from Milestone 1
            piece_id: Unique identifier for the piece

        Returns:
            features: Dictionary containing all extracted features
        """
        # Load edge image (grayscale)
        edge_img = cv2.imread(edge_image_path, cv2.IMREAD_GRAYSCALE)
        if edge_img is None:
            print(f"Warning: Could not load edge image {edge_image_path}")
            return None

        # Find contours in edge image
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            print(f"Warning: No contours found in {edge_image_path}")
            # Return default features for pieces with no contours
            return {
                'piece_id': piece_id,
                'contour': np.array([]),  # Empty contour
                'edge_type': 'unknown',
                'fourier_descriptors': np.zeros(16),  # Default length
                'hu_moments': np.zeros(7),
                'curvature_features': np.zeros(6),
                'area': 0,
                'perimeter': 0,
                'compactness': 0,
                'aspect_ratio': 0,
                'bounding_box': (0, 0, 0, 0)
            }

        # Use the largest contour (should be the main puzzle piece edge)
        main_contour = max(contours, key=cv2.contourArea)

        # Extract multiple shape descriptors
        fourier_desc = self.extract_fourier_descriptors(main_contour)
        hu_moments = self.extract_hu_moments(main_contour)
        curvature_feat = self.extract_curvature_features(main_contour)
        edge_type = self.classify_edge_type(main_contour)

        # Calculate additional geometric properties
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Bounding box properties
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 0

        # Compile all features
        features = {
            'piece_id': piece_id,
            'contour': main_contour,  # Store for visualization
            'edge_type': edge_type,

            # Shape descriptors
            'fourier_descriptors': fourier_desc,
            'hu_moments': hu_moments,
            'curvature_features': curvature_feat,

            # Geometric properties
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'aspect_ratio': aspect_ratio,
            'bounding_box': (x, y, w, h)
        }

        return features

    def load_and_process_all_pieces(self):
        """
        Load and process all puzzle pieces from Milestone 1 outputs.

        This method reads all the tiles, edges, and contours generated by Milestone 1
        and extracts comprehensive shape features for each piece.
        """
        print("🔍 Loading and processing puzzle pieces...")

        # Get all tile files (they follow naming pattern: tile_0.png, tile_1.png, etc.)
        tile_files = [f for f in os.listdir(self.tiles_dir)
                     if f.startswith('tile_') and f.endswith('.png')]
        tile_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        self.edge_features = {}

        processed_count = 0
        for tile_file in tile_files:
            piece_id = int(tile_file.split('_')[1].split('.')[0])

            # Corresponding edge and contour files
            edge_file = f"edges_{piece_id}.png"
            contour_file = f"contour_{piece_id}.png"

            edge_path = os.path.join(self.edges_dir, edge_file)
            contour_path = os.path.join(self.contours_dir, contour_file)

            if not os.path.exists(edge_path):
                print(f"Warning: Edge file not found: {edge_path}")
                continue

            # Extract features for this piece
            features = self.extract_edge_features(edge_path, contour_path, piece_id)

            if features is not None:
                self.edge_features[piece_id] = features
                processed_count += 1

                # Progress indicator
                edge_type = features['edge_type']
                area = features['area']
                print(f"  ✓ Piece {piece_id}: {edge_type} edge, area={area:.0f}")

        print(f"\n✅ Successfully processed {processed_count} puzzle pieces")
        return processed_count

    def calculate_feature_distance(self, features1, features2, weights=None):
        """
        Calculate weighted distance between two sets of edge features.

        We combine multiple distance metrics because no single descriptor is perfect
        for all types of shape matching. Different features capture different aspects:
        - Fourier: Global shape frequency content
        - Hu moments: Global shape characteristics
        - Curvature: Local shape variations
        - Geometric: Size and compactness

        Args:
            features1, features2: Feature dictionaries from extract_edge_features
            weights: Dictionary specifying weights for each feature type

        Returns:
            total_distance: Combined distance score
            distance_details: Dictionary with individual distance components
        """
        if weights is None:
            # Default weights based on empirical testing
            weights = {
                'fourier': 0.4,    # Most reliable for shape matching
                'hu_moments': 0.3, # Good for global shape
                'curvature': 0.2,  # Important for local variations
                'geometric': 0.1   # Less important but helps with tie-breaking
            }

        total_distance = 0
        distance_details = {}

        # 1. Fourier descriptors distance (cosine similarity converted to distance)
        fourier1 = features1['fourier_descriptors']
        fourier2 = features2['fourier_descriptors']

        if len(fourier1) > 0 and len(fourier2) > 0 and np.any(fourier1) and np.any(fourier2):
            try:
                # Ensure both arrays have the same length for cosine similarity
                min_len = min(len(fourier1), len(fourier2))
                f1 = fourier1[:min_len]
                f2 = fourier2[:min_len]

                # Cosine similarity measures shape similarity
                cos_sim = cosine_similarity([f1], [f2])[0][0]
                fourier_dist = 1 - cos_sim  # Convert to distance
            except Exception as e:
                # Fallback to Euclidean distance if cosine similarity fails
                fourier_dist = euclidean(fourier1[:min_len], fourier2[:min_len])
        else:
            fourier_dist = 1.0

        distance_details['fourier'] = fourier_dist

        # 2. Hu moments distance (Euclidean distance)
        hu1 = features1['hu_moments']
        hu2 = features2['hu_moments']
        if len(hu1) == len(hu2) and len(hu1) > 0:
            hu_dist = euclidean(hu1, hu2)
        else:
            hu_dist = 1.0  # Maximum distance if lengths don't match or empty
        distance_details['hu_moments'] = hu_dist

        # 3. Curvature features distance (Euclidean distance)
        curv1 = features1['curvature_features']
        curv2 = features2['curvature_features']
        if len(curv1) == len(curv2) and len(curv1) > 0:
            curv_dist = euclidean(curv1, curv2)
        else:
            curv_dist = 1.0  # Maximum distance if lengths don't match or empty
        distance_details['curvature'] = curv_dist

        # 4. Geometric features distance (normalized Euclidean)
        geom_features1 = np.array([
            features1['compactness'],
            features1['aspect_ratio'],
            features1['area'] / 10000,  # Normalize area
            features1['perimeter'] / 1000  # Normalize perimeter
        ])
        geom_features2 = np.array([
            features2['compactness'],
            features2['aspect_ratio'],
            features2['area'] / 10000,
            features2['perimeter'] / 1000
        ])
        geom_dist = euclidean(geom_features1, geom_features2)
        distance_details['geometric'] = geom_dist

        # Combine with weights
        total_distance = (
            weights['fourier'] * fourier_dist +
            weights['hu_moments'] * hu_dist +
            weights['curvature'] * curv_dist +
            weights['geometric'] * geom_dist
        )

        distance_details['total'] = total_distance

        return total_distance, distance_details

    def find_best_matches(self, target_piece_id, num_matches=5, edge_type_filter=None):
        """
        Find the best matching edges for a target piece.

        This method compares the target piece against all other pieces and ranks
        them by shape similarity.

        Args:
            target_piece_id: ID of piece to find matches for
            num_matches: Number of best matches to return
            edge_type_filter: Filter matches by edge type ('straight', 'curved', or None)

        Returns:
            matches: List of (piece_id, distance, distance_details) tuples, sorted by distance
        """
        if target_piece_id not in self.edge_features:
            print(f"Error: Piece {target_piece_id} not found in processed pieces")
            return []

        target_features = self.edge_features[target_piece_id]

        # Get all candidate pieces (excluding self)
        candidates = []
        for piece_id, features in self.edge_features.items():
            if piece_id == target_piece_id:
                continue
            if edge_type_filter and features['edge_type'] != edge_type_filter:
                continue
            candidates.append((piece_id, features))

        if not candidates:
            return []

        print(f"Comparing piece {target_piece_id} against {len(candidates)} candidates...")

        # Calculate distances to all candidates
        distances = []
        for piece_id, features in candidates:
            distance, distance_details = self.calculate_feature_distance(target_features, features)
            distances.append((piece_id, distance, distance_details))

        # Sort by distance (best matches first)
        distances.sort(key=lambda x: x[1])

        # Return top matches
        return distances[:num_matches]

    def create_match_visualization(self, piece1_id, piece2_id, save_path=None):
        """
        Create a visual comparison of two matching puzzle pieces.

        This visualization helps users understand why pieces are considered matches
        by showing the original pieces side-by-side with their contours overlaid.

        Args:
            piece1_id, piece2_id: IDs of pieces to compare
            save_path: Optional path to save the visualization

        Returns:
            visualization: OpenCV image showing the comparison
        """
        if piece1_id not in self.edge_features or piece2_id not in self.edge_features:
            print(f"Error: One or both pieces ({piece1_id}, {piece2_id}) not found")
            return None

        # Load tile images
        tile1_path = os.path.join(self.tiles_dir, f"tile_{piece1_id}.png")
        tile2_path = os.path.join(self.tiles_dir, f"tile_{piece2_id}.png")

        tile1 = cv2.imread(tile1_path)
        tile2 = cv2.imread(tile2_path)

        if tile1 is None or tile2 is None:
            print(f"Error: Could not load tile images for pieces {piece1_id} and {piece2_id}")
            return None

        # Get image dimensions
        h, w = tile1.shape[:2]

        # Create side-by-side visualization
        viz_width = w * 2 + 20  # 20px gap
        viz_height = h + 100   # Extra space for labels

        # Create white background
        visualization = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 240

        # Place tiles
        visualization[50:50+h, 10:10+w] = tile1
        visualization[50:50+h, 10+w+10:10+w+10+w] = tile2

        # Draw contours on tiles
        contour1 = self.edge_features[piece1_id]['contour']
        contour2 = self.edge_features[piece2_id]['contour']

        # Offset contour2 coordinates for second tile position
        offset_contour2 = contour2 + np.array([10+w+10, 50])

        # Draw contours (green for piece1, red for piece2)
        cv2.drawContours(visualization, [contour1 + np.array([10, 50])], -1, (0, 255, 0), 2)
        cv2.drawContours(visualization, [offset_contour2], -1, (255, 0, 0), 2)

        # Add labels
        cv2.putText(visualization, f"Piece {piece1_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(visualization, f"Piece {piece2_id}", (20+w, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add match quality information
        distance, _ = self.calculate_feature_distance(
            self.edge_features[piece1_id],
            self.edge_features[piece2_id]
        )
        cv2.putText(visualization, ".4f",
                   (10, viz_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, visualization)
            print(f"✓ Match visualization saved: {save_path}")

        return visualization

    def create_assembly_suggestions(self, grid_size=4):
        """
        Create assembly suggestions by finding potential neighboring pieces.

        This method analyzes all pieces and suggests which ones might fit together,
        similar to how humans look for matching puzzle edges.

        Args:
            grid_size: Size of puzzle grid (for context)

        Returns:
            suggestions: Dictionary with assembly suggestions for each piece
        """
        if not self.edge_features:
            print("Error: No pieces processed yet. Run load_and_process_all_pieces() first.")
            return None

        print("🧩 Creating assembly suggestions...")

        suggestions = {}
        piece_ids = list(self.edge_features.keys())

        # For each piece, find its best matches
        for i, piece_id in enumerate(piece_ids):
            if (i + 1) % 5 == 0:  # Progress indicator
                print(f"  Processing piece {i+1}/{len(piece_ids)}")

            matches = self.find_best_matches(piece_id, num_matches=4)

            suggestions[piece_id] = {
                'matches': matches,
                'edge_type': self.edge_features[piece_id]['edge_type'],
                'features': {
                    'area': self.edge_features[piece_id]['area'],
                    'perimeter': self.edge_features[piece_id]['perimeter'],
                    'compactness': self.edge_features[piece_id]['compactness']
                }
            }

        print(f"✅ Created assembly suggestions for {len(suggestions)} pieces")
        return suggestions

    def save_features_to_json(self, output_path=None):
        """
        Save extracted features to JSON file for analysis and debugging.

        This creates a human-readable record of all the features extracted
        from each puzzle piece, which can be useful for analysis and tuning.
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "edge_features.json")

        # Convert numpy arrays to lists for JSON serialization
        serializable_features = {}
        for piece_id, features in self.edge_features.items():
            serializable_features[str(piece_id)] = {
                'piece_id': features['piece_id'],
                'edge_type': features['edge_type'],
                'area': float(features['area']),
                'perimeter': float(features['perimeter']),
                'compactness': float(features['compactness']),
                'aspect_ratio': float(features['aspect_ratio']),
                'bounding_box': features['bounding_box'],
                'fourier_descriptors': features['fourier_descriptors'].tolist(),
                'hu_moments': features['hu_moments'].tolist(),
                'curvature_features': features['curvature_features'].tolist()
            }

        with open(output_path, 'w') as f:
            json.dump(serializable_features, f, indent=2)

        print(f"💾 Features saved to: {output_path}")

    def analyze_edge_types(self):
        """
        Analyze the distribution of edge types in the puzzle.

        This provides insights into the puzzle composition and can help
        validate that the edge classification is working correctly.
        """
        if not self.edge_features:
            return None

        edge_types = {}
        total_pieces = len(self.edge_features)

        for features in self.edge_features.values():
            edge_type = features['edge_type']
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        print("📊 Edge Type Analysis:")
        print(f"  Total pieces: {total_pieces}")
        for edge_type, count in edge_types.items():
            percentage = count / total_pieces * 100
            print(f"  - {edge_type}: {count} pieces ({percentage:.1f}%)")

        return edge_types

    def assemble_puzzle_grid(self, grid_size=4, output_path=None):
        """
        Attempt to assemble the puzzle pieces into a grid based on their positions.

        This is a simplified assembly that places pieces in a grid layout.
        A more sophisticated approach would use the matching results to determine
        actual spatial relationships.

        Args:
            grid_size: Size of the puzzle grid (NxN)
            output_path: Optional path to save the assembled image

        Returns:
            assembled_image: OpenCV image of the assembled puzzle
        """
        if not self.edge_features:
            print("Error: No pieces to assemble. Run load_and_process_all_pieces() first.")
            return None

        print("🧩 Assembling puzzle pieces into grid layout...")

        # Load the first tile to get dimensions
        first_tile_path = os.path.join(self.tiles_dir, "tile_0.png")
        first_tile = cv2.imread(first_tile_path)

        if first_tile is None:
            print("Error: Could not load tile images for assembly")
            return None

        tile_h, tile_w = first_tile.shape[:2]

        # Create the assembled image canvas
        assembled_h = tile_h * grid_size
        assembled_w = tile_w * grid_size
        assembled_image = np.zeros((assembled_h, assembled_w, 3), dtype=np.uint8)

        # Place each piece in its grid position
        # For a basic assembly, we assume pieces are numbered in row-major order
        for piece_id in range(grid_size * grid_size):
            tile_path = os.path.join(self.tiles_dir, f"tile_{piece_id}.png")
            tile = cv2.imread(tile_path)

            if tile is None:
                print(f"Warning: Could not load tile {piece_id}, skipping")
                continue

            # Calculate grid position
            row = piece_id // grid_size
            col = piece_id % grid_size

            # Calculate pixel position
            y_start = row * tile_h
            y_end = y_start + tile_h
            x_start = col * tile_w
            x_end = x_start + tile_w

            # Place the tile
            assembled_image[y_start:y_end, x_start:x_end] = tile

        print(f"✓ Assembled {grid_size}x{grid_size} puzzle grid")

        if output_path:
            cv2.imwrite(output_path, assembled_image)
            print(f"✓ Saved assembled puzzle to: {output_path}")

        return assembled_image

    def assemble_puzzle_from_matches(self, grid_size=4, output_path=None):
        """
        Attempt to assemble puzzle using matching results.

        This implementation creates a more intelligent assembly by:
        1. Identifying corner and edge pieces
        2. Using matching scores to suggest piece relationships
        3. Creating a structured layout based on piece connectivity

        Args:
            grid_size: Size of the puzzle grid
            output_path: Optional path to save the assembled image

        Returns:
            assembled_image: OpenCV image of the assembled puzzle
        """
        print("🔍 Attempting smart assembly using matching results...")

        if not self.edge_features:
            print("No pieces to assemble, using grid layout")
            return self.assemble_puzzle_grid(grid_size, output_path)

        # Try to create a more intelligent layout
        try:
            assembled_image = self._create_structured_assembly(grid_size)
            if assembled_image is not None:
                print("✓ Created structured assembly based on matches")
                if output_path:
                    cv2.imwrite(output_path, assembled_image)
                return assembled_image
        except Exception as e:
            print(f"Smart assembly failed ({e}), falling back to grid layout")

        # Fallback to grid assembly
        return self.assemble_puzzle_grid(grid_size, output_path)

    def _create_structured_assembly(self, grid_size=4):
        """
        Create a more intelligent assembly using piece matching relationships.
        This attempts to solve the puzzle by placing pieces next to their best matches.
        """
        # Get piece dimensions
        first_tile_path = os.path.join(self.tiles_dir, "tile_0.png")
        first_tile = cv2.imread(first_tile_path)

        if first_tile is None:
            raise ValueError("Could not load tile images")

        tile_h, tile_w = first_tile.shape[:2]

        # Create the assembled image canvas
        assembled_h = tile_h * grid_size
        assembled_w = tile_w * grid_size
        assembled_image = np.zeros((assembled_h, assembled_w, 3), dtype=np.uint8)

        # Create a placement map
        placement_map = {}
        placed_pieces = set()

        # Start with piece 0 in the top-left corner
        placement_map[(0, 0)] = 0
        placed_pieces.add(0)

        # Try to place pieces based on their best matches
        for current_pos in [(0, 0)]:
            current_piece = placement_map[current_pos]

            # Find the best matches for this piece
            matches = self.find_best_matches(current_piece, num_matches=4)

            # Try to place matches in adjacent positions
            row, col = current_pos

            # Possible adjacent positions (right, bottom, left, top)
            adjacent_positions = [
                (row, col + 1),      # right
                (row + 1, col),      # bottom
                (row, col - 1),      # left
                (row - 1, col)       # top
            ]

            match_idx = 0
            for adj_pos in adjacent_positions:
                adj_row, adj_col = adj_pos

                # Check if position is valid and empty
                if (0 <= adj_row < grid_size and 0 <= adj_col < grid_size and
                    adj_pos not in placement_map):

                    # Place the next best match
                    if match_idx < len(matches):
                        matched_piece, _, _ = matches[match_idx]
                        if matched_piece not in placed_pieces:
                            placement_map[adj_pos] = matched_piece
                            placed_pieces.add(matched_piece)
                            match_idx += 1

        # Fill remaining positions with unused pieces
        all_pieces = set(range(grid_size * grid_size))
        unused_pieces = all_pieces - placed_pieces

        for row in range(grid_size):
            for col in range(grid_size):
                pos = (row, col)
                if pos not in placement_map and unused_pieces:
                    # Place next unused piece
                    piece_id = unused_pieces.pop()
                    placement_map[pos] = piece_id

        # Now place all pieces according to the placement map
        for (row, col), piece_id in placement_map.items():
            self._place_piece_at_position(assembled_image, piece_id, row, col, tile_h, tile_w)

        return assembled_image

    def _place_piece_at_position(self, assembled_image, piece_id, row, col, tile_h, tile_w):
        """
        Place a specific piece at a grid position in the assembled image.
        """
        tile_path = os.path.join(self.tiles_dir, f"tile_{piece_id}.png")
        tile = cv2.imread(tile_path)

        if tile is None:
            # Create a placeholder if tile not found
            tile = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 128  # Gray placeholder
            cv2.putText(tile, f"?", (tile_w//2-10, tile_h//2+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Ensure tile is the right size
            tile = cv2.resize(tile, (tile_w, tile_h))

        # Calculate position
        y_start = row * tile_h
        y_end = y_start + tile_h
        x_start = col * tile_w
        x_end = x_start + tile_w

        # Place the tile
        assembled_image[y_start:y_end, x_start:x_end] = tile


def run_milestone2_pipeline(output_dir="output", grid_size=4):
    """
    Main pipeline function for Milestone 2 edge matching.

    This function orchestrates the entire edge matching process:
    1. Load and process all puzzle pieces
    2. Extract shape features
    3. Create assembly suggestions
    4. Generate visualizations

    Args:
        output_dir: Directory containing Milestone 1 outputs
        grid_size: Puzzle grid size (for context)

    Returns:
        matcher: Configured JigsawMatcher instance
        assembly_suggestions: Dictionary with matching suggestions
    """
    print("=" * 80)
    print("🧩 JIGSAW PUZZLE EDGE MATCHING — MILESTONE 2")
    print("=" * 80)

    try:
        # Initialize matcher
        matcher = JigsawMatcher(output_dir)

        # Process all pieces
        num_pieces = matcher.load_and_process_all_pieces()

        if num_pieces == 0:
            print("❌ Error: No pieces were successfully processed")
            return None, None

        # Analyze edge types
        matcher.analyze_edge_types()

        # Save features for analysis
        matcher.save_features_to_json()

        # Create assembly suggestions
        print("\n🔗 Generating assembly suggestions...")
        assembly_suggestions = matcher.create_assembly_suggestions(grid_size)

        # Create visualizations for top matches
        print("\n🎨 Creating match visualizations...")
        viz_dir = os.path.join(output_dir, "match_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Create visualizations for first few pieces
        piece_ids = list(assembly_suggestions.keys())[:3]  # Show top 3
        for piece_id in piece_ids:
            matches = assembly_suggestions[piece_id]['matches']
            if matches:
                best_match_id = matches[0][0]
                viz_path = os.path.join(viz_dir, f"match_{piece_id}_{best_match_id}.png")
                matcher.create_match_visualization(piece_id, best_match_id, viz_path)

        # Summary
        print("\n" + "=" * 80)
        print("✅ MILESTONE 2 PIPELINE COMPLETE")
        print("=" * 80)
        print(f"📁 Processed {num_pieces} puzzle pieces")
        print(f"🔍 Created {len(assembly_suggestions)} assembly suggestions")
        print(f"📊 Generated {len(piece_ids)} match visualizations")
        print(f"💾 Output directory: {output_dir}")
        print("=" * 80)

        return matcher, assembly_suggestions

    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def demonstrate_milestone2(image_path="puzzle_image.jpg", grid_size=4):
    """
    Complete demonstration of Milestone 2 functionality.

    This function shows how to use the edge matching system end-to-end,
    from processing an image through Milestone 1 to generating matches.

    Args:
        image_path: Path to puzzle image
        grid_size: Puzzle grid size
    """
    print("🚀 MILESTONE 2 DEMONSTRATION")
    print("=" * 50)

    # This would typically call Milestone 1 pipeline first
    print(f"📷 Would process image: {image_path}")
    print(f"🔢 Grid size: {grid_size}x{grid_size}")

    # For now, assume Milestone 1 outputs exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"❌ Output directory '{output_dir}' not found.")
        print("Please run Milestone 1 first to generate puzzle piece data.")
        return

    # Run Milestone 2
    matcher, suggestions = run_milestone2_pipeline(output_dir, grid_size)

    if matcher and suggestions:
        # Show example results
        print("\n🎯 EXAMPLE RESULTS:")

        # Pick first piece and show its matches
        first_piece = list(suggestions.keys())[0]
        matches = suggestions[first_piece]['matches']

        print(f"\nTop matches for piece {first_piece}:")
        for i, (match_id, distance, details) in enumerate(matches[:3]):
            print(".4f"
                  f"  (Fourier: {details['fourier']:.3f}, "
                  f"Hu: {details['hu_moments']:.3f}, "
                  f"Curvature: {details['curvature']:.3f})")

        print(f"\n📸 Match visualizations saved in: {output_dir}/match_visualizations/")
        print(f"📄 Feature data saved in: {output_dir}/edge_features.json")


if __name__ == "__main__":
    # Example usage
    print("Jigsaw Puzzle Edge Matching - Milestone 2")
    print("Usage: python jigsaw_matcher.py")
    print("\nFor demonstration with existing data:")
    demonstrate_milestone2()
