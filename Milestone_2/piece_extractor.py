# piece_features.py
# Milestone 2 - Edge Feature Extraction & Preprocessing

import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import euclidean
import sys

# ========== CONFIG ==========
TILES_DIR = Path("C:/Users/mmmsa/PycharmProjects/Image-Processing-Project/Milestone_1/output/tiles")
OUTPUT_DIR = Path("output/pieces")
DEBUG_VISUALIZATION = True
FIXED_EDGE_POINTS = 100  # Number of points to resample each edge to


# ============================

class PuzzlePiece:
    def __init__(self, piece_id, image):
        self.id = piece_id
        self.image = image
        self.edges = []  # List of 4 edges
        self.piece_type = None  # 'corner', 'edge', 'inner'
        self.centroid = None
        self.contour_points = None
        self.features = {}

    def classify_piece_type(self, puzzle_edges):
        """
        Classify piece type based on how many edges are puzzle boundaries.

        Args:
            puzzle_edges: List of 4 booleans indicating which edges are straight (puzzle boundaries)
        """
        num_straight_edges = sum(puzzle_edges)

        if num_straight_edges == 2:
            self.piece_type = 'corner'
        elif num_straight_edges == 1:
            self.piece_type = 'edge'
        elif num_straight_edges == 0:
            self.piece_type = 'inner'
        else:
            self.piece_type = 'unknown'

        return self.piece_type

    def extract_edges_with_canny(self):
        """
        Extract edge contours using Canny edge detection.
        Returns list of 4 edge contours.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection with adaptive thresholds
        v = np.median(blurred)
        lower = int(max(0, 0.5 * v))
        upper = int(min(255, 1.5 * v))
        edges = cv2.Canny(blurred, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Warning: No contours found for piece {self.id}")
            return []

        # Get the largest contour (should be the puzzle piece)
        contour = max(contours, key=cv2.contourArea)
        self.contour_points = contour.reshape(-1, 2)

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            self.centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Find corners using Harris corner detection
        corners = self.find_corners_harris()

        if len(corners) >= 4:
            # Sort corners clockwise
            sorted_corners = self.sort_corners_clockwise(corners[:4])

            # Extract edges between corners
            self.edges = self.extract_edges_from_corners(sorted_corners)

        return self.edges

    def find_corners_harris(self):
        """Find corners using Harris corner detection."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Harris corner detection
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # Dilate to mark corners
        dst = cv2.dilate(dst, None)

        # Threshold for corners
        corners = []
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # Find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # Define criteria for corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # Remove the first corner (it's the image center)
        corners = corners[1:]

        return corners

    def sort_corners_clockwise(self, corners):
        """Sort corners in clockwise order."""
        # Calculate centroid of corners
        center_x = np.mean([c[0] for c in corners])
        center_y = np.mean([c[1] for c in corners])
        center = np.array([center_x, center_y])

        # Calculate angles
        angles = []
        for corner in corners:
            dx = corner[0] - center[0]
            dy = corner[1] - center[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)

        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = [corners[i] for i in sorted_indices]

        return sorted_corners

    def extract_edges_from_corners(self, corners):
        """
        Extract edge contours between corners.

        Args:
            corners: List of 4 corners in clockwise order

        Returns:
            List of 4 edge contours
        """
        edges = []

        for i in range(4):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]

            # Extract edge points between these corners from the contour
            edge_points = self.get_edge_points_between_corners(start_corner, end_corner)

            if len(edge_points) > 10:  # Minimum points for a valid edge
                edges.append(edge_points)
            else:
                # If too few points, create a straight line
                edge_points = np.linspace(start_corner, end_corner, 50)
                edges.append(edge_points)

        return edges

    def get_edge_points_between_corners(self, corner1, corner2):
        """
        Extract contour points that lie between two corners.
        """
        if self.contour_points is None or len(self.contour_points) == 0:
            return np.array([])

        # Find indices of corners in contour
        distances1 = [euclidean(corner1, point) for point in self.contour_points]
        distances2 = [euclidean(corner2, point) for point in self.contour_points]

        idx1 = np.argmin(distances1)
        idx2 = np.argmin(distances2)

        # Extract points between these indices (going clockwise)
        if idx1 < idx2:
            edge_points = self.contour_points[idx1:idx2]
        else:
            edge_points = np.vstack([self.contour_points[idx1:], self.contour_points[:idx2]])

        return edge_points

    def resample_edge(self, edge_points, num_points=FIXED_EDGE_POINTS):
        """
        Resample edge to fixed number of points using interpolation.

        Args:
            edge_points: Original edge points
            num_points: Number of points in resampled edge

        Returns:
            Resampled edge points
        """
        if len(edge_points) < 2:
            return edge_points

        # Separate x and y coordinates
        x = edge_points[:, 0]
        y = edge_points[:, 1]

        # Calculate cumulative distance along edge
        distances = np.zeros(len(x))
        for i in range(1, len(x)):
            distances[i] = distances[i - 1] + np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)

        # Normalize distances to [0, 1]
        if distances[-1] > 0:
            distances = distances / distances[-1]

        # Create interpolation functions
        fx = interpolate.interp1d(distances, x, kind='linear')
        fy = interpolate.interp1d(distances, y, kind='linear')

        # Resample at fixed intervals
        new_distances = np.linspace(0, 1, num_points)
        new_x = fx(new_distances)
        new_y = fy(new_distances)

        resampled_points = np.column_stack([new_x, new_y])

        return resampled_points

    def compute_shape_features(self, edge_index, edge_points):
        """
        Compute shape features for an edge.

        Args:
            edge_index: Index of edge (0-3)
            edge_points: Points along the edge

        Returns:
            Dictionary of shape features
        """
        if len(edge_points) < 2:
            return {}

        features = {}

        # 1. Edge length
        length = 0
        for i in range(1, len(edge_points)):
            length += euclidean(edge_points[i], edge_points[i - 1])
        features['length'] = float(length)

        # 2. Straightness (how close to a straight line)
        start_point = edge_points[0]
        end_point = edge_points[-1]
        straight_line_length = euclidean(start_point, end_point)

        if straight_line_length > 0:
            features['straightness'] = float(straight_line_length / length)
        else:
            features['straightness'] = 0.0

        # 3. Curvature features
        if len(edge_points) >= 3:
            # Compute curvature at each point
            curvatures = []
            for i in range(1, len(edge_points) - 1):
                p1 = edge_points[i - 1]
                p2 = edge_points[i]
                p3 = edge_points[i + 1]

                # Area of triangle formed by three points
                area = 0.5 * abs(
                    p1[0] * (p2[1] - p3[1]) +
                    p2[0] * (p3[1] - p1[1]) +
                    p3[0] * (p1[1] - p2[1])
                )

                # Lengths of sides
                a = euclidean(p1, p2)
                b = euclidean(p2, p3)
                c = euclidean(p3, p1)

                # Curvature (inverse of radius of circumcircle)
                if a * b * c > 0:
                    curvature = 4 * area / (a * b * c)
                else:
                    curvature = 0

                curvatures.append(curvature)

            if curvatures:
                features['mean_curvature'] = float(np.mean(curvatures))
                features['max_curvature'] = float(np.max(curvatures))
                features['curvature_std'] = float(np.std(curvatures))

        # 4. Direction vectors
        features['start_point'] = [float(edge_points[0][0]), float(edge_points[0][1])]
        features['end_point'] = [float(edge_points[-1][0]), float(edge_points[-1][1])]

        # Calculate direction angle
        dx = edge_points[-1][0] - edge_points[0][0]
        dy = edge_points[-1][1] - edge_points[0][1]
        features['direction_angle'] = float(np.arctan2(dy, dx))

        return features

    def compute_color_features(self, edge_index, edge_points):
        """
        Compute color features along an edge.

        Args:
            edge_index: Index of edge (0-3)
            edge_points: Points along the edge

        Returns:
            Dictionary of color features
        """
        features = {}

        if len(edge_points) == 0:
            return features

        # Sample colors along the edge
        colors = []
        for point in edge_points:
            x, y = int(point[0]), int(point[1])

            # Ensure coordinates are within image bounds
            if 0 <= y < self.image.shape[0] and 0 <= x < self.image.shape[1]:
                color = self.image[y, x]
                colors.append(color)

        if colors:
            colors = np.array(colors)

            # Basic color statistics
            features['mean_color_bgr'] = colors.mean(axis=0).tolist()
            features['std_color_bgr'] = colors.std(axis=0).tolist()

            # Convert to HSV for additional features
            colors_hsv = []
            for color in colors:
                color_bgr = np.uint8([[color]])
                color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
                colors_hsv.append(color_hsv)

            colors_hsv = np.array(colors_hsv)
            features['mean_color_hsv'] = colors_hsv.mean(axis=0).tolist()

            # Color gradient along edge
            if len(colors) > 1:
                gradient = np.diff(colors, axis=0)
                features['mean_gradient'] = gradient.mean(axis=0).tolist()
                features['gradient_magnitude'] = float(np.mean(np.linalg.norm(gradient, axis=1)))

        return features

    def extract_all_features(self):
        """Extract all features for the piece."""
        self.features = {
            'piece_id': self.id,
            'piece_type': self.piece_type,
            'centroid': self.centroid,
            'image_shape': list(self.image.shape),
            'edges': []
        }

        # Determine which edges are straight (puzzle boundaries)
        puzzle_edges = self.detect_puzzle_boundaries()

        for i, edge in enumerate(self.edges):
            # Resample edge to fixed length
            resampled_edge = self.resample_edge(edge, FIXED_EDGE_POINTS)

            # Compute shape features
            shape_features = self.compute_shape_features(i, resampled_edge)

            # Compute color features
            color_features = self.compute_color_features(i, resampled_edge)

            # Edge metadata
            edge_data = {
                'edge_index': i,
                'is_puzzle_boundary': bool(puzzle_edges[i]) if i < len(puzzle_edges) else False,
                'resampled_points': resampled_edge.tolist(),
                'shape_features': shape_features,
                'color_features': color_features
            }

            self.features['edges'].append(edge_data)

        return self.features

    def detect_puzzle_boundaries(self):
        """
        Detect which edges are straight (puzzle boundaries).
        Returns list of 4 booleans.
        """
        puzzle_edges = []

        for i, edge in enumerate(self.edges):
            if len(edge) < 2:
                puzzle_edges.append(False)
                continue

            # Calculate straightness
            start_point = edge[0]
            end_point = edge[-1]
            straight_length = euclidean(start_point, end_point)

            # Calculate actual path length
            path_length = 0
            for j in range(1, len(edge)):
                path_length += euclidean(edge[j], edge[j - 1])

            # Edge is considered straight if path is close to straight line
            if path_length > 0 and straight_length / path_length > 0.95:
                puzzle_edges.append(True)  # Straight edge (puzzle boundary)
            else:
                puzzle_edges.append(False)  # Curved edge (connector tab)

        return puzzle_edges

    def save_features(self, output_dir):
        """Save piece features to JSON file."""
        output_path = output_dir / f"{self.id}.json"

        # Convert numpy arrays to lists for JSON serialization
        features_json = json.dumps(self.features, default=self.json_serializer, indent=2)

        with open(output_path, 'w') as f:
            f.write(features_json)

        print(f"  ✓ Saved features to: {output_path}")

        return output_path

    def json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)

    def create_debug_visualization(self, output_dir):
        """Create debug visualization of piece with edges marked."""
        if not DEBUG_VISUALIZATION:
            return

        # Create copy of image for drawing
        debug_img = self.image.copy()

        # Draw contour points
        if self.contour_points is not None:
            for point in self.contour_points:
                cv2.circle(debug_img, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)

        # Draw centroid
        if self.centroid:
            cv2.circle(debug_img, self.centroid, 5, (0, 0, 255), -1)

        # Draw edges with different colors
        edge_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for i, edge in enumerate(self.edges):
            if len(edge) < 2:
                continue

            color = edge_colors[i % len(edge_colors)]

            # Draw edge line
            for j in range(1, len(edge)):
                pt1 = (int(edge[j - 1][0]), int(edge[j - 1][1]))
                pt2 = (int(edge[j][0]), int(edge[j][1]))
                cv2.line(debug_img, pt1, pt2, color, 2)

            # Draw edge index
            if len(edge) > 0:
                mid_point = edge[len(edge) // 2]
                cv2.putText(debug_img, f'E{i}',
                            (int(mid_point[0]), int(mid_point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw piece type and ID
        cv2.putText(debug_img, f'{self.id} - {self.piece_type}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Determine puzzle boundaries
        puzzle_edges = self.detect_puzzle_boundaries()
        for i, is_boundary in enumerate(puzzle_edges):
            if i < len(self.edges) and len(self.edges[i]) > 0:
                end_point = self.edges[i][-1]
                label = 'B' if is_boundary else 'I'
                cv2.putText(debug_img, label,
                            (int(end_point[0]), int(end_point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Save debug image
        debug_path = output_dir / f"{self.id}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)

        print(f"  ✓ Saved debug visualization: {debug_path}")

        return debug_path


def load_tiles(tiles_dir):
    """Load all tile images."""
    tiles = {}

    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

    for ext in image_extensions:
        for img_path in tiles_dir.glob(f"*{ext}"):
            piece_id = img_path.stem  # e.g., "tile_0"
            image = cv2.imread(str(img_path))

            if image is not None:
                tiles[piece_id] = image
                print(f"  Loaded: {piece_id} - {image.shape}")
            else:
                print(f"  Warning: Failed to load {img_path}")

    return tiles


def process_all_pieces(tiles_dir, output_dir):
    """Process all puzzle pieces and extract features."""

    print("\n" + "=" * 60)
    print("MILESTONE 2: EDGE FEATURE EXTRACTION")
    print("=" * 60)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input tiles directory: {tiles_dir}")
    print(f"Output directory: {output_dir}")

    # Load tiles
    print("\nLoading tiles...")
    tiles = load_tiles(tiles_dir)

    if not tiles:
        print("No tiles found! Please check the tiles directory.")
        return {}

    print(f"Loaded {len(tiles)} tiles.")

    # Process each piece
    all_features = {}

    for piece_id, image in tiles.items():
        print(f"\nProcessing piece: {piece_id}")

        # Create PuzzlePiece object
        piece = PuzzlePiece(piece_id, image)

        # Extract edges using Canny
        print("  Extracting edges...")
        edges = piece.extract_edges_with_canny()

        if len(edges) != 4:
            print(f"  Warning: Found {len(edges)} edges (expected 4)")

        # Classify piece type
        puzzle_edges = piece.detect_puzzle_boundaries()
        piece_type = piece.classify_piece_type(puzzle_edges)
        print(f"  Piece type: {piece_type}")
        print(f"  Puzzle boundaries: {puzzle_edges}")

        # Extract all features
        print("  Extracting features...")
        features = piece.extract_all_features()

        # Save features to JSON
        json_path = piece.save_features(output_dir)

        # Create debug visualization
        debug_path = piece.create_debug_visualization(output_dir)

        # Store features
        all_features[piece_id] = features

    # Save summary of all pieces
    save_summary(all_features, output_dir)

    return all_features


def save_summary(all_features, output_dir):
    """Save summary of all pieces."""
    summary = {
        'total_pieces': len(all_features),
        'piece_types': {},
        'piece_list': list(all_features.keys())
    }

    # Count piece types
    for piece_id, features in all_features.items():
        piece_type = features.get('piece_type', 'unknown')
        if piece_type not in summary['piece_types']:
            summary['piece_types'][piece_type] = 0
        summary['piece_types'][piece_type] += 1

    # Save summary to JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total pieces processed: {summary['total_pieces']}")
    print(f"Piece types:")
    for piece_type, count in summary['piece_types'].items():
        print(f"  {piece_type}: {count}")
    print("=" * 60)

    return summary_path


def visualize_feature_distribution(all_features, output_dir):
    """Create visualizations of feature distributions."""
    if not all_features:
        return

    # Extract features for visualization
    piece_types = []
    edge_lengths = []
    edge_straightness = []

    for piece_id, features in all_features.items():
        piece_types.append(features.get('piece_type', 'unknown'))

        for edge in features.get('edges', []):
            shape_features = edge.get('shape_features', {})
            if 'length' in shape_features:
                edge_lengths.append(shape_features['length'])
            if 'straightness' in shape_features:
                edge_straightness.append(shape_features['straightness'])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Piece type distribution
    unique_types, type_counts = np.unique(piece_types, return_counts=True)
    axes[0, 0].bar(unique_types, type_counts)
    axes[0, 0].set_title('Piece Type Distribution')
    axes[0, 0].set_xlabel('Piece Type')
    axes[0, 0].set_ylabel('Count')

    # Plot 2: Edge length distribution
    if edge_lengths:
        axes[0, 1].hist(edge_lengths, bins=20, alpha=0.7)
        axes[0, 1].set_title('Edge Length Distribution')
        axes[0, 1].set_xlabel('Length (pixels)')
        axes[0, 1].set_ylabel('Frequency')

    # Plot 3: Edge straightness distribution
    if edge_straightness:
        axes[1, 0].hist(edge_straightness, bins=20, alpha=0.7)
        axes[1, 0].set_title('Edge Straightness Distribution')
        axes[1, 0].set_xlabel('Straightness (0=curved, 1=straight)')
        axes[1, 0].set_ylabel('Frequency')

    # Plot 4: Color feature example (first piece, first edge)
    if all_features:
        first_piece = list(all_features.values())[0]
        if first_piece.get('edges'):
            first_edge = first_piece['edges'][0]
            color_features = first_edge.get('color_features', {})

            if 'mean_color_bgr' in color_features:
                mean_color = color_features['mean_color_bgr']
                # Convert BGR to RGB for display
                color_patch = [[mean_color[2] / 255, mean_color[1] / 255, mean_color[0] / 255]]
                axes[1, 1].imshow([color_patch])
                axes[1, 1].set_title(f'Mean Edge Color (Piece 0, Edge 0)')
                axes[1, 1].axis('off')

    plt.tight_layout()

    # Save figure
    viz_path = output_dir / "feature_distributions.png"
    plt.savefig(viz_path, dpi=150)
    plt.close()

    print(f"✓ Feature distribution visualization saved: {viz_path}")


def main():
    """Main function."""
    print("Starting Milestone 2: Edge Feature Extraction")

    # Process all pieces
    all_features = process_all_pieces(TILES_DIR, OUTPUT_DIR)

    # Create visualizations
    if all_features:
        visualize_feature_distribution(all_features, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("MILESTONE 2 COMPLETE!")
    print("=" * 60)
    print(f"Output saved in: {OUTPUT_DIR}")
    print(f"Files created:")
    print(f"  - piece_XX.json: Feature files for each piece")
    print(f"  - piece_XX_debug.jpg: Debug visualizations")
    print(f"  - summary.json: Summary of all pieces")
    print(f"  - feature_distributions.png: Feature visualization")
    print("=" * 60)


if __name__ == "__main__":
    main()