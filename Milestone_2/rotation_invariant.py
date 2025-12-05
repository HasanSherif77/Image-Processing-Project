"""
Rotation-invariant feature extraction and comparison
"""
import cv2
import numpy as np
from typing import Tuple


class RotationInvariant:
    def __init__(self, rotation_steps: int = 8):
        self.rotation_steps = rotation_steps

    def rotate_contour(self, contour: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        Rotate contour around its centroid

        Args:
            contour: Input contour (N, 1, 2)
            angle_degrees: Rotation angle in degrees

        Returns:
            Rotated contour
        """
        # Convert to 2D array
        points = contour.reshape(-1, 2).astype(np.float32)

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return contour
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = np.array([cx, cy])

        # Create rotation matrix
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Translate to origin, rotate, translate back
        rotated = points - centroid
        rotated = np.dot(rotated, rotation_matrix.T)
        rotated = rotated + centroid

        return rotated.reshape(-1, 1, 2).astype(np.int32)

    def find_best_rotation_match(self, contour1: np.ndarray, contour2: np.ndarray,
                                 matcher) -> Tuple[float, float]:
        """
        Find best rotation angle that minimizes match score

        Returns:
            best_angle: Rotation angle in degrees
            best_score: Match score at best angle
        """
        best_score = float('inf')
        best_angle = 0

        angles = np.linspace(0, 360, self.rotation_steps, endpoint=False)

        for angle in angles:
            rotated_contour2 = self.rotate_contour(contour2, angle)
            score = matcher.match_contours(contour1, rotated_contour2)

            if score < best_score:
                best_score = score
                best_angle = angle

        return best_angle, best_score