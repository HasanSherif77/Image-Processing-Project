"""
Advanced contour matching for jigsaw puzzle pieces
Implements rotation-invariant shape matching algorithms
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import json


class ContourMatcher:
    def __init__(self, method="hu_moments"):
        """
        Initialize contour matcher

        Args:
            method: Matching method ('hu_moments', 'fourier', 'shape_context')
        """
        self.method = method
        self.matches = []

    def compute_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        """Compute Hu moments for a contour"""
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)

        # Apply logarithmic transformation for better scale invariance
        for i in range(7):
            if hu_moments[i] != 0:
                hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))

        return hu_moments.flatten()

    def compute_fourier_descriptors(self, contour: np.ndarray, num_descriptors: int = 20) -> np.ndarray:
        """Compute Fourier descriptors for rotation invariance"""
        # Convert contour to complex numbers
        complex_contour = contour[:, 0, 0] + 1j * contour[:, 0, 1]

        # Apply Fourier Transform
        fft_result = np.fft.fft(complex_contour)

        # Take magnitude (rotation invariant)
        magnitudes = np.abs(fft_result)

        # Normalize by DC component
        if magnitudes[0] != 0:
            magnitudes = magnitudes / magnitudes[0]

        # Return first N descriptors (excluding DC)
        return magnitudes[1:num_descriptors + 1]

    def match_contours(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """
        Match two contours using selected method

        Returns:
            match_score: Lower is better
        """
        if self.method == "hu_moments":
            hu1 = self.compute_hu_moments(contour1)
            hu2 = self.compute_hu_moments(contour2)
            score = np.sum(np.abs(hu1 - hu2))

        elif self.method == "fourier":
            fd1 = self.compute_fourier_descriptors(contour1)
            fd2 = self.compute_fourier_descriptors(contour2)
            score = np.sum(np.abs(fd1 - fd2))

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return score

    def find_best_matches(self, pieces: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """
        Find best matches among all puzzle pieces

        Args:
            pieces: List of piece dictionaries with 'contour' key
            threshold: Match threshold (lower = stricter)

        Returns:
            List of match dictionaries
        """
        matches = []
        n_pieces = len(pieces)

        for i in range(n_pieces):
            for j in range(i + 1, n_pieces):
                score = self.match_contours(pieces[i]["contour"], pieces[j]["contour"])

                if score < threshold:
                    matches.append({
                        "piece1": pieces[i]["id"],
                        "piece2": pieces[j]["id"],
                        "score": float(score),
                        "method": self.method
                    })

        # Sort by score (best matches first)
        matches.sort(key=lambda x: x["score"])
        self.matches = matches

        return matches