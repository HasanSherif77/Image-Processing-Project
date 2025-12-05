"""
Enhanced visualization for Milestone 2
"""
from typing import List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import os


class MatchVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_matches(self, pieces: List[Dict], matches: List[Dict],
                          top_n: int = 5) -> None:
        """
        Visualize top N matches

        Args:
            pieces: List of piece dictionaries
            matches: List of match dictionaries
            top_n: Number of top matches to visualize
        """
        # Create piece map
        piece_map = {piece["id"]: piece for piece in pieces}

        # Take top N matches
        top_matches = matches[:min(top_n, len(matches))]

        fig, axes = plt.subplots(top_n, 3, figsize=(15, 5 * top_n))

        if top_n == 1:
            axes = axes.reshape(1, -1)

        for idx, match in enumerate(top_matches):
            piece1 = piece_map[match["piece1"]]
            piece2 = piece_map[match["piece2"]]

            # Plot piece 1
            axes[idx, 0].imshow(cv2.cvtColor(piece1["image"], cv2.COLOR_BGR2RGB))
            axes[idx, 0].set_title(f"Piece {piece1['id']}")
            axes[idx, 0].axis('off')

            # Plot piece 2
            axes[idx, 2].imshow(cv2.cvtColor(piece2["image"], cv2.COLOR_BGR2RGB))
            axes[idx, 2].set_title(f"Piece {piece2['id']}")
            axes[idx, 2].axis('off')

            # Plot middle panel with info
            axes[idx, 1].axis('off')
            info_text = f"Match #{idx + 1}\n"
            info_text += f"Score: {match['score']:.4f}\n"
            if "rotation" in match:
                info_text += f"Rotation: {match['rotation']:.1f}°\n"
            info_text += f"Method: {match.get('method', 'N/A')}"
            axes[idx, 1].text(0.5, 0.5, info_text,
                              ha='center', va='center', fontsize=12,
                              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "top_matches.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_assembly(self, original_img: np.ndarray,
                           assembly_img: np.ndarray,
                           positions: Dict) -> None:
        """
        Visualize assembly result

        Args:
            original_img: Original puzzle image
            assembly_img: Assembled image
            positions: Piece positions dictionary
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Original image
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Puzzle", fontsize=14, weight='bold')
        axes[0].axis('off')

        # Assembly result
        axes[1].imshow(cv2.cvtColor(assembly_img, cv2.COLOR_BGR2RGB))

        # Add piece numbers
        for piece_id, (row, col) in positions.items():
            h, w = assembly_img.shape[:2]
            grid_h, grid_w = h // 8, w // 8  # Assuming 8x8 grid for display
            y = row * grid_h + grid_h // 2
            x = col * grid_w + grid_w // 2
            axes[1].text(x, y, str(piece_id),
                         ha='center', va='center',
                         fontsize=10, weight='bold',
                         color='yellow',
                         bbox=dict(boxstyle="circle", facecolor="red", alpha=0.7))

        axes[1].set_title("Assembly Result", fontsize=14, weight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "assembly_result.png"), dpi=150, bbox_inches='tight')
        plt.close()