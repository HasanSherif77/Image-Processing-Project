"""
Puzzle assembly using contour matches
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class PuzzleAssembler:
    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        self.positions = {}  # piece_id -> (row, col)
        self.assembly_grid = None

    def build_adjacency_graph(self, matches: List[Dict]) -> Dict:
        """
        Build adjacency graph from matches

        Returns:
            adjacency: {piece_id: [(neighbor_id, score, rotation), ...]}
        """
        adjacency = defaultdict(list)

        for match in matches:
            piece1 = match["piece1"]
            piece2 = match["piece2"]
            score = match["score"]
            rotation = match.get("rotation", 0)

            adjacency[piece1].append((piece2, score, rotation))
            adjacency[piece2].append((piece1, score, rotation))

        # Sort neighbors by match score (best first)
        for piece_id in adjacency:
            adjacency[piece_id].sort(key=lambda x: x[1])

        return adjacency

    def greedy_assemble(self, adjacency: Dict, start_piece: int = 0) -> Dict:
        """
        Greedy assembly starting from a given piece

        Returns:
            positions: {piece_id: (row, col)}
        """
        positions = {}
        placed = set()

        # Start with specified piece at center
        center = self.grid_size // 2
        positions[start_piece] = (center, center)
        placed.add(start_piece)

        # BFS queue
        queue = [start_piece]

        while queue and len(placed) < min(len(adjacency), self.grid_size * self.grid_size):
            current = queue.pop(0)
            current_row, current_col = positions[current]

            # Get best matches for current piece
            for neighbor, score, rotation in adjacency[current][:4]:  # Top 4 matches
                if neighbor in placed:
                    continue

                # Try to place in adjacent position
                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
                placed_successfully = False

                for dr, dc in directions:
                    new_row, new_col = current_row + dr, current_col + dc

                    # Check bounds
                    if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                        # Check if position is free
                        position_free = True
                        for pos in positions.values():
                            if pos == (new_row, new_col):
                                position_free = False
                                break

                        if position_free:
                            positions[neighbor] = (new_row, new_col)
                            placed.add(neighbor)
                            queue.append(neighbor)
                            placed_successfully = True
                            break

                if placed_successfully:
                    break

        self.positions = positions
        return positions

    def create_assembly_image(self, pieces: List[Dict], tile_size: Tuple[int, int]) -> np.ndarray:
        """
        Create visualization of assembled puzzle

        Args:
            pieces: List of piece dictionaries with 'image' and 'id'
            tile_size: Size of each tile (height, width)

        Returns:
            Assembled image
        """
        h, w = tile_size
        assembly_h = self.grid_size * h
        assembly_w = self.grid_size * w

        assembly_img = np.zeros((assembly_h, assembly_w, 3), dtype=np.uint8)
        assembly_img.fill(200)  # Gray background

        # Create mapping from piece_id to piece data
        piece_map = {piece["id"]: piece for piece in pieces}

        # Place pieces according to positions
        for piece_id, (row, col) in self.positions.items():
            if piece_id in piece_map:
                piece_img = piece_map[piece_id]["image"]
                piece_h, piece_w = piece_img.shape[:2]

                # Calculate position in assembly
                y_start = row * h
                y_end = y_start + min(piece_h, h)
                x_start = col * w
                x_end = x_start + min(piece_w, w)

                assembly_img[y_start:y_end, x_start:x_end] = piece_img[:min(piece_h, h), :min(piece_w, w)]

        return assembly_img