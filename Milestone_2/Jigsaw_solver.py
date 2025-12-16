# Jigsaw_Solver.py
import cv2
import numpy as np
import os

class JigsawSolver:
    def __init__(self, tiles_dir, grid_size=4, R=32, beam_width=20):
        self.tiles_dir = tiles_dir
        self.grid_size = grid_size
        self.R = R
        self.beam_width = beam_width

        self.tiles_original = []
        self.downsampled_tiles = []
        self.tile_edges_rotations = {}
        self.solution = None
        self.total_cost = None
        self.reconstructed_img = None

    # --------------------------
    # Load tiles
    # --------------------------
    def load_tiles(self):
        images = []
        for f in sorted(os.listdir(self.tiles_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(self.tiles_dir, f))
                if img is not None:
                    images.append(img)
        self.tiles_original = images
        return images

    # --------------------------
    # Utilities
    # --------------------------
    def downsample_images(self):
        self.downsampled_tiles = [
            cv2.resize(img, (self.R, self.R), interpolation=cv2.INTER_AREA)
            for img in self.tiles_original
        ]

    def extract_edges(self, tile):
        h, w, _ = tile.shape
        return {
            "top": tile[0, :, :],
            "bottom": tile[h-1, :, :],
            "left": tile[:, 0, :],
            "right": tile[:, w-1, :]
        }

    def rotate_tile(self, tile, angle):
        if angle == 0:
            return tile
        if angle == 90:
            return cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(tile, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def edge_cost(self, e1, e2):
        return np.mean(np.abs(e1.astype(np.float32) - e2.astype(np.float32)))

    # --------------------------
    # SOLVER
    # --------------------------
    def solve(self):
        self.load_tiles()
        self.downsample_images()

        N = len(self.tiles_original)
        G = self.grid_size

        # Precompute edges
        for idx, tile in enumerate(self.downsampled_tiles):
            self.tile_edges_rotations[idx] = {}
            for angle in [0, 90, 180, 270]:
                self.tile_edges_rotations[idx][angle] = self.extract_edges(
                    self.rotate_tile(tile, angle)
                )

        class State:
            def __init__(self, grid, used, cost):
                self.grid = grid
                self.used = used
                self.cost = cost

        beam = [State([], set(), 0.0)]

        for pos in range(G * G):
            r, c = divmod(pos, G)
            new_beam = []

            for state in beam:
                for t in range(N):
                    if t in state.used:
                        continue
                    for rot in [0, 90, 180, 270]:
                        cost = 0
                        if c > 0:
                            li, lr = state.grid[-1]
                            cost += self.edge_cost(
                                self.tile_edges_rotations[li][lr]["right"],
                                self.tile_edges_rotations[t][rot]["left"]
                            )
                        if r > 0:
                            ti, tr = state.grid[(r-1)*G + c]
                            cost += self.edge_cost(
                                self.tile_edges_rotations[ti][tr]["bottom"],
                                self.tile_edges_rotations[t][rot]["top"]
                            )

                        new_beam.append(
                            State(
                                state.grid + [(t, rot)],
                                state.used | {t},
                                state.cost + cost
                            )
                        )

            new_beam.sort(key=lambda s: s.cost)
            beam = new_beam[:self.beam_width]

        best = min(beam, key=lambda s: s.cost)
        self.solution = best.grid
        self.total_cost = best.cost

        self.reconstruct()
        return self.solution, self.total_cost

    # --------------------------
    # Reconstruct image
    # --------------------------
    def reconstruct(self):
        tile_h, tile_w, _ = self.tiles_original[0].shape
        G = self.grid_size

        img = np.zeros((G*tile_h, G*tile_w, 3), dtype=np.uint8)
        for idx, (t, rot) in enumerate(self.solution):
            r, c = divmod(idx, G)
            img[r*tile_h:(r+1)*tile_h,
                c*tile_w:(c+1)*tile_w] = self.rotate_tile(
                    self.tiles_original[t], rot
                )

        self.reconstructed_img = img
        return img
