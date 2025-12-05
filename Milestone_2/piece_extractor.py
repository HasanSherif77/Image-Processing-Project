# piece_extractor.py
# Step 6: Feature Extraction + Visualizations for existing puzzle tiles

import cv2
import numpy as np
import os
import json
from pathlib import Path
from matplotlib import pyplot as plt

# -----------------------
# CONFIG
# -----------------------
INPUT_FOLDER = "C:/Users\mmmsa\PycharmProjects\Image-Processing-Project\Milestone_1\output/tiles"  # folder containing tiles
OUTPUT_FOLDER = "Milestone_2/output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "visualizations"), exist_ok=True)

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def avg_color(image):
    return image.mean(axis=(0, 1)).tolist()

def color_histogram(image, bins=256):
    hist = []
    for i in range(3):  # B, G, R channels
        h = cv2.calcHist([image], [i], None, [bins], [0, 256])
        h = (h / h.sum()).flatten()  # normalize
        hist.extend(h.tolist())
    return hist

def borders_rgb(image):
    h, w = image.shape[:2]
    return {
        "top": image[0].tolist(),
        "bottom": image[h - 1].tolist(),
        "left": image[:, 0].tolist(),
        "right": image[:, w - 1].tolist()
    }

def detect_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    keypoints = sorted(keypoints, key=lambda k: -k.response)[:50]  # top 50 keypoints
    pts = [ [int(k.pt[0]), int(k.pt[1])] for k in keypoints ]
    return pts

def visualize_piece(piece_id, image, features):
    vis_dir = os.path.join(OUTPUT_FOLDER, "visualizations")

    # 1. Average color patch
    avg_patch = np.full((50, 50, 3), features["avg_color"], dtype=np.uint8)
    cv2.imwrite(os.path.join(vis_dir, f"{piece_id}_avg_color.png"), avg_patch)

    # 2. Histogram
    plt.figure(figsize=(6,3))
    colors = ('b', 'g', 'r')
    bins = 256
    for i, col in enumerate(colors):
        plt.plot(features["histogram"][i*bins:(i+1)*bins], color=col)
    plt.title(f"Piece {piece_id} Histogram")
    plt.xlabel("Bin")
    plt.ylabel("Normalized Count")
    plt.savefig(os.path.join(vis_dir, f"{piece_id}_hist.png"))
    plt.close()

    # 3. Borders
    borders = features["borders"]
    for side in ["top", "bottom", "left", "right"]:
        b = np.array(borders[side], dtype=np.uint8)
        if side in ["top", "bottom"]:
            b = b.reshape(1, -1, 3)
        else:
            b = b.reshape(-1, 1, 3)
        cv2.imwrite(os.path.join(vis_dir, f"{piece_id}_border_{side}.png"), b)

    # 4. Keypoints overlay
    kp_image = image.copy()
    for pt in features["keypoints"]:
        cv2.circle(kp_image, tuple(pt), 3, (0,255,0), -1)
    cv2.imwrite(os.path.join(vis_dir, f"{piece_id}_keypoints.png"), kp_image)

# -----------------------
# MAIN LOOP
# -----------------------

tiles = sorted(Path(INPUT_FOLDER).glob("*.*"))  # all image files
metadata = []

for piece_id, tile_path in enumerate(tiles):
    img = cv2.imread(str(tile_path))
    if img is None:
        print(f"Warning: failed to load {tile_path}")
        continue

    features = {
        "piece_id": piece_id,
        "width": img.shape[1],
        "height": img.shape[0],
        "avg_color": avg_color(img),
        "histogram": color_histogram(img),
        "borders": borders_rgb(img),
        "keypoints": detect_keypoints(img)
    }

    metadata.append(features)
    visualize_piece(piece_id, img, features)
    print(f"Processed piece {piece_id} ({tile_path.name})")

# Save metadata JSON
with open(os.path.join(OUTPUT_FOLDER, "tiles_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ All pieces processed and visualizations saved.")
