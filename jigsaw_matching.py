import os
import cv2
import numpy as np
import json

BASE_OUTDIR = os.path.join(os.getcwd(), "output")
TILES_DIR = os.path.join(BASE_OUTDIR, "tiles")
EDGES_DIR = os.path.join(BASE_OUTDIR, "edges")
MATCH_RESULTS_DIR = os.path.join(BASE_OUTDIR, "matching_results")
VIS_DIR = os.path.join(BASE_OUTDIR, "visualizations_matching")

os.makedirs(MATCH_RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


def load_tiles_and_edges():
    pieces = []
    tile_files = sorted(
        f for f in os.listdir(TILES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    for idx, fname in enumerate(tile_files):
        tile_path = os.path.join(TILES_DIR, fname)
        tile_img = cv2.imread(tile_path)
        edge_name = fname.replace("tile", "edges")
        edge_path = os.path.join(EDGES_DIR, edge_name)
        if not os.path.exists(edge_path):
            edge_img = None
        else:
            edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        pieces.append({
            "id": idx,
            "name": fname,
            "tile": tile_img,
            "edge": edge_img
        })
    return pieces


def build_mask_from_tile(tile, edge_img=None):
    if edge_img is not None:
        _, th = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(th) > 127:
            th = 255 - th
        return th
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = 255 - th
    return th


def get_outer_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]
    return contour


def split_contour_to_edges(contour, tol=0.12):
    xs = contour[:, 0]
    ys = contour[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min
    w = x_max - x_min
    top = []
    bottom = []
    left = []
    right = []
    for (x, y) in contour:
        if abs(y - y_min) < tol * h:
            top.append([x, y])
        elif abs(y - y_max) < tol * h:
            bottom.append([x, y])
        elif abs(x - x_min) < tol * w:
            left.append([x, y])
        elif abs(x - x_max) < tol * w:
            right.append([x, y])
    edges = {}
    for name, pts in zip(["top", "bottom", "left", "right"], [top, bottom, left, right]):
        if len(pts) > 0:
            pts = np.array(pts, dtype=np.float32)
            if name in ["top", "bottom"]:
                order = np.argsort(pts[:, 0])
            else:
                order = np.argsort(pts[:, 1])
            edges[name] = pts[order]
        else:
            edges[name] = None
    return edges


def resample_edge(points, num_samples=100):
    if points is None or len(points) < 2:
        return None
    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cumlen[-1]
    if total_len == 0:
        return None
    target = np.linspace(0, total_len, num_samples)
    new_pts = np.zeros((num_samples, 2), dtype=np.float32)
    for i, t in enumerate(target):
        idx = np.searchsorted(cumlen, t) - 1
        idx = max(0, min(idx, len(points) - 2))
        t0, t1 = cumlen[idx], cumlen[idx + 1]
        alpha = 0 if t1 == t0 else (t - t0) / (t1 - t0)
        new_pts[i] = (1 - alpha) * points[idx] + alpha * points[idx + 1]
    return new_pts


def descriptor_distance_from_chord(edge_pts, num_samples=100):
    pts = resample_edge(edge_pts, num_samples=num_samples)
    if pts is None:
        return None
    p0 = pts[0]
    p1 = pts[-1]
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)
    if chord_len == 0:
        return None
    n = np.array([-chord[1], chord[0]]) / chord_len
    vecs = pts - p0
    dists = vecs @ n
    max_abs = np.max(np.abs(dists)) + 1e-6
    dists = dists / max_abs
    return dists.astype(np.float32)


def build_all_edge_descriptors(pieces, num_samples=100):
    all_edges = []
    for piece in pieces:
        tile = piece["tile"]
        edge_img = piece["edge"]
        if tile is None:
            continue
        mask = build_mask_from_tile(tile, edge_img=edge_img)
        contour = get_outer_contour(mask)
        if contour is None:
            continue
        edges = split_contour_to_edges(contour)
        for side_name, pts in edges.items():
            if pts is None:
                continue
            desc = descriptor_distance_from_chord(pts, num_samples=num_samples)
            if desc is None:
                continue
            all_edges.append({
                "piece_id": piece["id"],
                "piece_name": piece["name"],
                "side": side_name,
                "points": pts,
                "descriptor": desc
            })
    return all_edges


def edge_descriptor_distance(d1, d2):
    if d1 is None or d2 is None:
        return np.inf
    n = min(len(d1), len(d2))
    d1 = d1[:n]
    d2 = d2[:n]
    return float(np.linalg.norm(d1 - d2))


def match_edges(all_edges, top_k=3):
    matches = []
    n = len(all_edges)
    for i in range(n):
        e1 = all_edges[i]
        best = []
        for j in range(n):
            if i == j:
                continue
            e2 = all_edges[j]
            d_norm = edge_descriptor_distance(e1["descriptor"], e2["descriptor"])
            d_rev = edge_descriptor_distance(e1["descriptor"], e2["descriptor"][::-1])
            d_inv = edge_descriptor_distance(e1["descriptor"], -e2["descriptor"])
            d_inv_rev = edge_descriptor_distance(e1["descriptor"], -e2["descriptor"][::-1])
            d = min(d_norm, d_rev, d_inv, d_inv_rev)
            best.append({
                "edge1_index": i,
                "edge2_index": j,
                "distance": d
            })
        best.sort(key=lambda x: x["distance"])
        matches.extend(best[:top_k])
    json_ready = []
    for m in matches:
        e1 = all_edges[m["edge1_index"]]
        e2 = all_edges[m["edge2_index"]]
        json_ready.append({
            "piece1_id": e1["piece_id"],
            "piece1_name": e1["piece_name"],
            "side1": e1["side"],
            "piece2_id": e2["piece_id"],
            "piece2_name": e2["piece_name"],
            "side2": e2["side"],
            "distance": m["distance"]
        })
    out_path = os.path.join(MATCH_RESULTS_DIR, "edge_matches.json")
    with open(out_path, "w") as f:
        json.dump(json_ready, f, indent=2)
    return matches


def visualize_edge_pair(pts1, pts2, out_path):
    canvas_size = 400
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    pts1_center = pts1.mean(axis=0)
    pts2_center = pts2.mean(axis=0)
    shift1 = np.array([canvas_size * 0.3, canvas_size * 0.5]) - pts1_center
    shift2 = np.array([canvas_size * 0.7, canvas_size * 0.5]) - pts2_center
    p1 = (pts1 + shift1).astype(np.int32)
    p2 = (pts2 + shift2).astype(np.int32)
    for i in range(len(p1) - 1):
        cv2.line(canvas, tuple(p1[i]), tuple(p1[i + 1]), (0, 0, 255), 2)
    for i in range(len(p2) - 1):
        cv2.line(canvas, tuple(p2[i]), tuple(p2[i + 1]), (0, 150, 0), 2)
    cv2.imwrite(out_path, canvas)


def main():
    print("Loading tiles...")
    pieces = load_tiles_and_edges()
    print("Pieces loaded:", len(pieces))

    all_edges = build_all_edge_descriptors(pieces, num_samples=120)
    print("Edges with descriptors:", len(all_edges))

    matches = match_edges(all_edges, top_k=3)
    print("Matches computed:", len(matches))

    if matches:
        matches_sorted = sorted(matches, key=lambda x: x["distance"])
        best = matches_sorted[0]
        e1 = all_edges[best["edge1_index"]]
        e2 = all_edges[best["edge2_index"]]
        out_img = os.path.join(VIS_DIR, "best_match_edges.png")
        visualize_edge_pair(e1["points"], e2["points"], out_img)
        print("Best match image saved at:", out_img)
    else:
        print("No matches found")


if __name__ == "_main_":
    main()