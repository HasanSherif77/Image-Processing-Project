"""
Main script to run Milestone 2 pipeline
"""
import os
import cv2
import json
from Milestone_1.jigsaw_pipeline import run_pipeline
from Milestone_2.contour_matcher import ContourMatcher
from Milestone_2.rotation_invariant import RotationInvariant
from Milestone_2.puzzle_assembler import PuzzleAssembler
from Milestone_2.visualization import MatchVisualizer


def run_complete_milestone2(image_path, output_dir, grid_size=4):
    """
    Run complete Milestone 2 pipeline

    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("RUNNING MILESTONE 2 - COMPLETE PIPELINE")
    print("=" * 60)

    # Step 1: Run Milestone 1 pipeline
    print("\n1. Running Milestone 1 pipeline...")
    final_img, contour_img, best_result = run_pipeline(
        image_path=image_path,
        output_dir=output_dir,
        grid_size=grid_size,
        apply_noise_reduction=True,
        apply_sharpening=True,
        apply_gamma_correction_option=True,
        apply_clahe_option=True,
        clean_output=True
    )

    # Step 2: Load extracted pieces
    print("\n2. Loading extracted pieces...")
    tiles_dir = os.path.join(output_dir, "tiles")
    pieces = []

    tile_files = sorted([f for f in os.listdir(tiles_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for idx, filename in enumerate(tile_files):
        tile_path = os.path.join(tiles_dir, filename)
        tile_img = cv2.imread(tile_path)

        if tile_img is not None:
            pieces.append({
                "id": idx,
                "filename": filename,
                "image": tile_img
            })

    print(f"   Loaded {len(pieces)} puzzle pieces")

    # Step 3: Extract contours from pieces
    print("\n3. Extracting contours from pieces...")
    contours = []

    for piece in pieces:
        gray = cv2.cvtColor(piece["image"], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        piece_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if piece_contours:
            largest_contour = max(piece_contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            contours.append({
                "piece_id": piece["id"],
                "contour": simplified_contour,
                "area": cv2.contourArea(simplified_contour)
            })

    print(f"   Extracted contours from {len(contours)} pieces")

    # Step 4: Match contours with rotation invariance
    print("\n4. Matching contours with rotation invariance...")
    matcher = ContourMatcher(method="hu_moments")
    rotator = RotationInvariant(rotation_steps=12)

    matches = []
    threshold = 0.5

    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            # Skip if area difference is too large
            area_ratio = min(contours[i]["area"], contours[j]["area"]) / max(contours[i]["area"], contours[j]["area"])
            if area_ratio < 0.5:
                continue

            # Find best rotation match
            best_angle, best_score = rotator.find_best_rotation_match(
                contours[i]["contour"],
                contours[j]["contour"],
                matcher
            )

            if best_score < threshold:
                matches.append({
                    "piece1": contours[i]["piece_id"],
                    "piece2": contours[j]["piece_id"],
                    "score": best_score,
                    "rotation": best_angle,
                    "method": "hu_moments_rotated"
                })

    # Sort matches by score
    matches.sort(key=lambda x: x["score"])
    print(f"   Found {len(matches)} potential matches")

    # Step 5: Assemble puzzle
    print("\n5. Assembling puzzle...")
    assembler = PuzzleAssembler(grid_size=grid_size)
    adjacency = assembler.build_adjacency_graph(matches)
    positions = assembler.greedy_assemble(adjacency, start_piece=0)

    # Create assembly image
    if pieces:
        tile_h, tile_w = pieces[0]["image"].shape[:2]
        assembly_img = assembler.create_assembly_image(pieces, (tile_h, tile_w))

        # Save assembly result
        assembly_path = os.path.join(output_dir, "assembly_result.jpg")
        cv2.imwrite(assembly_path, assembly_img)
        print(f"   Saved assembly result: {assembly_path}")

    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    visualizer = MatchVisualizer(output_dir)
    visualizer.visualize_matches(pieces, matches, top_n=5)
    visualizer.visualize_assembly(final_img, assembly_img, positions)

    # Step 7: Save results
    print("\n7. Saving results...")
    results = {
        "image_path": image_path,
        "grid_size": grid_size,
        "num_pieces": len(pieces),
        "num_contours": len(contours),
        "num_matches": len(matches),
        "assembly_positions": positions,
        "top_matches": matches[:10]
    }

    results_path = os.path.join(output_dir, "milestone2_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"   Saved results: {results_path}")

    print("\n" + "=" * 60)
    print("MILESTONE 2 PIPELINE COMPLETE!")
    print("=" * 60)

    return {
        "pieces": pieces,
        "contours": contours,
        "matches": matches,
        "assembly": assembly_img,
        "positions": positions
    }


if __name__ == "__main__":
    # Example usage
    image_path = "C:\men3em\semester 5\Image Processing\Gravity Falls\puzzle_2x2/1.jpg"
    output_dir = "milestone2_output"

    results = run_complete_milestone2(image_path, output_dir, grid_size=4)