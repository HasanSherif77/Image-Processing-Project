#!/usr/bin/env python3
"""
Milestone 2 Demo: Jigsaw Puzzle Edge Matching
Computer Vision Course Project

This script demonstrates the complete Milestone 2 edge matching pipeline.
It shows how to:
1. Process puzzle pieces from Milestone 1 outputs
2. Extract shape features using classical computer vision
3. Find potential matches between puzzle edges
4. Visualize matching results
5. Generate assembly suggestions

Usage:
    python milestone2_demo.py [output_dir] [grid_size]

Example:
    python milestone2_demo.py output 4

Requirements:
- Milestone 1 outputs in the specified directory
- jigsaw_matcher.py in the same directory
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Import our edge matching module
try:
    from jigsaw_matcher import JigsawMatcher, run_milestone2_pipeline, demonstrate_milestone2
    print("✓ Successfully imported jigsaw_matcher module")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure jigsaw_matcher.py is in the same directory.")
    sys.exit(1)


def run_interactive_demo(output_dir="output", grid_size=4):
    """
    Run an interactive demonstration of Milestone 2 functionality.

    This demo shows step-by-step how the edge matching works and provides
    detailed output for analysis.
    """
    print("🎯 INTERACTIVE MILESTONE 2 DEMONSTRATION")
    print("=" * 60)

    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"❌ Error: Output directory '{output_dir}' not found.")
        print("Please run Milestone 1 first to generate puzzle piece data.")
        print("\nExpected directory structure:")
        print(f"  {output_dir}/")
        print("  ├── tiles/tile_0.png, tile_1.png, ...")
        print("  ├── edges/edges_0.png, edges_1.png, ...")
        print("  ├── contours/contour_0.png, contour_1.png, ...")
        print("  └── final_image/*.jpg")
        return False

    try:
        # Initialize matcher
        print(f"🔧 Initializing matcher with output directory: {output_dir}")
        matcher = JigsawMatcher(output_dir)

        # Step 1: Load and process pieces
        print("\n📥 STEP 1: Loading and processing puzzle pieces")
        print("-" * 50)
        num_pieces = matcher.load_and_process_all_pieces()

        if num_pieces == 0:
            print("❌ No pieces were processed. Check your Milestone 1 outputs.")
            return False

        # Step 2: Analyze edge types
        print("\n📊 STEP 2: Analyzing edge type distribution")
        print("-" * 50)
        edge_analysis = matcher.analyze_edge_types()

        # Step 3: Demonstrate feature extraction
        print("\n🔬 STEP 3: Demonstrating feature extraction")
        print("-" * 50)

        # Show features for first piece
        first_piece_id = list(matcher.edge_features.keys())[0]
        features = matcher.edge_features[first_piece_id]

        print(f"Example features for piece {first_piece_id}:")
        print(f"  - Edge type: {features['edge_type']}")
        print(f"  - Area: {features['area']:.0f} pixels")
        print(f"  - Perimeter: {features['perimeter']:.0f} pixels")
        print(f"  - Compactness: {features['compactness']:.3f}")
        print(f"  - Fourier descriptors: {len(features['fourier_descriptors'])} values")
        print(f"  - Hu moments: {len(features['hu_moments'])} values")
        print(f"  - Curvature features: {len(features['curvature_features'])} values")

        # Step 4: Find matches for a piece
        print("\n🎯 STEP 4: Finding potential matches")
        print("-" * 50)

        target_piece = first_piece_id
        print(f"Finding matches for piece {target_piece}...")

        matches = matcher.find_best_matches(target_piece, num_matches=5)

        if matches:
            print(f"\nTop {len(matches)} matches for piece {target_piece}:")
            print("Rank | Piece ID | Distance | Edge Type")
            print("-" * 40)

            for rank, (match_id, distance, details) in enumerate(matches, 1):
                match_features = matcher.edge_features[match_id]
                edge_type = match_features['edge_type']
                print("4d")

            # Show detailed breakdown for best match
            best_match_id, best_distance, best_details = matches[0]
            print(f"\nDetailed breakdown for best match (Piece {best_match_id}):")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")
        # Step 5: Create visualizations
        print("\n🎨 STEP 5: Creating visualizations")
        print("-" * 50)

        viz_dir = os.path.join(output_dir, "match_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Create visualization for best match
        if matches:
            best_match_id = matches[0][0]
            viz_path = os.path.join(viz_dir, f"demo_match_{target_piece}_{best_match_id}.png")
            print(f"Creating visualization: {viz_path}")

            visualization = matcher.create_match_visualization(target_piece, best_match_id, viz_path)

            if visualization is not None:
                print("✓ Visualization created successfully")
            else:
                print("❌ Failed to create visualization")

        # Step 6: Generate assembly suggestions
        print("\n🧩 STEP 6: Generating assembly suggestions")
        print("-" * 50)

        assembly_suggestions = matcher.create_assembly_suggestions(grid_size)

        if assembly_suggestions:
            print(f"✓ Created assembly suggestions for {len(assembly_suggestions)} pieces")

            # Show summary statistics
            total_suggestions = sum(len(data['matches']) for data in assembly_suggestions.values())
            avg_distance = sum(match[1] for data in assembly_suggestions.values()
                             for match in data['matches']) / total_suggestions

            print(".3f")
            print(".3f")
        # Step 7: Save features for analysis
        print("\n💾 STEP 7: Saving feature data")
        print("-" * 50)

        matcher.save_features_to_json()

        # Summary
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("Summary of results:")
        print(f"  • Processed {num_pieces} puzzle pieces")
        print(f"  • Edge types: {', '.join(edge_analysis.keys()) if edge_analysis else 'None'}")
        print(f"  • Generated {len(matches) if matches else 0} matches for demonstration piece")
        print(f"  • Created {1 if matches else 0} visualization(s)")
        print(f"  • Saved feature data to JSON")
        print(f"  • Output directory: {output_dir}")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_comparison_report(output_dir="output"):
    """
    Create a detailed comparison report showing how different features contribute to matching.
    """
    print("📊 Creating detailed comparison report...")

    try:
        matcher = JigsawMatcher(output_dir)
        num_pieces = matcher.load_and_process_all_pieces()

        if num_pieces < 2:
            print("Need at least 2 pieces for comparison report")
            return

        # Compare first two pieces in detail
        piece1_id = list(matcher.edge_features.keys())[0]
        piece2_id = list(matcher.edge_features.keys())[1]

        features1 = matcher.edge_features[piece1_id]
        features2 = matcher.edge_features[piece2_id]

        distance, details = matcher.calculate_feature_distance(features1, features2)

        print(f"\nDetailed comparison between pieces {piece1_id} and {piece2_id}:")
        print(".4f")
        print(f"  - Fourier descriptors:   {details['fourier']:.4f}")
        print(f"  - Hu moments:           {details['hu_moments']:.4f}")
        print(f"  - Curvature features:   {details['curvature']:.4f}")
        print(f"  - Geometric features:   {details['geometric']:.4f}")

        # Show which features are most similar/different
        similarities = {
            'fourier': 1 - details['fourier'],  # Convert distance to similarity
            'hu_moments': 1 / (1 + details['hu_moments']),
            'curvature': 1 / (1 + details['curvature']),
            'geometric': 1 / (1 + details['geometric'])
        }

        best_feature = max(similarities, key=similarities.get)
        worst_feature = min(similarities, key=similarities.get)

        print(f"\nFeature similarity analysis:")
        print(".3f")
        print(".3f")
        print(f"  → {'Strong' if similarities[best_feature] > 0.8 else 'Moderate' if similarities[best_feature] > 0.6 else 'Weak'} match based on {best_feature}")

    except Exception as e:
        print(f"Failed to create comparison report: {str(e)}")


def main():
    """Main demo function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Jigsaw Puzzle Edge Matching Demo - Milestone 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python milestone2_demo.py                    # Use default settings
  python milestone2_demo.py -d output -g 4     # Specify output dir and grid size
  python milestone2_demo.py --compare         # Include detailed comparison report
        """
    )

    parser.add_argument('-d', '--output-dir',
                       default='output',
                       help='Directory containing Milestone 1 outputs (default: output)')
    parser.add_argument('-g', '--grid-size',
                       type=int, default=4, choices=[2, 4, 8],
                       help='Puzzle grid size (default: 4)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate detailed feature comparison report')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick automated demo instead of interactive')

    args = parser.parse_args()

    print("🧩 Jigsaw Puzzle Edge Matching Demo")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print("=" * 50)

    if args.quick:
        # Run the automated demo
        demonstrate_milestone2(grid_size=args.grid_size)
    else:
        # Run interactive demo
        success = run_interactive_demo(args.output_dir, args.grid_size)

        if success and args.compare:
            create_comparison_report(args.output_dir)

    if not args.quick:
        print("\n💡 Tips:")
        print("  • Check the 'match_visualizations' folder for visual results")
        print("  • Review 'edge_features.json' for detailed feature data")
        print("  • Try different pieces by modifying the target_piece variable")
        print("  • Use --compare flag for detailed feature analysis")


if __name__ == "__main__":
    main()
