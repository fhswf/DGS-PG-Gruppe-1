#!/usr/bin/env python3
"""
Simple command-line tool for pose estimation on images.

Usage:
    python simple_image_pose.py <image_path> [options]

Options:
    --output <path>     Save annotated image to specified path
    --no-image         Don't save annotated image (data only)
    --json <path>      Save keypoints as JSON
    --threshold <val>  Keypoint confidence threshold (default: 0.3)

Examples:
    python simple_image_pose.py ../data/test_pose.png
    python simple_image_pose.py ../data/test_pose.png --output result.png
    python simple_image_pose.py ../data/test_pose.png --no-image --json keypoints.json
    python simple_image_pose.py ../data/test_pose.png --threshold 0.5
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pose_estimator_2d import PoseEstimator2D
except ImportError:
    print("❌ Could not import PoseEstimator2D")
    sys.exit(1)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pose estimation for single images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_image_pose.py ../data/test_pose.png
  python simple_image_pose.py ../data/test_pose.png --output result.png
  python simple_image_pose.py ../data/test_pose.png --no-image --json keypoints.json
  python simple_image_pose.py ../data/test_pose.png --threshold 0.5
        """
    )
    
    parser.add_argument(
        'image_path',
        help='Path to input image'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Path for annotated output image'
    )
    
    parser.add_argument(
        '--no-image',
        action='store_true',
        help="Don't save annotated image (data only)"
    )
    
    parser.add_argument(
        '--json', '-j',
        help='Path to save keypoints as JSON'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.8,  # Erhöht auf 0.8 für hohe Qualität
        help='Keypoint confidence threshold (default: 0.8)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps'],
        default='cpu',
        help='Device to use for inference (default: cpu)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['performance', 'balanced', 'lightweight'],
        default='performance',  # Changed from 'balanced' to 'performance'
        help='Model mode (default: performance)'
    )
    
    return parser.parse_args()


def main():
    """Main function for command-line usage."""
    
    args = parse_arguments()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"❌ Input image not found: {args.image_path}")
        sys.exit(1)
    
    # Determine output settings
    save_image = not args.no_image
    output_path = None
    
    if save_image:
        if args.output:
            output_path = args.output
        else:
            # Generate default output path
            input_file = Path(args.image_path)
            output_dir = Path("../output/pose-estimation")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"annotated_{input_file.name}"
    
    # Print configuration
    print(f"🎯 Input: {args.image_path}")
    if save_image:
        print(f"💾 Image Output: {output_path}")
    else:
        print("📊 Image Output: Disabled (data only)")
    
    if args.json:
        print(f"📋 JSON Output: {args.json}")
    
    print(f"⚙️  Device: {args.device}, Mode: {args.mode}, Threshold: {args.threshold}")
    
    try:
        # Initialize estimator
        estimator = PoseEstimator2D(
            mode=args.mode,
            device=args.device,
            kpt_threshold=args.threshold
        )
        
        # Process image
        if save_image:
            result = estimator.process_image_with_annotation(
                image_path=args.image_path,
                output_path=output_path,
                draw_bbox=True,
                draw_keypoints=True,
                keypoint_threshold=args.threshold
            )
        else:
            result = estimator.process_image(args.image_path)
        
        # Save JSON if requested
        if args.json:
            estimator.export_to_json(result, args.json)
            print(f"💾 JSON saved to: {args.json}")
        
        # Print summary
        if result.num_persons > 0:
            print(f"\n✅ Successfully detected {result.num_persons} person(s)")
            for i in range(result.num_persons):
                visible_kpts = sum(result.scores[i] > args.threshold)
                avg_conf = result.scores[i][result.scores[i] > args.threshold].mean() if visible_kpts > 0 else 0
                
                print(f"   Person {i+1}: {visible_kpts}/133 keypoints, avg confidence: {avg_conf:.3f}")
                
                # Show keypoint breakdown with improved thresholds
                body_kpts = sum(result.scores[i][:17] > args.threshold)
                face_kpts = sum(result.scores[i][17:85] > args.threshold) 
                hand_kpts = sum(result.scores[i][91:] > args.threshold)
                
                print(f"             Body: {body_kpts}/17, Face: {face_kpts}/68, Hands: {hand_kpts}/42")
                
                # Quality analysis
                high_conf = sum(result.scores[i] > 0.8)
                medium_conf = sum((result.scores[i] > 0.5) & (result.scores[i] <= 0.8))
                low_conf = sum((result.scores[i] > args.threshold) & (result.scores[i] <= 0.5))
                
                print(f"             Quality: {high_conf} high (>0.8), {medium_conf} medium (0.5-0.8), {low_conf} low ({args.threshold}-0.5)")
        else:
            print("\n❌ No persons detected")
            
        print(f"\n🏁 Processing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
