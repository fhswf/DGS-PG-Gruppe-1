#!/usr/bin/env python3
"""
Analyze keypoint confidence thresholds for pose estimation.

This tool helps you understand how different confidence thresholds affect
the number and quality of detected keypoints.

Usage:
    python analyze_thresholds.py <image_path>

Example:
    python analyze_thresholds.py ../data/test_pose.png
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pose_estimator_2d import PoseEstimator2D
except ImportError:
    print("❌ Could not import PoseEstimator2D")
    sys.exit(1)


def analyze_thresholds(image_path, mode='performance'):
    """Analyze keypoint detection at various confidence thresholds."""
    
    print(f"🔍 Analyzing confidence thresholds for: {image_path}")
    print(f"📊 Using model mode: {mode}")
    print("=" * 60)
    
    # Initialize estimator
    estimator = PoseEstimator2D(mode=mode, device='cpu', kpt_threshold=0.0)
    
    # Process image
    result = estimator.process_image(image_path)
    
    if result.num_persons == 0:
        print("❌ No persons detected in image")
        return
    
    scores = result.scores[0]  # First person
    keypoints = result.keypoints[0]
    
    print(f"👥 Detected {result.num_persons} person(s)")
    print(f"📈 Raw confidence range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"📊 Mean confidence: {scores.mean():.3f}")
    print()
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    print("🎯 Keypoint Detection by Confidence Threshold:")
    print("Threshold | Total | Body  | Face  | Hands | Feet  | Avg Conf")
    print("----------|-------|-------|-------|-------|-------|----------")
    
    for thresh in thresholds:
        visible_mask = scores > thresh
        total_visible = np.sum(visible_mask)
        
        # Keypoint group ranges (COCO WholeBody format)
        body_visible = np.sum(visible_mask[:17])      # 0-16: Body
        face_visible = np.sum(visible_mask[17:85])    # 17-84: Face  
        lhand_visible = np.sum(visible_mask[91:112])  # 91-111: Left hand
        rhand_visible = np.sum(visible_mask[112:133]) # 112-132: Right hand
        hands_visible = lhand_visible + rhand_visible
        feet_visible = np.sum(visible_mask[85:91])    # 85-90: Feet
        
        if total_visible > 0:
            avg_conf = np.mean(scores[visible_mask])
        else:
            avg_conf = 0.0
        
        print(f"   {thresh:4.1f}   | {total_visible:3d}/133| {body_visible:2d}/17 | {face_visible:2d}/68 | {hands_visible:2d}/42 | {feet_visible:1d}/6  | {avg_conf:6.3f}")
    
    print()
    
    # Recommend optimal threshold
    print("💡 Threshold Recommendations:")
    
    # Find threshold where we still have good coverage but high confidence
    good_thresholds = []
    for thresh in [0.5, 0.6, 0.7, 0.8]:
        visible = np.sum(scores > thresh)
        avg_conf = np.mean(scores[scores > thresh]) if visible > 0 else 0
        coverage = visible / 133.0
        
        if coverage > 0.6 and avg_conf > thresh + 0.2:  # Good coverage + confident detections
            good_thresholds.append((thresh, visible, avg_conf, coverage))
    
    if good_thresholds:
        # Sort by balance of coverage and confidence
        best_thresh = max(good_thresholds, key=lambda x: x[2] * x[3])  # conf * coverage
        thresh, visible, avg_conf, coverage = best_thresh
        
        print(f"🎯 Recommended threshold: {thresh}")
        print(f"   - Keypoints detected: {visible}/133 ({coverage:.1%})")
        print(f"   - Average confidence: {avg_conf:.3f}")
        print(f"   - Good balance of quality and coverage")
    else:
        print("🔍 Try lower thresholds - image may have challenging conditions")
    
    print()
    
    # Show most/least confident keypoints
    print("🏆 Most Confident Keypoints:")
    top_indices = np.argsort(scores)[-5:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"   {i+1}. Keypoint {idx:3d}: {scores[idx]:.3f}")
    
    print()
    print("⚠️  Least Confident Keypoints:")
    bottom_indices = np.argsort(scores)[:5]
    for i, idx in enumerate(bottom_indices):
        print(f"   {i+1}. Keypoint {idx:3d}: {scores[idx]:.3f}")


def main():
    """Main function for command-line usage."""
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_thresholds.py <image_path> [mode]")
        print("Example: python analyze_thresholds.py ../data/test_pose.png")
        print("Modes: performance, balanced, lightweight")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'performance'
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    try:
        analyze_thresholds(image_path, mode)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
