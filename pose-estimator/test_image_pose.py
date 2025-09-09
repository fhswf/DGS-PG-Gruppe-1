#!/usr/bin/env python3
"""
Test script for processing single images with pose estimation.

This script extends the PoseEstimator2D functionality to process individual images
and save annotated results to the output folder.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add the pose-estimator directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import from the working parts of pose_estimator_2d
    import sys
    sys.path.insert(0, '.')
    
    # Import directly from RTMLib
    from rtmlib import Wholebody, draw_skeleton
    import cv2
    import numpy as np
    from pathlib import Path
    from dataclasses import dataclass
    
    # Define PoseResult locally to avoid import issues
    @dataclass
    class PoseResult:
        frame_idx: int
        keypoints: np.ndarray
        scores: np.ndarray  
        bboxes: np.ndarray
        num_persons: int
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure RTMLib is installed: pip install rtmlib")
    sys.exit(1)


class ImagePoseEstimator:
    """Standalone image pose estimator using RTMLib."""
    
    def __init__(
        self,
        mode: str = 'balanced',
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        kpt_threshold: float = 0.3,
        to_openpose: bool = False
    ):
        """Initialize the pose estimator."""
        self.mode = mode
        self.backend = backend
        self.device = device
        self.kpt_threshold = kpt_threshold
        self.to_openpose = to_openpose
        
        # Initialize RTMLib Wholebody model
        self.wholebody = Wholebody(
            mode=mode,
            backend=backend,
            device=device,
            to_openpose=to_openpose
        )
        print(f"✅ Pose estimator initialized: mode={mode}, device={device}")
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> PoseResult:
        """Process a single frame and extract pose keypoints."""
        try:
            # Run detection to get person bounding boxes
            bboxes = self.wholebody.det_model(frame)
            
            # Run pose estimation with detected bboxes
            keypoints, scores = self.wholebody.pose_model(frame, bboxes=bboxes)
            
            # Handle case where no persons are detected
            if keypoints is None or len(keypoints) == 0:
                keypoints = np.empty((0, 133, 2))
                scores = np.empty((0, 133))
                bboxes = np.empty((0, 5))
                num_persons = 0
            else:
                num_persons = keypoints.shape[0]
            
            return PoseResult(
                frame_idx=frame_idx,
                keypoints=keypoints,
                scores=scores,
                bboxes=bboxes,
                num_persons=num_persons
            )
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return PoseResult(
                frame_idx=frame_idx,
                keypoints=np.empty((0, 133, 2)),
                scores=np.empty((0, 133)),
                bboxes=np.empty((0, 5)),
                num_persons=0
            )
    
    def process_single_image(
        self,
        image_path: str,
        output_path: str = None,
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        keypoint_threshold: float = 0.3
    ) -> PoseResult:
        """
        Process a single image and optionally save annotated result.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            draw_bbox: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoints and skeleton
            keypoint_threshold: Minimum confidence for drawing keypoints
            
        Returns:
            PoseResult object with pose estimation results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        print(f"🖼️  Processing image: {image_path.name}")
        print(f"📐 Image dimensions: {frame.shape}")
        
        # Process the image using the frame processing method
        result = self._process_frame(frame, frame_idx=0)
        
        print(f"👥 Detected {result.num_persons} person(s)")
        
        # Create annotated image if output path is provided
        if output_path:
            annotated_frame = self._create_annotated_image(
                frame, result, draw_bbox, draw_keypoints, keypoint_threshold
            )
            
            # Save annotated image
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"💾 Annotated image saved: {output_path}")
        
        return result
    
    def _create_annotated_image(
        self,
        image: np.ndarray,
        result: PoseResult,
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        threshold: float = 0.3
    ) -> np.ndarray:
        """Create annotated image with pose estimation results."""
        annotated = image.copy()
        
        if result.num_persons == 0:
            return annotated
        
        # Draw bounding boxes
        if draw_bbox and len(result.bboxes) > 0:
            for i, bbox in enumerate(result.bboxes):
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    conf = bbox[4] if len(bbox) > 4 else 1.0
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence score
                    cv2.putText(
                        annotated, f'Person {i+1}: {conf:.2f}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1
                    )
        
        # Draw keypoints and skeleton
        if draw_keypoints:
            try:
                # Use RTMLib's draw_skeleton function
                annotated = draw_skeleton(
                    annotated,
                    result.keypoints,
                    result.scores,
                    kpt_thr=threshold,
                    openpose_skeleton=self.to_openpose
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not use RTMLib draw_skeleton: {e}")
                print("🎨 Using fallback keypoint drawing...")
                self._draw_keypoints_fallback(annotated, result, threshold)
        
        return annotated
    
    def _draw_keypoints_fallback(
        self,
        image: np.ndarray,
        result: PoseResult,
        threshold: float = 0.3
    ):
        """Fallback method to draw keypoints as colored circles."""
        # Define colors for different keypoint groups
        keypoint_colors = {
            'body': (0, 255, 0),      # Green for body (0-16)
            'face': (255, 0, 0),      # Blue for face (17-84)  
            'left_hand': (0, 0, 255), # Red for left hand (85-105)
            'right_hand': (255, 255, 0), # Cyan for right hand (106-126)
            'feet': (255, 0, 255)     # Magenta for feet (127-132)
        }
        
        for person_idx in range(result.num_persons):
            keypoints = result.keypoints[person_idx]
            scores = result.scores[person_idx]
            
            for kpt_idx, (kpt, score) in enumerate(zip(keypoints, scores)):
                if score > threshold:
                    x, y = int(kpt[0]), int(kpt[1])
                    
                    # Determine keypoint group and color
                    if kpt_idx < 17:  # Body keypoints
                        color = keypoint_colors['body']
                    elif kpt_idx < 85:  # Face keypoints (17-84)
                        color = keypoint_colors['face']
                    elif kpt_idx < 106:  # Left hand (85-105)
                        color = keypoint_colors['left_hand']
                    elif kpt_idx < 127:  # Right hand (106-126)
                        color = keypoint_colors['right_hand']
                    else:  # Feet (127-132)
                        color = keypoint_colors['feet']
                    
                    # Draw keypoint
                    cv2.circle(image, (x, y), 3, color, -1)
                    cv2.circle(image, (x, y), 4, (255, 255, 255), 1)  # White border
    
    def print_pose_summary(self, result: PoseResult, image_name: str):
        """Print a summary of pose estimation results."""
        print("\n" + "="*60)
        print(f"📊 POSE ESTIMATION SUMMARY for {image_name}")
        print("="*60)
        print(f"👥 Number of persons detected: {result.num_persons}")
        
        if result.num_persons > 0:
            for i in range(result.num_persons):
                keypoints = result.keypoints[i]
                scores = result.scores[i]
                bbox = result.bboxes[i] if len(result.bboxes) > i else None
                
                print(f"\n👤 Person {i+1}:")
                if bbox is not None and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    conf = bbox[4] if len(bbox) > 4 else 1.0
                    print(f"   📦 Bounding box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                    print(f"   🎯 Detection confidence: {conf:.3f}")
                
                # Count visible keypoints by category
                body_visible = np.sum(scores[:17] > 0.3)
                face_visible = np.sum(scores[17:85] > 0.3)
                hands_visible = np.sum(scores[85:127] > 0.3)
                feet_visible = np.sum(scores[127:133] > 0.3)
                
                print(f"   🦴 Body keypoints visible: {body_visible}/17")
                print(f"   😀 Face keypoints visible: {face_visible}/68") 
                print(f"   ✋ Hand keypoints visible: {hands_visible}/42")
                print(f"   🦶 Feet keypoints visible: {feet_visible}/6")
                print(f"   📈 Total keypoints visible: {np.sum(scores > 0.3)}/133")
                print(f"   ⭐ Average confidence: {np.mean(scores[scores > 0]):.3f}")
        else:
            print("❌ No persons detected in the image")
        
        print("="*60)


def test_image_pose_estimation():
    """Test function to process the test image."""
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output" / "pose-estimation"
    
    test_image = data_dir / "test_pose.png"
    
    # Check if test image exists
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Image Pose Estimation Test")
    print(f"📂 Input image: {test_image}")
    print(f"📂 Output directory: {output_dir}")
    
    try:
        # Initialize pose estimator
        estimator = ImagePoseEstimator(
            mode='balanced',
            backend='onnxruntime',
            device='cpu',
            kpt_threshold=0.3
        )
        
        # Process the test image
        output_path = output_dir / f"annotated_{test_image.name}"
        result = estimator.process_single_image(
            image_path=str(test_image),
            output_path=str(output_path),
            draw_bbox=True,
            draw_keypoints=True,
            keypoint_threshold=0.3
        )
        
        # Print detailed summary
        estimator.print_pose_summary(result, test_image.name)
        
        # Save keypoints to JSON for further analysis
        if result.num_persons > 0:
            import json
            keypoints_json = output_dir / f"keypoints_{test_image.stem}.json"
            
            json_data = {
                'image_info': {
                    'filename': test_image.name,
                    'num_persons': result.num_persons
                },
                'persons': []
            }
            
            for i in range(result.num_persons):
                person_data = {
                    'person_id': i,
                    'bbox': result.bboxes[i].tolist() if len(result.bboxes) > i else None,
                    'keypoints': result.keypoints[i].tolist(),
                    'scores': result.scores[i].tolist(),
                    'visible_keypoints': int(np.sum(result.scores[i] > 0.3))
                }
                json_data['persons'].append(person_data)
            
            with open(keypoints_json, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"📄 Keypoints saved to: {keypoints_json}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during pose estimation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_image_pose_estimation()
