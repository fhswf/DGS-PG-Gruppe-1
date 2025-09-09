"""
PoseEstimator2D: A Python wrapper class for 2D pose estimation using RTMLib

This module provides a convenient wrapper around the RTMLib library for performing
whole-body pose estimation on video files. It supports multiple performance modes
and outputs pose coordinates following the COCO WholeBody standard (133 keypoints).

Author: DGS Project Group 1
Date: September 2025
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass
import json
import time

try:
    from rtmlib import Wholebody, draw_skeleton
except ImportError:
    raise ImportError("RTMLib not found. Please install with: pip install rtmlib")


@dataclass
class PoseResult:
    """
    Data class to store pose estimation results for a single frame or image.
    
    Attributes:
        frame_idx: Frame index (0 for single images)
        keypoints: Array of shape (num_persons, 133, 2) containing x,y coordinates
        scores: Array of shape (num_persons, 133) containing confidence scores
        bboxes: Array of shape (num_persons, 5) containing bounding boxes [x1,y1,x2,y2,score]
        num_persons: Number of detected persons
    """
    frame_idx: int
    keypoints: np.ndarray
    scores: np.ndarray
    bboxes: np.ndarray
    num_persons: int


@dataclass
class VideoResult:
    """
    Data class to store video processing results.
    
    Attributes:
        frame_results: List of PoseResult objects for each frame
        total_frames: Total number of processed frames
        fps: Original video FPS
        processing_time: Total processing time in seconds
    """
    frame_results: List[PoseResult]
    total_frames: int
    fps: float
    processing_time: float


class PoseEstimator2D:
    """
    A wrapper class for RTMLib pose estimation.
    
    This class provides an easy-to-use interface for whole-body pose estimation
    supporting 133 keypoints (17 body + 68 face + 42 hands + 6 feet).
    """
    
    def __init__(
        self,
        mode: str = 'balanced',
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        to_openpose: bool = False,
        kpt_threshold: float = 0.3
    ):
        """
        Initialize the PoseEstimator2D.
        
        Args:
            mode: Performance mode ('performance', 'balanced', 'lightweight')
            backend: Backend to use ('onnxruntime', 'opencv', 'openvino')
            device: Device to use ('cpu', 'cuda', 'mps')
            to_openpose: Whether to convert to OpenPose format
            kpt_threshold: Keypoint confidence threshold
        """
        self.mode = mode
        self.backend = backend
        self.device = device
        self.to_openpose = to_openpose
        self.kpt_threshold = kpt_threshold
        
        # Initialize RTMLib Wholebody model for 133 keypoints
        try:
            self.model = Wholebody(
                mode=mode,
                backend=backend,
                device=device,
                to_openpose=to_openpose
            )
            print(f"Initialized RTMLib Wholebody with mode={mode}, backend={backend}, device={device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RTMLib: {e}")
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> PoseResult:
        """
        Process a single frame and extract pose keypoints.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            frame_idx: Frame index for tracking
            
        Returns:
            PoseResult object containing pose estimation results
        """
        try:
            # Perform pose estimation
            keypoints, scores = self.model(frame)
            
            # Handle empty results
            if keypoints is None or len(keypoints) == 0:
                return PoseResult(
                    frame_idx=frame_idx,
                    keypoints=np.empty((0, 133, 2)),
                    scores=np.empty((0, 133)),
                    bboxes=np.empty((0, 5)),
                    num_persons=0
                )
            
            # Ensure correct shape
            keypoints = np.array(keypoints)
            scores = np.array(scores)
            
            if keypoints.ndim == 2:
                keypoints = keypoints[np.newaxis, ...]
            if scores.ndim == 1:
                scores = scores[np.newaxis, ...]
            
            num_persons = keypoints.shape[0]
            
            # Calculate bounding boxes from keypoints
            bboxes = []
            for i in range(num_persons):
                valid_kpts = keypoints[i][scores[i] > self.kpt_threshold]
                if len(valid_kpts) > 0:
                    x_coords = valid_kpts[:, 0]
                    y_coords = valid_kpts[:, 1]
                    x1, y1 = np.min(x_coords), np.min(y_coords)
                    x2, y2 = np.max(x_coords), np.max(y_coords)
                    # Add some padding
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    confidence = np.mean(scores[i][scores[i] > self.kpt_threshold])
                    bboxes.append([x1, y1, x2, y2, confidence])
                else:
                    bboxes.append([0, 0, 0, 0, 0])
            
            bboxes = np.array(bboxes) if bboxes else np.empty((0, 5))
            
            return PoseResult(
                frame_idx=frame_idx,
                keypoints=keypoints,
                scores=scores,
                bboxes=bboxes,
                num_persons=num_persons
            )
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            # Return empty result on error
            return PoseResult(
                frame_idx=frame_idx,
                keypoints=np.empty((0, 133, 2)),
                scores=np.empty((0, 133)),
                bboxes=np.empty((0, 5)),
                num_persons=0
            )
    
    def process_image(
        self,
        image_path: Union[str, Path]
    ) -> PoseResult:
        """
        Process a single image and extract pose keypoints.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            PoseResult object containing pose estimation results
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot load image file: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {frame.shape}")
        
        # Process the image (frame_idx=0 for single image)
        result = self._process_frame(frame, frame_idx=0)
        
        print(f"Detected {result.num_persons} person(s) in the image")
        return result
    
    def process_image_with_annotation(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        keypoint_threshold: float = 0.3
    ) -> PoseResult:
        """
        Process a single image and optionally save the annotated result.
        
        Args:
            image_path: Path to the input image file
            output_path: Path to save the annotated image (optional)
            draw_bbox: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoints and skeleton
            keypoint_threshold: Minimum confidence threshold for drawing keypoints
            
        Returns:
            PoseResult object containing pose estimation results
        """
        image_path = Path(image_path)
        
        # Only create output directory if output_path is provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot load image file: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Process the image
        result = self._process_frame(frame, frame_idx=0)
        
        # Create annotated image
        annotated_frame = frame.copy()
        
        if result.num_persons > 0:
            # Draw bounding boxes
            if draw_bbox and len(result.bboxes) > 0:
                for bbox in result.bboxes:
                    x1, y1, x2, y2 = bbox[:4].astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints and skeleton
            if draw_keypoints:
                annotated_frame = draw_skeleton(
                    annotated_frame,
                    result.keypoints,
                    result.scores,
                    kpt_thr=keypoint_threshold
                )
        
        # Save annotated image only if output_path is provided
        if output_path is not None:
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"Saved annotated image to: {output_path}")
        
        return result
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_frames: bool = False,
        max_frames: Optional[int] = None
    ) -> VideoResult:
        """
        Process a video file and extract pose keypoints for all frames.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save output files (optional)
            save_frames: Whether to save annotated frames
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            VideoResult object containing all frame results
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
            ValueError: If the video cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        # Setup output directory if needed
        if output_dir and save_frames:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process frames
        frame_results = []
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self._process_frame(frame, frame_idx)
            frame_results.append(result)
            
            # Save annotated frame if requested
            if save_frames and output_dir and result.num_persons > 0:
                annotated_frame = draw_skeleton(
                    frame.copy(),
                    result.keypoints,
                    result.scores,
                    kpt_thr=self.kpt_threshold
                )
                frame_filename = output_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(frame_filename), annotated_frame)
            
            # Progress update
            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")
        
        cap.release()
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return VideoResult(
            frame_results=frame_results,
            total_frames=len(frame_results),
            fps=fps,
            processing_time=processing_time
        )
    
    def export_to_json(
        self,
        result: Union[PoseResult, VideoResult],
        output_path: Union[str, Path],
        include_scores: bool = True
    ) -> None:
        """
        Export pose estimation results to JSON format.
        
        Args:
            result: PoseResult or VideoResult object
            output_path: Path to save the JSON file
            include_scores: Whether to include confidence scores
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(result, PoseResult):
            # Single frame/image result
            data = {
                "frame_idx": int(result.frame_idx),
                "num_persons": int(result.num_persons),
                "keypoints": result.keypoints.tolist(),
                "bboxes": result.bboxes.tolist()
            }
            if include_scores:
                data["scores"] = result.scores.tolist()
        
        elif isinstance(result, VideoResult):
            # Video result
            data = {
                "total_frames": result.total_frames,
                "fps": result.fps,
                "processing_time": result.processing_time,
                "frames": []
            }
            
            for frame_result in result.frame_results:
                frame_data = {
                    "frame_idx": int(frame_result.frame_idx),
                    "num_persons": int(frame_result.num_persons),
                    "keypoints": frame_result.keypoints.tolist(),
                    "bboxes": frame_result.bboxes.tolist()
                }
                if include_scores:
                    frame_data["scores"] = frame_result.scores.tolist()
                
                data["frames"].append(frame_data)
        
        else:
            raise ValueError("Result must be PoseResult or VideoResult")
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported results to: {output_path}")
    
    def get_summary(self, result: Union[PoseResult, VideoResult]) -> str:
        """
        Get a summary string of the pose estimation results.
        
        Args:
            result: PoseResult or VideoResult object
            
        Returns:
            Formatted summary string
        """
        if isinstance(result, PoseResult):
            # Single frame/image summary
            summary = f"=== Pose Estimation Summary ===\n"
            summary += f"Frame: {result.frame_idx}\n"
            summary += f"Detected persons: {result.num_persons}\n"
            
            if result.num_persons > 0:
                for i in range(result.num_persons):
                    valid_kpts = np.sum(result.scores[i] > self.kpt_threshold)
                    avg_confidence = np.mean(result.scores[i][result.scores[i] > self.kpt_threshold])
                    summary += f"Person {i+1}: {valid_kpts}/133 keypoints, avg confidence: {avg_confidence:.3f}\n"
        
        elif isinstance(result, VideoResult):
            # Video summary
            total_persons = sum(fr.num_persons for fr in result.frame_results)
            frames_with_detection = sum(1 for fr in result.frame_results if fr.num_persons > 0)
            
            summary = f"=== Video Processing Summary ===\n"
            summary += f"Total frames: {result.total_frames}\n"
            summary += f"FPS: {result.fps:.2f}\n"
            summary += f"Processing time: {result.processing_time:.2f}s\n"
            summary += f"Frames with detection: {frames_with_detection}/{result.total_frames}\n"
            summary += f"Total person detections: {total_persons}\n"
            
            if total_persons > 0:
                avg_persons_per_frame = total_persons / result.total_frames
                summary += f"Average persons per frame: {avg_persons_per_frame:.2f}\n"
        
        else:
            summary = "Invalid result type"
        
        return summary


# Convenience functions for quick usage
def estimate_pose_image(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    mode: str = 'balanced',
    device: str = 'cpu'
) -> PoseResult:
    """
    Quick function to estimate pose in a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save annotated image (optional)
        mode: RTMLib mode ('performance', 'balanced', 'lightweight')
        device: Device to use ('cpu', 'cuda', 'mps')
        
    Returns:
        PoseResult object
    """
    estimator = PoseEstimator2D(mode=mode, device=device)
    
    if output_path:
        return estimator.process_image_with_annotation(image_path, output_path)
    else:
        return estimator.process_image(image_path)


def estimate_pose_video(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    mode: str = 'balanced',
    device: str = 'cpu',
    max_frames: Optional[int] = None
) -> VideoResult:
    """
    Quick function to estimate pose in a video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output files (optional)
        mode: RTMLib mode ('performance', 'balanced', 'lightweight')
        device: Device to use ('cpu', 'cuda', 'mps')
        max_frames: Maximum frames to process (None for all)
        
    Returns:
        VideoResult object
    """
    estimator = PoseEstimator2D(mode=mode, device=device)
    return estimator.process_video(
        video_path,
        output_dir=output_dir,
        save_frames=bool(output_dir),
        max_frames=max_frames
    )


if __name__ == "__main__":
    # Example usage
    print("RTMLib Pose Estimator 2D - Test Script")
    print("=" * 50)
    
    # Test with example image (if available)
    test_image = Path("../data/test_pose.png")
    if test_image.exists():
        print(f"Testing with image: {test_image}")
        
        estimator = PoseEstimator2D(mode='balanced', device='cpu')
        result = estimator.process_image(test_image)
        
        print(estimator.get_summary(result))
        
        # Save annotated result
        output_path = Path("../output/pose-estimation/test_result.png")
        estimator.process_image_with_annotation(
            test_image,
            output_path
        )
        
        # Export to JSON
        json_path = Path("../output/pose-estimation/test_result.json")
        estimator.export_to_json(result, json_path)
        
    else:
        print("No test image found. Please place an image at ../data/test_pose.png")
        print("Available convenience functions:")
        print("- estimate_pose_image(image_path, output_path)")
        print("- estimate_pose_video(video_path, output_dir)")