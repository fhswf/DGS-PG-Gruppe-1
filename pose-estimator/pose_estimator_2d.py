"""
PoseEstimator2D: A Python wrapper class for 2D pose estimation using RTMLib

This module provides a convenient wrapper around the RTMLib library for performing
whole-body pose estimation on video files. It supports multiple performance modes
and outputs pose coordinates following the COCO WholeBody standard (133 keypoints).

Author: Generated based on pose-estimation notebook and RTMLib documentation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import warnings

try:
    from rtmlib import Wholebody, draw_skeleton, draw_bbox
except ImportError:
    raise ImportError(
        "RTMLib is required. Install it with: pip install rtmlib"
    )


@dataclass
class PoseResult:
    """
    Container for pose estimation results from a single frame.
    
    Attributes:
        frame_idx: Index of the frame in the video
        keypoints: Array of shape (N, 133, 2) containing x,y coordinates for N persons
        scores: Array of shape (N, 133) containing confidence scores for each keypoint
        bboxes: Array of shape (N, 5) containing bounding boxes (x1, y1, x2, y2, score)
        num_persons: Number of detected persons in the frame
    """
    frame_idx: int
    keypoints: np.ndarray
    scores: np.ndarray  
    bboxes: np.ndarray
    num_persons: int


@dataclass 
class VideoResult:
    """
    Container for pose estimation results from an entire video.
    
    Attributes:
        video_path: Path to the input video file
        total_frames: Total number of frames processed
        frame_results: List of PoseResult objects, one per frame
        fps: Original video frame rate
        resolution: Original video resolution (width, height)
        processing_stats: Dictionary with processing statistics
    """
    video_path: str
    total_frames: int
    frame_results: List[PoseResult] = field(default_factory=list)
    fps: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    def get_keypoints_by_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get keypoints for a specific frame."""
        if 0 <= frame_idx < len(self.frame_results):
            return self.frame_results[frame_idx].keypoints
        return None
    
    def get_all_keypoints(self) -> List[np.ndarray]:
        """Get keypoints for all frames."""
        return [result.keypoints for result in self.frame_results]
    
    def get_person_trajectory(self, person_idx: int = 0) -> List[Optional[np.ndarray]]:
        """
        Get the keypoint trajectory for a specific person across all frames.
        
        Args:
            person_idx: Index of the person (0 for first detected person)
            
        Returns:
            List of keypoint arrays, None for frames where person was not detected
        """
        trajectory = []
        for result in self.frame_results:
            if result.num_persons > person_idx:
                trajectory.append(result.keypoints[person_idx])
            else:
                trajectory.append(None)
        return trajectory


class PoseEstimator2D:
    """
    A Python wrapper class for 2D pose estimation using RTMLib.
    
    This class provides an easy-to-use interface for performing whole-body pose 
    estimation on video files. It supports different performance modes and outputs
    pose coordinates following the COCO WholeBody standard (133 keypoints).
    
    Example:
        >>> estimator = PoseEstimator2D(mode='balanced')
        >>> result = estimator.process_video('path/to/video.mp4')
        >>> print(f"Processed {result.total_frames} frames")
        >>> print(f"Found {result.frame_results[0].num_persons} persons in first frame")
    """
    
    VALID_MODES = ['performance', 'lightweight', 'balanced']
    VALID_BACKENDS = ['onnxruntime', 'opencv', 'openvino']
    VALID_DEVICES = ['cpu', 'cuda', 'mps']
    
    def __init__(
        self,
        mode: str = 'balanced',
        backend: str = 'onnxruntime', 
        device: str = 'cpu',
        kpt_threshold: float = 0.3,
        to_openpose: bool = False
    ):
        """
        Initialize the PoseEstimator2D.
        
        Args:
            mode: Performance mode - 'performance', 'lightweight', or 'balanced'
            backend: Inference backend - 'onnxruntime', 'opencv', or 'openvino'  
            device: Compute device - 'cpu', 'cuda', or 'mps'
            kpt_threshold: Confidence threshold for keypoint detection (0.0-1.0)
            to_openpose: Whether to use OpenPose-style keypoint format
            
        Raises:
            ValueError: If invalid mode, backend, or device is specified
        """
        # Validate parameters
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}")
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Must be one of: {self.VALID_BACKENDS}")
        if device not in self.VALID_DEVICES:
            raise ValueError(f"Invalid device '{device}'. Must be one of: {self.VALID_DEVICES}")
        if not 0.0 <= kpt_threshold <= 1.0:
            raise ValueError("kpt_threshold must be between 0.0 and 1.0")
            
        self.mode = mode
        self.backend = backend
        self.device = device
        self.kpt_threshold = kpt_threshold
        self.to_openpose = to_openpose
        
        # Initialize the Wholebody model
        try:
            self.wholebody = Wholebody(
                mode=mode,
                backend=backend, 
                device=device,
                to_openpose=to_openpose
            )
            print(f"PoseEstimator2D initialized with mode='{mode}', backend='{backend}', device='{device}'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Wholebody model: {e}")
    
    @property 
    def mode(self) -> str:
        """Get the current performance mode."""
        return self._mode
    
    @mode.setter
    def mode(self, value: str):
        """Set the performance mode."""
        if value not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{value}'. Must be one of: {self.VALID_MODES}")
        self._mode = value
        
    def process_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        progress_callback: Optional[callable] = None
    ) -> VideoResult:
        """
        Process a video file and extract pose keypoints for all frames.
        
        Args:
            video_path: Path to the input video file
            max_frames: Maximum number of frames to process (None for all frames)
            start_frame: Frame index to start processing from 
            progress_callback: Optional callback function for progress updates
            
        Returns:
            VideoResult object containing all pose estimation results
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
            ValueError: If the video cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        try:
            # Get video properties
            total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Determine frames to process
            if max_frames is None:
                frames_to_process = total_frames_in_video - start_frame
            else:
                frames_to_process = min(max_frames, total_frames_in_video - start_frame)
                
            # Skip to start frame
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Initialize result container
            result = VideoResult(
                video_path=str(video_path),
                total_frames=frames_to_process,
                fps=fps,
                resolution=(width, height)
            )
            
            # Process frames
            frame_idx = start_frame
            processed_frames = 0
            frames_with_detections = 0
            total_persons_detected = 0
            
            print(f"Processing {frames_to_process} frames starting from frame {start_frame}...")
            
            while processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video reached at frame {frame_idx}")
                    break
                    
                # Process frame
                frame_result = self._process_frame(frame, frame_idx)
                result.frame_results.append(frame_result)
                
                # Update statistics
                if frame_result.num_persons > 0:
                    frames_with_detections += 1
                    total_persons_detected += frame_result.num_persons
                
                # Progress callback
                if progress_callback:
                    progress = (processed_frames + 1) / frames_to_process
                    progress_callback(progress, frame_idx, frame_result.num_persons)
                    
                processed_frames += 1
                frame_idx += 1
                
                # Print progress every 10 frames
                if processed_frames % 10 == 0:
                    print(f"Processed {processed_frames}/{frames_to_process} frames")
            
            # Update final statistics
            result.processing_stats = {
                'frames_with_detections': frames_with_detections,
                'total_persons_detected': total_persons_detected,
                'avg_persons_per_frame': total_persons_detected / max(1, processed_frames),
                'detection_rate': frames_with_detections / max(1, processed_frames)
            }
            
            print(f"Video processing complete!")
            print(f"- Processed: {processed_frames} frames")
            print(f"- Frames with detections: {frames_with_detections}")
            print(f"- Total persons detected: {total_persons_detected}")
            print(f"- Average persons per frame: {result.processing_stats['avg_persons_per_frame']:.2f}")
            
            return result
            
        finally:
            cap.release()
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> PoseResult:
        """
        Process a single frame and extract pose keypoints.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            frame_idx: Index of the frame
            
        Returns:
            PoseResult object containing pose estimation results for this frame
        """
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
            warnings.warn(f"Error processing frame {frame_idx}: {e}")
            # Return empty result on error
            return PoseResult(
                frame_idx=frame_idx,
                keypoints=np.empty((0, 133, 2)),
                scores=np.empty((0, 133)),
                bboxes=np.empty((0, 5)),
                num_persons=0
            )
    
    def process_frame(self, frame: np.ndarray) -> PoseResult:
        """
        Process a single frame (public interface).
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            PoseResult object containing pose estimation results
        """
        return self._process_frame(frame, 0)
    
    def visualize_frame(
        self,
        frame: np.ndarray, 
        pose_result: PoseResult,
        draw_bboxes: bool = True,
        draw_skeleton: bool = True,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        skeleton_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Visualize pose estimation results on a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            pose_result: PoseResult object with pose estimation results
            draw_bboxes: Whether to draw bounding boxes
            draw_skeleton: Whether to draw pose skeleton
            bbox_color: Color for bounding boxes (B, G, R)
            skeleton_color: Color for skeleton (B, G, R) - not used in current rtmlib
            
        Returns:
            Annotated frame as numpy array
        """
        canvas = frame.copy()
        
        if pose_result.num_persons > 0:
            if draw_bboxes and len(pose_result.bboxes) > 0:
                canvas = draw_bbox(canvas, pose_result.bboxes)
                
            if draw_skeleton and len(pose_result.keypoints) > 0:
                canvas = draw_skeleton(
                    canvas, 
                    pose_result.keypoints, 
                    pose_result.scores,
                    kpt_thr=self.kpt_threshold
                )
        
        return canvas
    
    def save_results_to_video(
        self,
        video_result: VideoResult,
        output_path: Union[str, Path],
        original_video_path: Union[str, Path],
        draw_bboxes: bool = True,
        draw_skeleton: bool = True,
        codec: str = 'mp4v'
    ):
        """
        Save annotated video with pose estimation results.
        
        Args:
            video_result: VideoResult object with pose estimation results
            output_path: Path for output video file
            original_video_path: Path to original video file for frame reading
            draw_bboxes: Whether to draw bounding boxes
            draw_skeleton: Whether to draw pose skeleton  
            codec: Video codec to use
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open original video for reading frames
        cap = cv2.VideoCapture(str(original_video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {original_video_path}")
            
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            print(f"Saving annotated video to {output_path}...")
            
            # Process each frame
            for i, pose_result in enumerate(video_result.frame_results):
                # Read corresponding frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, pose_result.frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {pose_result.frame_idx}")
                    continue
                
                # Annotate frame
                annotated_frame = self.visualize_frame(
                    frame, pose_result, draw_bboxes, draw_skeleton
                )
                
                # Write frame
                out.write(annotated_frame)
                
                if (i + 1) % 10 == 0:
                    print(f"Saved {i + 1}/{len(video_result.frame_results)} frames")
            
            print(f"Video saved successfully to {output_path}")
            
        finally:
            cap.release()
            out.release()
    
    def export_keypoints_to_json(self, video_result: VideoResult, output_path: Union[str, Path]):
        """
        Export pose keypoints to JSON format.
        
        Args:
            video_result: VideoResult object with pose estimation results
            output_path: Path for output JSON file
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_data = {
            'video_info': {
                'video_path': video_result.video_path,
                'total_frames': video_result.total_frames,
                'fps': video_result.fps,
                'resolution': video_result.resolution
            },
            'processing_stats': video_result.processing_stats,
            'frames': []
        }
        
        for result in video_result.frame_results:
            frame_data = {
                'frame_idx': result.frame_idx,
                'num_persons': result.num_persons,
                'persons': []
            }
            
            for person_idx in range(result.num_persons):
                person_data = {
                    'person_idx': person_idx,
                    'bbox': result.bboxes[person_idx].tolist() if len(result.bboxes) > person_idx else None,
                    'keypoints': result.keypoints[person_idx].tolist(),
                    'scores': result.scores[person_idx].tolist()
                }
                frame_data['persons'].append(person_data)
            
            json_data['frames'].append(frame_data)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        print(f"Keypoints exported to {output_path}")


# Convenience function for quick usage
def estimate_poses_from_video(
    video_path: Union[str, Path],
    mode: str = 'balanced',
    max_frames: Optional[int] = None,
    device: str = 'cpu',
    **kwargs
) -> VideoResult:
    """
    Convenience function to quickly estimate poses from a video file.
    
    Args:
        video_path: Path to input video file
        mode: Performance mode ('performance', 'lightweight', 'balanced')
        max_frames: Maximum number of frames to process (None for all)
        device: Compute device ('cpu', 'cuda', 'mps')
        **kwargs: Additional arguments passed to PoseEstimator2D
        
    Returns:
        VideoResult object with pose estimation results
    """
    estimator = PoseEstimator2D(mode=mode, device=device, **kwargs)
    return estimator.process_video(video_path, max_frames=max_frames)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="2D Pose Estimation using RTMLib")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--mode", choices=['performance', 'lightweight', 'balanced'], 
                       default='balanced', help="Performance mode")
    parser.add_argument("--device", choices=['cpu', 'cuda', 'mps'], 
                       default='cpu', help="Compute device")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--output-json", help="Path to save keypoints as JSON")
    parser.add_argument("--output-video", help="Path to save annotated video")
    
    args = parser.parse_args()
    
    # Run pose estimation
    estimator = PoseEstimator2D(mode=args.mode, device=args.device)
    result = estimator.process_video(args.video_path, max_frames=args.max_frames)
    
    # Save results
    if args.output_json:
        estimator.export_keypoints_to_json(result, args.output_json)
        
    if args.output_video:
        estimator.save_results_to_video(result, args.output_video, args.video_path)
