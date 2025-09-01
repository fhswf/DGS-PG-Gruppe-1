"""
RTMLib ML Backend for Label Studio

This module provides pose estimation capabilities using RTMLib
for automatic annotation in Label Studio.
"""

import os
import io
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import cv2
import requests
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_key
from rtmlib import Wholebody, draw_skeleton

logger = logging.getLogger(__name__)


class RTMLibPoseEstimator(LabelStudioMLBase):
    """RTMLib-based pose estimation model for Label Studio."""
    
    def __init__(self, **kwargs):
        """
        Initialize the RTMLib pose estimator.
        
        Args:
            **kwargs: Additional arguments passed to LabelStudioMLBase
        """
        super(RTMLibPoseEstimator, self).__init__(**kwargs)
        
        # Model configuration
        self.device = os.environ.get('DEVICE', 'cpu')  # cpu, cuda, mps
        self.backend = os.environ.get('BACKEND', 'onnxruntime')  # opencv, onnxruntime, openvino
        self.mode = os.environ.get('MODE', 'balanced')  # performance, lightweight, balanced
        self.confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.3'))
        
        # Initialize pose estimator
        self.pose_estimator = None
        self._initialize_model()
        
        # Label mapping for wholebody keypoints
        self.keypoint_labels = self._get_keypoint_labels()
        
        logger.info(f"RTMLib Pose Estimator initialized with device={self.device}, "
                   f"backend={self.backend}, mode={self.mode}")

    def _initialize_model(self):
        """Initialize the RTMLib wholebody pose estimator."""
        try:
            self.pose_estimator = Wholebody(
                to_openpose=False,  # Use mmpose-style keypoints
                mode=self.mode,
                backend=self.backend,
                device=self.device
            )
            logger.info("RTMLib Wholebody model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RTMLib model: {e}")
            raise

    def _get_keypoint_labels(self) -> List[str]:
        """
        Get the keypoint labels for wholebody pose estimation.
        
        Returns:
            List of keypoint label names
        """
        # Standard COCO wholebody keypoint labels (133 keypoints total)
        body_labels = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Foot keypoints (6 points)
        foot_labels = [
            'left_big_toe', 'left_small_toe', 'left_heel',
            'right_big_toe', 'right_small_toe', 'right_heel'
        ]
        
        # Face keypoints (68 points)
        face_labels = [f'face_{i}' for i in range(68)]
        
        # Hand keypoints (21 points each hand)
        left_hand_labels = [f'left_hand_{i}' for i in range(21)]
        right_hand_labels = [f'right_hand_{i}' for i in range(21)]
        
        return body_labels + foot_labels + face_labels + left_hand_labels + right_hand_labels

    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """
        Download image from URL and convert to OpenCV format.
        
        Args:
            url: Image URL
            
        Returns:
            OpenCV image array or None if failed
        """
        try:
            # Handle Label Studio local file URLs
            if url.startswith('/data/'):
                # Local file path
                local_path = os.path.join('/app/data', url.lstrip('/data/'))
                if os.path.exists(local_path):
                    img = cv2.imread(local_path)
                    return img
                else:
                    logger.warning(f"Local file not found: {local_path}")
                    return None
            
            # Download from HTTP URL
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert to OpenCV format
            image = Image.open(io.BytesIO(response.content))
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    def _format_keypoints_for_label_studio(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray, 
        image_width: int, 
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Format keypoints for Label Studio keypoint annotation format.
        
        Args:
            keypoints: Keypoint coordinates array (N, 2)
            scores: Confidence scores array (N,)
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            List of keypoint annotations in Label Studio format
        """
        results = []
        
        for idx, (keypoint, score) in enumerate(zip(keypoints, scores)):
            if score > self.confidence_threshold:
                x, y = keypoint
                
                # Convert to percentages for Label Studio
                x_percent = (x / image_width) * 100
                y_percent = (y / image_height) * 100
                
                # Create keypoint annotation
                keypoint_result = {
                    "from_name": "keypoints",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "keypointlabels": [self.keypoint_labels[idx]],
                        "width": 2,  # Point width for visualization
                        "height": 2  # Point height for visualization
                    },
                    "score": float(score)
                }
                results.append(keypoint_result)
        
        return results

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        Generate pose estimation predictions for given tasks.
        
        Args:
            tasks: List of Label Studio tasks
            context: Optional context information
            **kwargs: Additional arguments
            
        Returns:
            List of predictions in Label Studio format
        """
        if not self.pose_estimator:
            logger.error("Pose estimator not initialized")
            return []
        
        predictions = []
        
        for task in tasks:
            try:
                # Extract image URL from task data
                image_url = task['data'].get('image')
                if not image_url:
                    logger.warning("No image URL found in task data")
                    continue
                
                # Download and process image
                img = self._download_image(image_url)
                if img is None:
                    logger.warning(f"Failed to load image: {image_url}")
                    continue
                
                # Get image dimensions
                image_height, image_width = img.shape[:2]
                
                # Perform pose estimation
                keypoints, scores = self.pose_estimator(img)
                
                if keypoints is None or len(keypoints) == 0:
                    logger.info("No pose detected in image")
                    predictions.append({"result": []})
                    continue
                
                # Format results for Label Studio
                prediction_results = []
                
                # Process each detected person
                for person_idx, (person_keypoints, person_scores) in enumerate(zip(keypoints, scores)):
                    keypoint_annotations = self._format_keypoints_for_label_studio(
                        person_keypoints, person_scores, image_width, image_height
                    )
                    prediction_results.extend(keypoint_annotations)
                
                prediction = {
                    "result": prediction_results,
                    "score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
                    "model_version": f"rtmlib-{self.mode}"
                }
                
                predictions.append(prediction)
                logger.info(f"Generated prediction with {len(prediction_results)} keypoints")
                
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                predictions.append({"result": []})
        
        return predictions

    def fit(self, event: str, data: Dict, **kwargs):
        """
        Handle training/fitting events from Label Studio.
        
        Args:
            event: Event type (e.g., 'ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
            data: Event data from Label Studio
            **kwargs: Additional arguments
        """
        logger.info(f"Received fit event: {event}")
        
        # RTMLib models are pre-trained, so we don't implement custom training
        # But this method can be used for:
        # - Logging annotation statistics
        # - Updating confidence thresholds based on user feedback
        # - Model performance monitoring
        
        if event in ['ANNOTATION_CREATED', 'ANNOTATION_UPDATED']:
            # Log annotation activity
            annotation_id = data.get('annotation', {}).get('id')
            task_id = data.get('task', {}).get('id')
            logger.info(f"Annotation {annotation_id} for task {task_id}: {event}")
            
            # Here you could implement:
            # - Adaptive confidence threshold adjustment
            # - User feedback collection
            # - Performance metrics calculation

    def get_train_job_status(self, train_job_id: str) -> Dict[str, Any]:
        """
        Get the status of a training job.
        
        Args:
            train_job_id: Training job identifier
            
        Returns:
            Training job status information
        """
        # Since we don't implement custom training, return completed status
        return {
            "job_status": "completed",
            "error": None,
            "log": "RTMLib uses pre-trained models - no training required",
            "created_at": None,
            "finished_at": None
        }
