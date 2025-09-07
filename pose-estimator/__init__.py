"""
Pose Estimator 2D Package

A Python wrapper for 2D pose estimation using RTMLib with support for video processing
and COCO WholeBody standard (133 keypoints) output.
"""

from .pose_estimator_2d import (
    PoseEstimator2D,
    PoseResult, 
    VideoResult,
    estimate_poses_from_video
)

__version__ = "1.0.0"
__author__ = "DGS-PG-Gruppe-1"

__all__ = [
    "PoseEstimator2D",
    "PoseResult",
    "VideoResult", 
    "estimate_poses_from_video"
]
