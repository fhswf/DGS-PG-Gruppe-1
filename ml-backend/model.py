"""
RTMLib ML Backend for Label Studio

This module provides pose estimation capabilities using RTMLib
for automatic annotation in Label Studio.
"""

import os
import io
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import requests
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase

try:
	from rtmlib import Wholebody
except Exception:  # pragma: no cover - allow import to fail during type checks
	Wholebody = None  # type: ignore


logger = logging.getLogger(__name__)


class RTMLibPoseEstimator(LabelStudioMLBase):
	"""RTMLib-based pose estimation model for Label Studio."""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		# Model configuration via env
		self.device = os.environ.get("DEVICE", "cpu")  # cpu, cuda, mps
		self.backend = os.environ.get("BACKEND_TYPE", "onnxruntime")  # opencv, onnxruntime, openvino
		self.mode = os.environ.get("MODE", "balanced")  # performance, lightweight, balanced
		self.confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
		
		# Label Studio configuration
		self.label_studio_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
		self.label_studio_token = os.environ.get("LABEL_STUDIO_API_TOKEN", "")

		self.pose_estimator = None
		self._initialize_model()

		# WholeBody 133 KP names (17 body + 6 feet + 68 face + 21 left hand + 21 right hand)
		self.keypoint_labels = self._get_keypoint_labels()

		logger.info(
			f"RTMLib Pose Estimator initialized with device={self.device}, backend={self.backend}, mode={self.mode}"
		)

	def _initialize_model(self):
		if Wholebody is None:
			raise RuntimeError("rtmlib is not installed")
		try:
			print("Initializing RTMLib Wholebody model...")
			print(f"  Device: {self.device}")
			print(f"  Backend: {self.backend}")
			print(f"  Mode: {self.mode}")
			
			self.pose_estimator = Wholebody(
				to_openpose=False,  # mmpose-style ordering
				mode=self.mode,
				backend=self.backend,
				device=self.device,
			)
			print("RTMLib Wholebody model initialized successfully")
		except Exception as e:
			print(f"Failed to initialize RTMLib model: {e}")
			logger.exception(f"Failed to initialize RTMLib model: {e}")
			raise

	def _get_keypoint_labels(self) -> List[str]:
		body_labels = [
			"nose",
			"left_eye",
			"right_eye",
			"left_ear",
			"right_ear",
			"left_shoulder",
			"right_shoulder",
			"left_elbow",
			"right_elbow",
			"left_wrist",
			"right_wrist",
			"left_hip",
			"right_hip",
			"left_knee",
			"right_knee",
			"left_ankle",
			"right_ankle",
		]
		foot_labels = [
			"left_big_toe",
			"left_small_toe",
			"left_heel",
			"right_big_toe",
			"right_small_toe",
			"right_heel",
		]
		face_labels = [f"face_{i}" for i in range(68)]
		#left_hand_labels = [f"left_hand_{i}" for i in range(21)]
		#right_hand_labels = [f"right_hand_{i}" for i in range(21)]
		left_hand_labels = ["left_wrist_0","left_thumb_1", "left_thumb_2", "left_thumb_3", 
					"left_thumb_4", "left_index_5", "left_index_6", "left_index_7", "left_index_8",
    				"left_middle_9", "left_middle_10", "left_middle_11", "left_middle_12",
    				"left_ring_13", "left_ring_14", "left_ring_15", "left_ring_16",
    				"left_little_17", "left_little_18", "left_little_19", "left_little_20"]

		right_hand_labels = ["right_wrist_0", "right_thumb_1", "right_thumb_2", "right_thumb_3", 
					"right_thumb_4", "right_index_5", "right_index_6", "right_index_7", "right_index_8",
   					"right_middle_9", "right_middle_10", "right_middle_11", "right_middle_12",
    				"right_ring_13", "right_ring_14", "right_ring_15", "right_ring_16",
    				"right_little_17", "right_little_18", "right_little_19", "right_little_20"]

		return body_labels + foot_labels + face_labels + left_hand_labels + right_hand_labels

	def _download_image(self, url: str) -> Optional[np.ndarray]:
		try:
			# Handle Label Studio local file URLs and relative /data paths
			if url.startswith("/data/"):
				local_path = os.path.join("/app", url.lstrip("/"))
				if os.path.exists(local_path):
					img = cv2.imread(local_path)
					return img
				logger.warning(f"Local file not found: {local_path}")
				# Try to construct full URL for remote access
				url = f"{self.label_studio_url}{url}"
				logger.info(f"Trying remote URL: {url}")

			# Replace localhost with host.docker.internal for container-to-host communication
			if "localhost:8080" in url:
				url = url.replace("localhost:8080", "host.docker.internal:8080")
				logger.info(f"Adjusted URL for container access: {url}")

			# Fetch remote
			headers = {}
			if self.label_studio_token:
				headers["Authorization"] = f"Token {self.label_studio_token}"
			resp = requests.get(url, headers=headers, timeout=30)
			resp.raise_for_status()
			image = Image.open(io.BytesIO(resp.content)).convert("RGB")
			arr = np.array(image)  # RGB
			arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
			return arr
		except Exception as e:
			logger.exception(f"Failed to load image {url}: {e}")
			return None

	def _format_keypoints_for_label_studio(
		self,
		keypoints: np.ndarray,
		scores: np.ndarray,
		image_width: int,
		image_height: int,
	) -> List[Dict[str, Any]]:
		results: List[Dict[str, Any]] = []
		for idx, (pt, sc) in enumerate(zip(keypoints, scores)):
			if idx >= len(self.keypoint_labels):
				break
			if float(sc) < self.confidence_threshold:
				continue
			x, y = float(pt[0]), float(pt[1])
			x_pct = (x / image_width) * 100.0
			y_pct = (y / image_height) * 100.0
			results.append(
				{
					"from_name": "keypoints",
					"to_name": "image",
					"type": "keypointlabels",
					"value": {
						"x": x_pct,
						"y": y_pct,
						"keypointlabels": [self.keypoint_labels[idx]],
						# width/height are optional for points in LS; keep tiny for visibility
						"width": 0.5,
						"height": 0.5,
					},
					"score": float(sc),
				}
			)
		return results

	def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
		if not self.pose_estimator:
			logger.error("Pose estimator not initialized")
			return []

		predictions: List[Dict[str, Any]] = []
		for task in tasks:
			try:
				image_url = task.get("data", {}).get("image")
				if not image_url:
					logger.warning("Task has no 'image' field in data")
					predictions.append({"result": []})
					continue

				img = self._download_image(image_url)
				if img is None:
					predictions.append({"result": []})
					continue

				h, w = img.shape[:2]

				# rtmlib Wholebody returns keypoints and scores for each detected person
				keypoints_list, scores_list = self.pose_estimator(img)

				if keypoints_list is None or len(keypoints_list) == 0:
					predictions.append({"result": []})
					continue

				result_items: List[Dict[str, Any]] = []
				person_scores = []
				for person_kps, person_scs in zip(keypoints_list, scores_list):
					# Expected shapes: (133, 2) and (133,)
					person_scores.append(float(np.mean(person_scs)))
					result_items.extend(
						self._format_keypoints_for_label_studio(person_kps, person_scs, w, h)
					)

				predictions.append(
					{
						"result": result_items,
						"score": float(np.mean(person_scores)) if person_scores else 0.0,
						"model_version": f"rtmlib-{self.mode}",
					}
				)
			except Exception as e:
				logger.exception(f"Error during prediction: {e}")
				predictions.append({"result": []})

		return predictions

	# Optional training hook; no-op for pre-trained rtmlib models
	def fit(self, event: str, data: Dict, **kwargs):  # pragma: no cover - not used by default
		logger.info(f"Received event {event}")

