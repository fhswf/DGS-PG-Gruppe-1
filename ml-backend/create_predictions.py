"""
Test script for integrating RTMLibPoseEstimator with Label Studio.
This script fetches tasks from Label Studio and sends predictions using the local ML backend.
"""

import os
import io
import logging
from label_studio_sdk import LabelStudio
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import requests
from PIL import Image
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system environment variables

try:
	from rtmlib import Wholebody
except Exception:  # pragma: no cover - allow import to fail during type checks
	Wholebody = None  # type: ignore


logger = logging.getLogger(__name__)


class RTMLibPoseEstimator:
	"""RTMLib-based pose estimation model for Label Studio."""

	def __init__(self, **kwargs):

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

# ------------------------------
# Configuration - Load from environment variables
# ------------------------------
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "https://label-studio.fh-swf.cloud")
API_TOKEN = os.environ.get("LABEL_STUDIO_API_TOKEN", "your-api-token-here")
PROJECT_ID = os.environ.get("PROJECT_ID", "2")
TASK_LIMIT = int(os.environ.get("TASK_LIMIT", "1"))
START_TASK_ID = os.environ.get("START_TASK_ID")
if START_TASK_ID is not None and START_TASK_ID != "":
    START_TASK_ID = int(START_TASK_ID)
else:
    START_TASK_ID = None

# Rate limiting configuration
REQUEST_DELAY_SECONDS = float(os.environ.get("REQUEST_DELAY_SECONDS", "1.0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
MAX_BACKOFF_DELAY = int(os.environ.get("MAX_BACKOFF_DELAY", "60"))

# ------------------------------
# Set environment variables for ML model configuration (optional, uses defaults if not set)
# ------------------------------
os.environ.setdefault("DEVICE", "cpu")  # cpu, cuda, mps
os.environ.setdefault("BACKEND_TYPE", "onnxruntime")  # opencv, onnxruntime, openvino
os.environ.setdefault("MODE", "balanced")  # performance, lightweight, balanced
os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.3")
os.environ.setdefault("LABEL_STUDIO_URL", LABEL_STUDIO_URL)
os.environ.setdefault("LABEL_STUDIO_API_TOKEN", API_TOKEN)

# ------------------------------
# Test basic API functionality
# ------------------------------
def test_basic_api():
    try:
        print("Testing basic API calls...")
        
        # Try to get project tasks count
        print("Getting project info...")
        project = ls_client.projects.get(id=PROJECT_ID)
        print(f"Project has {project.task_number} tasks")
        
        # Try to get first task
        print("Getting first task...")
        tasks = ls_client.tasks.list(project=PROJECT_ID, page=1, page_size=1)
        first_task = next(iter(tasks), None)
        if first_task:
            print(f"First task ID: {first_task.id}")
            return True
        else:
            print("No tasks found")
            return False
            
    except Exception as e:
        print(f"Basic API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ------------------------------
# Test Label Studio connection
# ------------------------------
def test_label_studio_connection():
    try:
        print("Testing Label Studio connection...")
        # Try to get project info as a quick connectivity test
        project = ls_client.projects.get(id=PROJECT_ID)
        print(f"✅ Connected to Label Studio. Project: {project.title}")
        return test_basic_api()
    except Exception as e:
        print(f"❌ Failed to connect to Label Studio: {e}")
        print("Please check your LABEL_STUDIO_URL and API_TOKEN configuration.")
        return False

# ------------------------------
# Initialize Label Studio client
# ------------------------------
ls_client = LabelStudio(
    base_url=LABEL_STUDIO_URL,
    api_key=API_TOKEN
)
print("Label Studio client initialized.")

# Test connection before initializing ML model
if not test_label_studio_connection():
    print("Exiting due to Label Studio connection failure.")
    exit(1)

# ------------------------------
# Initialize ML model
# ------------------------------
ml_model = RTMLibPoseEstimator()
print("Local ML backend initialized.")

# ------------------------------
# Process a single task
# ------------------------------
def process_task(task):
    task_id = task.id
    
    # Check if task already has predictions (from task object)
    if hasattr(task, 'predictions') and task.predictions:
        print(f"Task {task_id} already has predictions, skipping.")
        return False
    
    # Check if task already has annotations (from task object)
    if hasattr(task, 'annotations') and task.annotations:
        print(f"Task {task_id} already has annotations, skipping.")
        return False

    image_url = task.data.get("image") if task.data else None
    if not image_url:
        print(f"Task {task_id} has no image, skipping.")
        return False

    # Run local ML model
    prediction_response = ml_model.predict([{"data": {"image": image_url}}])[0]

    # Prepare payload for Label Studio using SDK format
    payload = {
        "task": task_id,
        "result": prediction_response["result"],
        "model_version": prediction_response["model_version"],
        "score": prediction_response["score"]
    }

    # Log the prediction payload
    # print(f"Sending prediction for task {task_id}: {payload}")

    # Send prediction to Label Studio using SDK with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            prediction_response = ls_client.predictions.create(**payload)
            # Add delay between successful requests to prevent overwhelming the server
            if REQUEST_DELAY_SECONDS > 0:
                time.sleep(REQUEST_DELAY_SECONDS)
            return True
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "no available server" in error_msg:
                # Server overload - use exponential backoff
                if attempt < MAX_RETRIES - 1:
                    backoff_delay = min(REQUEST_DELAY_SECONDS * (2 ** attempt), MAX_BACKOFF_DELAY)
                    print(f"⚠️  Server overload (503) for task {task_id}, retrying in {backoff_delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(backoff_delay)
                    continue
                else:
                    print(f"❌ Failed for task {task_id} after {MAX_RETRIES} attempts: {e}")
                    return False
            else:
                # Other error - don't retry
                print(f"❌ Failed for task {task_id}: {e}")
                return False

    # This should not be reached, but just in case
    print(f"❌ Failed for task {task_id} after all retries")
    return False

# ------------------------------
# Create predictions for tasks
# ------------------------------
def create_predictions(start_task_id=None):
    predictions_sent = 0
    batch_size = 1  # Keep small for now
    
    if start_task_id:
        print(f"Looking for task ID {start_task_id} to start processing from...")
        # Try to estimate starting page (assuming roughly sequential task IDs)
        # Get the first task ID to calculate offset
        try:
            first_task_response = ls_client.tasks.list(project=PROJECT_ID, page=1, page_size=1)
            first_task_iter = iter(first_task_response)
            first_task = next(first_task_iter)
            first_task_id = first_task.id
            
            # Estimate starting page (task IDs might not be perfectly sequential)
            estimated_page = max(1, (start_task_id - first_task_id) + 1)
            page = estimated_page
            print(f"First task ID: {first_task_id}, estimated starting page: {page}")
        except Exception as e:
            print(f"Could not determine first task ID, starting from page 1: {e}")
            page = 1
    else:
        page = 1
    
    found_start_task = start_task_id is None  # If no start_id, start immediately
    
    while predictions_sent < TASK_LIMIT:
        try:
            print(f"Fetching page {page} (page_size={batch_size})...")
            import time
            start_time = time.time()
            
            # Use direct API call instead of iterator to avoid hanging
            response = ls_client.tasks.list(project=PROJECT_ID, page=page, page_size=batch_size)
            
            # Check if response has tasks without converting to list
            if hasattr(response, '__iter__'):
                # Try to get just the first item to test
                task_iter = iter(response)
                try:
                    task = next(task_iter)
                    tasks = [task]  # Just process one task at a time
                    fetch_time = time.time() - start_time
                    print(f"Page {page} fetched in {fetch_time:.2f}s - got 1 task")
                except StopIteration:
                    print("No more tasks available.")
                    break
            else:
                print("Unexpected response format")
                break
                
            print(f"Processing task from page {page}...")
            
            for task in tasks:
                if predictions_sent >= TASK_LIMIT:
                    break
                
                # Check if we've reached the start task ID
                if not found_start_task:
                    if task.id >= start_task_id:
                        found_start_task = True
                        print(f"Found start task ID {task.id}, beginning processing...")
                    else:
                        print(f"Skipping task {task.id} (before start ID {start_task_id})")
                        continue
                
                if found_start_task and process_task(task):
                    predictions_sent += 1
            
            page += 1
            
        except Exception as e:
            print(f"Error fetching tasks from project {PROJECT_ID}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"Finished sending predictions. Total successful: {predictions_sent}")

# ------------------------------
# Main flow
# ------------------------------
if __name__ == "__main__":
    create_predictions(start_task_id=START_TASK_ID)