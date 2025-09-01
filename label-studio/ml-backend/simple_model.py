"""
Simplified RTMLib ML Backend for Label Studio

This is a minimal version that works without the full label-studio-ml-backend package.
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
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import RTMLib
try:
    from rtmlib import Wholebody, draw_skeleton
    RTMLIB_AVAILABLE = True
    logger.info("RTMLib successfully imported")
except ImportError as e:
    logger.error(f"RTMLib import failed: {e}")
    RTMLIB_AVAILABLE = False

class SimpleRTMLibBackend:
    """Simplified RTMLib-based pose estimation backend."""
    
    def __init__(self):
        """Initialize the pose estimator."""
        self.device = os.environ.get('DEVICE', 'cpu')
        self.backend = os.environ.get('BACKEND', 'onnxruntime')
        self.mode = os.environ.get('MODE', 'balanced')
        self.confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.3'))
        
        self.pose_estimator = None
        if RTMLIB_AVAILABLE:
            self._initialize_model()
        
        # Keypoint labels for wholebody
        self.keypoint_labels = self._get_keypoint_labels()
        
        logger.info(f"Backend initialized: device={self.device}, mode={self.mode}")

    def _initialize_model(self):
        """Initialize the RTMLib model."""
        try:
            self.pose_estimator = Wholebody(
                to_openpose=False,
                mode=self.mode,
                backend=self.backend,
                device=self.device
            )
            logger.info("RTMLib model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RTMLib model: {e}")
            self.pose_estimator = None

    def _get_keypoint_labels(self) -> List[str]:
        """Get keypoint labels."""
        # Simplified keypoint labels
        body_labels = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Add simplified additional keypoints to reach expected count
        additional_labels = [f'keypoint_{i}' for i in range(17, 133)]
        
        return body_labels + additional_labels

    def _get_skeleton_connections(self):
        """Define skeleton connections for body pose."""
        # Body skeleton connections (COCO format)
        body_connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes, eyes-ears
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders-hips
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders-elbows-wrists
            # Legs  
            (11, 13), (13, 15), (12, 14), (14, 16),  # hips-knees-ankles
            # Feet
            (15, 17), (15, 18), (15, 19),  # left ankle-foot
            (16, 20), (16, 21), (16, 22),  # right ankle-foot
        ]
        return body_connections

    def _create_skeleton_polygon(self, keypoints, scores, image_width, image_height):
        """Create skeleton polygon annotations from keypoints."""
        connections = self._get_skeleton_connections()
        skeleton_lines = []
        
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                start_idx < len(scores) and end_idx < len(scores)):
                
                if scores[start_idx] > self.confidence_threshold and scores[end_idx] > self.confidence_threshold:
                    start_point = keypoints[start_idx]
                    end_point = keypoints[end_idx]
                    
                    # Convert to percentages
                    start_x = (start_point[0] / image_width) * 100
                    start_y = (start_point[1] / image_height) * 100
                    end_x = (end_point[0] / image_width) * 100
                    end_y = (end_point[1] / image_height) * 100
                    
                    skeleton_lines.extend([[start_x, start_y], [end_x, end_y]])
        
        if len(skeleton_lines) > 0:
            return {
                "from_name": "skeleton",
                "to_name": "image",
                "type": "polygonlabels", 
                "value": {
                    "points": skeleton_lines,
                    "polygonlabels": ["body_skeleton"]
                }
            }
        return None

    def _download_image(self, url: str, request_headers: dict = None) -> Optional[np.ndarray]:
        """Download image from URL."""
        try:
            # Handle relative URLs starting with /data/
            if url.startswith('/data/'):
                # Try local file first (shared volume) - add 'media' to path
                local_path = f"/label-studio/data/media{url[5:]}"  # Remove '/data' and add to media path
                if os.path.exists(local_path):
                    logger.info(f"Loading image from local path: {local_path}")
                    return cv2.imread(local_path)
                
                # Fallback to app data path
                app_local_path = f"/app{url}"
                if os.path.exists(app_local_path):
                    logger.info(f"Loading image from app path: {app_local_path}")
                    return cv2.imread(app_local_path)
                
                # Convert to full URL as fallback
                url = f"http://label-studio:8080{url}"
                    
            elif url.startswith('data/'):
                # Try local file first - add 'media' to path
                local_path = f"/label-studio/data/media/{url[5:]}"  # Remove 'data/' and add to media path
                if os.path.exists(local_path):
                    logger.info(f"Loading image from local path: {local_path}")
                    return cv2.imread(local_path)
                
                # Fallback to app data path
                app_local_path = f"/app/{url}"
                if os.path.exists(app_local_path):
                    logger.info(f"Loading image from app path: {app_local_path}")
                    return cv2.imread(app_local_path)
                
                # Convert to full URL as fallback
                url = f"http://label-studio:8080/{url}"
            
            # Handle local files (absolute paths)
            if url.startswith('/app/data/') or url.startswith('/label-studio/data/'):
                if os.path.exists(url):
                    logger.info(f"Loading image from absolute path: {url}")
                    return cv2.imread(url)
            
            # Handle data URLs
            if url.startswith('data:image'):
                import base64
                header, data = url.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_array
            
            # Download from HTTP with authentication for Label Studio
            logger.info(f"Attempting to download image from URL: {url}")
            headers = {}
            if 'label-studio' in url:
                # Try to use Label Studio token if available
                token = os.getenv('LABEL_STUDIO_TOKEN', '')
                if token:
                    headers['Authorization'] = f'Token {token}'
                
                # Forward cookies and session info from original request
                if request_headers:
                    if 'Cookie' in request_headers:
                        headers['Cookie'] = request_headers['Cookie']
                    if 'Authorization' in request_headers:
                        headers['Authorization'] = request_headers['Authorization']
                    if 'X-Csrftoken' in request_headers:
                        headers['X-Csrftoken'] = request_headers['X-Csrftoken']
                
                # Set user agent
                headers['User-Agent'] = 'Label-Studio-ML/1.0'
            
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            img_array = np.array(image)
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    def predict(self, tasks: List[Dict], request_headers: dict = None) -> List[Dict]:
        """Generate pose predictions."""
        if not RTMLIB_AVAILABLE or not self.pose_estimator:
            logger.warning("RTMLib not available, returning empty predictions")
            return [{"result": []} for _ in tasks]
        
        predictions = []
        
        for task in tasks:
            try:
                image_url = task['data'].get('image')
                if not image_url:
                    predictions.append({"result": []})
                    continue
                
                img = self._download_image(image_url, request_headers)
                if img is None:
                    predictions.append({"result": []})
                    continue
                
                image_height, image_width = img.shape[:2]
                
                # Perform pose estimation
                keypoints, scores = self.pose_estimator(img)
                
                if keypoints is None or len(keypoints) == 0:
                    predictions.append({"result": []})
                    continue
                
                # Format results
                result = []
                for person_idx, (person_keypoints, person_scores) in enumerate(zip(keypoints, scores)):
                    # Add skeleton connections
                    skeleton = self._create_skeleton_polygon(person_keypoints, person_scores, image_width, image_height)
                    if skeleton:
                        result.append(skeleton)
                    
                    # Add individual keypoints
                    for kp_idx, (keypoint, score) in enumerate(zip(person_keypoints, person_scores)):
                        if score > self.confidence_threshold and kp_idx < len(self.keypoint_labels):
                            x, y = keypoint
                            x_percent = (x / image_width) * 100
                            y_percent = (y / image_height) * 100
                            
                            result.append({
                                "from_name": "keypoints",
                                "to_name": "image", 
                                "type": "keypointlabels",
                                "value": {
                                    "x": x_percent,
                                    "y": y_percent,
                                    "keypointlabels": [self.keypoint_labels[kp_idx]],
                                    "width": 2,
                                    "height": 2
                                },
                                "score": float(score)
                            })
                
                prediction = {
                    "result": result,
                    "score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
                    "model_version": f"rtmlib-{self.mode}"
                }
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                predictions.append({"result": []})
        
        return predictions

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize backend
backend = SimpleRTMLibBackend()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "rtmlib_available": RTMLIB_AVAILABLE,
        "model_loaded": backend.pose_estimator is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        if not data or 'tasks' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        
        tasks = data['tasks']
        # Forward request headers for authentication
        request_headers = dict(request.headers)
        predictions = backend.predict(tasks, request_headers)
        
        # Label Studio expects specific format
        formatted_predictions = []
        for i, prediction in enumerate(predictions):
            formatted_predictions.append({
                "id": i,
                "result": prediction.get("result", []),
                "score": prediction.get("score", 0.0),
                "model_version": prediction.get("model_version", "rtmlib-v1.0.0")
            })
        
        return jsonify({"results": formatted_predictions})
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/setup', methods=['POST'])
def setup():
    """Setup endpoint for Label Studio compatibility."""
    try:
        data = request.get_json()
        
        # Label Studio sends the labeling configuration during setup
        labeling_config = data.get('schema', '')
        project_id = data.get('project', 'unknown')
        
        logger.info(f"Setup called for project {project_id}")
        logger.info(f"Labeling config received: {labeling_config[:200]}...")
        
        # Return setup response with model information
        return jsonify({
            "model_version": f"rtmlib-{backend.mode}-v1.0.0",
            "score_threshold": backend.confidence_threshold,
            "supported_formats": ["image"],
            "max_batch_size": 1,
            "setup_success": True
        })
        
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return jsonify({
            "error": str(e),
            "setup_success": False
        }), 500

@app.route('/fit', methods=['POST'])
def fit():
    """Fit/training endpoint."""
    try:
        data = request.get_json()
        event = data.get('event', 'UNKNOWN')
        logger.info(f"Received fit event: {event}")
        return jsonify({"status": "ok", "message": "Event logged"})
    except Exception as e:
        logger.error(f"Fit error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Training endpoint (alias for fit)."""
    return fit()

@app.route('/is_training', methods=['GET'])
def is_training():
    """Check if model is currently training."""
    return jsonify({"is_training": False})

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for Label Studio events."""
    try:
        data = request.get_json()
        action = data.get('action', 'unknown')
        logger.info(f"Webhook received: {action}")
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "RTMLib ML Backend",
        "version": "1.0.0",
        "status": "running"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9090))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting RTMLib ML Backend on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
