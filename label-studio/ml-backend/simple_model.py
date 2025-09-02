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
        
        # Set up model cache directory
        self.model_dir = os.environ.get('MODEL_DIR', '/app/models')
        self._setup_model_cache()
        
        # Initialize keypoint labels first
        self.keypoint_labels = self._get_keypoint_labels()
        
        self.pose_estimator = None
        if RTMLIB_AVAILABLE:
            self._initialize_model()
        
        logger.info(f"Backend initialized: device={self.device}, mode={self.mode}")

    def _setup_model_cache(self):
        """Set up persistent model cache directory."""
        try:
            # Create cache directories
            os.makedirs(self.model_dir, exist_ok=True)
            cache_dir = os.path.join(self.model_dir, 'rtmlib_cache')
            hub_dir = os.path.join(cache_dir, 'hub')
            checkpoints_dir = os.path.join(hub_dir, 'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # Set RTMLib cache environment variables
            os.environ['RTMLIB_HOME'] = cache_dir
            os.environ['HUB_CACHE_DIR'] = hub_dir
            
            # Also set torch hub cache if available
            torch_cache_dir = os.path.join(self.model_dir, 'torch_cache')
            os.makedirs(torch_cache_dir, exist_ok=True)
            os.environ['TORCH_HOME'] = torch_cache_dir
            
            # Check if models are already cached
            cached_files = []
            cache_paths = [
                checkpoints_dir,  # our configured path
                os.path.join(torch_cache_dir, 'hub', 'checkpoints'),  # torch hub path
                os.path.expanduser('~/.cache/rtmlib/hub/checkpoints'),  # default rtmlib path
            ]
            
            for cache_path in cache_paths:
                if os.path.exists(cache_path):
                    files = [f for f in os.listdir(cache_path) if f.endswith(('.onnx', '.zip'))]
                    cached_files.extend(files)
            
            if cached_files:
                logger.info(f"Found {len(cached_files)} cached model files")
            else:
                logger.info("No cached models found, will download on first use")
            
            logger.info(f"Model cache configured at: {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to setup model cache: {e}")

    def _initialize_model(self):
        """Initialize the RTMLib model."""
        try:
            logger.info("Initializing RTMLib model with persistent cache...")
            self.pose_estimator = Wholebody(
                to_openpose=False,
                mode=self.mode,
                backend=self.backend,
                device=self.device
            )
            logger.info("RTMLib model initialized successfully")
            
            # Log expected keypoint count
            logger.info(f"Expected keypoint labels: {len(self.keypoint_labels)}")
        except Exception as e:
            logger.error(f"Failed to initialize RTMLib model: {e}")
            self.pose_estimator = None

    def _get_keypoint_labels(self) -> List[str]:
        """Get keypoint labels matching the labeling config."""
        # Body keypoints (first 17)
        body_labels = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Foot keypoints (3 per foot = 6 total)
        foot_labels = [
            'left_big_toe', 'left_small_toe', 'left_heel',
            'right_big_toe', 'right_small_toe', 'right_heel'
        ]
        
        # Face keypoints (68 points)
        face_labels = [f'face_{i}' for i in range(68)]
        
        # Left hand keypoints (21 points)
        left_hand_labels = [f'left_hand_{i}' for i in range(21)]
        
        # Right hand keypoints (21 points)
        right_hand_labels = [f'right_hand_{i}' for i in range(21)]
        
        # Combine all labels: 17 + 6 + 68 + 21 + 21 = 133 total
        all_labels = body_labels + foot_labels + face_labels + left_hand_labels + right_hand_labels
        
        return all_labels

    def _download_image(self, url: str, request_headers: dict = None) -> Optional[np.ndarray]:
        """Download image from URL."""
        try:
            # Handle relative URLs starting with /data/
            if url.startswith('/data/'):
                # Try multiple local file paths
                local_paths = [
                    f"/app{url}",  # /app/data/...
                    f"/label-studio{url}",  # /label-studio/data/...
                    f"/label-studio/data/media{url[5:]}",  # Remove '/data' and add to media path
                ]
                
                for local_path in local_paths:
                    if os.path.exists(local_path):
                        logger.info(f"Loading image from local path: {local_path}")
                        img = cv2.imread(local_path)
                        if img is not None:
                            return img
                
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
                    # Add individual keypoints only (no skeleton)
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
                                    "width": 0.1,  # Default keypoint size as percentage
                                    "keypointlabels": [self.keypoint_labels[kp_idx]]
                                },
                                "score": float(score)
                            })
                        elif kp_idx >= len(self.keypoint_labels):
                            # Log if we have more keypoints than labels
                            logger.warning(f"Keypoint index {kp_idx} exceeds available labels ({len(self.keypoint_labels)})")
                
                # Log keypoint counts for debugging
                if len(keypoints) > 0:
                    logger.info(f"Detected {len(keypoints[0])} keypoints, using {len([r for r in result if r['type'] == 'keypointlabels'])} above threshold")
                
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
