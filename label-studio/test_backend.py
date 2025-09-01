"""
Example script demonstrating how to use the RTMLib ML Backend
for pose estimation in Label Studio.
"""

import requests
import json
import base64
import cv2
import numpy as np
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def test_ml_backend_prediction(backend_url: str, image_path: str):
    """
    Test the ML backend prediction endpoint.
    
    Args:
        backend_url: URL of the ML backend (e.g., http://localhost:9090)
        image_path: Path to the test image
    """
    # Prepare the request data
    tasks = [{
        'data': {
            'image': f'data:image/jpeg;base64,{encode_image_to_base64(image_path)}'
        }
    }]
    
    request_data = {
        'tasks': tasks,
        'label_config': '''
        <View>
          <Image name="image" value="$image"/>
          <KeyPointLabels name="keypoints" toName="image">
            <Label value="nose"/>
            <Label value="left_eye"/>
            <Label value="right_eye"/>
            <!-- Add more labels as needed -->
          </KeyPointLabels>
        </View>
        '''
    }
    
    try:
        # Send prediction request
        response = requests.post(
            f"{backend_url}/predict",
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Prediction successful!")
            print(f"Number of predictions: {len(results.get('results', []))}")
            
            for i, prediction in enumerate(results.get('results', [])):
                keypoints = prediction.get('result', [])
                score = prediction.get('score', 0)
                print(f"  Prediction {i+1}: {len(keypoints)} keypoints, score: {score:.3f}")
            
            return results
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def test_health_check(backend_url: str):
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{backend_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ ML Backend is healthy")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def visualize_predictions(image_path: str, predictions: dict, output_path: str = None):
    """
    Visualize pose predictions on the image.
    
    Args:
        image_path: Path to the original image
        predictions: Prediction results from ML backend
        output_path: Path to save the visualization (optional)
    """
    try:
        from rtmlib import draw_skeleton
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        height, width = img.shape[:2]
        
        # Extract keypoints from predictions
        for prediction in predictions.get('results', []):
            keypoints = []
            scores = []
            
            for result in prediction.get('result', []):
                if result.get('type') == 'keypointlabels':
                    value = result.get('value', {})
                    x_percent = value.get('x', 0)
                    y_percent = value.get('y', 0)
                    score = result.get('score', 0)
                    
                    # Convert percentages to pixels
                    x = int(x_percent * width / 100)
                    y = int(y_percent * height / 100)
                    
                    keypoints.append([x, y])
                    scores.append(score)
            
            if keypoints:
                keypoints = np.array([keypoints])  # Add batch dimension
                scores = np.array([scores])  # Add batch dimension
                
                # Draw skeleton
                img = draw_skeleton(img, keypoints, scores, kpt_thr=0.3)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"✅ Visualization saved to: {output_path}")
        else:
            cv2.imshow('Pose Estimation Results', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except ImportError:
        print("⚠️  rtmlib not available for visualization")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


def main():
    """Main function to test the ML backend."""
    backend_url = "http://localhost:9090"
    
    # Test image path (you can change this to your test image)
    test_image = "../data/test.mov"  # Adjust path as needed
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        print("Please provide a valid image path or place a test image in the data directory.")
        return
    
    print("🧪 Testing RTMLib ML Backend")
    print("=" * 40)
    
    # Test health check
    print("1. Testing health check...")
    if not test_health_check(backend_url):
        print("Please make sure the ML backend is running with: docker-compose up")
        return
    
    # Test prediction
    print("\n2. Testing pose prediction...")
    predictions = test_ml_backend_prediction(backend_url, test_image)
    
    if predictions:
        # Visualize results
        print("\n3. Generating visualization...")
        output_file = "pose_prediction_result.jpg"
        visualize_predictions(test_image, predictions, output_file)
        
        print(f"\n🎉 Test completed successfully!")
        print(f"Check {output_file} for the visualization.")
    else:
        print("\n❌ Test failed!")


if __name__ == "__main__":
    main()
