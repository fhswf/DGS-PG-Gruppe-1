"""
Unit tests for the RTMLib ML Backend.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import cv2
from model import RTMLibPoseEstimator


class TestRTMLibPoseEstimator:
    """Test suite for RTMLibPoseEstimator class."""

    @pytest.fixture
    def mock_pose_estimator(self):
        """Create a mock pose estimator for testing."""
        with patch('model.Wholebody') as mock_wholebody:
            mock_instance = Mock()
            mock_wholebody.return_value = mock_instance
            
            estimator = RTMLibPoseEstimator()
            estimator.pose_estimator = mock_instance
            
            yield estimator, mock_instance

    def test_initialization(self):
        """Test proper initialization of the RTMLibPoseEstimator."""
        with patch('model.Wholebody') as mock_wholebody:
            mock_wholebody.return_value = Mock()
            
            estimator = RTMLibPoseEstimator()
            
            assert estimator.device == 'cpu'
            assert estimator.backend == 'onnxruntime'
            assert estimator.mode == 'balanced'
            assert estimator.confidence_threshold == 0.3
            assert estimator.pose_estimator is not None
            assert len(estimator.keypoint_labels) == 133  # Wholebody keypoints

    def test_keypoint_labels(self):
        """Test that keypoint labels are properly defined."""
        with patch('model.Wholebody'):
            estimator = RTMLibPoseEstimator()
            labels = estimator.keypoint_labels
            
            # Check total count (17 body + 6 foot + 68 face + 21*2 hands = 133)
            assert len(labels) == 133
            
            # Check some key labels
            assert 'nose' in labels
            assert 'left_shoulder' in labels
            assert 'right_ankle' in labels
            assert 'left_big_toe' in labels
            assert 'face_0' in labels
            assert 'left_hand_0' in labels
            assert 'right_hand_20' in labels

    @patch('model.cv2.imread')
    @patch('model.os.path.exists')
    def test_download_image_local_file(self, mock_exists, mock_imread):
        """Test downloading image from local file path."""
        with patch('model.Wholebody'):
            estimator = RTMLibPoseEstimator()
            
            # Setup mocks
            mock_exists.return_value = True
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            mock_imread.return_value = test_image
            
            # Test local file URL
            result = estimator._download_image('/data/test_image.jpg')
            
            assert result is not None
            np.testing.assert_array_equal(result, test_image)
            mock_exists.assert_called_once()
            mock_imread.assert_called_once()

    @patch('model.requests.get')
    def test_download_image_http_url(self, mock_get):
        """Test downloading image from HTTP URL."""
        with patch('model.Wholebody'):
            estimator = RTMLibPoseEstimator()
            
            # Create a mock image
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Mock PIL Image and requests
            with patch('model.Image.open') as mock_image_open:
                mock_pil_image = Mock()
                mock_pil_image.size = (100, 100)
                mock_image_open.return_value = mock_pil_image
                
                # Mock numpy array conversion
                with patch('model.np.array') as mock_array:
                    mock_array.return_value = test_image
                    
                    # Mock cv2.cvtColor
                    with patch('model.cv2.cvtColor') as mock_cvtColor:
                        mock_cvtColor.return_value = test_image
                        
                        # Setup requests mock
                        mock_response = Mock()
                        mock_response.content = b'fake_image_data'
                        mock_get.return_value = mock_response
                        
                        result = estimator._download_image('http://example.com/image.jpg')
                        
                        assert result is not None
                        mock_get.assert_called_once()

    def test_format_keypoints_for_label_studio(self):
        """Test keypoint formatting for Label Studio."""
        with patch('model.Wholebody'):
            estimator = RTMLibPoseEstimator()
            
            # Test data
            keypoints = np.array([[50, 60], [70, 80], [10, 20]])
            scores = np.array([0.9, 0.8, 0.2])  # Last one below threshold
            image_width, image_height = 200, 150
            
            result = estimator._format_keypoints_for_label_studio(
                keypoints, scores, image_width, image_height
            )
            
            # Should only return 2 keypoints (score >= 0.3)
            assert len(result) == 2
            
            # Check first keypoint
            first_kp = result[0]
            assert first_kp['type'] == 'keypointlabels'
            assert first_kp['value']['x'] == 25.0  # 50/200 * 100
            assert first_kp['value']['y'] == 40.0  # 60/150 * 100
            assert first_kp['score'] == 0.9
            assert first_kp['value']['keypointlabels'] == ['nose']  # First label

    def test_predict_success(self, mock_pose_estimator):
        """Test successful pose prediction."""
        estimator, mock_instance = mock_pose_estimator
        
        # Mock pose estimation results
        mock_keypoints = np.array([[[50, 60], [70, 80]]])  # 1 person, 2 keypoints
        mock_scores = np.array([[0.9, 0.8]])  # 1 person, 2 scores
        mock_instance.return_value = (mock_keypoints, mock_scores)
        
        # Mock image download
        test_image = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
        with patch.object(estimator, '_download_image') as mock_download:
            mock_download.return_value = test_image
            
            # Test task
            tasks = [{'data': {'image': 'http://example.com/test.jpg'}}]
            
            predictions = estimator.predict(tasks)
            
            assert len(predictions) == 1
            prediction = predictions[0]
            assert 'result' in prediction
            assert 'score' in prediction
            assert 'model_version' in prediction
            assert len(prediction['result']) == 2  # 2 keypoints above threshold

    def test_predict_no_image_url(self, mock_pose_estimator):
        """Test prediction with missing image URL."""
        estimator, _ = mock_pose_estimator
        
        tasks = [{'data': {}}]  # No image URL
        predictions = estimator.predict(tasks)
        
        assert len(predictions) == 0

    def test_predict_image_download_failure(self, mock_pose_estimator):
        """Test prediction when image download fails."""
        estimator, _ = mock_pose_estimator
        
        with patch.object(estimator, '_download_image') as mock_download:
            mock_download.return_value = None  # Download failure
            
            tasks = [{'data': {'image': 'http://example.com/test.jpg'}}]
            predictions = estimator.predict(tasks)
            
            assert len(predictions) == 0

    def test_predict_no_pose_detected(self, mock_pose_estimator):
        """Test prediction when no pose is detected."""
        estimator, mock_instance = mock_pose_estimator
        
        # Mock no pose detection
        mock_instance.return_value = (None, None)
        
        test_image = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
        with patch.object(estimator, '_download_image') as mock_download:
            mock_download.return_value = test_image
            
            tasks = [{'data': {'image': 'http://example.com/test.jpg'}}]
            predictions = estimator.predict(tasks)
            
            assert len(predictions) == 1
            assert predictions[0]['result'] == []

    def test_fit_method(self, mock_pose_estimator):
        """Test the fit method for handling Label Studio events."""
        estimator, _ = mock_pose_estimator
        
        # Test annotation created event
        event = 'ANNOTATION_CREATED'
        data = {
            'annotation': {'id': 123},
            'task': {'id': 456}
        }
        
        # Should not raise any exceptions
        estimator.fit(event, data)

    def test_get_train_job_status(self, mock_pose_estimator):
        """Test getting training job status."""
        estimator, _ = mock_pose_estimator
        
        status = estimator.get_train_job_status('test_job_id')
        
        assert status['job_status'] == 'completed'
        assert status['error'] is None
        assert 'pre-trained' in status['log']

    @patch.dict('os.environ', {
        'DEVICE': 'cuda',
        'BACKEND': 'onnxruntime',
        'MODE': 'performance',
        'CONFIDENCE_THRESHOLD': '0.5'
    })
    def test_environment_configuration(self):
        """Test configuration from environment variables."""
        with patch('model.Wholebody') as mock_wholebody:
            mock_wholebody.return_value = Mock()
            
            estimator = RTMLibPoseEstimator()
            
            assert estimator.device == 'cuda'
            assert estimator.backend == 'onnxruntime'
            assert estimator.mode == 'performance'
            assert estimator.confidence_threshold == 0.5


if __name__ == '__main__':
    pytest.main([__file__])
