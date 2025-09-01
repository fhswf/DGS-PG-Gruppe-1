"""
Integration tests for the RTMLib ML Backend API.
"""

import json
import pytest
import responses
from unittest.mock import patch, Mock
import numpy as np


class TestMLBackendAPI:
    """Test suite for the ML Backend API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask application."""
        with patch('model.Wholebody') as mock_wholebody:
            mock_wholebody.return_value = Mock()
            
            from _wsgi import create_app
            app = create_app()
            app.config['TESTING'] = True
            
            with app.test_client() as client:
                yield client

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200

    @responses.activate
    def test_predict_endpoint(self, client):
        """Test the predict endpoint with a valid request."""
        # Mock external image URL
        responses.add(
            responses.GET,
            'http://example.com/test.jpg',
            body=b'fake_image_data',
            status=200,
            content_type='image/jpeg'
        )
        
        # Mock the pose estimation
        with patch('model.RTMLibPoseEstimator.predict') as mock_predict:
            mock_predict.return_value = [{
                'result': [{
                    'from_name': 'keypoints',
                    'to_name': 'image',
                    'type': 'keypointlabels',
                    'value': {
                        'x': 25.0,
                        'y': 40.0,
                        'keypointlabels': ['nose'],
                        'width': 2,
                        'height': 2
                    },
                    'score': 0.9
                }],
                'score': 0.9,
                'model_version': 'rtmlib-balanced'
            }]
            
            # Test request
            request_data = {
                'tasks': [{
                    'data': {'image': 'http://example.com/test.jpg'}
                }]
            }
            
            response = client.post('/predict',
                                 data=json.dumps(request_data),
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'results' in data
            assert len(data['results']) == 1

    def test_predict_endpoint_invalid_data(self, client):
        """Test the predict endpoint with invalid data."""
        response = client.post('/predict',
                             data='invalid json',
                             content_type='application/json')
        
        assert response.status_code == 400

    def test_fit_endpoint(self, client):
        """Test the fit endpoint."""
        request_data = {
            'event': 'ANNOTATION_CREATED',
            'data': {
                'annotation': {'id': 123},
                'task': {'id': 456}
            }
        }
        
        response = client.post('/fit',
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        assert response.status_code == 200

    def test_train_job_status_endpoint(self, client):
        """Test the train job status endpoint."""
        response = client.get('/train_job_status/test_job_id')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['job_status'] == 'completed'


if __name__ == '__main__':
    pytest.main([__file__])
