"""
WSGI application entry point for the RTMLib ML backend.
"""

import os
import logging
from label_studio_ml.api import init_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    try:
        # Initialize the Label Studio ML backend app
        app = init_app(
            model_class='model.RTMLibPoseEstimator',
            model_dir=os.environ.get('MODEL_DIR', '/app/models'),
            redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
            redis_host=os.environ.get('REDIS_HOST', 'localhost'),
            redis_port=int(os.environ.get('REDIS_PORT', 6379)),
            redis_db=int(os.environ.get('REDIS_DB', 0))
        )
        
        logger.info("RTMLib ML Backend application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        raise

# Create the application instance
app = create_app()

if __name__ == '__main__':
    # Run the application for development
    port = int(os.environ.get('PORT', 9090))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting RTMLib ML Backend on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
