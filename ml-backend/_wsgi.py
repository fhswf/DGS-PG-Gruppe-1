"""
WSGI application entry point for the RTMLib ML backend.
"""

import os
import logging
from label_studio_ml.api import init_app
from model import RTMLibPoseEstimator


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def create_app():
	"""Create and configure the Flask application."""
	try:
		app = init_app(model_class=RTMLibPoseEstimator)
		logger.info("RTMLib ML Backend application created successfully")
		return app
	except Exception as e:
		logger.exception(f"Failed to create application: {e}")
		raise


app = create_app()


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 9090))
	host = os.environ.get("HOST", "0.0.0.0")
	debug = os.environ.get("DEBUG", "False").lower() == "true"

	logger.info(f"Starting RTMLib ML Backend on {host}:{port}")
	app.run(host=host, port=port, debug=debug)

