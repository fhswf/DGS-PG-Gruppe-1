# Label Studio with RTMLib ML Backend

This project provides a Docker setup for Label Studio with an integrated machine learning backend using RTMLib for pose estimation.

## Features

- **Label Studio**: Web-based data labeling platform
- **ML Backend**: RTMLib-based pose estimation for automatic annotation
- **Docker Integration**: Easy deployment with Docker Compose
- **Pose Estimation**: Wholebody pose estimation with balanced mode

## Project Structure

```
label-studio/
├── docker-compose.yml          # Docker services configuration
├── ml-backend/                 # ML backend service
│   ├── Dockerfile             # ML backend container
│   ├── requirements.txt       # Python dependencies
│   ├── model.py              # Main ML logic
│   ├── _wsgi.py              # WSGI application
│   └── tests/                # Unit tests
├── data/                     # Shared data volume
└── README.md                # This file
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for containers
- Internet connection for model downloads (first run only)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd label-studio
   ```

2. **Start the services:**
   ```bash
   docker-compose up -d
   ```
   
   **Note**: The first startup may take 5-10 minutes as the ML backend downloads the RTMLib models (approximately 200MB).

3. **Wait for model download:**
   ```bash
   # Check if the ML backend is ready
   curl http://localhost:9090/health
   
   # Monitor the download progress
   docker-compose logs -f ml-backend
   ```

4. **Access Label Studio:**
   - Open http://localhost:8080 in your browser
   - Create an admin account on first launch

5. **Configure ML Backend:**
   - Go to Settings > Model
   - Add ML Backend URL: `http://ml-backend:9090`
   - The backend will provide automatic pose estimation predictions

### Quick Test

Run the system test to verify everything is working:
```bash
./test_system.sh
```

### Usage

1. **Create a Project:**
   - Import images for pose estimation
   - Use the pose estimation labeling interface
   - The ML backend will automatically provide wholebody pose predictions

2. **Review and Correct:**
   - ML predictions appear as pre-annotations
   - Review and correct as needed
   - Submit final annotations

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
LABEL_STUDIO_API_KEY=your_api_key_here
```

### ML Backend Configuration

The ML backend uses RTMLib with the following default settings:
- **Model**: Wholebody pose estimation
- **Mode**: Balanced (performance vs accuracy)
- **Backend**: ONNX Runtime
- **Device**: CPU (configurable for GPU)

## Development

### Running Tests

```bash
docker-compose exec ml-backend python -m pytest tests/
```

### Debugging

View logs:
```bash
docker-compose logs -f ml-backend
docker-compose logs -f label-studio
```

### Custom Configuration

Modify `ml-backend/model.py` to:
- Change pose estimation models
- Adjust confidence thresholds
- Add custom preprocessing

## API Documentation

The ML backend provides these endpoints:

- `GET /health` - Health check
- `POST /predict` - Generate pose predictions
- `POST /fit` - Train/update model (optional)

## Troubleshooting

### Common Issues

1. **Container Memory Issues:**
   - Increase Docker memory allocation to 4GB+

2. **Model Download Failures:**
   - Check internet connection
   - Verify model URLs in configuration

3. **Connection Issues:**
   - Ensure both containers are on the same network
   - Check firewall settings

### Performance Optimization

- Use GPU-enabled Docker images for faster inference
- Adjust batch sizes in model configuration
- Cache models locally to avoid repeated downloads

## License

This project follows the Apache 2.0 license for RTMLib components and Label Studio's licensing terms.
