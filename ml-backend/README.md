# RTMLib ML Backend for Label Studio

This ML backend integrates **rtmlib Wholebody** pose estimation (133 keypoints) with Label Studio for automatic keypoint annotation. It provides real-time pose detection for body, face, and hands in images.

## Features

- **133 Keypoints Detection:** 17 body + 6 feet + 68 face + 21 left hand + 21 right hand keypoints
- **Multiple Backends:** ONNX Runtime, OpenCV, OpenVINO support
- **Flexible Device Support:** CPU, CUDA, MPS (Apple Silicon)
- **Label Studio Integration:** Seamless ML-assisted annotation workflow
- **Docker Support:** Easy deployment and scaling

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Label Studio running (e.g., on http://localhost:8080)
- API Token from Label Studio

### 2. Setup Environment

Copy the environment template and configure:

```bash
cp .env.example .env
# Edit .env with your Label Studio URL and API token
```

### 3. Build and Run

```bash
# Build the Docker image
docker compose build

# Start the ML backend
docker compose up -d

# Check health
curl http://localhost:9090/health
```

### 4. Connect to Label Studio

1. Go to Label Studio: **Settings** → **Model** → **Add Model**
2. Set URL: `http://localhost:9090`
3. Enable **Use for interactive preannotations**

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Label Studio Configuration
LABEL_STUDIO_URL=http://host.docker.internal:8080
LABEL_STUDIO_API_TOKEN=your_api_token_here

# ML Backend Configuration
BACKEND_PORT=9090
DEVICE=cpu                    # cpu, cuda, mps
BACKEND_TYPE=onnxruntime      # onnxruntime, opencv, openvino
MODE=balanced                 # performance, lightweight, balanced

# Optional: Debug settings
DEBUG=false
LOG_LEVEL=INFO
```

### Getting Label Studio API Token

1. **Login to Label Studio**
2. **Go to Account & Settings:**
   - Click your profile picture (top right)
   - Select "Account & Settings"
3. **Access Token:**
   - Go to "Access Token" tab
   - Copy your existing token or click "Reset Token" to generate new one
   
   **Alternative method for legacy tokens:**
   - Go to Django Admin: `http://localhost:8080/admin/`
   - Login with admin credentials
   - Navigate to: **AUTH TOKEN** → **Tokens**
   - Find your user and copy the token

⚠️ **Important:** Keep your API token secure and never commit it to version control.

## Label Studio Project Configuration

### Labeling Configuration (labeling-config.xml)

Use this XML configuration in your Label Studio project:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomBy="1.5" zoomControl="true"/>
  
  <KeyPointLabels name="keypoints" toName="image" strokeWidth="2" pointSize="small">
    
    <!-- Body Keypoints (17) -->
    <Label value="nose" background="red"/>
    <Label value="left_eye" background="blue"/>
    <Label value="right_eye" background="blue"/>
    <Label value="left_ear" background="green"/>
    <Label value="right_ear" background="green"/>
    <Label value="left_shoulder" background="purple"/>
    <Label value="right_shoulder" background="purple"/>
    <Label value="left_elbow" background="orange"/>
    <Label value="right_elbow" background="orange"/>
    <Label value="left_wrist" background="yellow"/>
    <Label value="right_wrist" background="yellow"/>
    <Label value="left_hip" background="pink"/>
    <Label value="right_hip" background="pink"/>
    <Label value="left_knee" background="cyan"/>
    <Label value="right_knee" background="cyan"/>
    <Label value="left_ankle" background="brown"/>
    <Label value="right_ankle" background="brown"/>
    
    <!-- Feet Keypoints (6) -->
    <Label value="left_big_toe" background="darkred"/>
    <Label value="left_small_toe" background="darkred"/>
    <Label value="left_heel" background="darkred"/>
    <Label value="right_big_toe" background="darkblue"/>
    <Label value="right_small_toe" background="darkblue"/>
    <Label value="right_heel" background="darkblue"/>
    
    <!-- Face Keypoints (68) -->
    <!-- Face keypoints: face_0 to face_67 -->
    <Label value="face_0" background="lightgray"/>
    <Label value="face_1" background="lightgray"/>
    <!-- ... add face_2 through face_66 ... -->
    <Label value="face_67" background="lightgray"/>
    
    <!-- Left Hand Keypoints (21) -->
    <!-- Hand keypoints: left_hand_0 to left_hand_20 -->
    <Label value="left_hand_0" background="lightgreen"/>
    <Label value="left_hand_1" background="lightgreen"/>
    <!-- ... add left_hand_2 through left_hand_19 ... -->
    <Label value="left_hand_20" background="lightgreen"/>
    
    <!-- Right Hand Keypoints (21) -->
    <!-- Hand keypoints: right_hand_0 to right_hand_20 -->
    <Label value="right_hand_0" background="lightcoral"/>
    <Label value="right_hand_1" background="lightcoral"/>
    <!-- ... add right_hand_2 through right_hand_19 ... -->
    <Label value="right_hand_20" background="lightcoral"/>
    
  </KeyPointLabels>
</View>
```

### Setting up Label Studio Project

1. **Create New Project**
2. **Import Data:** Upload your images
3. **Labeling Setup:** 
   - Use the configuration above
   - Or use the provided `labeling-config.xml` file
4. **Settings → Model:**
   - Add ML Backend URL: `http://localhost:9090`
   - Enable "Use for interactive preannotations"
   - Test connection

## Development

### Manual Setup (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export LABEL_STUDIO_URL=http://localhost:8080
export LABEL_STUDIO_API_TOKEN=your_token

# Run the backend
python -m label_studio_ml.api

# Or using the WSGI app directly
python _wsgi.py
```

### Testing

```bash
# Test health endpoint
curl http://localhost:9090/health

# Test prediction
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "data": {
        "image": "http://localhost:8080/data/upload/1/your-image.jpg"
      }
    }]
  }'
```

### Building Custom Image

```bash
# Build with specific tag
docker build -t your-registry/rtmlib-ml-backend:v1.0 .

# Push to registry
docker push your-registry/rtmlib-ml-backend:v1.0
```

## Keypoint Schema

The model detects 133 keypoints following the COCO-WholeBody format:

- **Body (17):** Standard COCO pose keypoints
- **Feet (6):** Toe and heel points for both feet  
- **Face (68):** Facial landmarks following 68-point standard
- **Hands (42):** 21 keypoints per hand (left_hand_*, right_hand_*)

## API Endpoints

- `GET /health` - Health check and model status
- `POST /predict` - Generate keypoint predictions
- `POST /setup` - Initialize model (called by Label Studio)
- `POST /train` - Training endpoint (not implemented)
- `POST /webhook` - Webhook for Label Studio events

## Troubleshooting

### Common Issues

1. **Container can't reach Label Studio:**
   - Use `host.docker.internal:8080` instead of `localhost:8080`
   - Check Label Studio is running and accessible

2. **API Authentication errors:**
   - Verify API token is correct and not expired
   - Check Label Studio user permissions

3. **No keypoints detected:**
   - Check image format and quality
   - Adjust `CONFIDENCE_THRESHOLD` in environment
   - Verify labeling configuration matches model output

4. **Performance issues:**
   - Switch to GPU: set `DEVICE=cuda`
   - Use performance mode: set `MODE=performance`
   - Consider using lighter backend: `BACKEND_TYPE=opencv`

### Logs

```bash
# View real-time logs
docker compose logs -f ml-backend

# Debug mode
# Set DEBUG=true in .env and restart
```

## License

This project integrates with RTMLib and Label Studio. Please check their respective licenses for usage terms.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test with Label Studio
5. Submit pull request
- POST `/train` – optional training hook (no-op here)

For the full API and format details, see:
- https://labelstud.io/guide/ml_create
- https://labelstud.io/guide/task_format
- https://labelstud.io/guide/export#Label-Studio-JSON-format-of-annotated-tasks

## License
Apache-2.0 for rtmlib; follow dependencies' licenses.
