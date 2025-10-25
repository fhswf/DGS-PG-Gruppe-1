# Label Studio Pose Estimation Integration

This script automatically generates pose estimation predictions for Label Studio tasks using RTMLib.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install label-studio-sdk rtmlib opencv-python numpy requests pillow python-dotenv
```

### 2. Configure Environment
Copy and edit the environment file:
```bash
cp .env.example .env
```

Edit `.env` with your Label Studio credentials:
```bash
LABEL_STUDIO_URL=https://your-labelstudio-instance.com
LABEL_STUDIO_API_TOKEN=your-api-token-here
PROJECT_ID=your-project-id
```

### 3. Run the Script
```bash
python create_predictions.py
```

## ⚙️ Configuration Options

### Environment Variables
- `LABEL_STUDIO_URL`: Your Label Studio instance URL
- `LABEL_STUDIO_API_TOKEN`: API token for authentication
- `PROJECT_ID`: Label Studio project ID to process
- `TASK_LIMIT`: Maximum tasks to process (default: 1)
- `START_TASK_ID`: Resume from specific task ID (optional)
- `REQUEST_DELAY_SECONDS`: Delay between API requests to prevent server overload (default: 1.0)
- `MAX_RETRIES`: Maximum retry attempts for failed requests (default: 3)
- `MAX_BACKOFF_DELAY`: Maximum delay for exponential backoff in seconds (default: 60)

### Optional Settings
- `DEVICE`: cpu/cuda/mps (default: cpu)
- `BACKEND_TYPE`: onnxruntime/opencv/openvino (default: onnxruntime)
- `MODE`: performance/balanced/lightweight (default: balanced)
- `CONFIDENCE_THRESHOLD`: Minimum confidence score (default: 0.3)

## 📋 Usage Examples

### Process All Tasks
```bash
python create_predictions.py
```

### Start from Specific Task
```bash
START_TASK_ID=100 python create_predictions.py
```

### Limit Number of Tasks
```bash
TASK_LIMIT=50 python create_predictions.py
```

### Custom Configuration
```bash
LABEL_STUDIO_URL=https://custom-instance.com \
PROJECT_ID=123 \
TASK_LIMIT=25 \
python create_predictions.py
```

## 🔍 What It Does

1. **Connects** to Label Studio using provided credentials
2. **Fetches tasks** from the specified project (starting from START_TASK_ID if provided)
3. **Skips tasks** that already have predictions or annotations
4. **Downloads images** from Label Studio
5. **Runs pose estimation** using RTMLib Wholebody model (133 keypoints)
6. **Sends predictions** back to Label Studio with rate limiting to prevent server overload
7. **Automatically retries** failed requests (503 errors) with exponential backoff
8. **Reports progress** and any errors

### Rate Limiting & Error Handling

The script includes built-in rate limiting to prevent server overload:
- **Request Delays**: Configurable delay between API requests (default: 1 second)
- **Retry Logic**: Automatic retry for 503 Service Unavailable errors
- **Exponential Backoff**: Increasing delays between retry attempts (1s → 2s → 4s → ...)
- **Maximum Retries**: Configurable limit on retry attempts (default: 3)
- **Backoff Cap**: Maximum delay between retries (default: 60 seconds)

## 🛠️ Troubleshooting

### Connection Issues
- Verify `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_TOKEN`
- Check network connectivity to Label Studio instance

### Model Loading Issues
- Ensure internet connection for first-time model download (~200MB)
- Check available disk space

### Permission Errors
- Verify API token has write permissions for the project
- Check project ID is correct

### Rate Limiting & Server Overload
- **503 Errors**: Increase `REQUEST_DELAY_SECONDS` (try 2.0 or 3.0)
- **Timeout Issues**: Reduce `TASK_LIMIT` or increase delays between requests
- **Server Capacity**: Monitor server load and adjust rate limiting accordingly
- **Retry Configuration**: Adjust `MAX_RETRIES` and `MAX_BACKOFF_DELAY` based on server responsiveness

## 📊 Output

The script provides minimal output during processing:
- ✅ Connection status and model initialization
- 📋 Task processing progress (skipping existing predictions)
- ❌ Error messages for failed predictions
- 📈 Final summary of successful predictions

## 🔒 Security

- Never commit `.env` files containing real API tokens
- Use `.env.example` as a template for team members
- Each developer should maintain their own `.env` file