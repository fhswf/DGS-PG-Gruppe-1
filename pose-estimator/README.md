# RTMLib Pose Estimator

This module provides 2D whole-body pose estimation capabilities for both **videos** and **individual images** using RTMLib, specifically designed for sign language analysis.

## 🎯 **Features**

- **133-keypoint detection**: Full body (17) + Face (68) + Hands (42) + Feet (6)
- **Video processing**: Process entire video files frame by frame
- **Image processing**: Process individual images with annotation
- **Multiple performance modes**: performance, balanced, lightweight
- **Multiple backends**: ONNX Runtime, OpenCV, OpenVINO
- **Device support**: CPU, CUDA, MPS (Apple Silicon)

## 📁 **Files Overview**

- **`pose_estimator_2d.py`** - ⭐ **Main PoseEstimator2D class** (recommended)
- **`simple_image_pose.py`** - Command-line tool using PoseEstimator2D
- **`test_image_pose.py`** - Alternative implementation for testing
- **`quick_start_demo.py`** - Demo script
- **`example_usage.py`** - Usage examples

## 🚀 **Quick Start**

### **Command-Line Usage (Recommended)**

```bash
# Simple usage with high-quality defaults (performance mode, 0.8 threshold)
python simple_image_pose.py ../data/test_pose.png

# Data-only analysis (no image output)
python simple_image_pose.py ../data/test_pose.png --no-image

# Custom output path
python simple_image_pose.py ../data/test_pose.png --output my_result.png

# Export keypoints as JSON
python simple_image_pose.py ../data/test_pose.png --json keypoints.json

# Adjust quality settings
python simple_image_pose.py ../data/test_pose.png --threshold 0.5  # More keypoints
python simple_image_pose.py ../data/test_pose.png --mode balanced   # Faster processing
```

### **Programmatic Usage**

```python
from pose_estimator_2d import PoseEstimator2D

# Initialize estimator
estimator = PoseEstimator2D(
    mode='performance',     # performance, balanced, or lightweight
    device='cpu',          # cpu, cuda, or mps
    kpt_threshold=0.8      # confidence threshold
)

# Process single image (data only)
result = estimator.process_image('../data/test_pose.png')
print(f"Detected {result.num_persons} persons")

# Process image with annotation
result = estimator.process_image_with_annotation(
    image_path='../data/test_pose.png',
    output_path='../output/annotated.png'
)

# Process video
video_result = estimator.process_video('../data/video.mp4')
print(f"Processed {video_result.total_frames} frames")
```
# Process video
video_result = estimator.process_video('../data/video.mp4')
print(f"Processed {video_result.total_frames} frames")
```

## 🔧 **Installation**

```bash
# Install RTMLib
pip install rtmlib

# Install additional dependencies
pip install opencv-python numpy pathlib
```

## 📊 **PoseEstimator2D Class Reference**

### **Initialization Parameters**

```python
PoseEstimator2D(
    mode='performance',          # Model quality: 'performance', 'balanced', 'lightweight'
    backend='onnxruntime',       # Backend: 'onnxruntime', 'opencv', 'openvino'
    device='cpu',               # Device: 'cpu', 'cuda', 'mps'
    kpt_threshold=0.8,          # Confidence threshold for keypoint detection
    to_openpose=False           # Convert to OpenPose format (optional)
)
```

### **Main Methods**

#### **`process_image(image_path)`**
Process image and return keypoint data only.

```python
result = estimator.process_image('../data/test_pose.png')
# Returns: PoseResult with keypoints, scores, bboxes, etc.
```

#### **`process_image_with_annotation(image_path, output_path=None)`**
Process image and optionally save annotated version.

```python
# Save annotated image
result = estimator.process_image_with_annotation(
    '../data/test_pose.png', 
    '../output/annotated.png'
)

# Process without saving (data only)
result = estimator.process_image_with_annotation('../data/test_pose.png', None)
```

#### **`process_video(video_path, output_path=None)`**
Process entire video file.

```python
video_result = estimator.process_video('../data/video.mp4', '../output/annotated_video.mp4')
```

### **Result Objects**

#### **PoseResult (for images)**
```python
result.keypoints        # Array of keypoint coordinates [num_persons, 133, 2]
result.scores          # Confidence scores [num_persons, 133]
result.bboxes          # Bounding boxes [num_persons, 4]
result.num_persons     # Number of detected persons
```

#### **VideoResult (for videos)**
```python
video_result.frame_results    # List of PoseResult objects (one per frame)
video_result.total_frames     # Total number of frames processed
video_result.fps             # Frames per second
video_result.processing_time # Total processing time in seconds
```

## 📊 **Supported Keypoint Groups**

| Group | Keypoints | Range | Color |
|-------|-----------|-------|-------|
| **Body** | 17 | 0-16 | 🟢 Green |
| **Face** | 68 | 17-84 | 🔵 Blue |
| **Left Hand** | 21 | 85-105 | 🔴 Red |
| **Right Hand** | 21 | 106-126 | 🟡 Yellow |
| **Feet** | 6 | 127-132 | 🟣 Purple |

**Note**: All 133 keypoint coordinates are always returned, but only those above the confidence threshold are considered "detected".

## 🎨 **Output Examples**

### **Command-Line Output Structure**

```
output/pose-estimation/
├── annotated_test_pose.png     # Annotated image with keypoints
└── keypoints_test_pose.json    # JSON with all keypoint data (if --json used)
```

### **JSON Export Structure**

```json
{
  "frame_idx": 0,
  "num_persons": 1,
  "keypoints": [[x1, y1], [x2, y2], ...],  // Always 133 coordinate pairs
  "scores": [0.95, 0.87, 0.23, ...],       // Always 133 confidence scores
  "bboxes": [[x1, y1, x2, y2, conf]]       // Bounding boxes
}
```

### **Sample Output Structure**

```
output/pose-estimation/
├── annotated_test_pose.png     # Annotated image with keypoints
└── keypoints_test_pose.json    # JSON with all keypoint data
```

### **JSON Structure**

```json
{
  "image_info": {
    "filename": "test_pose.png",
    "num_persons": 1
  },
  "persons": [
    {
      "person_id": 0,
      "bbox": [x1, y1, x2, y2],
      "keypoints": [[x, y], ...],  // 133 keypoints
      "scores": [0.95, 0.87, ...], // 133 confidence scores
      "visible_keypoints": 105
    }
  ]
}
```

### **Configuration Options**

### **Performance Modes**

- **`performance`**: Highest accuracy, slower inference ⭐ **Default & Recommended**
- **`balanced`**: Good balance of speed and accuracy
- **`lightweight`**: Fastest inference, lower accuracy

### **Confidence Thresholds**

- **`0.8`**: High quality keypoints only ⭐ **Default**
- **`0.5`**: Good balance of coverage and quality
- **`0.3`**: Maximum coverage, includes lower confidence detections

### **Backends**

- **`onnxruntime`**: Best compatibility ⭐ **Recommended**
- **`opencv`**: Good for CPU-only environments
- **`openvino`**: Optimized for Intel hardware

### **Devices**

- **`cpu`**: Universal compatibility ⭐ **Default**
- **`cuda`**: NVIDIA GPU acceleration
- **`mps`**: Apple Silicon GPU acceleration

## 📈 **Performance Examples**

Based on test image (1326x1916 pixels):

| Configuration | Time | Keypoints Detected | Avg Confidence | Quality |
|---------------|------|-------------------|----------------|---------|
| `performance + 0.8` ⭐ | ~4s | **117/133** | **6.522** | **High** |
| `performance + 0.5` | ~4s | 132/133 | 5.857 | Mixed |
| `balanced + 0.8` | ~2s | 110/133 | 6.200 | High |
| `lightweight + 0.8` | ~1s | 95/133 | 5.800 | High |

## 🔍 **Detailed Usage Examples**

### **1. Basic Image Processing**

```python
from test_image_pose import ImagePoseEstimator

estimator = ImagePoseEstimator(mode='balanced', device='cpu')
result = estimator.process_single_image(
    image_path='../data/test_pose.png',
    output_path='../output/result.png'
)

print(f"Detected {result.num_persons} persons")
```

### **2. Custom Configuration**

```python
estimator = ImagePoseEstimator(
    mode='performance',
    backend='onnxruntime',
    device='cpu',
    kpt_threshold=0.5,  # Higher threshold for better quality
    to_openpose=False
)
```

### **3. Batch Processing**

```python
import glob
from pathlib import Path

estimator = ImagePoseEstimator()
input_dir = Path('../data/images')
output_dir = Path('../output/batch_results')

for image_path in input_dir.glob('*.png'):
    output_path = output_dir / f"annotated_{image_path.name}"
    result = estimator.process_single_image(str(image_path), str(output_path))
    print(f"Processed {image_path.name}: {result.num_persons} persons")
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **RTMLib not found**
   ```bash
   pip install rtmlib
   ```

2. **ONNX Runtime issues**
   ```bash
   pip install onnxruntime
   ```

3. **No persons detected**
   - Try lowering `kpt_threshold` (e.g., 0.1)
   - Check image quality and lighting
   - Ensure persons are clearly visible

4. **Low confidence scores**
   - Use `performance` mode for better accuracy
   - Ensure good image quality
   - Check for occlusions or motion blur

## 🤝 **Integration with Label Studio**

The pose estimation results can be directly used with the ML backend for Label Studio annotation. See the [`ml-backend/README.md`](../ml-backend/README.md) for integration details.

---

**For sign language research at FH Südwestfalen - DGS Project Group 1**
