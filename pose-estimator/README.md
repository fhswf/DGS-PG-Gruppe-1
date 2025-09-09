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

- **`pose_estimator_2d.py`** - Main PoseEstimator2D class for video processing
- **`test_image_pose.py`** - Extended ImagePoseEstimator for single images
- **`simple_image_pose.py`** - Command-line tool for quick image processing
- **`quick_start_demo.py`** - Demo script
- **`example_usage.py`** - Usage examples

## 🚀 **Quick Start**

### **Process a Single Image**

```bash
# Simple usage
python simple_image_pose.py ../data/test_pose.png

# Specify output path
python simple_image_pose.py ../data/test_pose.png ../output/my_result.png
```

### **Comprehensive Image Analysis**

```bash
# Run detailed analysis with JSON export
python test_image_pose.py
```

### **Process a Video**

```python
from pose_estimator_2d import PoseEstimator2D

estimator = PoseEstimator2D(mode='balanced', device='cpu')
result = estimator.process_video('../data/video.mp4')
print(f"Processed {result.total_frames} frames")
```

## 🔧 **Installation**

```bash
# Install RTMLib
pip install rtmlib

# Install additional dependencies
pip install opencv-python numpy pathlib
```

## 📊 **Supported Keypoint Groups**

| Group | Keypoints | Range | Color |
|-------|-----------|-------|-------|
| **Body** | 17 | 0-16 | 🟢 Green |
| **Face** | 68 | 17-84 | 🔵 Blue |
| **Left Hand** | 21 | 85-105 | 🔴 Red |
| **Right Hand** | 21 | 106-126 | 🟡 Yellow |
| **Feet** | 6 | 127-132 | 🟣 Purple |

## 🎨 **Output Examples**

The image processing creates:

1. **Annotated Image**: Visual representation with keypoints and skeleton
2. **JSON Data**: Detailed keypoint coordinates and confidence scores
3. **Console Summary**: Quick overview of detection results

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

## ⚙️ **Configuration Options**

### **Performance Modes**

- **`performance`**: Highest accuracy, slower inference
- **`balanced`**: Good balance of speed and accuracy ⭐ **Recommended**
- **`lightweight`**: Fastest inference, lower accuracy

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

| Configuration | Time | Keypoints Detected | Avg Confidence |
|---------------|------|-------------------|----------------|
| `balanced + cpu` | ~2s | 105/133 | 0.840 |
| `performance + cpu` | ~4s | 110/133 | 0.865 |
| `lightweight + cpu` | ~1s | 95/133 | 0.780 |

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
