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

### **Process a 3D Side-by-Side Video**

You can process 3D side-by-side (SBS) videos, where each frame contains a left and right image, using the following method:

```python
from pose_estimator_2d import PoseEstimator2D

estimator = PoseEstimator2D(mode='balanced', device='cpu')
results = estimator.process_side_by_side_video(
  '../data/3d_sbs_video.mp4',
  output_json_path='../output/poses_sbs.json',
  show_first_annotated=True  # Show annotated left/right images for the first frame
)
print(f"Processed {len(results)} frames (left/right)")
```

Each frame is split into left and right images, and pose coordinates are extracted for both. The results are saved as a JSON file if `output_json_path` is provided.

If `show_first_annotated=True`, the first frame's left and right images will be saved as PNG files (`first_frame_left_annotated.png` and `first_frame_right_annotated.png`) with pose estimations overlaid in the current directory.

### **Process a Single Image**

```bash
# Simple usage with high-quality defaults
python simple_image_pose.py ../data/test_pose.png

# Specify output path (still high-quality by default)
python simple_image_pose.py ../data/test_pose.png ../output/my_result.png

# Adjust threshold if needed
python simple_image_pose.py ../data/test_pose.png --threshold 0.5  # More keypoints
python simple_image_pose.py ../data/test_pose.png --threshold 0.9  # Only highest quality

# Data-only analysis (no image output)
python simple_image_pose.py ../data/test_pose.png --no-image

# Custom output path
python simple_image_pose.py ../data/test_pose.png --output my_result.png

# Export keypoints as JSON
python simple_image_pose.py ../data/test_pose.png --json keypoints.json
```

### **Comprehensive Image Analysis**

```bash
# Run detailed analysis with JSON export
python test_image_pose.py
```

### **Advanced Image Annotation Options**

You can control annotation details with extra arguments:

```python
result = estimator.process_image_with_annotation(
    '../data/test_pose.png',
    output_path='../output/annotated.png',
    draw_bbox=True,            # Draw bounding boxes (default: True)
    draw_keypoints=True,       # Draw keypoints/skeleton (default: True)
    keypoint_threshold=0.5     # Only draw keypoints above this confidence (default: 0.3)
)
```

### **Process a Video**

```python
from pose_estimator_2d import PoseEstimator2D

estimator = PoseEstimator2D(mode='balanced', device='cpu')
result = estimator.process_video(
    '../data/video.mp4',
    output_dir='../output/annotated_frames',  # Save annotated frames here (optional)
    save_frames=True,                        # Set to True to save annotated frames
    max_frames=100                           # Limit number of frames (optional)
)
print(f"Processed {result.total_frames} frames")

# You can also process the full video without saving frames:
# result = estimator.process_video('../data/video.mp4')
```

## 🔧 **Installation**

```bash
# Install RTMLib
pip install rtmlib

# Install additional dependencies
pip install opencv-python numpy pathlib
```

## 📊 PoseEstimator2D Class Reference

**Initialization Parameters**
```python
PoseEstimator2D(
    mode='performance',          # Model quality: 'performance', 'balanced', 'lightweight'
    backend='onnxruntime',       # Backend: 'onnxruntime', 'opencv', 'openvino'
    device='cpu',                # Device: 'cpu', 'cuda', 'mps'
    kpt_threshold=0.8,           # Confidence threshold for keypoint detection
    to_openpose=False            # Convert to OpenPose format (optional)
)
```

**Main Methods**
- `process_image(image_path)`: Process image and return keypoint data only.
- `process_image_with_annotation(image_path, output_path=None, draw_bbox=True, draw_keypoints=True, keypoint_threshold=0.3)`: Process image and optionally save annotated version, with control over annotation details.
- `process_video(video_path, output_dir=None, save_frames=False, max_frames=None)`: Process entire video file, optionally save annotated frames.
- `process_side_by_side_video(...)`: Process 3D side-by-side video (see above).
- `export_to_json(result, output_path)`: Export a PoseResult or VideoResult to a JSON file.
- `get_summary(result)`: Get a human-readable summary of a PoseResult or VideoResult.

**Result Objects**
- `PoseResult` (for images):  
  - `result.keypoints` – Array of keypoint coordinates [num_persons, 133, 2]  
  - `result.scores` – Confidence scores [num_persons, 133]  
  - `result.bboxes` – Bounding boxes [num_persons, 4]  
  - `result.num_persons` – Number of detected persons  
- `VideoResult` (for videos):  
  - `video_result.frame_results` – List of PoseResult objects (one per frame)  
  - `video_result.total_frames` – Total number of frames processed  
  - `video_result.fps` – Frames per second  
  - `video_result.processing_time` – Total processing time in seconds  

## � Exporting Results to JSON

You can export any PoseResult or VideoResult to a JSON file:

```python
estimator.export_to_json(result, 'output.json')
```

## 📝 Getting a Summary of Results

You can get a human-readable summary string for any result:

```python
summary = estimator.get_summary(result)
print(summary)
```

## �📊 **Supported Keypoint Groups**

| Group | Keypoints | Range | Color |
|-------|-----------|-------|-------|
| **Body** | 17 | 0-16 | 🟢 Green |
| **Face** | 68 | 17-84 | 🔵 Blue |
| **Left Hand** | 21 | 85-105 | 🔴 Red |
| **Right Hand** | 21 | 106-126 | 🟡 Yellow |
| **Feet** | 6 | 127-132 | 🟣 Purple |

> **Note:** All 133 keypoint coordinates are always returned, but only those above the confidence threshold are considered "detected".

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

### **Frame-level JSON Export Structure (for videos)**

```json
{
  "frame_idx": 0,
  "num_persons": 1,
  "keypoints": [[x1, y1], [x2, y2], ...],  // Always 133 coordinate pairs
  "scores": [0.95, 0.87, 0.23, ...],       // Always 133 confidence scores
  "bboxes": [[x1, y1, x2, y2, conf]]       // Bounding boxes
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

5. **Other tips**
  - Try different performance modes (`performance`, `balanced`, `lightweight`)
  - Use the CLI options for more control over output and quality

## 🤝 **Integration with Label Studio**

The pose estimation results can be directly used with the ML backend for Label Studio annotation. See the [`ml-backend/README.md`](../ml-backend/README.md) for integration details.

---

**For sign language research at FH Südwestfalen - DGS Project Group 1**
