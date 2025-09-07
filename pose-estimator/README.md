# PoseEstimator2D

Eine Python Wrapper Klasse für 2D Pose Estimation basierend auf RTMLib. Diese Bibliothek ermöglicht es, Ganzkörper-Pose-Estimation auf Videodateien durchzuführen und folgt dem COCO WholeBody Standard mit 133 Keypoints.

## Features

- **Drei Performance Modi**: `performance`, `lightweight`, `balanced`
- **Video-Verarbeitung**: Unterstützt gängige Videoformate
- **COCO WholeBody Standard**: 133 Keypoints pro Person
- **Flexible Ausgabe**: Strukturierte Ergebnisse mit Koordinaten und Konfidenzwerten
- **Visualisierung**: Automatische Annotation von Videos mit Pose-Skeletten
- **Export-Funktionen**: JSON-Export für weitere Verarbeitung

## Installation

Zuerst muss RTMLib installiert werden:

```bash
pip install rtmlib
```

Zusätzliche Dependencies für GPU-Unterstützung (optional):

```bash
# Für NVIDIA GPUs
pip install onnxruntime-gpu

# Für Intel OpenVINO
pip install openvino
```

## Schnellstart

### Einfache Verwendung

```python
from pose_estimator import PoseEstimator2D

# Initialisierung
estimator = PoseEstimator2D(mode='balanced')

# Video verarbeiten
result = estimator.process_video('path/to/video.mp4')

# Ergebnisse anzeigen
print(f"Verarbeitete Frames: {result.total_frames}")
print(f"Personen im ersten Frame: {result.frame_results[0].num_persons}")

# Keypoints für erstes Frame abrufen
keypoints = result.get_keypoints_by_frame(0)
print(f"Keypoint-Shape: {keypoints.shape}")  # (N, 133, 2) für N Personen
```

### Erweiterte Verwendung

```python
# Mit benutzerdefinierten Parametern
estimator = PoseEstimator2D(
    mode='performance',      # Beste Performance
    device='cuda',           # GPU-Unterstützung
    backend='onnxruntime',   # Inference Backend
    kpt_threshold=0.3        # Konfidenz-Schwellwert
)

# Nur die ersten 100 Frames verarbeiten
result = estimator.process_video(
    'video.mp4', 
    max_frames=100,
    start_frame=50
)

# Progress Callback
def progress_callback(progress, frame_idx, num_persons):
    print(f"Fortschritt: {progress*100:.1f}% - Frame {frame_idx} - {num_persons} Personen")

result = estimator.process_video(
    'video.mp4',
    progress_callback=progress_callback
)
```

### Einzelne Frames verarbeiten

```python
import cv2

# Frame von Video oder Kamera laden
frame = cv2.imread('image.jpg')

# Pose estimation für einzelnes Frame
pose_result = estimator.process_frame(frame)

print(f"Gefundene Personen: {pose_result.num_persons}")
print(f"Keypoints Shape: {pose_result.keypoints.shape}")
print(f"Konfidenzwerte Shape: {pose_result.scores.shape}")
```

### Visualisierung

```python
# Frame mit Pose-Annotation
annotated_frame = estimator.visualize_frame(frame, pose_result)

# Annotiertes Video speichern
estimator.save_results_to_video(
    result, 
    'output_annotated.mp4',
    'original_video.mp4'
)
```

### Datenexport

```python
# Keypoints als JSON exportieren
estimator.export_keypoints_to_json(result, 'poses.json')

# Einzelne Person über Zeit verfolgen
person_trajectory = result.get_person_trajectory(person_idx=0)
for frame_idx, keypoints in enumerate(person_trajectory):
    if keypoints is not None:
        print(f"Frame {frame_idx}: Person gefunden")
```

## API Referenz

### PoseEstimator2D

Hauptklasse für Pose Estimation.

#### Parameter

- `mode`: Performance-Modus (`'performance'`, `'lightweight'`, `'balanced'`)
- `backend`: Inference Backend (`'onnxruntime'`, `'opencv'`, `'openvino'`)
- `device`: Compute-Device (`'cpu'`, `'cuda'`, `'mps'`)
- `kpt_threshold`: Konfidenz-Schwellwert für Keypoints (0.0-1.0)
- `to_openpose`: OpenPose-Format verwenden (Standard: False)

#### Methoden

##### `process_video(video_path, max_frames=None, start_frame=0, progress_callback=None)`

Verarbeitet eine Videodatei und extrahiert Pose-Keypoints.

**Parameter:**
- `video_path`: Pfad zur Videodatei
- `max_frames`: Maximale Anzahl zu verarbeitender Frames
- `start_frame`: Startframe-Index
- `progress_callback`: Callback-Funktion für Fortschrittsanzeigen

**Rückgabe:** `VideoResult` Objekt

##### `process_frame(frame)`

Verarbeitet einen einzelnen Frame.

**Parameter:**
- `frame`: Input-Frame als numpy array (BGR Format)

**Rückgabe:** `PoseResult` Objekt

##### `visualize_frame(frame, pose_result, draw_bboxes=True, draw_skeleton=True)`

Visualisiert Pose-Estimation-Ergebnisse auf einem Frame.

**Rückgabe:** Annotierter Frame als numpy array

### Datenstrukturen

#### PoseResult

Container für Pose-Estimation-Ergebnisse eines einzelnen Frames.

**Attribute:**
- `frame_idx`: Frame-Index
- `keypoints`: Array (N, 133, 2) mit x,y-Koordinaten für N Personen  
- `scores`: Array (N, 133) mit Konfidenzwerten für jeden Keypoint
- `bboxes`: Array (N, 5) mit Bounding Boxes (x1, y1, x2, y2, score)
- `num_persons`: Anzahl erkannter Personen

#### VideoResult

Container für Pose-Estimation-Ergebnisse eines gesamten Videos.

**Attribute:**
- `video_path`: Pfad zur Input-Videodatei
- `total_frames`: Gesamtanzahl verarbeiteter Frames
- `frame_results`: Liste von PoseResult-Objekten
- `fps`: Original-Video-Framerate
- `resolution`: Original-Video-Auflösung (width, height)
- `processing_stats`: Dictionary mit Verarbeitungsstatistiken

**Methoden:**
- `get_keypoints_by_frame(frame_idx)`: Keypoints für spezifischen Frame
- `get_all_keypoints()`: Keypoints für alle Frames
- `get_person_trajectory(person_idx)`: Keypoint-Trajektorie einer Person

## COCO WholeBody Keypoints

Die Klasse gibt 133 Keypoints pro Person im COCO WholeBody Format zurück:

- **Body (17 Keypoints)**: Hauptkörper-Gelenke
- **Feet (6 Keypoints)**: Füße
- **Face (68 Keypoints)**: Gesichts-Landmarks  
- **Hands (42 Keypoints)**: Hände (21 pro Hand)

Jeder Keypoint hat x,y-Koordinaten und einen Konfidenzwert.

## Performance Modi

### `'performance'`
- Höchste Genauigkeit
- Langsamste Verarbeitung
- Empfohlen für Offline-Analyse

### `'lightweight'`  
- Schnellste Verarbeitung
- Geringste Genauigkeit
- Empfohlen für Echtzeit-Anwendungen

### `'balanced'`
- Ausgewogenes Verhältnis von Geschwindigkeit und Genauigkeit
- Empfohlen für die meisten Anwendungen

## Beispiele

### Video-Batch-Verarbeitung

```python
import os
from pathlib import Path

# Mehrere Videos verarbeiten
video_dir = Path('videos/')
output_dir = Path('results/')

estimator = PoseEstimator2D(mode='balanced')

for video_file in video_dir.glob('*.mp4'):
    print(f"Verarbeite: {video_file}")
    
    result = estimator.process_video(video_file)
    
    # JSON Export
    json_path = output_dir / f"{video_file.stem}_poses.json"
    estimator.export_keypoints_to_json(result, json_path)
    
    # Annotiertes Video
    video_path = output_dir / f"{video_file.stem}_annotated.mp4" 
    estimator.save_results_to_video(result, video_path, video_file)
```

### Bewegungsanalyse

```python
# Bewegung einer spezifischen Person analysieren
result = estimator.process_video('dance_video.mp4')

# Trajektorie der ersten Person
trajectory = result.get_person_trajectory(0)

# Handgelenk-Position über Zeit (Keypoint 9 = rechtes Handgelenk)
hand_positions = []
for keypoints in trajectory:
    if keypoints is not None:
        hand_pos = keypoints[9]  # Rechtes Handgelenk
        hand_positions.append(hand_pos)

# Bewegungsgeschwindigkeit berechnen
import numpy as np
velocities = []
for i in range(1, len(hand_positions)):
    pos_diff = np.array(hand_positions[i]) - np.array(hand_positions[i-1])
    velocity = np.linalg.norm(pos_diff)
    velocities.append(velocity)

print(f"Durchschnittliche Handgeschwindigkeit: {np.mean(velocities):.2f} Pixel/Frame")
```

## Convenience Function

Für schnelle Verwendung steht eine Convenience-Funktion zur Verfügung:

```python
from pose_estimator import estimate_poses_from_video

# Einfache Verwendung
result = estimate_poses_from_video(
    'video.mp4',
    mode='balanced',
    max_frames=100,
    device='cpu'
)
```

## Command Line Interface

Die Klasse kann auch über die Kommandozeile verwendet werden:

```bash
python pose_estimator_2d.py video.mp4 --mode balanced --device cpu --max-frames 100 --output-json poses.json --output-video annotated.mp4
```

## Fehlerbehandlung

Die Klasse behandelt häufige Fehler robust:

- Ungültige Videodateien
- Frames ohne erkannte Personen
- Hardware-Inkompatibilitäten
- Speicherprobleme bei großen Videos

Bei Fehlern werden Warnungen ausgegeben und die Verarbeitung fortsetzt wo möglich.

## Performance-Tipps

1. **GPU verwenden**: `device='cuda'` für NVIDIA GPUs
2. **Batch-Verarbeitung**: Mehrere kleine Videos anstatt einem sehr großen
3. **Frame-Sampling**: `max_frames` Parameter für lange Videos
4. **Threshold anpassen**: Höhere `kpt_threshold` für weniger falsch-positive
5. **Backend optimieren**: `openvino` für Intel CPUs, `onnxruntime-gpu` für GPUs
