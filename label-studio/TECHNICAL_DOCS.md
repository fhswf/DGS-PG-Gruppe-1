# Label Studio mit RTMLib ML Backend - Technische Dokumentation

## Architektur-Übersicht

Das System besteht aus zwei Docker-Containern, die über ein gemeinsames Netzwerk kommunizieren:

### Container 1: Label Studio
- **Image**: `heartexlabs/label-studio:latest`
- **Port**: 8080
- **Funktion**: Web-basierte Annotationsplattform
- **Volumes**: Gemeinsames Datenverzeichnis für Bilder

### Container 2: ML Backend
- **Base Image**: `python:3.9-slim`
- **Port**: 9090
- **Funktion**: RTMLib-basierte Pose-Estimation API
- **Volumes**: Gemeinsames Datenverzeichnis und Modell-Cache

## Technische Details

### RTMLib Integration

Das ML Backend nutzt RTMLib mit folgender Konfiguration:

```python
wholebody = Wholebody(
    to_openpose=False,      # MMPose-Format
    mode='balanced',        # Ausgewogen zwischen Performance und Genauigkeit
    backend='onnxruntime',  # ONNX Runtime für Cross-Platform Support
    device='cpu'           # CPU-Verarbeitung (GPU optional)
)
```

### Pose Estimation Pipeline

1. **Bildverarbeitung**: Download und Konvertierung der Eingabebilder
2. **Pose Detection**: Wholebody-Pose-Estimation mit 133 Keypoints
3. **Formatierung**: Konvertierung in Label Studio JSON-Format
4. **Rückgabe**: Predictions als Pre-Annotations

### Keypoint-Schema

Das System unterstützt 133 Wholebody-Keypoints:
- **Körper**: 17 Keypoints (COCO-Standard)
- **Füße**: 6 Keypoints 
- **Gesicht**: 68 Keypoints
- **Hände**: 42 Keypoints (21 pro Hand)

## API-Endpunkte

### ML Backend (`http://localhost:9090`)

#### `GET /health`
Gesundheitsprüfung des Backends.

**Response:**
```json
{
    "status": "ok",
    "model_loaded": true
}
```

#### `POST /predict`
Pose-Estimation für gegebene Bilder.

**Request:**
```json
{
    "tasks": [
        {
            "data": {
                "image": "http://example.com/image.jpg"
            }
        }
    ]
}
```

**Response:**
```json
{
    "results": [
        {
            "result": [
                {
                    "from_name": "keypoints",
                    "to_name": "image",
                    "type": "keypointlabels",
                    "value": {
                        "x": 25.0,
                        "y": 40.0,
                        "keypointlabels": ["nose"],
                        "width": 2,
                        "height": 2
                    },
                    "score": 0.9
                }
            ],
            "score": 0.85,
            "model_version": "rtmlib-balanced"
        }
    ]
}
```

#### `POST /fit`
Training/Update-Events von Label Studio verarbeiten.

**Request:**
```json
{
    "event": "ANNOTATION_CREATED",
    "data": {
        "annotation": {"id": 123},
        "task": {"id": 456}
    }
}
```

## Konfiguration

### Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| `DEVICE` | `cpu` | Verarbeitungsgerät (cpu, cuda, mps) |
| `BACKEND` | `onnxruntime` | Inference-Backend |
| `MODE` | `balanced` | Modell-Modus (performance, lightweight, balanced) |
| `CONFIDENCE_THRESHOLD` | `0.3` | Mindest-Konfidenz für Keypoints |
| `MODEL_DIR` | `/app/models` | Modell-Cache-Verzeichnis |

### Docker-Netzwerk

Die Container kommunizieren über das interne Netzwerk `label-studio-network`:
- Label Studio erreicht ML Backend über `http://ml-backend:9090`
- Externe Zugriffe über `localhost:8080` (Label Studio) und `localhost:9090` (ML Backend)

## Datenfluss

### 1. Bildimport in Label Studio
```
Benutzer → Label Studio → Lokaler Dateispeicher
                      ↓
                   /data Volume (gemeinsam)
```

### 2. Pose-Prediction
```
Label Studio → ML Backend → RTMLib → Pose Keypoints → Label Studio
             (HTTP API)     (Inference)              (JSON Response)
```

### 3. Annotation und Training
```
Benutzer → Label Studio → Annotation speichern → Webhook → ML Backend
        (Korrektur)      (JSON Format)          (fit())    (Logging)
```

## Performance-Optimierung

### CPU-Optimierung
- ONNX Runtime für optimierte CPU-Inferenz
- Batch-Verarbeitung für mehrere Bilder
- Modell-Caching im Container

### Speicher-Management
- Lazy Loading der RTMLib-Modelle
- Bildkompression für Netzwerk-Transfer
- Docker-Volume für persistenten Modell-Cache

### Skalierung
- Horizontale Skalierung durch mehrere ML Backend Container
- Load Balancing mit Docker Compose
- Redis für Background-Job-Queue

## Fehlerbehandlung

### Robuste Bildverarbeitung
```python
def _download_image(self, url: str) -> Optional[np.ndarray]:
    try:
        # Lokale Dateien bevorzugen
        if url.startswith('/data/'):
            return cv2.imread(local_path)
        
        # HTTP-Download mit Timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Format-Konvertierung
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        logger.error(f"Image download failed: {e}")
        return None
```

### Graceful Degradation
- Leere Predictions bei Fehlern
- Fallback auf CPU bei GPU-Problemen  
- Retry-Mechanismen für Netzwerk-Fehler

## Monitoring und Logging

### Container-Logs
```bash
# Alle Logs anzeigen
docker-compose logs -f

# ML Backend spezifisch
docker-compose logs -f ml-backend

# Label Studio spezifisch  
docker-compose logs -f label-studio
```

### Metriken
- Prediction Latenz
- Modell-Konfidenz-Scores
- Annotation-Statistiken
- Container-Ressourcenverbrauch

## Sicherheit

### Container-Isolation
- Separate Netzwerk-Namespaces
- Minimale Basis-Images
- Non-Root-User für Anwendungen

### API-Sicherheit
- CORS-Konfiguration
- Request-Validierung
- Rate Limiting (optional)

### Datenschutz
- Lokale Datenverarbeitung
- Keine externen API-Calls
- Verschlüsselte Container-Kommunikation

## Entwicklung und Tests

### Lokale Entwicklung
```bash
# Installation
make dev-install

# Tests ausführen
make test

# API testen
make test-api
```

### CI/CD Integration
```yaml
# GitHub Actions Beispiel
- name: Test ML Backend
  run: |
    docker-compose up -d
    sleep 30
    python test_backend.py
    docker-compose down
```

## Deployment

### Produktionsumgebung
```yaml
# docker-compose.prod.yml
services:
  ml-backend:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    environment:
      - DEVICE=cuda  # GPU-Unterstützung
```

### Kubernetes (optional)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-backend
  template:
    spec:
      containers:
      - name: ml-backend
        image: ml-backend:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```
