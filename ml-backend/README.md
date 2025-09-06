# RTMLib ML Backend for Label Studio

This ML backend wraps rtmlib Wholebody (133 keypoints) to provide automatic keypoint pre-annotations in Label Studio.

- Inference: Wholebody(mode="balanced", backend=..., device=...)
- Output: Label Studio predictions with `KeyPointLabels` results for 133 COCO-WholeBody keypoints
- Endpoints and auth: Uses Label Studio ML Backend SDK (`/health`, `/predict`, `/setup`, `/train`, etc.). If your data requires auth, set `LABEL_STUDIO_API_KEY` and `LABEL_STUDIO_URL`.

## Build & Run

Docker Compose (recommended):

```bash
# From this folder
docker compose build
docker compose up -d
# Health
curl -fsSL http://localhost:9090/health
```

Connect to Label Studio: Settings > Model > Add Model, URL: `http://localhost:9090`

Environment variables:
- DEVICE: cpu | cuda | mps (default cpu)
- BACKEND: onnxruntime | opencv | openvino (default onnxruntime)
- MODE: performance | lightweight | balanced (default balanced)
- CONFIDENCE_THRESHOLD: default 0.3
- LABEL_STUDIO_URL: e.g. http://host.docker.internal:8080
- LABEL_STUDIO_API_KEY: your token (for downloading protected data)

## Labeling Config

Make sure your project uses:

```xml
<Image name="image" value="$image"/>
<KeyPointLabels name="keypoints" toName="image"> ... 133 labels ... </KeyPointLabels>
```

`from_name` must be `keypoints` and `to_name` must be `image` (matches the model output).

## Notes
- The backend returns predictions in the LS format with x/y in percent and one result per visible keypoint.
- If your tasks store images under `/data/`, mount the same data dir to the container (already mounted as `../data:/app/data:ro`).

## Endpoints
The SDK exposes:
- GET `/health` – health check
- POST `/predict` – predictions for tasks (Label Studio calls this)
- POST `/setup` – called by LS when adding a model
- POST `/train` – optional training hook (no-op here)

For the full API and format details, see:
- https://labelstud.io/guide/ml_create
- https://labelstud.io/guide/task_format
- https://labelstud.io/guide/export#Label-Studio-JSON-format-of-annotated-tasks

## License
Apache-2.0 for rtmlib; follow dependencies' licenses.
