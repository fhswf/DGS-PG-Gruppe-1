FROM python:3.10-slim

# System packages for OpenCV/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
	git curl build-essential \
	libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
	&& pip install -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash app \
	&& mkdir -p /app/models /app/data \
	&& chown -R app:app /app
USER app

ENV PYTHONPATH=/app \
	MODEL_DIR=/app/models

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
	CMD curl -fsSL http://localhost:9090/health || exit 1

CMD ["python", "_wsgi.py"]

