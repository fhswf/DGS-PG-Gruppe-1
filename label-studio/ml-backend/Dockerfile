FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies in stages
COPY requirements.txt .

# Install dependencies with retries and better error handling
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt || \
    (echo "First install attempt failed, retrying with individual packages..." && \
     pip install --no-cache-dir numpy opencv-python-headless pillow requests flask flask-cors && \
     pip install --no-cache-dir onnxruntime && \
     pip install --no-cache-dir rtmlib && \
     pip install --no-cache-dir redis rq uwsgi tqdm && \
     pip install --no-cache-dir git+https://github.com/HumanSignal/label-studio-ml-backend.git)

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 9090

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Start the application
CMD ["python", "_wsgi.py"]
