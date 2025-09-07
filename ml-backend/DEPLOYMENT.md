# Deployment Guide

## Quick Deployment

### 1. Setup
```bash
# Clone and navigate
git clone <repository>
cd ml-backend

# Configure environment
cp .env.example .env
# Edit .env with your Label Studio URL and API token
```

### 2. Docker Deployment (Recommended)
```bash
# Build and start
docker compose up -d

# Check status
docker compose ps
curl http://localhost:9090/health
```

### 3. Connect to Label Studio
1. Open Label Studio → Settings → Model
2. Add Model URL: `http://localhost:9090`
3. Enable "Use for interactive preannotations"

## Production Deployment

### Docker Swarm
```bash
# Deploy to swarm
docker stack deploy -c docker-compose.yml rtmlib-stack
```

### Kubernetes
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtmlib-ml-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rtmlib-ml-backend
  template:
    metadata:
      labels:
        app: rtmlib-ml-backend
    spec:
      containers:
      - name: ml-backend
        image: your-registry/rtmlib-ml-backend:latest
        ports:
        - containerPort: 9090
        env:
        - name: LABEL_STUDIO_URL
          value: "http://label-studio-service:8080"
        - name: LABEL_STUDIO_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: label-studio-secret
              key: api-token
```

### Environment-Specific Configs

#### Development
```bash
DEVICE=cpu
MODE=balanced
DEBUG=true
```

#### Production
```bash
DEVICE=cuda
MODE=performance
DEBUG=false
LOG_LEVEL=WARNING
```

## Monitoring

### Health Checks
```bash
# Basic health
curl http://localhost:9090/health

# Docker health
docker compose exec ml-backend curl localhost:9090/health
```

### Performance Monitoring
```bash
# Container stats
docker stats rtmlib-ml-backend

# Logs
docker compose logs -f ml-backend
```

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  ml-backend:
    # ...
    deploy:
      replicas: 3
```

### Load Balancing
Use nginx or traefik to balance requests across multiple backend instances.

## Troubleshooting

### Common Issues
1. **Port conflicts:** Change BACKEND_PORT in .env
2. **Memory issues:** Reduce batch size or use lighter model
3. **Network issues:** Check docker network configuration

### Debug Mode
```bash
# Enable debug
echo "DEBUG=true" >> .env
docker compose restart ml-backend

# View detailed logs
docker compose logs -f ml-backend
```
