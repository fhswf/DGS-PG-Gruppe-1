# Kubernetes Deployment Guide

This guide explains how to deploy the RTMLib ML Backend to a Kubernetes cluster using GitHub Actions and Traefik ingress.

## 📋 Prerequisites

### 1. Kubernetes Cluster
- Kubernetes cluster (v1.20+)
- **Traefik ingress controller** (as used in your existing infrastructure)
- **cert-manager** with Let's Encrypt for TLS certificates
- Optional: Prometheus for monitoring

### 2. GitHub Repository Setup
- Container registry access (GitHub Container Registry)
- Kubernetes cluster access credentials
- **Semantic release** configured with `release_config.json`

### 3. Required GitHub Secrets

Set up the following secrets in your GitHub repository:

#### **For Production Environment:**
```
KUBE_CONFIG                      # Base64 encoded kubeconfig file
LABEL_STUDIO_API_TOKEN          # Label Studio API token
LABEL_STUDIO_URL                # Label Studio URL (e.g., https://label-studio.yourdomain.com)
```

## 🚀 Deployment Process

### Automatic Deployment

The GitHub Actions workflow will automatically deploy when:

1. **Production**: Push to `main` branch
2. **Manual**: Workflow dispatch trigger

### Domain Configuration

The deployment is configured for your existing domain infrastructure:

- **Primary**: `rtmlib.gawron.cloud`
- **FH-SWF**: `rtmlib.fh-swf.cloud`

Both domains use **Traefik** with **Let's Encrypt** certificates.

## 📁 Kubernetes Manifests

### Architecture Overview

- **Unified Namespace**: All environments deploy to `rtmlib` namespace
- **Traefik Ingress**: Integrated with your existing Traefik setup
- **Kustomize Overlays**: Environment-specific configurations
- **Semantic Versioning**: Automatic version management

### Core Components

- **`deployment.yaml`** - Main application deployment with rolling updates
- **`service.yaml`** - ClusterIP service exposing port 9090
- **`ingress.yaml`** - Traefik ingress with Let's Encrypt TLS
- **`configmap.yaml`** - Configuration settings
- **`hpa.yaml`** - Auto-scaling based on CPU/memory
- **`monitoring.yaml`** - Prometheus monitoring setup

### Environment Overlays

```
k8s/overlays/
├── staging/           # Development environment
│   ├── kustomization.yaml
│   └── staging-deployment-patch.yaml
└── production/        # Production environment
    ├── kustomization.yaml
    └── production-deployment-patch.yaml
```

### Resource Allocation

#### Staging Environment:
```yaml
Replicas: 1
Resources per pod:
- Requests: 1GB RAM, 0.5 CPU
- Limits: 2GB RAM, 1 CPU
- Log Level: DEBUG
```

#### Production Environment:
```yaml
Replicas: 3
Resources per pod:
- Requests: 2GB RAM, 1 CPU
- Limits: 4GB RAM, 2 CPU
- Log Level: INFO
```

### Auto-scaling

- **Min replicas**: 2 (staging: 1)
- **Max replicas**: 10
- **Scale up**: CPU > 70% or Memory > 80%
- **Scale down**: Gradual with 5-minute stabilization

## 🛠️ Configuration

### Environment Variables

Configurations are managed via Kustomize overlays:

#### Staging (`k8s/overlays/staging/kustomization.yaml`):
```yaml
configMapGenerator:
- name: rtmlib-ml-backend-config
  literals:
  - DEVICE=cpu
  - BACKEND_TYPE=onnxruntime
  - MODE=balanced
  - LOG_LEVEL=DEBUG
  - DEBUG=true
  - CONFIDENCE_THRESHOLD=0.3
```

#### Production (`k8s/overlays/production/kustomization.yaml`):
```yaml
configMapGenerator:
- name: rtmlib-ml-backend-config
  literals:
  - DEVICE=cpu
  - BACKEND_TYPE=onnxruntime
  - MODE=balanced
  - LOG_LEVEL=INFO
  - DEBUG=false
  - CONFIDENCE_THRESHOLD=0.5
```

### GPU Support

For GPU-enabled nodes, update the overlay:

```yaml
# In overlay kustomization.yaml
configMapGenerator:
- name: rtmlib-ml-backend-config
  literals:
  - DEVICE=cuda
  - MODE=performance
```

### Traefik Ingress Configuration

The ingress is pre-configured for your Traefik setup:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    traefik.frontend.passHostHeader: 'true'
    traefik.ingress.kubernetes.io/router.entrypoints: websecure
    traefik.ingress.kubernetes.io/router.tls.certresolver: letsencrypt
    traefik.ingress.kubernetes.io/router.middlewares: default-cors@kubernetescrd
spec:
  rules:
  - host: rtmlib.gawron.cloud
  - host: rtmlib.fh-swf.cloud
```

## 🔧 Setup Instructions

### 1. Prepare Kubernetes Cluster

Your existing Traefik setup should work without additional configuration. Verify:

```bash
# Check Traefik is running
kubectl get pods -n traefik-system

# Verify cert-manager
kubectl get pods -n cert-manager
```

### 2. Configure GitHub Secrets

```bash
# Get kubeconfig (base64 encoded)
cat ~/.kube/config | base64 | pbcopy

# Add to GitHub:
# Settings > Secrets and variables > Actions > New repository secret
```

### 3. Local Testing with Deploy Script

```bash
# Make script executable
chmod +x deploy.sh

# Deploy to staging
./deploy.sh staging

# Deploy to production  
./deploy.sh production

# Check deployment
kubectl get pods -n rtmlib
```

### 4. Deploy

```bash
# Commit and push to trigger deployment
git add .
git commit -m "feat: deploy RTMLib ML backend to production"
git push origin main
```

## 📊 Monitoring & Troubleshooting

### Health Checks

```bash
# Check deployment status
kubectl get deployments -n rtmlib

# Check pod status
kubectl get pods -n rtmlib -l app=rtmlib-ml-backend

# Check service
kubectl get services -n rtmlib

# Check ingress
kubectl get ingress -n rtmlib
```

### Access URLs

Once deployed, the service will be available at:

- **Primary**: https://rtmlib.gawron.cloud
- **FH-SWF**: https://rtmlib.fh-swf.cloud

### API Endpoints

```bash
# Health check
curl https://rtmlib.gawron.cloud/health

# Test prediction
curl -X POST https://rtmlib.gawron.cloud/predict \
  -H "Content-Type: application/json" \
  -d '{"tasks": [{"data": {"image": "https://example.com/image.jpg"}}]}'
```

### Logs

```bash
# View application logs
kubectl logs -f deployment/rtmlib-ml-backend -n rtmlib

# View logs from specific pod
kubectl logs <pod-name> -n rtmlib

# Follow logs with labels
kubectl logs -f -l app=rtmlib-ml-backend -n rtmlib
```

### Debugging

```bash
# Get pod details
kubectl describe pod <pod-name> -n rtmlib

# Execute into pod
kubectl exec -it <pod-name> -n rtmlib -- /bin/bash

# Port forward for local testing
kubectl port-forward service/rtmlib-ml-backend-service 9090:9090 -n rtmlib

# Test locally
curl http://localhost:9090/health
```

### Common Issues

1. **ImagePullError**
   - Check GitHub Container Registry permissions
   - Verify image tag exists: `ghcr.io/fhswf/dgs-pg-gruppe-1/rtmlib-ml-backend`
   - Check if image was built successfully in Actions

2. **CrashLoopBackOff**
   - Check environment variables in secrets
   - Verify Label Studio connectivity
   - Check resource limits and requests
   - Review pod logs for errors

3. **Ingress not working**
   - Verify Traefik is running and configured
   - Check DNS records for `rtmlib.gawron.cloud` and `rtmlib.fh-swf.cloud`
   - Verify Let's Encrypt certificates are issued
   - Check middleware configuration

4. **Kustomize errors**
   - Verify overlay structure is correct
   - Check image names match in `kustomization.yaml`
   - Ensure all referenced files exist

## 🔒 Security Considerations

### Pod Security

- **Non-root user**: Runs as UID 1000
- **Read-only filesystem**: Where possible  
- **No privilege escalation**
- **Security context**: Restrictive settings
- **Resource limits**: Prevent resource exhaustion

### Network Security

The Traefik configuration includes:

- **TLS termination**: Automatic Let's Encrypt certificates
- **CORS middleware**: Configured for API access
- **Rate limiting**: Can be added via Traefik middleware

### Secrets Management

- **Kubernetes secrets**: API tokens stored securely
- **GitHub secrets**: Kubeconfig and credentials
- **No hardcoded values**: All sensitive data externalized

## 📈 Performance Tuning

### Resource Optimization

#### For high-traffic scenarios:
```yaml
# In production overlay
patchesStrategicMerge:
- |
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: rtmlib-ml-backend
  spec:
    template:
      spec:
        containers:
        - name: ml-backend
          resources:
            requests:
              memory: "4Gi"
              cpu: "2000m"
            limits:
              memory: "8Gi" 
              cpu: "4000m"
```

### Horizontal Scaling

```yaml
# Aggressive scaling for high load
spec:
  minReplicas: 5
  maxReplicas: 20
  targetCPUUtilizationPercentage: 50
  targetMemoryUtilizationPercentage: 70
```

### Caching Strategy

- **Model cache**: 10GB emptyDir volume for ONNX models
- **Shared cache**: Consider PersistentVolume for multiple pods
- **Prediction cache**: Implement Redis for repeated predictions

## 🔄 CI/CD Pipeline Features

### Semantic Release Integration

The pipeline uses semantic versioning:

```json
{
  "branch": "main",
  "branches": [
    "main",
    {"name": "dev", "prerelease": "beta"}
  ]
}
```

### Build Process

1. **Multi-platform builds**: AMD64 and ARM64 support
2. **Layer caching**: GitHub Actions cache for faster builds
3. **Security scanning**: Trivy vulnerability scanner
4. **Version management**: Automatic semantic versioning

### Deployment Process

1. **Environment separation**: Kustomize overlays for staging/production
2. **Secret management**: Kubernetes secrets with proper scoping
3. **Rolling updates**: Zero-downtime deployments
4. **Health checks**: Comprehensive readiness and liveness probes

### Quality Gates

1. **Vulnerability scanning**: Trivy security scan with SARIF upload
2. **Manifest validation**: Kubernetes YAML validation
3. **Environment protection**: GitHub environment protection rules

## 🎯 Access Patterns

### External Access (via Traefik)
- **Primary**: https://rtmlib.gawron.cloud
- **FH-SWF**: https://rtmlib.fh-swf.cloud

### Internal Access (within cluster)
```yaml
# Service name for internal communication
rtmlib-ml-backend-service.rtmlib.svc.cluster.local:9090
```

### Label Studio Integration
```yaml
# Add to Label Studio ML backend configuration
{
  "url": "https://rtmlib.gawron.cloud",
  "name": "RTMLib Wholebody Detection",
  "title": "RTMLib 133-keypoint pose estimation"
}
```

This setup provides a production-ready, scalable, and secure deployment of the RTMLib ML Backend integrated with your existing Traefik infrastructure! 🎉
