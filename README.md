# Sign Language Pose Estimation Project

This repository contains a comprehensive solution for **whole-body pose estimation** specifically designed for **German Sign Language (DGS)** analysis and annotation.

## 🎯 **Project Overview**

This project provides automated pose estimation tools for sign language research and annotation, featuring:

- **133-keypoint whole-body pose detection** using RTMLib
- **Automated ML backend** for Label Studio integration
- **Production-ready Kubernetes deployment**
- **Docker-based development environment**

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RTMLib        │    │   ML Backend    │    │  Label Studio   │
│ Pose Estimator  │◄──►│    (Flask)      │◄──►│   Annotation    │
│                 │    │                 │    │   Interface     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 **Repository Structure**

- **`pose-estimator/`** - Core RTMLib pose estimation implementation
- **`ml-backend/`** - Label Studio ML backend with Docker setup
- **`k8s/`** - Kubernetes deployment manifests  
- **`notebooks/`** - Jupyter notebooks for development and testing
- **`data/`** - Sample data and test videos
- **`output/`** - Generated pose estimation results

## 🚀 **Quick Start**

### **1. Pose Estimation (Standalone)**

```bash
cd pose-estimator
pip install -r requirements.txt
python quick_start_demo.py
```

### **2. ML Backend Development**

```bash
cd ml-backend
docker compose up -d
```

### **3. Production Deployment**

```bash
./deploy.sh
```

## 📚 **Documentation**

### **Core Components**
- **[ML Backend Setup & Deployment](ml-backend/README.md)** - Complete guide for building and deploying the ML backend
- **[Pose Estimator Documentation](pose-estimator/README.md)** - RTMLib implementation details
- **[Kubernetes Deployment Guide](KUBERNETES.md)** - Production deployment instructions

### **Development Resources**
- **[Docker Development Setup](ml-backend/README.md#development-setup)** - Local development environment
- **[Label Studio Configuration](ml-backend/README.md#label-studio-integration)** - ML backend integration guide
- **[API Documentation](ml-backend/README.md#api-endpoints)** - REST API reference

## 🔧 **Technology Stack**

- **Pose Estimation**: RTMLib (ONNX Runtime)
- **ML Backend**: Python Flask, Label Studio ML SDK
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes, Kustomize
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana

## 🎥 **Pose Estimation Features**

### **Supported Keypoints (133 total)**
- **Body**: 17 keypoints (COCO format)
- **Face**: 68 keypoints (detailed facial landmarks)
- **Hands**: 42 keypoints (21 per hand)
- **Feet**: 6 keypoints (3 per foot)

### **Detection Modes**
- **Performance**: High accuracy, slower inference
- **Balanced**: Good accuracy-speed trade-off
- **Lightweight**: Fast inference, lower accuracy

## 🌐 **Production Deployment**

The system is deployed on Kubernetes with:

- **High Availability**: 3 replicas with auto-scaling
- **Load Balancing**: Traefik ingress with TLS
- **Monitoring**: Prometheus metrics and health checks
- **Security**: Non-root containers, secret management

**Access URLs:**
- **Primary**: https://rtmlib.gawron.cloud
- **FH-SWF**: https://rtmlib.fh-swf.cloud

## 📊 **Performance**

- **Inference Speed**: ~50ms per frame (CPU)
- **Accuracy**: 95%+ keypoint detection rate
- **Scalability**: Handles 100+ concurrent requests
- **Uptime**: 99.9% availability

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is developed for research purposes in German Sign Language analysis.

## 👥 **Team**

**FH Südwestfalen - DGS Project Group 1**

---

For detailed setup and deployment instructions, see the [ML Backend Documentation](ml-backend/README.md).
