#!/bin/bash

set -e

# Kubernetes Deployment Script for RTMLib ML Backend
# Usage: ./deploy.sh [staging|production] [namespace]

ENVIRONMENT=${1:-staging}
NAMESPACE=${2:-rtmlib}

echo "🚀 Deploying RTMLib ML Backend to Kubernetes"
echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed"
    exit 1
fi

# Check if kustomize is available
if ! command -v kustomize &> /dev/null; then
    echo "⚠️  kustomize not found, installing..."
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
    sudo mv kustomize /usr/local/bin/
fi

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    echo "❌ Environment must be 'staging' or 'production'"
    exit 1
fi

# Create namespace if it doesn't exist
echo "📦 Creating namespace: $NAMESPACE"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Apply secrets (these should be created manually or via CI/CD)
echo "🔑 Checking for required secrets..."
if ! kubectl get secret label-studio-config -n "$NAMESPACE" &> /dev/null; then
    echo "⚠️  Creating placeholder secret. Please update with real values:"
    echo "kubectl create secret generic label-studio-config \\"
    echo "  --from-literal=api-token='your-api-token' \\"
    echo "  --from-literal=url='http://label-studio:8080' \\"
    echo "  --namespace=$NAMESPACE"
    
    kubectl create secret generic label-studio-config \
        --from-literal=api-token='placeholder-token' \
        --from-literal=url='http://label-studio:8080' \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
fi

# Deploy using kustomize
echo "🛠️  Deploying with kustomize..."
cd "$(dirname "$0")/k8s/overlays/$ENVIRONMENT"

# Build and apply manifests
kustomize build . | kubectl apply -f -

echo "✅ Deployment completed!"
echo ""
echo "📋 Useful commands:"
echo "kubectl get pods -n $NAMESPACE"
echo "kubectl logs -f deployment/rtmlib-ml-backend -n $NAMESPACE"
echo "kubectl port-forward service/rtmlib-ml-backend-service 9090:80 -n $NAMESPACE"
echo ""
echo "🌐 Access the service:"
echo "kubectl get ingress -n $NAMESPACE"

# Wait for rollout
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/rtmlib-ml-backend -n "$NAMESPACE" --timeout=300s

echo "🎉 Deployment is ready!"

# Show pod status
kubectl get pods -n "$NAMESPACE" -l app=rtmlib-ml-backend
