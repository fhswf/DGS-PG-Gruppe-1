#!/bin/bash

# Setup script for Label Studio with RTMLib ML Backend

set -e

echo "Setting up Label Studio with RTMLib ML Backend..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration if needed."
fi

# Create data directory if it doesn't exist
mkdir -p data

# Build and start the services
echo "Building and starting Docker containers..."
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check if Label Studio is running
echo "Checking Label Studio status..."
if curl -f http://localhost:8080/api/health > /dev/null 2>&1; then
    echo "✅ Label Studio is running at http://localhost:8080"
else
    echo "⚠️  Label Studio may still be starting up. Please check logs with: docker-compose logs label-studio"
fi

# Check if ML Backend is running
echo "Checking ML Backend status..."
if curl -f http://localhost:9090/health > /dev/null 2>&1; then
    echo "✅ ML Backend is running at http://localhost:9090"
else
    echo "⚠️  ML Backend may still be starting up. Please check logs with: docker-compose logs ml-backend"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Open http://localhost:8080 in your browser"
echo "2. Create an admin account"
echo "3. Create a new project"
echo "4. Import the labeling configuration from labeling-config.xml"
echo "5. Add the ML Backend URL: http://ml-backend:9090"
echo "6. Upload your images and start labeling!"
echo ""
echo "Useful commands:"
echo "  View logs:           docker-compose logs -f"
echo "  Stop services:       docker-compose down"
echo "  Restart services:    docker-compose restart"
echo "  Update containers:   docker-compose pull && docker-compose up -d"
