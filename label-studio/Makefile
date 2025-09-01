.PHONY: help build up down logs test clean install dev-install

# Default target
help:
	@echo "Available commands:"
	@echo "  build        - Build Docker containers"
	@echo "  up          - Start all services"
	@echo "  down        - Stop all services"
	@echo "  logs        - View logs from all services"
	@echo "  logs-ls     - View Label Studio logs"
	@echo "  logs-ml     - View ML Backend logs"
	@echo "  test        - Run ML Backend tests"
	@echo "  test-api    - Test ML Backend API"
	@echo "  clean       - Clean up containers and volumes"
	@echo "  install     - Install dependencies locally"
	@echo "  dev-install - Install development dependencies"
	@echo "  setup       - Run complete setup"

# Build containers
build:
	docker-compose build --no-cache

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

logs-ls:
	docker-compose logs -f label-studio

logs-ml:
	docker-compose logs -f ml-backend

# Run tests
test:
	docker-compose exec ml-backend python -m pytest tests/ -v

test-api:
	python test_backend.py

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Install dependencies locally (for development)
install:
	pip install -r ml-backend/requirements.txt

dev-install:
	pip install -r ml-backend/requirements.txt
	pip install -r ml-backend/requirements-test.txt

# Complete setup
setup:
	./setup.sh

# Restart services
restart:
	docker-compose restart

# Update containers
update:
	docker-compose pull
	docker-compose up -d

# Check service status
status:
	docker-compose ps
	@echo ""
	@echo "Label Studio: http://localhost:8080"
	@echo "ML Backend: http://localhost:9090"
	@echo ""
	@echo "Health checks:"
	@curl -s http://localhost:8080/api/health > /dev/null && echo "✅ Label Studio is healthy" || echo "❌ Label Studio is not responding"
	@curl -s http://localhost:9090/health > /dev/null && echo "✅ ML Backend is healthy" || echo "❌ ML Backend is not responding"
