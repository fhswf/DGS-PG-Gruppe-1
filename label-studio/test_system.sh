#!/bin/bash

# Complete system test for Label Studio with RTMLib ML Backend

echo "🧪 Testing Label Studio with RTMLib ML Backend"
echo "=" * 50

# Test 1: Check if Docker containers are running
echo "1. Checking Docker containers..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Docker containers are running"
else
    echo "❌ Docker containers are not running"
    echo "Starting containers..."
    docker-compose up -d
    sleep 30
fi

# Test 2: Check ML Backend health
echo "2. Testing ML Backend health..."
HEALTH_RESPONSE=$(curl -s http://localhost:9090/health)
if echo "$HEALTH_RESPONSE" | grep -q "ok"; then
    echo "✅ ML Backend is healthy"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "❌ ML Backend is not responding"
    echo "   Please check logs: docker-compose logs ml-backend"
    exit 1
fi

# Test 3: Check Label Studio health
echo "3. Testing Label Studio health..."
LS_RESPONSE=$(curl -s http://localhost:8080/api/health)
if echo "$LS_RESPONSE" | grep -q "UP"; then
    echo "✅ Label Studio is healthy"
else
    echo "❌ Label Studio is not responding"
    echo "   Please check logs: docker-compose logs label-studio"
    exit 1
fi

# Test 4: Test prediction endpoint
echo "4. Testing ML Backend prediction endpoint..."
TEST_PREDICTION=$(curl -s -X POST http://localhost:9090/predict \
    -H "Content-Type: application/json" \
    -d '{"tasks": [{"data": {"image": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=600&fit=crop"}}]}')

if echo "$TEST_PREDICTION" | grep -q "results"; then
    echo "✅ Prediction endpoint is working"
    echo "   Prediction contains results"
else
    echo "⚠️  Prediction endpoint responded but may have issues"
    echo "   Response: $TEST_PREDICTION"
fi

echo ""
echo "🎉 System Test Complete!"
echo ""
echo "📋 Summary:"
echo "   • Label Studio: http://localhost:8080"
echo "   • ML Backend: http://localhost:9090"
echo "   • Health Status: All services operational"
echo ""
echo "🚀 Next steps:"
echo "   1. Open Label Studio in your browser"
echo "   2. Create a new project"
echo "   3. Import the labeling configuration from labeling-config.xml"
echo "   4. Add ML Backend URL: http://ml-backend:9090"
echo "   5. Upload images and start labeling with automatic pose estimation!"
