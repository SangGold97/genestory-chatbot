#!/bin/bash

# Genestory Chatbot Startup Script
# This script starts all components of the chatbot application

set -e

echo "🚀 Starting Genestory Chatbot Application..."

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            echo "✅ $service_name is ready"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start within timeout"
    return 1
}

# # Check if virtual environment exists
# if [ ! -d "venv" ]; then
#     echo "❌ Virtual environment not found. Please run setup.sh first."
#     exit 1
# fi

# # Activate virtual environment
# echo "🔄 Activating virtual environment..."
# source venv/bin/activate

# Create logs directory
mkdir -p logs

# Start Milvus database
echo "🗄️ Starting Milvus database..."
if ! docker ps | grep -q milvus-standalone; then
    docker-compose up -d
    
    # Wait for Milvus to be ready
    if ! wait_for_service "http://localhost:19530" "Milvus"; then
        echo "❌ Failed to start Milvus database"
        exit 1
    fi
else
    echo "✅ Milvus is already running"
fi

# Check ports
echo "🔍 Checking ports..."
if ! check_port 8000; then
    echo "❌ Backend port 8000 is already in use"
    exit 1
fi

if ! check_port 8501; then
    echo "❌ Frontend port 8501 is already in use"
    exit 1
fi

# Start backend in background
echo "⚙️ Starting backend server..."
nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
if ! wait_for_service "http://localhost:8000/health" "Backend API"; then
    echo "❌ Failed to start backend"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Initialize system models
echo "🤖 Initializing system models..."
curl -X POST "http://localhost:8000/system/initialize" \
    -H "Content-Type: application/json" \
    > /dev/null 2>&1 || echo "⚠️ Could not initialize models automatically"

# Load sample documents
echo "📚 Loading sample documents..."
python scripts/load_documents.py > logs/document_loader.log 2>&1 &
LOADER_PID=$!

# Start frontend
echo "🌐 Starting frontend..."
nohup streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to be ready
if ! wait_for_service "http://localhost:8501" "Frontend"; then
    echo "❌ Failed to start frontend"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

# Create PID file for shutdown script
echo "$BACKEND_PID $FRONTEND_PID $LOADER_PID" > .pids

echo "🎉 Genestory Chatbot is now running!"
echo ""
echo "📊 Services:"
echo "  • Backend API: http://localhost:8000"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • Frontend Web UI: http://localhost:8501"
echo "  • Milvus Database: http://localhost:19530"
echo "  • Milvus Web UI: http://localhost:3000"
echo ""
echo "📋 Logs:"
echo "  • Backend: logs/backend.log"
echo "  • Frontend: logs/frontend.log"
echo "  • Document Loader: logs/document_loader.log"
echo ""
echo "🛑 To stop the application, run: ./scripts/stop.sh"
echo ""
echo "⏳ Note: Model initialization may take a few minutes in the background."
