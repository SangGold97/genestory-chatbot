#!/bin/bash

# Genestory Chatbot Stop Script
# This script stops all components of the chatbot application

echo "🛑 Stopping Genestory Chatbot Application..."

# Function to kill process gracefully
kill_process() {
    local pid=$1
    local name=$2
    
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "🔄 Stopping $name (PID: $pid)..."
        kill -TERM "$pid"
        
        # Wait for graceful shutdown
        local attempts=10
        while [ $attempts -gt 0 ] && kill -0 "$pid" 2>/dev/null; do
            sleep 1
            attempts=$((attempts - 1))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚡ Force stopping $name..."
            kill -KILL "$pid"
        fi
        
        echo "✅ $name stopped"
    else
        echo "ℹ️ $name is not running"
    fi
}

# Read PIDs from file
if [ -f ".pids" ]; then
    read BACKEND_PID FRONTEND_PID LOADER_PID < .pids
    
    # Stop processes
    kill_process "$FRONTEND_PID" "Frontend"
    kill_process "$BACKEND_PID" "Backend"
    kill_process "$LOADER_PID" "Document Loader"
    
    # Remove PID file
    rm -f .pids
else
    echo "ℹ️ PID file not found, attempting to find and stop processes..."
    
    # Try to find and stop processes by name
    pkill -f "uvicorn backend.main:app" && echo "✅ Backend stopped" || echo "ℹ️ Backend not running"
    pkill -f "streamlit run frontend/app.py" && echo "✅ Frontend stopped" || echo "ℹ️ Frontend not running"
    pkill -f "scripts/load_documents.py" && echo "✅ Document loader stopped" || echo "ℹ️ Document loader not running"
fi

# Stop Docker containers
echo "🐳 Stopping Docker containers..."
docker-compose down

echo "🏁 Genestory Chatbot stopped successfully!"
