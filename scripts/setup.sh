#!/bin/bash

# Genestory Chatbot Setup Script
# This script sets up the development environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up Genestory Chatbot Environment..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p models/downloads
mkdir -p knowledge_base/uploaded

# Check if Docker is available for Milvus
if command -v docker &> /dev/null; then
    echo "✅ Docker is available"
    
    # Check if Milvus is running
    if ! docker ps | grep -q milvus; then
        echo "🔄 Starting Milvus database..."
        # Download and start Milvus standalone
        curl -L https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -o docker-compose.yml
        docker-compose up -d
        echo "⏳ Waiting for Milvus to start..."
        sleep 30
    else
        echo "✅ Milvus is already running"
    fi
else
    echo "⚠️ Docker not found. Please install Docker to run Milvus database."
    echo "You can also use a remote Milvus instance by updating the configuration."
fi

# Set up environment variables
if [ ! -f ".env" ]; then
    echo "📝 Environment file already exists"
else
    echo "✅ Environment file configured"
fi

echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 To start the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the backend: python -m uvicorn backend.main:app --reload"
echo "3. Start the frontend: streamlit run frontend/app.py"
echo ""
echo "📚 API Documentation will be available at: http://localhost:8000/docs"
echo "🌐 Web Interface will be available at: http://localhost:8501"
