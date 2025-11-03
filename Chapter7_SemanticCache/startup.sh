#!/bin/bash
# startup.sh - Script to start the AI Service Dashboard

echo "🚀 Starting AI Service Dashboard..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create tmp directory if it doesn't exist
mkdir -p /tmp

# Start FastAPI backend in background
echo "🔧 Starting FastAPI backend..."
python fastapi_backend.py &
FASTAPI_PID=$!

# Wait a moment for backend to start
sleep 3

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend..."
streamlit run streamlit_app.py

# Cleanup when script exits
trap "kill $FASTAPI_PID" EXIT
