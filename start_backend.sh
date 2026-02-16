#!/bin/bash
# Start the FastAPI backend server
# Usage: ./start_backend.sh

echo "🚀 Starting Research Assistant Backend..."
echo ""

# Check if we're in the right directory
if [ ! -f "backend/app/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import fastapi" 2>/dev/null || {
    echo "⚠️  FastAPI not found. Installing backend dependencies..."
    pip install -r backend/requirements.txt
}

# Create logs directory if not exists
mkdir -p backend/logs

echo ""
echo "✅ Starting server on http://localhost:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo "📝 Logs: backend/logs/app.log"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server with auto-reload for development
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
