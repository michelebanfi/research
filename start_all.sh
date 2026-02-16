#!/bin/bash
# Start both backend and frontend
# Usage: ./start_all.sh

echo "🚀 Starting Research Assistant (Backend + Frontend)..."
echo ""

# Check if we're in the right directory
if [ ! -f "backend/app/main.py" ] || [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup INT TERM

# Start backend in background
echo "📡 Starting Backend..."
./start_backend.sh &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend in background
echo ""
echo "🎨 Starting Frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

echo ""
echo "✅ Both services started!"
echo "📚 API: http://localhost:8000"
echo "🌐 App: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
