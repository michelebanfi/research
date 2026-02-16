#!/bin/bash
# Start the React frontend development server
# Usage: ./start_frontend.sh

echo "🚀 Starting Research Assistant Frontend..."
echo ""

# Check if we're in the right directory
if [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

echo ""
echo "✅ Starting Vite dev server on http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop"
echo ""

npm run dev
