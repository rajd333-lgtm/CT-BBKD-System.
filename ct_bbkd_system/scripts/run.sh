#!/bin/bash
# CT-BBKD System Startup Script
# Usage: bash scripts/run.sh

set -e
echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║    CT-BBKD System — Starting Up       ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

# Check Python
python3 --version || { echo "ERROR: Python 3 required"; exit 1; }

# Install deps if needed
echo "  [1/3] Checking dependencies..."
pip install flask psutil -q 2>/dev/null || true

# Start backend in background
echo "  [2/3] Starting Backend API (port 5000)..."
cd "$(dirname "$0")/.."
python backend/app.py &
BACKEND_PID=$!
echo "        Backend PID: $BACKEND_PID"

sleep 1

# Open dashboard
echo "  [3/3] Opening Dashboard..."
if command -v open &>/dev/null; then
    open frontend/dashboard.html
elif command -v xdg-open &>/dev/null; then
    xdg-open frontend/dashboard.html
else
    echo "        Open in browser: $(pwd)/frontend/dashboard.html"
fi

echo ""
echo "  ✅ System Running!"
echo "     API:       http://localhost:5000"
echo "     Dashboard: frontend/dashboard.html"
echo "     Health:    http://localhost:5000/api/v1/health"
echo ""
echo "  Press Ctrl+C to stop..."
wait $BACKEND_PID
