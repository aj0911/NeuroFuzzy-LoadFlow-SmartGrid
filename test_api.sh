#!/bin/bash

# Test script for Neuro-Fuzzy API

echo "======================================================================"
echo "                    TESTING NEURO-FUZZY API"
echo "======================================================================"

# Start the API in background
echo ""
echo "[1/3] Starting API server..."
python api/main.py &
API_PID=$!
sleep 3

# Test health endpoint
echo ""
echo "[2/3] Testing /health endpoint..."
curl -s http://localhost:8000/health | python -m json.tool

# Test example endpoint
echo ""
echo "[3/3] Testing /example endpoint..."
curl -s http://localhost:8000/example | python -m json.tool

# Kill the server
echo ""
echo "Stopping API server..."
kill $API_PID

echo ""
echo "======================================================================"
echo "✅ API TEST COMPLETE"
echo "======================================================================"
echo ""
echo "To start the server manually:"
echo "  python api/main.py"
echo ""
echo "API docs: http://localhost:8000/docs"
echo "======================================================================"
