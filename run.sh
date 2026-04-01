#!/bin/bash
set -e

echo "=== QuranClip Setup ==="

# Install Python dependencies
pip install -q -r backend/requirements.txt

# Install frontend dependencies and build
cd frontend
npm install --silent
npm run build
cd ..

# Start the API server (serves both API + built frontend)
echo "=== Starting QuranClip API on port 8000 ==="
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
