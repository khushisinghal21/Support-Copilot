#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
echo "🚀 Starting Hiver AI Support Agent (Frontend + Backend) on http://localhost:8000"
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
