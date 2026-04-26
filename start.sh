#!/bin/bash

# Start the FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit UI in the foreground
echo "Starting Streamlit UI..."
streamlit run ui.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.maxUploadSize=200
