#!/bin/bash

# Define memory limit in bytes (2GB = 2 * 1024 * 1024 * 1024 bytes)
MEMORY_LIMIT=4294967296

# Define the working directory
WORK_DIR="$(pwd)"

# Check if a process is already running on port 8080
if lsof -i :8080; then
    echo "Port 8080 is already in use. Stopping the existing process..."
    kill $(lsof -t -i:8080) || true
    sleep 2
fi

echo "Starting BNN Backend with $MEMORY_LIMIT memory limit..."

# Use systemd-run to create a transient service with memory limits
systemd-run --user --scope -p MemoryMax=$MEMORY_LIMIT -p MemorySwapMax=$MEMORY_LIMIT \
  --working-directory="$WORK_DIR" \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# Note: For production use, you might want to use:
# systemd-run --user --unit=bnn-api --scope -p MemoryMax=$MEMORY_LIMIT -p MemorySwapMax=$MEMORY_LIMIT \
#   --working-directory="$WORK_DIR" \
#   python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 