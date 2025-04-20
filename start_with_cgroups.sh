#!/bin/bash

# Define memory limit in bytes (2GB)
MEMORY_LIMIT=4294967296

# Define cgroup name
CGROUP_NAME="bnn_cgroup"

# Check if cgroups is available and create a memory cgroup
if [ -d "/sys/fs/cgroup/memory" ]; then
    echo "Using cgroups v1 to limit memory"
    
    # Make sure we have permission to create cgroups
    if [ ! -w "/sys/fs/cgroup/memory" ]; then
        echo "Error: No permission to write to cgroups. Try running with sudo."
        exit 1
    fi
    
    # Create cgroup if it doesn't exist
    if [ ! -d "/sys/fs/cgroup/memory/$CGROUP_NAME" ]; then
        sudo mkdir -p "/sys/fs/cgroup/memory/$CGROUP_NAME"
    fi
    
    # Set memory limit for the cgroup
    echo $MEMORY_LIMIT | sudo tee "/sys/fs/cgroup/memory/$CGROUP_NAME/memory.limit_in_bytes"
    
    # Configure OOM behavior - allow the process to be killed when it exceeds memory
    echo 1 | sudo tee "/sys/fs/cgroup/memory/$CGROUP_NAME/memory.oom_control"
    
    # Check if a process is already running on port 8080
    if lsof -i :8080; then
        echo "Port 8080 is already in use. Stopping the existing process..."
        kill $(lsof -t -i:8080) || true
        sleep 2
    fi
    
    echo "Starting BNN Backend with $MEMORY_LIMIT bytes memory limit using cgroups..."
    
    # Start the process and add its PID to the cgroup
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 &
    PID=$!
    echo $PID | sudo tee "/sys/fs/cgroup/memory/$CGROUP_NAME/cgroup.procs"
    
    # Keep this script running to maintain the cgroup
    wait $PID
    
    # Cleanup
    sudo rmdir "/sys/fs/cgroup/memory/$CGROUP_NAME"
else
    echo "Cgroups v1 not available. Falling back to unlimited memory."
    echo "Consider using systemd-run or Docker for memory limits."
    
    # Check if a process is already running on port 8080
    if lsof -i :8080; then
        echo "Port 8080 is already in use. Stopping the existing process..."
        kill $(lsof -t -i:8080) || true
        sleep 2
    fi
    
    # Start without memory limits
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
fi 