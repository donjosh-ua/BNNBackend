#!/bin/bash

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build and start the Docker container with memory limits
echo "Building and starting the BNN Backend with memory limits..."
docker-compose up --build -d

# Print logs
echo "Container is starting. Here are the logs:"
docker-compose logs -f

# To stop the container, run:
# docker-compose down 