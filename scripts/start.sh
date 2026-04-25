#!/bin/bash
# Start the vLLM inference infrastructure
set -e

echo "Starting vLLM inference infrastructure..."

# Check if multi-model profile is requested
if [ "$1" = "--multi-model" ]; then
    echo "Starting with multi-model support..."
    docker compose --profile multi-model up -d --build
else
    echo "Starting single model..."
    docker compose up -d --build
fi

echo "Waiting for services to be healthy..."
docker compose ps

echo "Infrastructure started. API available at http://localhost:8000"
echo "Swagger docs at http://localhost:8000/docs"
