#!/bin/bash
# Check health of the vLLM inference infrastructure
set -e

API_URL="${API_URL:-http://localhost:8000}"

echo "Checking API Gateway health..."
curl -s "${API_URL}/health" | python3 -m json.tool

echo ""
echo "Checking system metrics..."
curl -s "${API_URL}/metrics" | python3 -m json.tool

echo ""
echo "Listing available models..."
curl -s "${API_URL}/v1/models" | python3 -m json.tool
