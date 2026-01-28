#!/bin/bash

echo "🐳 Building Docker image..."
docker-compose build

echo "🚀 Starting container..."
docker-compose up -d

echo "✅ Container is ready!"
echo ""
echo "To access the container:"
echo "  ./shell.sh"
echo ""
echo "To run a Python script:"
echo "  docker-compose exec zimuav python your_script.py"