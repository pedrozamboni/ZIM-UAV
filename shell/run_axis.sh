#!/bin/bash
echo "🚀 Starting axis computation..."


# Default config file
DEFAULT_CONFIG="./configs/axis_config.yml"
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"

# Set to "true" to stop container after running, "false" to keep it running
STOP_AFTER="${STOP_AFTER:-true}"

echo "🚀 Running axis computation with config: $CONFIG_FILE"

# Start container
docker-compose up -d

# Check if config exists
if ! docker-compose exec -T zimuav test -f "$CONFIG_FILE" > /dev/null 2>&1; then
    echo "❌ Config file not found: $CONFIG_FILE"
    if [ "$STOP_AFTER" = "true" ]; then
        docker-compose down
    fi
    exit 1
fi

# Run axis computation
docker-compose exec zimuav python ./scripts/axis_extractor.py "$CONFIG_FILE"
EXIT_CODE=$?

# Stop container if configured
if [ "$STOP_AFTER" = "true" ]; then
    echo ""
    echo "🛑 Stopping container..."
    docker-compose down
fi

# Report result
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Completed!"
else
    echo "❌ Failed!"
fi

exit $EXIT_CODE
