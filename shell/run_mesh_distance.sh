#!/bin/bash

echo "🚀 Starting mesh and distance processing pipeline..."

# Start container
docker-compose up -d

# Run mesh.py
echo ""
echo "📐 Step 1/2: Running mesh.py..."
docker-compose exec zimuav python ./scripts/mesh.py "${1:-./configs/mesh_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ mesh.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ mesh.py completed!"

# Wait 5 seconds with countdown
echo ""
for i in 5 4 3 2 1; do
    echo "⏳ Waiting $i seconds..."
    sleep 1
done

# Run distance.py
echo ""
echo "📏 Step 2/2: Running distance.py..."
docker-compose exec zimuav python ./scripts/distance.py "${2:-./configs/distance_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ distance.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ distance.py completed!"

# Stop container
echo ""
docker-compose down

echo "🎉 Pipeline completed successfully!"