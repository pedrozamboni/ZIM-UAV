#!/bin/bash

echo "🚀 Starting roof extraction..."
echo "🚀 Getting roof points from segmented point cloud"

# Start container
docker-compose up -d

# Run mesh.py
echo ""
echo "📐 Step 1/2: Running roof_pcd.py..."
docker-compose exec zimuav python ./scripts/roof_pcd.py "${1:-./configs/roof_pcd_split.yml}"

if [ $? -ne 0 ]; then
    echo "❌ roof_pcd.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ split.py completed!"

# Wait 5 seconds with countdown
echo ""
for i in 5 4 3 2 1; do
    echo "⏳ Waiting $i seconds..."
    sleep 1
done

# Run roof_extractor.py
echo ""
echo "📏 Step 2/2: Running roof_extractor.py..."
docker-compose exec zimuav python ./scripts/roof_extractor.py "${2:-./configs/roof_extractor.yml}"

if [ $? -ne 0 ]; then
    echo "❌ roof_extractor.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ roof_extractor.py completed!"

# Stop container
echo ""
docker-compose down

echo "🎉 Pipeline completed successfully!"