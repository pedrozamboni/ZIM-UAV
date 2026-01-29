#!/bin/bash

echo "🚀 Starting curb segmentation..."
echo "🚀 Cropping point cloud using street shapefile"

# Start container
docker-compose up -d

# Run mesh.py
echo ""
echo "📐 Step 1/2: Running crop_point_cloud.py..."
docker-compose exec zimuav python ./scripts/crop_point_cloud.py "${1:-./configs/crop_point_cloud_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ crop_point_cloud.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ crop_point_cloud.py completed!"

# Wait 5 seconds with countdown
echo ""
for i in 5 4 3 2 1; do
    echo "⏳ Waiting $i seconds..."
    sleep 1
done

# Run curb_inference.py
echo ""
echo "📏 Step 2/2: Running curb_inference.py..."
docker-compose exec zimuav python ./scripts/curb_inference.py "${2:-./configs/curb_inference.yml}"

if [ $? -ne 0 ]; then
    echo "❌ curb_inference.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ curb_inference.py completed!"

# Stop container
echo ""
docker-compose down

echo "🎉 Pipeline completed successfully!"