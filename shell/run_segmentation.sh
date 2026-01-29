#!/bin/bash

echo "🚀 Starting point cloud segmentation..."

# Start container
docker-compose up -d

# Run batch_inf.py
echo ""
echo "📐 Step 1/2: Running batch_infe.py..."
docker-compose exec zimuav python ./scripts/batch_infe.py "${1:-./configs/batch_infe_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ batch_inf.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ batch_inf.py completed!"

# Wait 5 seconds with countdown
echo ""
for i in 5 4 3 2 1; do
    echo "⏳ Waiting $i seconds..."
    sleep 1
done

# Run merge.py
echo ""
echo "📏 Step 2/2: Running merge.py..."
docker-compose exec zimuav python ./scripts/merge.py "${2:-./configs/merge_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ merge.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ merge.py completed!"

# Stop container
echo ""
docker-compose down

echo "🎉 Point cloud segmentation pipeline completed successfully!"