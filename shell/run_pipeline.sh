#!/bin/bash

echo "🚀 Running end-to-end pipeline..."
# Start container
docker-compose up -d

# Run mesh.py
echo ""
echo "📐 Step 1/6: Running csf.py..."
docker-compose exec zimuav python ./scripts/csf.py "${1:-./configs/csf_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ csf.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ csf.py completed!"

# Wait 5 seconds with countdown
echo ""
for i in 5 4 3 2 1; do
    echo "⏳ Waiting $i seconds..."
    sleep 1
done

echo ""
echo "📏 Step 2/6: Running mesh.py and distance.py..."

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


docker-compose exec zimuav python ./scripts/distance.py "${2:-./configs/distance_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ distance.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ distance.py completed!"


echo ""
echo "📏 Step 3/6: Running batch_infe.py and merge.py"
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


docker-compose exec zimuav python ./scripts/merge.py "${2:-./configs/merge_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ merge.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ merge.py completed!"

echo ""
echo "📏 Step 4/6: Running crop_point_cloud.py and curb_inference.py..."

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
docker-compose exec zimuav python ./scripts/curb_inference.py "${2:-./configs/curb_inference.yml}"

if [ $? -ne 0 ]; then
    echo "❌ curb_inference.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ curb_inference.py completed!"

echo ""
echo "📏 Step 5/6: Running axis_extractor.py..."
echo ""
docker-compose exec zimuav python ./scripts/axis_extractor.py "${2:-./configs/axis_config.yml}"

if [ $? -ne 0 ]; then
    echo "❌ axis_extractor.py failed!"
    docker-compose down
    exit 1
fi

echo "✅ axis_extractor.py completed!"

echo "📏 Step 6/6: Running roof_pcd.py and roof_extractor.py..."

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