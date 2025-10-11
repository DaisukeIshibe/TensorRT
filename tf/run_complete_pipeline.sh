#!/bin/bash
# Complete pipeline runner - executes all steps in sequence

echo "🚀 Complete TensorRT Verification Pipeline"
echo "=========================================="
echo ""
echo "This script will run the complete pipeline:"
echo "1. Train CIFAR-10 model with Keras (TensorFlow)"
echo "2. Convert SavedModel to ONNX"
echo "3. Convert ONNX to TensorRT"
echo "4. Run Python TensorRT inference and compare results"
echo "5. Build and run C++ TensorRT inference"
echo ""

# Function to check if Docker image exists
check_docker_image() {
    local image=$1
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "❌ Docker image $image not found"
        echo "Please pull the image first: docker pull $image"
        return 1
    fi
    return 0
}

# Check required Docker images
echo "🔍 Checking Docker images..."
check_docker_image "nvcr.io/nvidia/tensorflow:25.02-tf2-py3" || exit 1
check_docker_image "nvcr.io/nvidia/tensorrt:25.06-py3" || exit 1
echo "✅ All required Docker images found"
echo ""

# Make scripts executable
chmod +x run_tensorflow_steps.sh
chmod +x run_tensorrt_steps.sh

# Run TensorFlow steps
echo "🏃 Running TensorFlow steps..."
./run_tensorflow_steps.sh

if [ $? -ne 0 ]; then
    echo "❌ TensorFlow steps failed"
    exit 1
fi

echo ""
echo "⏳ Waiting 3 seconds before TensorRT steps..."
sleep 3

# Run TensorRT steps
echo "🏃 Running TensorRT steps..."
./run_tensorrt_steps.sh

if [ $? -ne 0 ]; then
    echo "❌ TensorRT steps failed"
    exit 1
fi

echo ""
echo "🎉 Complete pipeline executed successfully!"
echo ""
echo "📁 Generated files and directories:"
echo "=================================="
ls -la cifar10_vgg_model/ model.onnx model.trt test_samples.npy test_labels.npy *_predictions*.npy cpp_tensorrt_results.csv build/ 2>/dev/null || echo "Some files may not be present"

echo ""
echo "🔍 Quick verification:"
echo "====================="

# Check key files
key_files=("cifar10_vgg_model/saved_model.pb" "model.onnx" "model.trt" "test_samples.npy")
all_present=true

for file in "${key_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
        all_present=false
    fi
done

echo ""
if [ "$all_present" = true ]; then
    echo "🎉 All key files are present! Pipeline completed successfully."
    echo ""
    echo "🔬 To analyze results further, you can:"
    echo "- Check *_predictions*.npy files for numerical comparison"
    echo "- Review cpp_tensorrt_results.csv for C++ inference results"
    echo "- Compare predictions across different model formats"
else
    echo "⚠️  Some files are missing. Please check the logs above for errors."
fi