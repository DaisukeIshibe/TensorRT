#!/bin/bash
# Run TensorRT conversion and inference steps
# Uses TensorRT Docker image

echo "🚀 Step 3, 4 & 5: TensorRT Conversion and Python/C++ Inference"
echo "=============================================================="

# Check if Docker image exists
if ! docker image inspect nvcr.io/nvidia/tensorrt:25.06-py3 >/dev/null 2>&1; then
    echo "❌ Docker image nvcr.io/nvidia/tensorrt:25.06-py3 not found"
    echo "Please pull the image first: docker pull nvcr.io/nvidia/tensorrt:25.06-py3"
    exit 1
fi

# Check if required files exist
if [ ! -f "model.onnx" ]; then
    echo "❌ model.onnx not found. Please run ./run_tensorflow_steps.sh first"
    exit 1
fi

echo "📂 Current directory: $(pwd)"
echo "🐳 Running TensorRT container..."

# Run TensorRT container
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:25.06-py3 \
    bash -c "
        echo '🔧 Installing additional packages...'
        pip install numpy tabulate onnxruntime-gpu
        
        echo ''
        echo '🏃 Step 3: Converting ONNX to TensorRT...'
        python convert_to_tensorrt.py
        
        echo ''
        echo '🏃 Step 4: Python TensorRT inference and comparison...'
        python compare_models.py
        
        echo ''
        echo '🏃 Step 5: Building C++ TensorRT inference program...'
        
        # Install cmake and build tools
        apt-get update && apt-get install -y cmake build-essential
        
        # Create build directory
        mkdir -p build
        cd build
        
        # Build C++ program
        echo '🔨 Building C++ TensorRT inference...'
        cmake ..
        make -j\$(nproc)
        
        if [ -f tensorrt_inference ]; then
            echo '✅ C++ program built successfully'
            echo '🏃 Running C++ TensorRT inference...'
            ./tensorrt_inference ../model.trt ../test_samples.npy ../cpp_tensorrt_results.csv
            
            # Move results to main directory
            mv cpp_tensorrt_results.csv ../ 2>/dev/null || echo 'C++ results file not created'
        else
            echo '❌ Failed to build C++ program'
        fi
        
        cd ..
        
        echo ''
        echo '✅ All TensorRT steps completed!'
        echo 'Generated files:'
        ls -la model.trt *_predictions*.npy cpp_tensorrt_results.csv 2>/dev/null || echo 'Some files may not be present'
    "

echo ""
echo "🎉 All TensorRT steps completed!"
echo ""
echo "📊 Results Summary:"
echo "- SavedModel: cifar10_vgg_model/"
echo "- ONNX Model: model.onnx"
echo "- TensorRT Engine: model.trt"
echo "- Python predictions: *_predictions*.npy files"
echo "- C++ results: cpp_tensorrt_results.csv"