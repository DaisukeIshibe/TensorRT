#!/bin/bash
# Run TensorFlow/Keras model training and ONNX conversion steps
# Uses TensorFlow Docker image

echo "🚀 Step 1 & 2: TensorFlow Model Training and ONNX Conversion"
echo "============================================================"

# Check if Docker image exists
if ! docker image inspect nvcr.io/nvidia/tensorflow:25.02-tf2-py3 >/dev/null 2>&1; then
    echo "❌ Docker image nvcr.io/nvidia/tensorflow:25.02-tf2-py3 not found"
    echo "Please pull the image first: docker pull nvcr.io/nvidia/tensorflow:25.02-tf2-py3"
    exit 1
fi

echo "📂 Current directory: $(pwd)"
echo "🐳 Running TensorFlow container..."

# Run TensorFlow container
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorflow:25.02-tf2-py3 \
    bash -c "
        echo '🔧 Installing additional packages...'
        pip install tf2onnx onnxruntime-gpu tabulate
        
        echo ''
        echo '🏃 Step 1: Training CIFAR-10 model with Keras...'
        python cifar10.py
        
        echo ''
        echo '🔄 Step 2: Converting SavedModel to ONNX...'
        python convert_to_onnx.py
        
        echo ''
        echo '✅ TensorFlow steps completed!'
        echo 'Generated files:'
        ls -la cifar10_vgg_model/ model.onnx test_samples.npy test_labels.npy *.npy 2>/dev/null || echo 'Some files may not be present'
    "

echo ""
echo "🎉 TensorFlow and ONNX conversion completed!"
echo "Next, run: ./run_tensorrt_steps.sh"