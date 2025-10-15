#!/bin/bash

# TensorRT 8.x (23.03-py3) Docker Container Build and Run Script

echo "🐳 TensorRT 8.x C++ Compilation and Execution"
echo "=============================================="

# Check if container exists
if ! docker images | grep -q "nvcr.io/nvidia/tensorrt.*23.03-py3"; then
    echo "❌ TensorRT 23.03-py3 container not found"
    echo "   Please pull the container first:"
    echo "   docker pull nvcr.io/nvidia/tensorrt:23.03-py3"
    exit 1
fi

echo "✅ TensorRT 23.03-py3 container found"

# Run TensorRT 8.x container with C++ compilation and execution
docker run -v /etc/group:/etc/group:ro \
           -v /etc/passwd:/etc/passwd:ro \
           -v $HOME:$HOME \
           -u $(id -u $USER):$(id -g $USER) \
           --gpus all \
           --ipc=host \
           --rm \
           --workdir $(pwd) \
           nvcr.io/nvidia/tensorrt:23.03-py3 \
           bash -c "
echo '🔧 TensorRT 8.x Container Environment'
echo '===================================='
echo 'TensorRT Version:' \$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || echo 'Not available in Python')
echo 'CUDA Version:' \$(nvcc --version | grep release | awk '{print \$6}' | cut -c2-)
echo 'GPU Info:'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits

echo ''
echo '🛠️  Compiling TensorRT 8.x C++ program...'
echo '========================================='

# Compilation flags for TensorRT 8.x
INCLUDE_DIRS='-I/usr/include/x86_64-linux-gnu -I/usr/local/cuda/include -I/opt/tensorrt/include'
LIBRARY_DIRS='-L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64 -L/opt/tensorrt/lib'
LIBRARIES='-lnvinfer -lnvonnxparser -lcudart'
CPPFLAGS='-std=c++14 -O2 -DWITH_OPENCV'

# Check if TensorRT headers exist
if [ -f '/opt/tensorrt/include/NvInfer.h' ]; then
    echo '✅ TensorRT headers found in /opt/tensorrt/include'
else
    echo '⚠️  TensorRT headers not found in /opt/tensorrt/include, trying /usr/include'
    INCLUDE_DIRS='-I/usr/include -I/usr/local/cuda/include'
    LIBRARY_DIRS='-L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64'
fi

# Compile the program
g++ \$CPPFLAGS \$INCLUDE_DIRS tensorrt_inference_8x.cpp -o tensorrt_inference_8x \$LIBRARY_DIRS \$LIBRARIES

if [ \$? -eq 0 ]; then
    echo '✅ Compilation successful'
    echo ''
    
    echo '🚀 Running TensorRT 8.x C++ Inference...'
    echo '======================================='
    ./tensorrt_inference_8x
    
    echo ''
    echo '📊 Performance Summary'
    echo '====================='
    echo 'Framework: TensorRT 8.x (Container 23.03-py3)'
    echo 'Language: C++'
    echo 'Architecture: x86_64'
    
else
    echo '❌ Compilation failed'
    echo 'Checking available libraries...'
    find /usr -name '*nvinfer*' 2>/dev/null | head -5
    find /opt -name '*nvinfer*' 2>/dev/null | head -5
    exit 1
fi
"