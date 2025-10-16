#!/bin/bash
# TensorRT 8.x Precision Engine Generation Test Script

echo "🚀 TensorRT 8.x Multiple Precision Engine Generation"
echo "=================================================="

# Check if ONNX model exists
if [ ! -f "model.onnx" ]; then
    echo "❌ model.onnx not found! Please run convert_to_onnx.py first."
    exit 1
fi

echo "📋 Available options:"
echo "  1. Generate FP32 engine only"
echo "  2. Generate FP16 engine only" 
echo "  3. Generate INT8 engine only"
echo "  4. Generate ALL precision engines"
echo ""

# Check if running in TensorRT container
if ! python3 -c "import tensorrt" 2>/dev/null; then
    echo "⚠️  TensorRT not available. Running in TensorRT 8.x container..."
    echo ""
    docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
        nvcr.io/nvidia/tensorrt:23.03-py3 bash -c "
        echo '🔧 Generating ALL precision engines...'
        python3 generate_trt8x_engine.py --all
        echo ''
        echo '📊 Generated engines:'
        ls -lh model_trt8x_*.trt 2>/dev/null || echo 'No engines found'
        "
else
    echo "🔧 TensorRT environment detected. Generating ALL precision engines..."
    python3 generate_trt8x_engine.py --all
    echo ""
    echo "📊 Generated engines:"
    ls -lh model_trt8x_*.trt 2>/dev/null || echo "No engines found"
fi

echo ""
echo "✅ TensorRT 8.x precision engine generation completed!"
echo ""
echo "Usage examples:"
echo "  python3 generate_trt8x_engine.py --precision fp32"
echo "  python3 generate_trt8x_engine.py --precision fp16"
echo "  python3 generate_trt8x_engine.py --precision int8"
echo "  python3 generate_trt8x_engine.py --all"