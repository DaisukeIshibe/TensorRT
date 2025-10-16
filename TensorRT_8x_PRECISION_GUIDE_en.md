# TensorRT 8.x Precision Engine Generation

## Overview
`generate_trt8x_engine.py` has been updated to support precision-specific engine generation for FP32, FP16, and INT8.

## New Features
- **Precision Selection**: Specify FP32/FP16/INT8 with `--precision` option
- **Batch Generation**: Generate engines for all precisions at once with `--all` option
- **Automatic File Naming**: Save as `model_trt8x_{precision}.trt` according to precision

## Usage

### Individual Precision Specification
```bash
# FP32 engine generation (default)
python3 generate_trt8x_engine.py --precision fp32

# FP16 engine generation
python3 generate_trt8x_engine.py --precision fp16

# INT8 engine generation (Note: currently limited)
python3 generate_trt8x_engine.py --precision int8
```

### Batch Generation for All Precisions
```bash
# Generate engines for all precisions
python3 generate_trt8x_engine.py --all
```

### Docker Execution Example
```bash
# Execute in TensorRT 8.x container
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:23.03-py3 \
  python3 generate_trt8x_engine.py --all
```

## Performance Comparison Results

| Precision | Engine Size | Size Reduction | Status |
|-----------|-------------|----------------|---------|
| **FP32** | 15.7 MB | - | ✅ Verified |
| **FP16** | 7.9 MB | **50% reduction** | ✅ Verified |
| INT8 | - | - | ⚠️ Limited |

### Size Reduction Benefits
- **FP16**: Achieves approximately 50% size reduction compared to FP32
- **Memory Efficiency**: Enables inference with smaller GPU memory
- **Transfer Speed**: Faster model loading with lightweight engines

## Technical Details

### Optimization Profile
All precisions support the following dynamic batch sizes:
```
Minimum batch: 1
Optimal batch: 16
Maximum batch: 32
```

### TensorRT 8.x API Support
- Uses `builder.build_engine()`
- `max_workspace_size` configuration (1GB)
- Dynamic batch size support

### Known Limitations

#### INT8 Calibration
The current environment has the following limitations for INT8 calibration:
- **pycuda dependency**: GPU memory management required for calibration
- **Data requirements**: Appropriate representative dataset needed
- **Current status**: Limited due to technical constraints

#### Recommendations
For INT8 usage in production environments, consider:
- Using `tensorrt_int8_comparison.py` in TensorRT 10.x environment
- Preparing appropriate calibration datasets
- Gradual precision evaluation

## File Output

### Generated Engine Files
```
model_trt8x_fp32.trt    # FP32 precision engine (15.7 MB)
model_trt8x_fp16.trt    # FP16 precision engine (7.9 MB)
model_trt8x_int8.trt    # INT8 precision engine (limited)
```

### Cache Files
```
calibration_cache_8x.cache    # For INT8 calibration
```

## Usage in C++ Inference

Generated engines can be used with `tensorrt_inference_8x.cpp`:

```cpp
// Specify engine file
std::string engine_file = "model_trt8x_fp16.trt";  // FP16 engine usage example
```

## Troubleshooting

### Error: "ONNX model not found"
```bash
# Generate ONNX model first
python3 convert_to_onnx.py
```

### Error: "TensorRT not found"
```bash
# Execute in TensorRT 8.x container
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:23.03-py3 bash
```

### Warning: "Detected subnormal FP16 values"
- Normal warning during FP16 conversion
- Minimal impact on accuracy
- Consider model retraining if necessary

## Summary

✅ **Successful Items**:
- Full support for FP32/FP16 engine generation
- 50% size reduction effect
- Flexible operation through command-line arguments
- Complete TensorRT 8.x API support

⚠️ **Attention Items**:
- INT8 is limited in current environment
- TensorRT 10.x recommended for production environments

This feature enables precision selection for FP32/FP16 even in TensorRT 8.x, allowing for a balance between memory efficiency and performance.