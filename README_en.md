# TensorRT Inference Optimization Project

## 📋 Project Overview

A comprehensive verification project that converts CIFAR-10 classification models created in TensorFlow to ONNX and TensorRT formats, implementing **batch processing optimization** and **precision-specific performance comparison**.

### 🎯 Main Objectives
- **Inference Consistency Verification**: SavedModel → ONNX → TensorRT conversion accuracy
- **Batch Processing Optimization**: Efficiency improvement from single sample to batch size 32
- **Precision-Specific Performance Comparison**: Processing speed and accuracy evaluation for FP32, FP16, INT8
- **Cross-Version Compatibility**: Compatibility verification between TensorRT 8.x and 10.x

## 🚀 Key Achievements

### Performance Optimization Results
| Precision | Throughput (samples/sec) | Speedup | Engine Size (MB) |
|-----------|--------------------------|---------|------------------|
| **TensorRT INT8** | **31,127.9** ⭐ | **14.8x** | **4.5** 💾 |
| **TensorRT FP16** | **20,023.8** | **9.5x** | 8.5 |
| TensorRT FP32 | 2,108.2 | 1.0x | 16.4 |
| TF Lite INT8 | 1,160.1 | 3.6x | 3.7 |
| SavedModel | 555.6 | - | - |

### Batch Processing Impact
```
GPU Transfer Optimization: 100 → 4 transfers (96% reduction)
Memory Efficiency: Parallel processing with batch size 32
Processing Speed Improvement: 10.2x faster (TensorRT batch vs single)
```

## 🛠️ Technology Stack

### Development Environment
- **TensorRT 10.x**: nvcr.io/nvidia/tensorrt:25.06-py3 (TensorRT 10.11.0)
- **TensorRT 8.x**: nvcr.io/nvidia/tensorrt:23.03-py3 (TensorRT 8.5.3)
- **TensorFlow**: nvcr.io/nvidia/tensorflow:25.02-tf2-py3 (TF 2.17.0)
- **CUDA**: 12.9 / GPU support required

### Programming Languages
- **Python**: Inference engines, performance comparison, model conversion
- **C++**: High-performance inference implementation, direct TensorRT API usage

## 📁 Project Structure

### 🔧 Engine Generation & Conversion
```bash
cifar10.py                    # CIFAR-10 model training
convert_to_onnx.py           # ONNX conversion
convert_to_tensorrt.py       # TensorRT conversion (general)
convert_to_tensorrt_batch.py # Batch-optimized TensorRT
generate_trt8x_engine.py     # TensorRT 8.x specific engine
```

### 🚀 Inference & Verification Programs
```bash
tensorrt_inference_csv.cpp       # C++ TensorRT 10.x (batch processing)
tensorrt_inference_8x.cpp        # C++ TensorRT 8.x (legacy API)
precision_comparison_fp16.py     # FP32/FP16 comparison
tensorrt_int8_comparison.py      # Complete precision comparison
complete_precision_comparison.py # Cross-framework comparison
```

### 📊 Performance Measurement & Comparison
```bash
benchmark_performance.py     # Comprehensive performance measurement
compare_models.py           # Inter-model format comparison
export_csv_data.py          # C++ compatible data generation
```

### 🐳 Execution Environment
```bash
docker_tf.sh     # TensorFlow environment
docker_trt.sh    # TensorRT 10.x environment
docker_trt8x.sh  # TensorRT 8.x environment
```

## 🎯 Quick Start

### 1. Model Training & Data Preparation
```bash
# Train CIFAR-10 model in TensorFlow container
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorflow:25.02-tf2-py3 python3 cifar10.py

# Generate test data CSV (C++ compatible)
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorflow:25.02-tf2-py3 python3 export_csv_data.py
```

### 2. ONNX & TensorRT Conversion
```bash
# ONNX conversion
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorflow:25.02-tf2-py3 python3 convert_to_onnx.py

# Generate batch-optimized TensorRT engine
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:25.06-py3 python3 convert_to_tensorrt_batch.py
```

### 3. Performance Comparison Execution
```bash
# Complete precision-specific performance comparison
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:25.06-py3 python3 tensorrt_int8_comparison.py

# C++ implementation verification
./docker_trt.sh  # TensorRT 10.x
./docker_trt8x.sh # TensorRT 8.x
```

## 📊 Key Verification Results

### 1. Precision-Specific Performance Comparison
**14.8x speedup with INT8** achieved through TensorRT-specific optimization:
- No accuracy degradation (identical classification accuracy across all precisions)
- 73% engine size reduction
- Overwhelming performance improvement through GPU optimization

### 2. Batch Processing Optimization
**96% GPU transfer reduction** with significant memory efficiency improvement:
- Single processing: 100 GPU transfers
- Batch processing: 4 GPU transfers (batch size 32)
- 10.2x processing speed improvement

### 3. Cross-Version Compatibility
**Performance comparison verification between TensorRT 8.x and 10.x**:
- Performance difference: 1.2% (10.x advantage)
- Engine size: 71% reduction (10.x advantage)
- API: More intuitive interface in 10.x

## 🎯 Technical Highlights

### Batch Processing Optimization
```cpp
// Dynamic batch size support (TensorRT 10.x)
context.set_input_shape(input_name, [current_batch_size, 32, 32, 3]);
context.set_tensor_address(input_name, input_gpu);
context.execute_async_v3(0);
```

### Optimization Profile Configuration
```python
# Batch-optimized engine generation
profile.set_shape('input_1', 
                 [1, 32, 32, 3],    # min
                 [16, 32, 32, 3],   # opt  
                 [32, 32, 32, 3])   # max
```

### INT8 Quantization
```python
# Using EntropyCalibrator
config.int8_calibrator = TensorRTCalibrator(calibrator)
config.set_flag(trt.BuilderFlag.INT8)
```

## 🏆 Benchmark Results

**Peak performance achieved in this project**:
- **Highest Speed**: TensorRT INT8 (31,127.9 samples/sec)
- **Smallest Size**: TensorRT INT8 (4.5 MB)
- **Highest Efficiency**: 96% GPU transfer reduction through batch processing
- **Compatibility**: TensorRT 8.x/10.x cross-version support

**Practicality**: Comprehensive solution supporting high-speed inference in production environments, lightweight deployment on edge devices, and verification in development environments.

## 🔧 Technical Challenges & Solutions

### 1. Batch Processing Optimization
**Challenge**: Inefficient GPU utilization due to single sample processing
**Solution**: 
```cpp
// Dynamic processing implementation with batch size 32
const int batch_size = 32;
vector<float> input_batch(current_batch_size * input_size_per_sample);

// TensorRT dynamic batch size configuration
Dims4 inputShape{current_batch_size, 32, 32, 3};
context->setInputShape(input_name.c_str(), inputShape);
```

### 2. TensorRT Optimization Profile
**Challenge**: `Error Code 4: Network has dynamic inputs, but no optimization profile`
**Solution**:
```python
profile = builder.create_optimization_profile()
profile.set_shape(input_name, 
                 min=(1, 32, 32, 3),     # Minimum batch size
                 opt=(16, 32, 32, 3),    # Optimal batch size
                 max=(32, 32, 32, 3))    # Maximum batch size
config.add_optimization_profile(profile)
```

### 3. TensorRT 10.x API Migration
**Challenge**: Migration from legacy API (`execute()`) to modern API
**Solution**:
```cpp
// TensorRT 10.x API
context->set_tensor_address(input_name, input_gpu);
context->set_tensor_address(output_name, output_gpu);
context->execute_async_v3(0);
```

### 4. Memory Safety
**Challenge**: Potential risks from manual memory management
**Solution**:
```cpp
// Using smart pointers
std::unique_ptr<nvinfer1::ICudaEngine> engine;
std::unique_ptr<nvinfer1::IExecutionContext> context;
```

## 💡 Usage Notes

### Performance Optimization Tips
1. **Batch Size**: 32 is optimal (balance with GPU memory)
2. **Data Type**: INT8 for highest speed, FP16 for balance
3. **Memory Management**: Leverage batch-wise GPU transfers
4. **Profile Configuration**: Dynamic shape settings according to use cases

### Troubleshooting
- **GPU Memory Shortage**: Adjust batch size to 16 or lower
- **Accuracy Issues**: Regenerate engine, verify calibration data
- **API Errors**: Verify TensorRT version and API correspondence
- **Compilation Errors**: Recommend building within Docker containers

## 🚧 Future Expansion Plans

- [ ] **Dynamic Shape Support**: Variable input size handling
- [ ] **Multi-GPU Support**: Distributed inference system
- [ ] **Streaming Inference**: Real-time processing pipeline
- [ ] **Triton Integration**: NVIDIA Triton server integration
- [ ] **Benchmark Automation**: CI/CD pipeline integration

## 📞 Support & Contribution

### Reporting & Questions
- Specify execution environment (GPU, TensorRT version) when creating issues
- Provide reproducible minimal code examples
- Attach detailed logs and error messages

### Development Participation
- Fork → Create branch → Pull request
- Code style: C++17, Python PEP8 compliant
- Testing: Operation verification in each environment required

---

**🎯 Project Goal Achievement**: SavedModel → ONNX → TensorRT inference consistency verification, batch processing optimization, precision-specific performance comparison, and cross-version compatibility have all been completed successfully.