#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT format with batch optimization - TensorRT 10.x compatible
"""
import os
import numpy as np
import tensorrt as trt

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_with_batch_optimization(onnx_file_path, engine_file_path, max_batch_size=32):
    """
    Build TensorRT engine from ONNX model with batch optimization using TensorRT 10.x API
    """
    print(f"Building TensorRT engine from '{onnx_file_path}' with batch optimization (max batch: {max_batch_size})...")
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Configure builder
    config = builder.create_builder_config()
    
    # Set memory pool limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    print("Using FP32 precision")
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Add optimization profile for dynamic batch size
    print("Adding optimization profile for dynamic batch size...")
    profile = builder.create_optimization_profile()
    
    # Find input layer
    input_name = network.get_input(0).name
    print(f"Input layer name: {input_name}")
    
    # Set optimization profile (min, opt, max batch sizes)
    # NHWC format: [batch, height, width, channels]
    profile.set_shape(input_name, 
                     min=(1, 32, 32, 3),     # Minimum batch size: 1
                     opt=(max_batch_size // 2, 32, 32, 3),  # Optimal batch size: 16 
                     max=(max_batch_size, 32, 32, 3))       # Maximum batch size: 32
    
    config.add_optimization_profile(profile)
    print(f"✅ Added optimization profile: min=1, opt={max_batch_size//2}, max={max_batch_size}")
    
    # Build engine (returns serialized engine in TensorRT 10+)
    print("Building TensorRT engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build TensorRT engine")
        return None
    
    # Save engine
    print(f"Saving TensorRT engine to '{engine_file_path}'...")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ TensorRT engine saved successfully to '{engine_file_path}'")
    return serialized_engine

if __name__ == "__main__":
    print(f"🔧 TensorRT Version: {trt.__version__}")
    
    # Configuration
    onnx_file_path = 'model.onnx'
    engine_file_path = 'model.trt'
    max_batch_size = 32
    
    if not os.path.exists(onnx_file_path):
        print(f"❌ ONNX model '{onnx_file_path}' not found. Please run convert_to_onnx.py first.")
        exit(1)
    
    try:
        # Build TensorRT engine from ONNX with batch optimization
        engine_data = build_engine_with_batch_optimization(onnx_file_path, engine_file_path, max_batch_size)
        
        if engine_data is None:
            print("❌ Failed to build TensorRT engine")
            exit(1)
        
        print("\n🎉 TensorRT batch-optimized engine conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during TensorRT conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise