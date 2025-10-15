#!/usr/bin/env python3
"""
TensorRT 8.x Engine Generator
TensorRT 8.x向けのエンジン生成スクリプト
"""
import tensorrt as trt
import numpy as np
import os

def build_engine_for_trt8x():
    """TensorRT 8.x向けエンジンの生成"""
    print("🔧 Building TensorRT 8.x engine...")
    
    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Check TensorRT version
    print(f"📋 TensorRT Version: {trt.__version__}")
    
    # ONNXモデル確認
    onnx_path = 'model.onnx'
    if not os.path.exists(onnx_path):
        print("❌ ONNX model not found!")
        return False
    
    # Builder and network creation (TensorRT 8.x style)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Create network with explicit batch (recommended for TensorRT 8.x)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(f"   {parser.get_error(error)}")
            return False
    
    print("✅ ONNX model parsed successfully")
    
    # Configuration for TensorRT 8.x
    config.max_workspace_size = 1 << 30  # 1GB workspace
    
    # For TensorRT 8.x, we can set optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Set dynamic batch size (TensorRT 8.x style)
    input_name = "input_1"  # CIFAR-10 input
    profile.set_shape(input_name, 
                     [1, 32, 32, 3],    # min
                     [16, 32, 32, 3],   # opt
                     [32, 32, 32, 3])   # max
    
    config.add_optimization_profile(profile)
    
    print("✅ Optimization profile configured")
    print("   Min batch: 1, Optimal batch: 16, Max batch: 32")
    
    # Build engine (TensorRT 8.x style)
    print("🔨 Building engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("❌ Failed to build engine")
        return False
    
    # Serialize and save engine
    engine_path = "model_trt8x.trt"
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    print(f"✅ TensorRT 8.x engine saved as: {engine_path}")
    
    # Engine info
    engine_size = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"📊 Engine size: {engine_size:.1f} MB")
    print(f"📊 Number of bindings: {engine.num_bindings}")
    
    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        binding_type = "Input" if is_input else "Output"
        print(f"   {binding_type}: {binding_name}, Shape: {binding_shape}")
    
    return True

if __name__ == "__main__":
    print("🚀 TensorRT 8.x Engine Generation")
    print("=" * 40)
    
    success = build_engine_for_trt8x()
    
    if success:
        print("\n✅ TensorRT 8.x engine generation completed!")
    else:
        print("\n❌ TensorRT 8.x engine generation failed!")
        exit(1)