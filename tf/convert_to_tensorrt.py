#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT format
This script converts an ONNX model to TensorRT engine using Python API
"""
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx(onnx_file_path, engine_file_path, max_batch_size=1, precision='fp32'):
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_file_path (str): Path to ONNX model file
        engine_file_path (str): Path where TensorRT engine will be saved
        max_batch_size (int): Maximum batch size for the engine
        precision (str): Precision mode ('fp32', 'fp16', 'int8')
    """
    print(f"Building TensorRT engine from '{onnx_file_path}'...")
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        else:
            print("FP16 not supported, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision")
        else:
            print("INT8 not supported, using FP32")
    else:
        print("Using FP32 precision")
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build engine
    print("Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("Failed to build TensorRT engine")
        return None
    
    # Save engine
    print(f"Saving TensorRT engine to '{engine_file_path}'...")
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"✅ TensorRT engine saved successfully to '{engine_file_path}'")
    return engine

def load_engine(engine_file_path):
    """
    Load TensorRT engine from file
    
    Args:
        engine_file_path (str): Path to TensorRT engine file
    
    Returns:
        trt.ICudaEngine: TensorRT engine
    """
    print(f"Loading TensorRT engine from '{engine_file_path}'...")
    
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("Failed to load TensorRT engine")
        return None
    
    print("✅ TensorRT engine loaded successfully")
    return engine

def allocate_buffers(engine):
    """
    Allocate host and device buffers for TensorRT engine
    
    Args:
        engine (trt.ICudaEngine): TensorRT engine
    
    Returns:
        tuple: (inputs, outputs, bindings, stream)
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append to the appropriate list
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """
    Run inference with TensorRT engine
    
    Args:
        context (trt.IExecutionContext): TensorRT execution context
        bindings (list): Device memory bindings
        inputs (list): Input buffer info
        outputs (list): Output buffer info
        stream (cuda.Stream): CUDA stream
        batch_size (int): Batch size
    
    Returns:
        list: Output arrays
    """
    # Transfer input data to the GPU
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    
    # Run inference
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return only the host outputs
    return [out['host'] for out in outputs]

def test_tensorrt_engine(engine_file_path, test_data):
    """
    Test TensorRT engine with sample data
    
    Args:
        engine_file_path (str): Path to TensorRT engine file
        test_data (np.array): Test input data
    
    Returns:
        np.array: Prediction results
    """
    print(f"Testing TensorRT engine '{engine_file_path}'...")
    
    # Load engine
    engine = load_engine(engine_file_path)
    if engine is None:
        return None
    
    # Create execution context
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # Prepare input data
    input_data = test_data.astype(np.float32).ravel()
    np.copyto(inputs[0]['host'], input_data)
    
    # Run inference
    output_data = do_inference(context, bindings, inputs, outputs, stream, batch_size=len(test_data))
    
    # Reshape output
    output_shape = (len(test_data), -1)  # Batch size x num_classes
    predictions = output_data[0].reshape(output_shape)
    
    print(f"TensorRT inference completed. Output shape: {predictions.shape}")
    return predictions

if __name__ == "__main__":
    # Configuration
    onnx_file_path = 'model.onnx'
    engine_file_path = 'model.trt'
    precision = 'fp32'  # Can be 'fp32', 'fp16', or 'int8'
    
    if not os.path.exists(onnx_file_path):
        print(f"❌ ONNX model '{onnx_file_path}' not found. Please run convert_to_onnx.py first.")
        exit(1)
    
    try:
        # Build TensorRT engine from ONNX
        engine = build_engine_from_onnx(onnx_file_path, engine_file_path, precision=precision)
        
        if engine is None:
            print("❌ Failed to build TensorRT engine")
            exit(1)
        
        # Test with sample data if available
        if os.path.exists('test_samples.npy'):
            test_samples = np.load('test_samples.npy')
            print(f"Testing with {len(test_samples)} test samples...")
            
            # Run TensorRT inference
            trt_predictions = test_tensorrt_engine(engine_file_path, test_samples)
            
            if trt_predictions is not None:
                # Save TensorRT predictions
                np.save('tensorrt_predictions.npy', trt_predictions)
                print("TensorRT predictions saved to 'tensorrt_predictions.npy'")
                
                # Compare with ONNX predictions if available
                if os.path.exists('onnx_predictions.npy'):
                    onnx_predictions = np.load('onnx_predictions.npy')
                    max_diff = np.max(np.abs(trt_predictions - onnx_predictions))
                    is_consistent = np.allclose(trt_predictions, onnx_predictions, rtol=1e-3, atol=1e-4)
                    
                    print(f"Maximum difference vs ONNX: {max_diff}")
                    print(f"TensorRT vs ONNX consistency: {'YES' if is_consistent else 'NO'}")
                    
                    if is_consistent:
                        print("✅ ONNX and TensorRT models produce consistent results")
                    else:
                        print("⚠️  ONNX and TensorRT models have slight differences (may be acceptable)")
        else:
            print("No test samples found. Engine built successfully but not tested.")
        
        print("\n🎉 TensorRT conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during TensorRT conversion: {str(e)}")
        raise