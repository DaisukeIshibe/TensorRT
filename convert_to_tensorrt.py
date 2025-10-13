#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT format - TensorRT 10.x compatible
This script converts an ONNX model to TensorRT engine using TensorRT 10.x Python API
"""
import os
import numpy as np
import tensorrt as trt

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx_trt10(onnx_file_path, engine_file_path, precision='fp32'):
    """
    Build TensorRT engine from ONNX model using TensorRT 10.x API
    
    Args:
        onnx_file_path (str): Path to ONNX model file
        engine_file_path (str): Path where TensorRT engine will be saved
        precision (str): Precision mode ('fp32', 'fp16', 'int8')
    """
    print(f"Building TensorRT engine from '{onnx_file_path}' using TensorRT 10.x API...")
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Configure builder
    config = builder.create_builder_config()
    
    # Set memory pool limit (replaces max_workspace_size in TensorRT 10+)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
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

def test_tensorrt_engine_trt10(engine_file_path, test_data):
    """
    Test TensorRT engine with sample data using TensorRT 10.x API
    
    Args:
        engine_file_path (str): Path to TensorRT engine file
        test_data (np.array): Test input data
    
    Returns:
        np.array: Prediction results
    """
    print(f"Testing TensorRT engine '{engine_file_path}' with TensorRT 10.x API...")
    
    try:
        # Load engine
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_file_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("Failed to load TensorRT engine")
            return None
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Get tensor info
        input_names = []
        output_names = []
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        
        if not input_names or not output_names:
            print("Failed to find input/output tensors")
            return None
        
        input_name = input_names[0]
        output_name = output_names[0]
        
        # Set input shape for dynamic batch size if needed
        input_shape = engine.get_tensor_shape(input_name)
        batch_size = len(test_data)
        
        if input_shape[0] == -1:  # Dynamic batch size
            new_shape = [batch_size] + list(input_shape[1:])
            context.set_input_shape(input_name, new_shape)
            print(f"Set dynamic input shape: {new_shape}")
        
        # Get output shape after setting input shape
        output_shape = context.get_tensor_shape(output_name)
        print(f"Output shape: {output_shape}")
        
        # Prepare input data
        input_data = test_data.astype(np.float32)
        output_data = np.empty(output_shape, dtype=np.float32)
        
        # Allocate GPU memory
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        input_gpu = cuda.mem_alloc(input_data.nbytes)
        output_gpu = cuda.mem_alloc(output_data.nbytes)
        
        # Copy input to GPU
        cuda.memcpy_htod(input_gpu, input_data)
        
        # Set tensor addresses
        context.set_tensor_address(input_name, input_gpu)
        context.set_tensor_address(output_name, output_gpu)
        
        # Run inference
        context.execute_async_v3(0)  # Use default stream
        
        # Copy output back to host
        cuda.memcpy_dtoh(output_data, output_gpu)
        
        # Cleanup GPU memory
        input_gpu.free()
        output_gpu.free()
        
        print(f"TensorRT inference completed. Output shape: {output_data.shape}")
        return output_data
        
    except Exception as e:
        print(f"Error during TensorRT inference: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_model_consistency_trt10(savedmodel_path, onnx_model_path, tensorrt_engine_path, test_data):
    """
    Verify that SavedModel, ONNX, and TensorRT models produce consistent results
    """
    print("Verifying model consistency across all formats...")
    
    results = {}
    
    # Test SavedModel
    if os.path.exists(savedmodel_path):
        try:
            import tensorflow as tf
            saved_model = tf.keras.models.load_model(savedmodel_path)
            tf_predictions = saved_model.predict(test_data, verbose=0)
            results['SavedModel'] = tf_predictions
            print("✅ SavedModel predictions obtained")
        except Exception as e:
            print(f"❌ SavedModel test failed: {e}")
    
    # Test ONNX
    if os.path.exists(onnx_model_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_model_path)
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: test_data.astype(np.float32)}
            onnx_predictions = session.run(None, ort_inputs)[0]
            results['ONNX'] = onnx_predictions
            print("✅ ONNX predictions obtained")
        except Exception as e:
            print(f"❌ ONNX test failed: {e}")
    
    # Test TensorRT
    if os.path.exists(tensorrt_engine_path):
        trt_predictions = test_tensorrt_engine_trt10(tensorrt_engine_path, test_data)
        if trt_predictions is not None:
            results['TensorRT'] = trt_predictions
            print("✅ TensorRT predictions obtained")
    
    # Compare results
    if len(results) >= 2:
        print("\n🔍 Consistency Analysis:")
        model_names = list(results.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                pred1, pred2 = results[name1], results[name2]
                
                if pred1.shape == pred2.shape:
                    max_diff = np.max(np.abs(pred1 - pred2))
                    mean_diff = np.mean(np.abs(pred1 - pred2))
                    is_consistent = np.allclose(pred1, pred2, rtol=1e-3, atol=1e-4)
                    
                    status = "✅ CONSISTENT" if is_consistent else "⚠️  DIFFERENT"
                    print(f"  {name1} vs {name2}: Max diff={max_diff:.6f}, Mean diff={mean_diff:.6f} - {status}")
                else:
                    print(f"  {name1} vs {name2}: ❌ Shape mismatch ({pred1.shape} vs {pred2.shape})")
        
        # Save results
        for name, predictions in results.items():
            filename = f"{name.lower()}_predictions_trt10.npy"
            np.save(filename, predictions)
            print(f"💾 {name} predictions saved to {filename}")
    
    return results

if __name__ == "__main__":
    print(f"🔧 TensorRT Version: {trt.__version__}")
    
    # Configuration
    onnx_file_path = 'model.onnx'
    engine_file_path = 'model.trt'
    precision = 'fp32'  # Can be 'fp32', 'fp16', or 'int8'
    
    if not os.path.exists(onnx_file_path):
        print(f"❌ ONNX model '{onnx_file_path}' not found. Please run convert_to_onnx.py first.")
        exit(1)
    
    try:
        # Build TensorRT engine from ONNX
        engine_data = build_engine_from_onnx_trt10(onnx_file_path, engine_file_path, precision=precision)
        
        if engine_data is None:
            print("❌ Failed to build TensorRT engine")
            exit(1)
        
        # Test with sample data if available
        if os.path.exists('test_samples.npy'):
            test_samples = np.load('test_samples.npy')
            print(f"Testing with {len(test_samples)} test samples...")
            
            # Run comprehensive verification
            results = verify_model_consistency_trt10(
                'cifar10_vgg_model', 
                onnx_file_path, 
                engine_file_path, 
                test_samples
            )
            
            if results:
                print(f"\n🎉 Verification completed with {len(results)} models!")
            else:
                print("❌ No successful model tests")
        else:
            print("No test samples found. Engine built successfully but not tested.")
        
        print("\n🎉 TensorRT 10.x conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during TensorRT conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise