#!/usr/bin/env python3
"""
Convert TensorFlow SavedModel to ONNX format
This script converts a SavedModel to ONNX format using tf2onnx
"""
import os
import tensorflow as tf
import tf2onnx
import numpy as np

def convert_savedmodel_to_onnx(savedmodel_path, onnx_output_path):
    """
    Convert a TensorFlow SavedModel to ONNX format
    
    Args:
        savedmodel_path (str): Path to the SavedModel directory
        onnx_output_path (str): Path where ONNX model will be saved
    """
    if not os.path.exists(savedmodel_path):
        raise FileNotFoundError(f"SavedModel directory '{savedmodel_path}' does not exist.")
    
    print(f"Converting SavedModel '{savedmodel_path}' to ONNX...")
    
    # Convert SavedModel to ONNX
    model_proto, _ = tf2onnx.convert.from_saved_model(
        savedmodel_path, 
        output_path=onnx_output_path,
        opset=13  # Use ONNX opset 13 for better TensorRT compatibility
    )
    
    print(f"ONNX model saved to '{onnx_output_path}'")
    return onnx_output_path

def test_onnx_model(onnx_model_path, test_data):
    """
    Test the ONNX model with sample data
    
    Args:
        onnx_model_path (str): Path to the ONNX model
        test_data (np.array): Test input data
    """
    import onnxruntime as ort
    
    print(f"Testing ONNX model '{onnx_model_path}'...")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    
    # Run inference
    ort_inputs = {input_name: test_data.astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX model inference completed. Output shape: {ort_outputs[0].shape}")
    return ort_outputs[0]

def verify_model_consistency(savedmodel_path, onnx_model_path, test_data):
    """
    Verify that SavedModel and ONNX model produce consistent results
    
    Args:
        savedmodel_path (str): Path to SavedModel
        onnx_model_path (str): Path to ONNX model
        test_data (np.array): Test input data
    """
    print("Verifying model consistency...")
    
    # Get SavedModel predictions
    saved_model = tf.keras.models.load_model(savedmodel_path)
    tf_predictions = saved_model.predict(test_data)
    
    # Get ONNX model predictions
    onnx_predictions = test_onnx_model(onnx_model_path, test_data)
    
    # Compare predictions
    max_diff = np.max(np.abs(tf_predictions - onnx_predictions))
    is_consistent = np.allclose(tf_predictions, onnx_predictions, rtol=1e-4, atol=1e-5)
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Models are consistent: {'YES' if is_consistent else 'NO'}")
    
    if is_consistent:
        print("✅ SavedModel and ONNX model produce consistent results")
    else:
        print("❌ SavedModel and ONNX model produce inconsistent results")
    
    # Save ONNX predictions for later comparison
    np.save('onnx_predictions.npy', onnx_predictions)
    print("ONNX predictions saved to 'onnx_predictions.npy'")
    
    return is_consistent

if __name__ == "__main__":
    # Configuration
    savedmodel_path = 'cifar10_vgg_model'
    onnx_output_path = 'model.onnx'
    
    # Load test samples if available
    if os.path.exists('test_samples.npy'):
        test_samples = np.load('test_samples.npy')
        print(f"Loaded {len(test_samples)} test samples")
    else:
        print("No test samples found. Please run cifar10.py first to generate test samples.")
        # Generate dummy test data as fallback
        test_samples = np.random.random((10, 32, 32, 3)).astype(np.float32)
        print("Using dummy test data for conversion verification")
    
    try:
        # Convert SavedModel to ONNX
        onnx_path = convert_savedmodel_to_onnx(savedmodel_path, onnx_output_path)
        
        # Verify model consistency
        verify_model_consistency(savedmodel_path, onnx_path, test_samples)
        
        print("\n🎉 ONNX conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during ONNX conversion: {str(e)}")
        raise