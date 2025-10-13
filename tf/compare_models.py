#!/usr/bin/env python3
"""
TensorRT 10.x compatible model comparison script
Updated for TensorRT 10.11.0+ API
"""
import os
import numpy as np
import tensorflow as tf
import keras
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tabulate import tabulate

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRT10ModelComparator:
    def __init__(self):
        self.savedmodel_path = 'cifar10_vgg_model'
        self.onnx_path = 'model.onnx'
        self.tensorrt_path = 'model.trt'
        self.test_samples_path = 'test_samples.npy'
        self.test_labels_path = 'test_labels.npy'
        
        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Load test data
        self.test_samples = None
        self.test_labels = None
        self.load_test_data()
    
    def load_test_data(self):
        """Load test samples and labels"""
        if os.path.exists(self.test_samples_path) and os.path.exists(self.test_labels_path):
            self.test_samples = np.load(self.test_samples_path)
            self.test_labels = np.load(self.test_labels_path)
            print(f"✅ Loaded {len(self.test_samples)} test samples")
        else:
            print("❌ Test samples not found. Please run cifar10.py first.")
            return False
        return True
    
    def predict_savedmodel(self):
        """SavedModelで予測を実行"""
        if not os.path.exists(self.savedmodel_path):
            print(f"❌ SavedModel not found: {self.savedmodel_path}")
            return None
        
        print("🔄 Running SavedModel inference...")
        try:
            # Keras 3.x compatible loading using TFSMLayer
            model = keras.layers.TFSMLayer(self.savedmodel_path, call_endpoint='serving_default')
            # Create a functional model for easier use
            inputs = keras.Input(shape=(32, 32, 3))
            outputs = model(inputs)
            functional_model = keras.Model(inputs, outputs)
            predictions = functional_model.predict(self.test_samples, verbose=0)
            
            # Handle TFSMLayer output which might be a dict
            if isinstance(predictions, dict):
                # Get the first output if it's a dictionary
                key = list(predictions.keys())[0]
                predictions = predictions[key]
                
        except Exception as e:
            print(f"⚠️ TFSMLayer failed, trying low-level SavedModel API: {e}")
            # Fallback to low-level TensorFlow SavedModel API
            imported = tf.saved_model.load(self.savedmodel_path)
            infer_func = imported.signatures['serving_default']
            
            # Convert test samples to tensor and run inference
            test_tensor = tf.convert_to_tensor(self.test_samples, dtype=tf.float32)
            result = infer_func(test_tensor)
            
            # Extract the output (assuming the output key is the model output)
            output_key = list(result.keys())[0]
            predictions = result[output_key].numpy()
        
        # Debug output shape
        print(f"📝 SavedModel output type: {type(predictions)}")
        if hasattr(predictions, 'shape'):
            print(f"📝 SavedModel output shape: {predictions.shape}")
        else:
            print(f"📝 SavedModel output: {predictions}")
            return None
        
        print(f"📝 First prediction sample: {predictions[0] if len(predictions.shape) > 1 else predictions[:5]}")
        
        # Ensure predictions have correct shape for CIFAR-10 (10 classes)
        if len(predictions.shape) == 1:
            # If 1D, reshape to (batch_size, num_classes)
            predictions = predictions.reshape(-1, 10)
        elif predictions.shape[-1] != 10:
            print(f"⚠️ Unexpected output shape: {predictions.shape}, expected (batch_size, 10)")
        
        print("✅ SavedModel inference completed")
        return predictions
    
    def predict_onnx(self):
        """Run inference with ONNX model"""
        if not os.path.exists(self.onnx_path):
            print(f"❌ ONNX model not found: {self.onnx_path}")
            return None
        
        print("🔄 Running ONNX inference...")
        session = ort.InferenceSession(self.onnx_path)
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: self.test_samples.astype(np.float32)}
        predictions = session.run(None, ort_inputs)[0]
        print("✅ ONNX inference completed")
        return predictions
    
    def create_simple_tensorrt_engine(self, onnx_path, trt_path):
        """Create a simple TensorRT engine using trtexec-style approach"""
        print("🔧 Creating TensorRT engine using Python API...")
        
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        
        # Set memory pool size (replaces max_workspace_size in TensorRT 10+)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Create network from ONNX
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Get input layer and set optimization profile for dynamic shapes
        input_tensor = network.get_input(0)
        print(f"📝 Input tensor shape: {input_tensor.shape}")
        
        # Create optimization profile if needed
        if -1 in input_tensor.shape:
            profile = builder.create_optimization_profile()
            # Set profile for batch dimension (assuming batch is first dimension)
            profile.set_shape(input_tensor.name, (1, 32, 32, 3), (10, 32, 32, 3), (32, 32, 32, 3))
            config.add_optimization_profile(profile)
            print("✅ Added optimization profile for dynamic batch size")
        
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("❌ Failed to build TensorRT engine")
            return None
        
        # Save engine
        with open(trt_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ TensorRT engine saved to {trt_path}")
        return serialized_engine
    
    def predict_tensorrt_simple(self):
        """Simple TensorRT inference using Python API"""
        if not os.path.exists(self.tensorrt_path):
            print(f"🔧 TensorRT engine not found. Creating from ONNX...")
            if not os.path.exists(self.onnx_path):
                print(f"❌ ONNX model not found: {self.onnx_path}")
                return None
            
            # Create engine if it doesn't exist
            engine_data = self.create_simple_tensorrt_engine(self.onnx_path, self.tensorrt_path)
            if engine_data is None:
                return None
        
        print("🔄 Running TensorRT inference...")
        
        try:
            # Load engine
            runtime = trt.Runtime(TRT_LOGGER)
            with open(self.tensorrt_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            if engine is None:
                print("❌ Failed to load TensorRT engine")
                return None
            
            # Simple approach: use execute_v2 if available
            context = engine.create_execution_context()
            
            # Get input/output info
            input_names = []
            output_names = []
            input_shapes = []
            output_shapes = []
            
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_names.append(name)
                    input_shapes.append(engine.get_tensor_shape(name))
                else:
                    output_names.append(name)
                    output_shapes.append(engine.get_tensor_shape(name))
            
            # Prepare buffers
            input_data = self.test_samples.astype(np.float32)
            batch_size = len(input_data)
            
            # Set shapes if dynamic
            for i, name in enumerate(input_names):
                shape = list(input_shapes[i])
                if shape[0] == -1:
                    shape[0] = batch_size
                context.set_input_shape(name, shape)
            
            # Allocate GPU memory
            input_gpu = cuda.mem_alloc(input_data.nbytes)
            
            # Calculate output size
            output_shape = context.get_tensor_shape(output_names[0])
            output_size = int(np.prod(output_shape))
            output_gpu = cuda.mem_alloc(output_size * np.float32().itemsize)
            
            # Copy input to GPU
            cuda.memcpy_htod(input_gpu, input_data)
            
            # Set tensor addresses
            context.set_tensor_address(input_names[0], input_gpu)
            context.set_tensor_address(output_names[0], output_gpu)
            
            # Run inference
            context.execute_async_v3(0)  # Use default stream
            
            # Copy output back
            output_data = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_data, output_gpu)
            
            # Cleanup
            input_gpu.free()
            output_gpu.free()
            
            print("✅ TensorRT inference completed")
            return output_data
            
        except Exception as e:
            print(f"❌ TensorRT inference failed: {e}")
            return None
    
    def calculate_accuracy(self, predictions):
        """Calculate accuracy for predictions"""
        if predictions is None or self.test_labels is None:
            return 0.0
        
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_labels.flatten()
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy
    
    def compare_predictions(self, pred1, pred2, name1, name2):
        """Compare two sets of predictions"""
        if pred1 is None or pred2 is None:
            return {"max_diff": float('inf'), "mean_diff": float('inf'), "is_consistent": False}
        
        # Ensure same shape
        if pred1.shape != pred2.shape:
            print(f"⚠️  Shape mismatch: {name1} {pred1.shape} vs {name2} {pred2.shape}")
            return {"max_diff": float('inf'), "mean_diff": float('inf'), "is_consistent": False}
        
        max_diff = np.max(np.abs(pred1 - pred2))
        mean_diff = np.mean(np.abs(pred1 - pred2))
        is_consistent = np.allclose(pred1, pred2, rtol=1e-3, atol=1e-4)
        
        # Classification agreement
        pred1_classes = np.argmax(pred1, axis=1)
        pred2_classes = np.argmax(pred2, axis=1)
        class_agreement = np.mean(pred1_classes == pred2_classes)
        
        return {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "is_consistent": is_consistent,
            "class_agreement": class_agreement
        }
    
    def print_sample_predictions(self, savedmodel_pred, onnx_pred, tensorrt_pred, num_samples=5):
        """Print detailed predictions for first few samples"""
        print(f"\n📊 Detailed predictions for first {num_samples} samples:")
        
        for i in range(min(num_samples, len(self.test_samples))):
            true_label = self.test_labels[i, 0]
            true_class = self.class_names[true_label]
            
            print(f"\nSample {i+1} - True label: {true_label} ({true_class})")
            
            if savedmodel_pred is not None:
                sm_pred = np.argmax(savedmodel_pred[i])
                sm_conf = savedmodel_pred[i, sm_pred]
                print(f"  SavedModel:  {sm_pred} ({self.class_names[sm_pred]}) - Confidence: {sm_conf:.4f}")
            
            if onnx_pred is not None:
                onnx_pred_class = np.argmax(onnx_pred[i])
                onnx_conf = onnx_pred[i, onnx_pred_class]
                print(f"  ONNX:        {onnx_pred_class} ({self.class_names[onnx_pred_class]}) - Confidence: {onnx_conf:.4f}")
            
            if tensorrt_pred is not None:
                trt_pred_class = np.argmax(tensorrt_pred[i])
                trt_conf = tensorrt_pred[i, trt_pred_class]
                print(f"  TensorRT:    {trt_pred_class} ({self.class_names[trt_pred_class]}) - Confidence: {trt_conf:.4f}")
    
    def run_comparison(self):
        """Run complete model comparison"""
        if self.test_samples is None:
            return
        
        print("🚀 Starting TensorRT 10.x model comparison...")
        print(f"📝 Test samples shape: {self.test_samples.shape}")
        print("="*70)
        
        # Run predictions
        savedmodel_pred = self.predict_savedmodel()
        onnx_pred = self.predict_onnx()
        tensorrt_pred = self.predict_tensorrt_simple()
        
        # Calculate accuracies
        results = []
        
        if savedmodel_pred is not None:
            sm_accuracy = self.calculate_accuracy(savedmodel_pred)
            results.append(["SavedModel", f"{sm_accuracy:.4f}", "✅"])
        
        if onnx_pred is not None:
            onnx_accuracy = self.calculate_accuracy(onnx_pred)
            results.append(["ONNX", f"{onnx_accuracy:.4f}", "✅"])
        
        if tensorrt_pred is not None:
            trt_accuracy = self.calculate_accuracy(tensorrt_pred)
            results.append(["TensorRT", f"{trt_accuracy:.4f}", "✅"])
        
        # Print accuracy results
        print("\n📈 Accuracy Results:")
        print(tabulate(results, headers=["Model", "Accuracy", "Status"], tablefmt="grid"))
        
        # Compare predictions
        print("\n🔍 Model Consistency Comparison:")
        comparisons = []
        
        if savedmodel_pred is not None and onnx_pred is not None:
            comp = self.compare_predictions(savedmodel_pred, onnx_pred, "SavedModel", "ONNX")
            status = "✅ CONSISTENT" if comp["is_consistent"] else "⚠️  DIFFERENT"
            comparisons.append(["SavedModel vs ONNX", f"{comp['max_diff']:.6f}", f"{comp['mean_diff']:.6f}", status])
        
        if savedmodel_pred is not None and tensorrt_pred is not None:
            comp = self.compare_predictions(savedmodel_pred, tensorrt_pred, "SavedModel", "TensorRT")
            status = "✅ CONSISTENT" if comp["is_consistent"] else "⚠️  DIFFERENT"
            comparisons.append(["SavedModel vs TensorRT", f"{comp['max_diff']:.6f}", f"{comp['mean_diff']:.6f}", status])
        
        if onnx_pred is not None and tensorrt_pred is not None:
            comp = self.compare_predictions(onnx_pred, tensorrt_pred, "ONNX", "TensorRT")
            status = "✅ CONSISTENT" if comp["is_consistent"] else "⚠️  DIFFERENT"
            comparisons.append(["ONNX vs TensorRT", f"{comp['max_diff']:.6f}", f"{comp['mean_diff']:.6f}", status])
        
        if comparisons:
            print(tabulate(comparisons, 
                         headers=["Comparison", "Max Diff", "Mean Diff", "Status"], 
                         tablefmt="grid"))
        
        # Print sample predictions
        self.print_sample_predictions(savedmodel_pred, onnx_pred, tensorrt_pred)
        
        # Save results
        if savedmodel_pred is not None:
            np.save('savedmodel_predictions_final.npy', savedmodel_pred)
        if onnx_pred is not None:
            np.save('onnx_predictions_final.npy', onnx_pred)
        if tensorrt_pred is not None:
            np.save('tensorrt_predictions_final.npy', tensorrt_pred)
        
        print("\n💾 Prediction results saved to *_predictions_final.npy files")
        
        # Final summary
        all_consistent = True
        for comp in comparisons:
            if "DIFFERENT" in comp[3]:
                all_consistent = False
                break
        
        print("\n" + "="*70)
        if all_consistent and len(comparisons) > 0:
            print("🎉 SUCCESS: All models produce consistent results!")
        elif len(comparisons) > 0:
            print("⚠️  WARNING: Some models show differences. This may be acceptable depending on tolerance.")
        else:
            print("❌ ERROR: Unable to compare models. Check that all models are available.")
        print("="*70)

if __name__ == "__main__":
    try:
        print(f"🔧 TensorRT Version: {trt.__version__}")
        comparator = TensorRT10ModelComparator()
        comparator.run_comparison()
    except Exception as e:
        print(f"❌ Error during model comparison: {str(e)}")
        import traceback
        traceback.print_exc()