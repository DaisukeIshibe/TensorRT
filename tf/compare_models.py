#!/usr/bin/env python3
"""
Compare inference results between SavedModel, ONNX, and TensorRT models
This script loads all three model formats and compares their predictions
"""
import os
import numpy as np
import tensorflow as tf
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tabulate import tabulate

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelComparator:
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
        """Run inference with SavedModel"""
        if not os.path.exists(self.savedmodel_path):
            print(f"❌ SavedModel not found: {self.savedmodel_path}")
            return None
        
        print("🔄 Running SavedModel inference...")
        model = tf.keras.models.load_model(self.savedmodel_path)
        predictions = model.predict(self.test_samples, verbose=0)
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
    
    def allocate_buffers(self, engine):
        """Allocate buffers for TensorRT inference"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def do_inference(self, context, bindings, inputs, outputs, stream):
        """Run TensorRT inference"""
        [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
        context.execute_async(batch_size=len(self.test_samples), bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
        stream.synchronize()
        return [out['host'] for out in outputs]
    
    def predict_tensorrt(self):
        """Run inference with TensorRT engine"""
        if not os.path.exists(self.tensorrt_path):
            print(f"❌ TensorRT engine not found: {self.tensorrt_path}")
            return None
        
        print("🔄 Running TensorRT inference...")
        
        # Load engine
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.tensorrt_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("❌ Failed to load TensorRT engine")
            return None
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs, outputs, bindings, stream = self.allocate_buffers(engine)
        
        # Prepare input data
        input_data = self.test_samples.astype(np.float32).ravel()
        np.copyto(inputs[0]['host'], input_data)
        
        # Run inference
        output_data = self.do_inference(context, bindings, inputs, outputs, stream)
        
        # Reshape output
        predictions = output_data[0].reshape(len(self.test_samples), -1)
        print("✅ TensorRT inference completed")
        return predictions
    
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
        
        max_diff = np.max(np.abs(pred1 - pred2))
        mean_diff = np.mean(np.abs(pred1 - pred2))
        is_consistent = np.allclose(pred1, pred2, rtol=1e-3, atol=1e-4)
        
        return {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "is_consistent": is_consistent
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
        
        print("🚀 Starting model comparison...")
        print(f"📝 Test samples shape: {self.test_samples.shape}")
        print("="*70)
        
        # Run predictions
        savedmodel_pred = self.predict_savedmodel()
        onnx_pred = self.predict_onnx()
        tensorrt_pred = self.predict_tensorrt()
        
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
        comparator = ModelComparator()
        comparator.run_comparison()
    except Exception as e:
        print(f"❌ Error during model comparison: {str(e)}")
        raise