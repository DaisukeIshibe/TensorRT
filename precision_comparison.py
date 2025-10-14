#!/usr/bin/env python3
"""
TensorRT Precision Comparison: FP16 vs INT8
処理速度と分類精度の詳細比較
"""
import time
import numpy as np
import statistics
import json
from typing import Dict, List, Tuple, Any

def load_test_data():
    """テストデータの読み込み"""
    try:
        test_samples = np.load('test_samples.npy')
        test_labels = np.load('test_labels.npy')
        print(f"✅ Loaded {len(test_samples)} test samples: {test_samples.shape}")
        return test_samples, test_labels
    except FileNotFoundError:
        print("❌ Test data files not found!")
        return None, None

def calculate_accuracy(predictions: np.ndarray, true_labels: np.ndarray) -> float:
    """分類精度の計算"""
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1) if true_labels.ndim > 1 else true_labels
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy

def benchmark_tensorrt_precision(engine_path: str, test_samples: np.ndarray, 
                                test_labels: np.ndarray, precision_name: str,
                                batch_size: int = 32, num_runs: int = 5) -> Dict[str, Any]:
    """指定精度でのTensorRT推論ベンチマーク"""
    print(f"\n🔄 Benchmarking TensorRT {precision_name}...")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # エンジン読み込み時間測定
        start_time = time.time()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        load_time = time.time() - start_time
        
        # テンソル情報取得
        input_names = []
        output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        
        input_name = input_names[0]
        output_name = output_names[0]
        
        # ウォームアップ
        warmup_batch = test_samples[:min(batch_size, len(test_samples))]
        context.set_input_shape(input_name, [len(warmup_batch), 32, 32, 3])
        warmup_output_shape = context.get_tensor_shape(output_name)
        warmup_input_data = warmup_batch.astype(np.float32)
        warmup_output_data = np.empty(warmup_output_shape, dtype=np.float32)
        
        warmup_input_gpu = cuda.mem_alloc(warmup_input_data.nbytes)
        warmup_output_gpu = cuda.mem_alloc(warmup_output_data.nbytes)
        cuda.memcpy_htod(warmup_input_gpu, warmup_input_data)
        context.set_tensor_address(input_name, warmup_input_gpu)
        context.set_tensor_address(output_name, warmup_output_gpu)
        context.execute_async_v3(0)
        cuda.memcpy_dtoh(warmup_output_data, warmup_output_gpu)
        warmup_input_gpu.free()
        warmup_output_gpu.free()
        
        # バッチ処理で測定
        num_batches = (len(test_samples) + batch_size - 1) // batch_size
        all_predictions = []
        inference_times = []
        
        for run in range(num_runs):
            run_predictions = []
            start_time = time.time()
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(test_samples))
                current_batch_size = end_idx - start_idx
                
                batch_data = test_samples[start_idx:end_idx]
                
                # 入力形状設定
                context.set_input_shape(input_name, [current_batch_size, 32, 32, 3])
                output_shape = context.get_tensor_shape(output_name)
                
                # データ準備
                input_data = batch_data.astype(np.float32)
                output_data = np.empty(output_shape, dtype=np.float32)
                
                # GPU メモリ確保
                input_gpu = cuda.mem_alloc(input_data.nbytes)
                output_gpu = cuda.mem_alloc(output_data.nbytes)
                
                # データ転送・推論・結果取得
                cuda.memcpy_htod(input_gpu, input_data)
                context.set_tensor_address(input_name, input_gpu)
                context.set_tensor_address(output_name, output_gpu)
                context.execute_async_v3(0)
                cuda.memcpy_dtoh(output_data, output_gpu)
                
                # GPU メモリ解放
                input_gpu.free()
                output_gpu.free()
                
                run_predictions.append(output_data)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            if run == 0:  # 最初の実行結果を精度計算用に保存
                all_predictions = np.vstack(run_predictions)
            
            print(f"  Run {run+1}: {inference_time:.4f}s ({num_batches} batches)")
        
        # 精度計算
        accuracy = calculate_accuracy(all_predictions, test_labels)
        
        result = {
            'precision': precision_name,
            'load_time': load_time,
            'inference_times': inference_times,
            'mean_time': statistics.mean(inference_times),
            'std_time': statistics.stdev(inference_times),
            'samples_per_second': len(test_samples) / statistics.mean(inference_times),
            'accuracy': accuracy,
            'num_batches': num_batches,
            'batch_size': batch_size
        }
        
        print(f"✅ {precision_name} - Load: {load_time:.3f}s, Inference: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
        print(f"   Throughput: {result['samples_per_second']:.1f} samples/sec, Accuracy: {accuracy:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error benchmarking {precision_name}: {e}")
        return None

def generate_tensorrt_engines():
    """FP16とINT8のTensorRTエンジンを生成"""
    print("🔧 Generating TensorRT engines for FP16 and INT8...")
    
    try:
        import tensorrt as trt
        import onnx
        
        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # ONNXモデル読み込み
        onnx_model_path = 'model.onnx'
        if not os.path.exists(onnx_model_path):
            print("❌ ONNX model not found! Please convert the model first.")
            return False
        
        builder = trt.Builder(TRT_LOGGER)
        
        # FP16エンジン生成
        print("\n📦 Building FP16 engine...")
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model for FP16")
                return False
        
        config = builder.create_builder_config()
        
        # FP16設定
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 precision enabled")
        else:
            print("⚠️ FP16 not supported, using FP32")
        
        # 最適化プロファイル設定
        profile = builder.create_optimization_profile()
        profile.set_shape('input_1', [1, 32, 32, 3], [16, 32, 32, 3], [32, 32, 32, 3])
        config.add_optimization_profile(profile)
        
        # エンジンビルド
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine:
            with open('model_fp16.trt', 'wb') as f:
                f.write(serialized_engine)
            print("✅ FP16 engine saved as model_fp16.trt")
        else:
            print("❌ Failed to build FP16 engine")
            return False
        
        # INT8エンジン生成
        print("\n📦 Building INT8 engine...")
        network_int8 = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser_int8 = trt.OnnxParser(network_int8, TRT_LOGGER)
        
        with open(onnx_model_path, 'rb') as model:
            if not parser_int8.parse(model.read()):
                print("❌ Failed to parse ONNX model for INT8")
                return False
        
        config_int8 = builder.create_builder_config()
        
        # INT8設定
        if builder.platform_has_fast_int8:
            config_int8.set_flag(trt.BuilderFlag.INT8)
            print("✅ INT8 precision enabled")
            
            # INT8キャリブレーション（簡易版）
            import pycuda.driver as cuda_cal
            import pycuda.autoinit
            
            class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, data):
                    trt.IInt8EntropyCalibrator2.__init__(self)
                    self.data = data.astype(np.float32)
                    self.batch_idx = 0
                    self.batch_size = 32
                    self.device_input = None
                    
                def get_batch_size(self):
                    return self.batch_size
                    
                def get_batch(self, names):
                    if self.batch_idx < len(self.data) // self.batch_size:
                        start_idx = self.batch_idx * self.batch_size
                        end_idx = start_idx + self.batch_size
                        batch = self.data[start_idx:end_idx]
                        
                        if self.device_input is None:
                            self.device_input = cuda_cal.mem_alloc(batch.nbytes)
                        
                        cuda_cal.memcpy_htod(self.device_input, batch)
                        self.batch_idx += 1
                        return [int(self.device_input)]
                    return None
                    
                def read_calibration_cache(self):
                    return None
                    
                def write_calibration_cache(self, cache):
                    pass
                    
                def __del__(self):
                    if self.device_input:
                        self.device_input.free()
            
            # キャリブレーションデータとして最初の100サンプルを使用
            test_samples_cal, _ = load_test_data()
            if test_samples_cal is not None:
                calibrator = SimpleCalibrator(test_samples_cal[:100])
                config_int8.int8_calibrator = calibrator
                print("✅ INT8 calibrator configured")
            
        else:
            print("⚠️ INT8 not supported, using FP32")
        
        # 最適化プロファイル設定
        profile_int8 = builder.create_optimization_profile()
        profile_int8.set_shape('input_1', [1, 32, 32, 3], [16, 32, 32, 3], [32, 32, 32, 3])
        config_int8.add_optimization_profile(profile_int8)
        
        # エンジンビルド
        serialized_engine_int8 = builder.build_serialized_network(network_int8, config_int8)
        if serialized_engine_int8:
            with open('model_int8.trt', 'wb') as f:
                f.write(serialized_engine_int8)
            print("✅ INT8 engine saved as model_int8.trt")
        else:
            print("❌ Failed to build INT8 engine")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating engines: {e}")
        return False

def main():
    """メイン実行関数"""
    print("🚀 TensorRT Precision Comparison: FP16 vs INT8")
    print("=" * 60)
    
    # テストデータ読み込み
    test_samples, test_labels = load_test_data()
    if test_samples is None:
        return
    
    # TensorRTエンジン生成
    if not generate_tensorrt_engines():
        print("❌ Failed to generate TensorRT engines")
        return
    
    # FP16ベンチマーク
    fp16_result = benchmark_tensorrt_precision(
        'model_fp16.trt', test_samples, test_labels, 'FP16'
    )
    
    # INT8ベンチマーク
    int8_result = benchmark_tensorrt_precision(
        'model_int8.trt', test_samples, test_labels, 'INT8'
    )
    
    # 結果比較
    if fp16_result and int8_result:
        print("\n📊 Precision Comparison Results:")
        print("=" * 60)
        print(f"{'Metric':<20} {'FP16':<15} {'INT8':<15} {'Ratio (INT8/FP16)':<15}")
        print("-" * 60)
        
        # 推論時間比較
        fp16_time = fp16_result['mean_time']
        int8_time = int8_result['mean_time']
        time_ratio = int8_time / fp16_time
        print(f"{'Inference Time':<20} {fp16_time:.4f}s {int8_time:.4f}s {time_ratio:.2f}x")
        
        # スループット比較
        fp16_throughput = fp16_result['samples_per_second']
        int8_throughput = int8_result['samples_per_second']
        throughput_ratio = int8_throughput / fp16_throughput
        print(f"{'Throughput':<20} {fp16_throughput:.1f}/s {int8_throughput:.1f}/s {throughput_ratio:.2f}x")
        
        # 精度比較
        fp16_accuracy = fp16_result['accuracy']
        int8_accuracy = int8_result['accuracy']
        accuracy_diff = int8_accuracy - fp16_accuracy
        print(f"{'Accuracy':<20} {fp16_accuracy:.4f} {int8_accuracy:.4f} {accuracy_diff:+.4f}")
        
        # ロード時間比較
        fp16_load = fp16_result['load_time']
        int8_load = int8_result['load_time']
        load_ratio = int8_load / fp16_load
        print(f"{'Load Time':<20} {fp16_load:.3f}s {int8_load:.3f}s {load_ratio:.2f}x")
        
        print("\n🎯 Summary:")
        if throughput_ratio > 1.0:
            print(f"✅ INT8 is {throughput_ratio:.1f}x faster than FP16")
        else:
            print(f"⚠️ FP16 is {1/throughput_ratio:.1f}x faster than INT8")
        
        if abs(accuracy_diff) < 0.01:
            print(f"✅ Accuracy difference is minimal: {accuracy_diff:+.4f}")
        else:
            print(f"⚠️ Significant accuracy difference: {accuracy_diff:+.4f}")
        
        # 結果をJSONファイルに保存
        results = {
            'fp16': fp16_result,
            'int8': int8_result,
            'comparison': {
                'time_ratio': time_ratio,
                'throughput_ratio': throughput_ratio,
                'accuracy_difference': accuracy_diff,
                'load_time_ratio': load_ratio
            }
        }
        
        with open('precision_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to precision_comparison_results.json")

if __name__ == "__main__":
    import os
    main()