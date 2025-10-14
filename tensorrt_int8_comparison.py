#!/usr/bin/env python3
"""
TensorRT INT8 Engine Generator with Proper Calibration
TensorRT専用INT8エンジンの生成と性能比較
"""
import time
import numpy as np
import statistics
import json
import os
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

class EntropyCalibrator:
    """TensorRT INT8キャリブレーション用クラス"""
    
    def __init__(self, calibration_data, batch_size=32, cache_file="calibration.cache"):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.calibration_data = calibration_data.astype(np.float32)
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0
        self.device_input = None
        self.cuda = cuda
        
        # GPU メモリ確保
        if len(self.calibration_data) > 0:
            self.device_input = cuda.mem_alloc(self.calibration_data[0:self.batch_size].nbytes)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """キャリブレーション用バッチデータ取得"""
        if self.current_index + self.batch_size > len(self.calibration_data):
            return None
        
        batch = self.calibration_data[self.current_index:self.current_index + self.batch_size]
        self.cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        """キャリブレーションキャッシュ読み込み"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """キャリブレーションキャッシュ書き込み"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_tensorrt_int8_engine():
    """TensorRT INT8エンジンのビルド"""
    print("🔧 Building TensorRT INT8 engine with proper calibration...")
    
    try:
        import tensorrt as trt
        
        # テストデータ読み込み（キャリブレーション用）
        test_samples, _ = load_test_data()
        if test_samples is None:
            return False
        
        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # ONNXモデル読み込み
        onnx_model_path = 'model.onnx'
        if not os.path.exists(onnx_model_path):
            print("❌ ONNX model not found!")
            return False
        
        builder = trt.Builder(TRT_LOGGER)
        
        # ネットワーク作成
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"   {parser.get_error(error)}")
                return False
        
        config = builder.create_builder_config()
        
        # INT8設定
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✅ INT8 precision enabled")
            
            # キャリブレーション設定（より多くのデータを使用）
            calibration_data = test_samples[:min(320, len(test_samples))]  # 10バッチ分
            calibrator = EntropyCalibrator(calibration_data, batch_size=32)
            
            # TensorRT 10.x では IInt8EntropyCalibrator2 を継承
            class TensorRTCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, calibrator):
                    trt.IInt8EntropyCalibrator2.__init__(self)
                    self.calibrator = calibrator
                
                def get_batch_size(self):
                    return self.calibrator.get_batch_size()
                
                def get_batch(self, names):
                    return self.calibrator.get_batch(names)
                
                def read_calibration_cache(self):
                    return self.calibrator.read_calibration_cache()
                
                def write_calibration_cache(self, cache):
                    return self.calibrator.write_calibration_cache(cache)
            
            config.int8_calibrator = TensorRTCalibrator(calibrator)
            print("✅ INT8 calibrator configured")
            
        else:
            print("⚠️ INT8 not supported, using FP32")
            return False
        
        # 最適化プロファイル設定
        profile = builder.create_optimization_profile()
        profile.set_shape('input_1', [1, 32, 32, 3], [16, 32, 32, 3], [32, 32, 32, 3])
        config.add_optimization_profile(profile)
        
        # エンジンビルド
        print("   Building INT8 engine (this may take a while)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine:
            with open('model_tensorrt_int8.trt', 'wb') as f:
                f.write(serialized_engine)
            print("✅ TensorRT INT8 engine saved as model_tensorrt_int8.trt")
            return True
        else:
            print("❌ Failed to build INT8 engine")
            return False
        
    except Exception as e:
        print(f"❌ Error building INT8 engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_tensorrt_precision(engine_path: str, test_samples: np.ndarray, 
                                test_labels: np.ndarray, precision_name: str,
                                batch_size: int = 32, num_runs: int = 5) -> Dict[str, Any]:
    """TensorRT推論ベンチマーク"""
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
            'framework': 'TensorRT',
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

def main():
    """メイン実行関数"""
    print("🚀 TensorRT Complete Precision Comparison: FP32 vs FP16 vs INT8")
    print("=" * 70)
    
    # テストデータ読み込み
    test_samples, test_labels = load_test_data()
    if test_samples is None:
        return
    
    # TensorRT INT8エンジンをビルド
    if not build_tensorrt_int8_engine():
        print("❌ Failed to build TensorRT INT8 engine")
        return
    
    print("\n" + "="*70)
    print("🔄 Running TensorRT Precision Benchmarks...")
    
    # 既存エンジンの確認
    engines = {
        'FP32': 'model.trt',
        'FP16': 'model_fp16.trt',
        'INT8': 'model_tensorrt_int8.trt'
    }
    
    results = {}
    
    for precision, engine_path in engines.items():
        if os.path.exists(engine_path):
            result = benchmark_tensorrt_precision(
                engine_path, test_samples, test_labels, precision
            )
            if result:
                results[precision] = result
        else:
            print(f"⚠️ {precision} engine not found: {engine_path}")
    
    # 結果比較
    if len(results) >= 2:
        print("\n📊 TensorRT Precision Comparison Results:")
        print("=" * 90)
        print(f"{'Precision':<10} {'Time (s)':<10} {'Throughput':<15} {'Accuracy':<10} {'Size (MB)':<12} {'Speedup':<10}")
        print("-" * 90)
        
        # ベースライン（FP32）
        baseline_throughput = results.get('FP32', {}).get('samples_per_second', 1)
        
        for precision in ['FP32', 'FP16', 'INT8']:
            if precision in results:
                r = results[precision]
                engine_file = engines[precision]
                size_mb = os.path.getsize(engine_file) / (1024*1024) if os.path.exists(engine_file) else 0
                speedup = r['samples_per_second'] / baseline_throughput
                
                print(f"{precision:<10} {r['mean_time']:<10.4f} {r['samples_per_second']:<15.1f} {r['accuracy']:<10.4f} {size_mb:<12.1f} {speedup:<10.1f}x")
        
        print("\n🎯 TensorRT Precision Analysis:")
        
        # 速度比較
        if 'FP16' in results and 'FP32' in results:
            fp16_speedup = results['FP16']['samples_per_second'] / results['FP32']['samples_per_second']
            print(f"⚡ FP16 speedup: {fp16_speedup:.1f}x over FP32")
        
        if 'INT8' in results and 'FP32' in results:
            int8_speedup = results['INT8']['samples_per_second'] / results['FP32']['samples_per_second']
            print(f"⚡ INT8 speedup: {int8_speedup:.1f}x over FP32")
        
        # 精度比較
        accuracies = [r['accuracy'] for r in results.values()]
        accuracy_range = max(accuracies) - min(accuracies)
        print(f"🎯 Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f} (diff: {accuracy_range:.4f})")
        
        # サイズ比較
        sizes = {}
        for precision, engine_path in engines.items():
            if os.path.exists(engine_path):
                sizes[precision] = os.path.getsize(engine_path) / (1024*1024)
        
        if 'FP32' in sizes and 'INT8' in sizes:
            size_reduction = sizes['FP32'] / sizes['INT8']
            print(f"💾 INT8 size reduction: {size_reduction:.1f}x smaller than FP32")
        
        # 結果をJSONファイルに保存
        with open('tensorrt_precision_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to tensorrt_precision_comparison.json")

if __name__ == "__main__":
    main()