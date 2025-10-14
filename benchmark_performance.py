#!/usr/bin/env python3
"""
Performance Benchmark: SavedModel vs ONNX vs TensorRT (Batch-optimized)
各モデル形式での推論速度を詳細に比較測定
"""
import time
import numpy as np
import os
import statistics
from typing import List, Dict, Any

def load_test_data(num_samples: int = 100):
    """テストデータの読み込み"""
    try:
        test_samples = np.load('test_samples.npy')[:num_samples]
        test_labels = np.load('test_labels.npy')[:num_samples]
        print(f"✅ Loaded {len(test_samples)} test samples: {test_samples.shape}")
        return test_samples, test_labels
    except FileNotFoundError:
        print("❌ Test data not found. Please run data generation first.")
        return None, None

def benchmark_savedmodel(test_samples: np.ndarray, num_runs: int = 5) -> Dict[str, Any]:
    """SavedModel推論速度測定"""
    print("\n🔄 Benchmarking SavedModel...")
    
    try:
        import tensorflow as tf
        
        # モデル読み込み時間測定
        start_time = time.time()
        model = tf.keras.models.load_model('cifar10_vgg_model')
        load_time = time.time() - start_time
        
        # ウォームアップ
        _ = model.predict(test_samples[:1], verbose=0)
        
        # 複数回測定
        inference_times = []
        for run in range(num_runs):
            start_time = time.time()
            predictions = model.predict(test_samples, verbose=0)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            print(f"  Run {run+1}: {inference_time:.4f}s")
        
        return {
            'name': 'SavedModel',
            'load_time': load_time,
            'inference_times': inference_times,
            'mean_time': statistics.mean(inference_times),
            'std_time': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'samples_per_second': len(test_samples) / statistics.mean(inference_times),
            'status': '✅ Success'
        }
    except Exception as e:
        print(f"❌ SavedModel benchmark failed: {e}")
        return {'name': 'SavedModel', 'status': f'❌ Failed: {e}'}

def benchmark_onnx(test_samples: np.ndarray, num_runs: int = 5) -> Dict[str, Any]:
    """ONNX推論速度測定"""
    print("\n🔄 Benchmarking ONNX...")
    
    try:
        import onnxruntime as ort
        
        # セッション作成時間測定
        start_time = time.time()
        session = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        load_time = time.time() - start_time
        
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: test_samples.astype(np.float32)}
        
        # ウォームアップ
        _ = session.run(None, {input_name: test_samples[:1].astype(np.float32)})
        
        # 複数回測定
        inference_times = []
        for run in range(num_runs):
            start_time = time.time()
            predictions = session.run(None, ort_inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            print(f"  Run {run+1}: {inference_time:.4f}s")
        
        return {
            'name': 'ONNX',
            'load_time': load_time,
            'inference_times': inference_times,
            'mean_time': statistics.mean(inference_times),
            'std_time': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'samples_per_second': len(test_samples) / statistics.mean(inference_times),
            'status': '✅ Success'
        }
    except Exception as e:
        print(f"❌ ONNX benchmark failed: {e}")
        return {'name': 'ONNX', 'status': f'❌ Failed: {e}'}

def benchmark_tensorrt_batch(test_samples: np.ndarray, batch_size: int = 32, num_runs: int = 5) -> Dict[str, Any]:
    """TensorRT バッチ処理推論速度測定"""
    print(f"\n🔄 Benchmarking TensorRT (Batch size: {batch_size})...")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # エンジン読み込み時間測定
        start_time = time.time()
        runtime = trt.Runtime(TRT_LOGGER)
        with open('model.trt', 'rb') as f:
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
        
        # バッチ処理
        num_batches = (len(test_samples) + batch_size - 1) // batch_size
        
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
        
        # 複数回測定
        inference_times = []
        for run in range(num_runs):
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
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            print(f"  Run {run+1}: {inference_time:.4f}s ({num_batches} batches)")
        
        return {
            'name': f'TensorRT (Batch {batch_size})',
            'load_time': load_time,
            'inference_times': inference_times,
            'mean_time': statistics.mean(inference_times),
            'std_time': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'samples_per_second': len(test_samples) / statistics.mean(inference_times),
            'batch_size': batch_size,
            'num_batches': num_batches,
            'status': '✅ Success'
        }
    except Exception as e:
        print(f"❌ TensorRT benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'name': f'TensorRT (Batch {batch_size})', 'status': f'❌ Failed: {e}'}

def benchmark_tensorrt_single(test_samples: np.ndarray, num_runs: int = 5) -> Dict[str, Any]:
    """TensorRT 単一サンプル処理推論速度測定（比較用）"""
    print(f"\n🔄 Benchmarking TensorRT (Single sample processing)...")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # エンジン読み込み
        start_time = time.time()
        runtime = trt.Runtime(TRT_LOGGER)
        with open('model.trt', 'rb') as f:
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
        context.set_input_shape(input_name, [1, 32, 32, 3])
        warmup_output_shape = context.get_tensor_shape(output_name)
        warmup_input_data = test_samples[:1].astype(np.float32)
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
        
        # 複数回測定（最初の10サンプルのみ）
        limited_samples = test_samples[:10]  # 時間短縮のため10サンプルのみ
        inference_times = []
        for run in range(num_runs):
            start_time = time.time()
            
            for i, sample in enumerate(limited_samples):
                # 入力形状設定
                context.set_input_shape(input_name, [1, 32, 32, 3])
                output_shape = context.get_tensor_shape(output_name)
                
                # データ準備
                input_data = sample.reshape(1, 32, 32, 3).astype(np.float32)
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
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            print(f"  Run {run+1}: {inference_time:.4f}s ({len(limited_samples)} samples)")
        
        # 100サンプル相当に換算
        scaled_time = statistics.mean(inference_times) * (len(test_samples) / len(limited_samples))
        
        return {
            'name': 'TensorRT (Single)',
            'load_time': load_time,
            'inference_times': inference_times,
            'mean_time': statistics.mean(inference_times),
            'scaled_time_100': scaled_time,
            'std_time': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'samples_per_second': len(limited_samples) / statistics.mean(inference_times),
            'scaled_samples_per_second': len(test_samples) / scaled_time,
            'samples_tested': len(limited_samples),
            'status': '✅ Success'
        }
    except Exception as e:
        print(f"❌ TensorRT Single benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {'name': 'TensorRT (Single)', 'status': f'❌ Failed: {e}'}

def print_benchmark_results(results: List[Dict[str, Any]]):
    """ベンチマーク結果の表示"""
    print("\n" + "="*80)
    print("🏁 BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # 成功した結果のみフィルタ
    successful_results = [r for r in results if '✅ Success' in r.get('status', '')]
    
    if not successful_results:
        print("❌ No successful benchmarks to compare")
        return
    
    # 結果テーブル
    print(f"{'Model':<25} {'Load Time':<12} {'Avg Inference':<15} {'Std Dev':<10} {'Samples/sec':<12} {'Status'}")
    print("-" * 80)
    
    for result in results:
        if '✅ Success' in result.get('status', ''):
            load_time = f"{result['load_time']:.3f}s"
            if 'scaled_time_100' in result:
                avg_time = f"{result['scaled_time_100']:.3f}s*"
                samples_per_sec = f"{result['scaled_samples_per_second']:.1f}*"
            else:
                avg_time = f"{result['mean_time']:.3f}s"
                samples_per_sec = f"{result['samples_per_second']:.1f}"
            std_time = f"{result['std_time']:.3f}s"
            
            print(f"{result['name']:<25} {load_time:<12} {avg_time:<15} {std_time:<10} {samples_per_sec:<12} {result['status']}")
        else:
            print(f"{result['name']:<25} {'N/A':<12} {'N/A':<15} {'N/A':<10} {'N/A':<12} {result['status']}")
    
    print("\n* TensorRT Single results are scaled to 100 samples")
    
    # パフォーマンス比較
    if len(successful_results) > 1:
        print("\n📊 PERFORMANCE COMPARISON:")
        
        # 基準となる最速のモデルを見つける
        fastest_result = min(successful_results, 
                           key=lambda x: x.get('scaled_time_100', x.get('mean_time', float('inf'))))
        fastest_time = fastest_result.get('scaled_time_100', fastest_result.get('mean_time'))
        
        print(f"Baseline (fastest): {fastest_result['name']} - {fastest_time:.3f}s")
        print("-" * 50)
        
        for result in successful_results:
            current_time = result.get('scaled_time_100', result.get('mean_time'))
            speedup = current_time / fastest_time
            if speedup < 1.1:
                status = "🟢 Similar"
            elif speedup < 2.0:
                status = "🟡 Slower"
            else:
                status = "🔴 Much slower"
            
            print(f"{result['name']:<25} {speedup:.2f}x {status}")
    
    # GPU効率性の表示
    print(f"\n🚀 GPU EFFICIENCY ANALYSIS:")
    batch_results = [r for r in successful_results if 'Batch' in r['name']]
    single_results = [r for r in successful_results if 'Single' in r['name']]
    
    if batch_results and single_results:
        batch_time = batch_results[0].get('mean_time', 0)
        single_time = single_results[0].get('scaled_time_100', 0)
        if single_time > 0:
            efficiency_gain = single_time / batch_time
            print(f"Batch processing efficiency: {efficiency_gain:.1f}x faster than single processing")
            print(f"GPU transfer reduction: ~{((100-32)/100)*100:.0f}% fewer transfers")

def main():
    """メイン実行関数"""
    print("🏃‍♂️ Performance Benchmark: SavedModel vs ONNX vs TensorRT")
    print("="*60)
    
    # テストデータ読み込み
    test_samples, test_labels = load_test_data(100)
    if test_samples is None:
        return
    
    print(f"Testing with {len(test_samples)} samples")
    print(f"Input shape: {test_samples.shape}")
    
    # 各モデルのベンチマーク実行
    results = []
    
    # SavedModel
    if os.path.exists('cifar10_vgg_model'):
        results.append(benchmark_savedmodel(test_samples))
    else:
        print("⚠️ SavedModel not found, skipping...")
        results.append({'name': 'SavedModel', 'status': '❌ Model file not found'})
    
    # ONNX
    if os.path.exists('model.onnx'):
        results.append(benchmark_onnx(test_samples))
    else:
        print("⚠️ ONNX model not found, skipping...")
        results.append({'name': 'ONNX', 'status': '❌ Model file not found'})
    
    # TensorRT バッチ処理
    if os.path.exists('model.trt'):
        results.append(benchmark_tensorrt_batch(test_samples, batch_size=32))
        results.append(benchmark_tensorrt_single(test_samples))
    else:
        print("⚠️ TensorRT engine not found, skipping...")
        results.append({'name': 'TensorRT (Batch 32)', 'status': '❌ Engine file not found'})
        results.append({'name': 'TensorRT (Single)', 'status': '❌ Engine file not found'})
    
    # 結果表示
    print_benchmark_results(results)
    
    print(f"\n🎯 Benchmark completed successfully!")
    print(f"📝 All results based on {len(test_samples)} CIFAR-10 test samples")

if __name__ == "__main__":
    main()