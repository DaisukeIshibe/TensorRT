#!/usr/bin/env python3
"""
TensorRT FP32 vs FP16 Precision Comparison
処理速度と分類精度の比較（INT8は別途PyTorchで実装）
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

def generate_fp16_engine():
    """FP16のTensorRTエンジンを生成"""
    print("🔧 Generating TensorRT FP16 engine...")
    
    try:
        import tensorrt as trt
        
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
                for error in range(parser.num_errors):
                    print(f"   {parser.get_error(error)}")
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
            return True
        else:
            print("❌ Failed to build FP16 engine")
            return False
        
    except Exception as e:
        print(f"❌ Error generating FP16 engine: {e}")
        return False

def benchmark_pytorch_quantized():
    """PyTorchでのINT8量子化モデルベンチマーク"""
    print("\n🔄 Benchmarking PyTorch INT8 Quantization...")
    
    try:
        # TensorFlowコンテナでも利用可能なtorch（別途実装が必要）
        print("⚠️ PyTorch INT8 quantization requires separate implementation")
        print("   This would need PyTorch with quantization support")
        
        # 仮の結果（実際の実装では適切な量子化を行う）
        mock_result = {
            'precision': 'INT8 (PyTorch)',
            'load_time': 0.2,
            'mean_time': 0.008,
            'std_time': 0.001,
            'samples_per_second': 12500.0,
            'accuracy': 0.78,  # 量子化による精度低下を想定
            'num_batches': 4,
            'batch_size': 32,
            'note': 'Mock result - requires PyTorch implementation'
        }
        
        return mock_result
        
    except Exception as e:
        print(f"❌ Error with PyTorch quantization: {e}")
        return None

def main():
    """メイン実行関数"""
    print("🚀 TensorRT Precision Comparison: FP32 vs FP16")
    print("=" * 50)
    
    # テストデータ読み込み
    test_samples, test_labels = load_test_data()
    if test_samples is None:
        return
    
    # 既存のFP32エンジンを使用
    fp32_engine_path = 'model.trt'
    if not os.path.exists(fp32_engine_path):
        print("❌ FP32 TensorRT engine not found!")
        return
    
    # FP16エンジン生成
    if not generate_fp16_engine():
        print("❌ Failed to generate FP16 engine")
        return
    
    # FP32ベンチマーク
    print("\n" + "="*50)
    fp32_result = benchmark_tensorrt_precision(
        fp32_engine_path, test_samples, test_labels, 'FP32'
    )
    
    # FP16ベンチマーク
    fp16_result = benchmark_tensorrt_precision(
        'model_fp16.trt', test_samples, test_labels, 'FP16'
    )
    
    # PyTorch INT8比較（モック）
    int8_result = benchmark_pytorch_quantized()
    
    # 結果比較
    if fp32_result and fp16_result:
        print("\n📊 Precision Comparison Results:")
        print("=" * 80)
        print(f"{'Metric':<20} {'FP32':<15} {'FP16':<15} {'Speedup (FP16/FP32)':<20}")
        print("-" * 80)
        
        # 推論時間比較
        fp32_time = fp32_result['mean_time']
        fp16_time = fp16_result['mean_time']
        time_speedup = fp32_time / fp16_time
        print(f"{'Inference Time':<20} {fp32_time:.4f}s {fp16_time:.4f}s {time_speedup:.2f}x")
        
        # スループット比較
        fp32_throughput = fp32_result['samples_per_second']
        fp16_throughput = fp16_result['samples_per_second']
        throughput_speedup = fp16_throughput / fp32_throughput
        print(f"{'Throughput':<20} {fp32_throughput:.1f}/s {fp16_throughput:.1f}/s {throughput_speedup:.2f}x")
        
        # 精度比較
        fp32_accuracy = fp32_result['accuracy']
        fp16_accuracy = fp16_result['accuracy']
        accuracy_diff = fp16_accuracy - fp32_accuracy
        print(f"{'Accuracy':<20} {fp32_accuracy:.4f} {fp16_accuracy:.4f} {accuracy_diff:+.4f}")
        
        # ロード時間比較
        fp32_load = fp32_result['load_time']
        fp16_load = fp16_result['load_time']
        load_speedup = fp32_load / fp16_load
        print(f"{'Load Time':<20} {fp32_load:.3f}s {fp16_load:.3f}s {load_speedup:.2f}x")
        
        # エンジンサイズ比較
        fp32_size = os.path.getsize(fp32_engine_path) / (1024*1024)
        fp16_size = os.path.getsize('model_fp16.trt') / (1024*1024)
        size_reduction = fp32_size / fp16_size
        print(f"{'Engine Size':<20} {fp32_size:.1f}MB {fp16_size:.1f}MB {size_reduction:.2f}x smaller")
        
        print("\n🎯 FP32 vs FP16 Summary:")
        print(f"✅ FP16 is {throughput_speedup:.1f}x faster than FP32")
        print(f"✅ FP16 engine is {size_reduction:.1f}x smaller than FP32")
        if abs(accuracy_diff) < 0.01:
            print(f"✅ Accuracy difference is minimal: {accuracy_diff:+.4f}")
        else:
            print(f"⚠️ Accuracy difference: {accuracy_diff:+.4f}")
        
        # INT8結果も表示（モック）
        if int8_result:
            print(f"\n📋 INT8 Reference (Mock):")
            print(f"   Throughput: {int8_result['samples_per_second']:.1f} samples/sec")
            print(f"   Accuracy: {int8_result['accuracy']:.4f}")
            print(f"   Note: {int8_result['note']}")
        
        # 結果をJSONファイルに保存
        results = {
            'fp32': fp32_result,
            'fp16': fp16_result,
            'int8_mock': int8_result,
            'comparison': {
                'fp16_speedup': throughput_speedup,
                'accuracy_difference': accuracy_diff,
                'load_time_speedup': load_speedup,
                'size_reduction': size_reduction
            }
        }
        
        with open('precision_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to precision_comparison_results.json")

if __name__ == "__main__":
    main()