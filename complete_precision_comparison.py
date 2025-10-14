#!/usr/bin/env python3
"""
Complete Precision Comparison: FP32, FP16, INT8
TensorRT FP32/FP16 + TensorFlow Lite INT8の完全比較
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

def create_int8_tflite_model():
    """INT8量子化TensorFlow Liteモデルを作成"""
    print("🔧 Creating INT8 quantized TensorFlow Lite model...")
    
    try:
        import tensorflow as tf
        
        # SavedModelを読み込み
        model = tf.keras.models.load_model('cifar10_vgg_model')
        
        # テストデータを代表データセットとして使用
        test_samples, _ = load_test_data()
        if test_samples is None:
            return False
        
        # 代表データセット関数
        def representative_dataset():
            for i in range(min(100, len(test_samples))):
                yield [test_samples[i:i+1].astype(np.float32)]
        
        # TensorFlow Lite変換設定
        converter = tf.lite.TFLiteConverter.from_saved_model('cifar10_vgg_model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # 量子化モデル生成
        print("   Converting to INT8 quantized model...")
        tflite_quant_model = converter.convert()
        
        # モデル保存
        with open('model_int8.tflite', 'wb') as f:
            f.write(tflite_quant_model)
        
        print("✅ INT8 quantized model saved as model_int8.tflite")
        
        # FP32 TFLiteモデルも作成（比較用）
        converter_fp32 = tf.lite.TFLiteConverter.from_saved_model('cifar10_vgg_model')
        tflite_fp32_model = converter_fp32.convert()
        
        with open('model_fp32.tflite', 'wb') as f:
            f.write(tflite_fp32_model)
        
        print("✅ FP32 TFLite model saved as model_fp32.tflite")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating INT8 model: {e}")
        return False

def benchmark_tflite_model(model_path: str, test_samples: np.ndarray, 
                          test_labels: np.ndarray, precision_name: str,
                          num_runs: int = 5) -> Dict[str, Any]:
    """TensorFlow Liteモデルのベンチマーク"""
    print(f"\n🔄 Benchmarking TensorFlow Lite {precision_name}...")
    
    try:
        import tensorflow as tf
        
        # モデル読み込み時間測定
        start_time = time.time()
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        load_time = time.time() - start_time
        
        # 入出力テンソル情報取得
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        output_dtype = output_details[0]['dtype']
        
        print(f"   Input: {input_shape}, dtype: {input_dtype}")
        print(f"   Output dtype: {output_dtype}")
        
        # ウォームアップ
        test_input = test_samples[0:1].astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # バッチ処理（TFLiteは通常バッチサイズ1）
        all_predictions = []
        inference_times = []
        
        for run in range(num_runs):
            run_predictions = []
            start_time = time.time()
            
            for i in range(len(test_samples)):
                # 入力データ準備
                if input_dtype == np.uint8:
                    # INT8モデルの場合、入力を正規化してuint8に変換
                    input_data = test_samples[i:i+1].astype(np.float32)
                    # 0-255の範囲に正規化
                    input_data = (input_data + 1.0) * 127.5  # [-1,1] -> [0,255]
                    input_data = np.clip(input_data, 0, 255).astype(np.uint8)
                else:
                    input_data = test_samples[i:i+1].astype(input_dtype)
                
                # 推論実行
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                # 出力取得
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # INT8出力の場合はfloatに変換
                if output_dtype == np.uint8:
                    output_data = output_data.astype(np.float32)
                    # dequantization parameters取得
                    scale = output_details[0]['quantization'][0]
                    zero_point = output_details[0]['quantization'][1]
                    if scale != 0:
                        output_data = scale * (output_data - zero_point)
                
                run_predictions.append(output_data)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            if run == 0:  # 最初の実行結果を精度計算用に保存
                all_predictions = np.vstack(run_predictions)
            
            print(f"  Run {run+1}: {inference_time:.4f}s ({len(test_samples)} samples)")
        
        # 精度計算
        accuracy = calculate_accuracy(all_predictions, test_labels)
        
        result = {
            'precision': precision_name,
            'framework': 'TensorFlow Lite',
            'load_time': load_time,
            'inference_times': inference_times,
            'mean_time': statistics.mean(inference_times),
            'std_time': statistics.stdev(inference_times),
            'samples_per_second': len(test_samples) / statistics.mean(inference_times),
            'accuracy': accuracy,
            'input_dtype': str(input_dtype),
            'output_dtype': str(output_dtype)
        }
        
        print(f"✅ {precision_name} - Load: {load_time:.3f}s, Inference: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
        print(f"   Throughput: {result['samples_per_second']:.1f} samples/sec, Accuracy: {accuracy:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error benchmarking {precision_name}: {e}")
        return None

def get_tensorrt_results():
    """既存のTensorRT結果を読み込み"""
    try:
        with open('precision_comparison_results.json', 'r') as f:
            results = json.load(f)
        return results['fp32'], results['fp16']
    except:
        return None, None

def main():
    """メイン実行関数"""
    print("🚀 Complete Precision Comparison: FP32, FP16, INT8")
    print("=" * 60)
    
    # テストデータ読み込み
    test_samples, test_labels = load_test_data()
    if test_samples is None:
        return
    
    # INT8量子化モデル作成
    if not create_int8_tflite_model():
        print("❌ Failed to create INT8 model")
        return
    
    # TensorFlow Lite FP32ベンチマーク
    tflite_fp32_result = benchmark_tflite_model(
        'model_fp32.tflite', test_samples, test_labels, 'FP32'
    )
    
    # TensorFlow Lite INT8ベンチマーク
    tflite_int8_result = benchmark_tflite_model(
        'model_int8.tflite', test_samples, test_labels, 'INT8'
    )
    
    # TensorRT結果読み込み
    tensorrt_fp32, tensorrt_fp16 = get_tensorrt_results()
    
    # 総合結果比較
    if all([tflite_fp32_result, tflite_int8_result, tensorrt_fp32, tensorrt_fp16]):
        print("\n📊 Complete Precision Comparison Results:")
        print("=" * 100)
        print(f"{'Model':<25} {'Framework':<15} {'Time (s)':<10} {'Throughput':<15} {'Accuracy':<10} {'Size (MB)':<10}")
        print("-" * 100)
        
        # TensorRT結果
        trt_fp32_size = os.path.getsize('model.trt') / (1024*1024) if os.path.exists('model.trt') else 0
        trt_fp16_size = os.path.getsize('model_fp16.trt') / (1024*1024) if os.path.exists('model_fp16.trt') else 0
        
        print(f"{'TensorRT FP32':<25} {'TensorRT':<15} {tensorrt_fp32['mean_time']:<10.4f} {tensorrt_fp32['samples_per_second']:<15.1f} {tensorrt_fp32['accuracy']:<10.4f} {trt_fp32_size:<10.1f}")
        print(f"{'TensorRT FP16':<25} {'TensorRT':<15} {tensorrt_fp16['mean_time']:<10.4f} {tensorrt_fp16['samples_per_second']:<15.1f} {tensorrt_fp16['accuracy']:<10.4f} {trt_fp16_size:<10.1f}")
        
        # TensorFlow Lite結果
        tflite_fp32_size = os.path.getsize('model_fp32.tflite') / (1024*1024)
        tflite_int8_size = os.path.getsize('model_int8.tflite') / (1024*1024)
        
        print(f"{'TensorFlow Lite FP32':<25} {'TF Lite':<15} {tflite_fp32_result['mean_time']:<10.4f} {tflite_fp32_result['samples_per_second']:<15.1f} {tflite_fp32_result['accuracy']:<10.4f} {tflite_fp32_size:<10.1f}")
        print(f"{'TensorFlow Lite INT8':<25} {'TF Lite':<15} {tflite_int8_result['mean_time']:<10.4f} {tflite_int8_result['samples_per_second']:<15.1f} {tflite_int8_result['accuracy']:<10.4f} {tflite_int8_size:<10.1f}")
        
        print("\n🎯 Key Insights:")
        
        # 最高速度
        all_results = [
            ('TensorRT FP32', tensorrt_fp32['samples_per_second']),
            ('TensorRT FP16', tensorrt_fp16['samples_per_second']),
            ('TF Lite FP32', tflite_fp32_result['samples_per_second']),
            ('TF Lite INT8', tflite_int8_result['samples_per_second'])
        ]
        fastest = max(all_results, key=lambda x: x[1])
        print(f"🚀 Fastest: {fastest[0]} ({fastest[1]:.1f} samples/sec)")
        
        # 最小サイズ
        all_sizes = [
            ('TensorRT FP32', trt_fp32_size),
            ('TensorRT FP16', trt_fp16_size),
            ('TF Lite FP32', tflite_fp32_size),
            ('TF Lite INT8', tflite_int8_size)
        ]
        smallest = min(all_sizes, key=lambda x: x[1])
        print(f"💾 Smallest: {smallest[0]} ({smallest[1]:.1f} MB)")
        
        # 精度比較
        accuracies = [
            tensorrt_fp32['accuracy'],
            tensorrt_fp16['accuracy'], 
            tflite_fp32_result['accuracy'],
            tflite_int8_result['accuracy']
        ]
        accuracy_range = max(accuracies) - min(accuracies)
        print(f"🎯 Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f} (diff: {accuracy_range:.4f})")
        
        # 効率比較
        trt_speedup = tensorrt_fp16['samples_per_second'] / tensorrt_fp32['samples_per_second']
        int8_speedup = tflite_int8_result['samples_per_second'] / tflite_fp32_result['samples_per_second']
        print(f"⚡ TensorRT FP16 speedup: {trt_speedup:.1f}x over FP32")
        print(f"⚡ TF Lite INT8 speedup: {int8_speedup:.1f}x over FP32")
        
        # 完全な結果保存
        complete_results = {
            'tensorrt_fp32': tensorrt_fp32,
            'tensorrt_fp16': tensorrt_fp16,
            'tflite_fp32': tflite_fp32_result,
            'tflite_int8': tflite_int8_result,
            'summary': {
                'fastest_model': fastest[0],
                'fastest_throughput': fastest[1],
                'smallest_model': smallest[0],
                'smallest_size_mb': smallest[1],
                'accuracy_range': accuracy_range,
                'tensorrt_fp16_speedup': trt_speedup,
                'tflite_int8_speedup': int8_speedup
            }
        }
        
        with open('complete_precision_comparison.json', 'w') as f:
            json.dump(complete_results, f, indent=2)
        print(f"\n💾 Complete results saved to complete_precision_comparison.json")

if __name__ == "__main__":
    main()