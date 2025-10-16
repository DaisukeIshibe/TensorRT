#!/usr/bin/env python3
"""
TensorRT 8.x Engine Generator with Precision Options
TensorRT 8.x向けの精度別エンジン生成スクリプト
"""
import tensorrt as trt
import numpy as np
import os
import argparse
import random

def create_calibration_cache():
    """INT8キャリブレーション用の簡易キャッシュファイル作成"""
    cache_file = "calibration_cache_8x.cache"
    
    # 事前に生成されたキャリブレーションキャッシュがある場合はそれを使用
    if os.path.exists(cache_file):
        print(f"✅ Using existing calibration cache: {cache_file}")
        return cache_file
    
    # 簡易的なキャリブレーションキャッシュを作成
    print("🔧 Creating simple calibration cache for INT8...")
    
    # 最小限のキャッシュデータを作成（実用的ではないが、エンジン生成テスト用）
    cache_data = b"""TRT-8503-EntropyCalibrator2
input_1: 3c010a3c
"""
    
    with open(cache_file, 'wb') as f:
        f.write(cache_data)
    
    print(f"✅ Simple calibration cache created: {cache_file}")
    return cache_file

class SimpleTensorRTCalibrator(trt.IInt8EntropyCalibrator2):
    """簡易INT8 Calibrator for TensorRT 8.x"""
    
    def __init__(self, cache_file="calibration_cache_8x.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        
    def get_batch_size(self):
        return 1
        
    def get_batch(self, names):
        # キャッシュファイルを使用するため、実際のデータは不要
        return None
        
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
        
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def prepare_calibration_data(num_samples=1000):
    """キャリブレーション用データ準備"""
    print(f"📊 Preparing calibration data ({num_samples} samples)...")
    
    # CSVファイルを使用してキャリブレーションデータを読み込み
    csv_file = "test_samples.csv"
    if os.path.exists(csv_file):
        print(f"📄 Loading calibration data from {csv_file}...")
        calibration_data = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)
        
        # Reshape to CIFAR-10 format (32x32x3)
        calibration_data = calibration_data.reshape(-1, 32, 32, 3)
        
        # Limit number of samples
        if len(calibration_data) > num_samples:
            indices = random.sample(range(len(calibration_data)), num_samples)
            calibration_data = calibration_data[indices]
            
        print(f"✅ Calibration data prepared from CSV: {calibration_data.shape}")
        return calibration_data
    else:
        print("⚠️ test_samples.csv not found. Creating synthetic calibration data...")
        # Synthetic calibration data as fallback
        calibration_data = np.random.rand(num_samples, 32, 32, 3).astype(np.float32)
        print(f"✅ Synthetic calibration data prepared: {calibration_data.shape}")
        return calibration_data

def build_engine_for_trt8x(precision="fp32"):
    """TensorRT 8.x向けエンジンの生成 (精度指定対応)"""
    print(f"🔧 Building TensorRT 8.x engine with {precision.upper()} precision...")
    
    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Check TensorRT version
    print(f"📋 TensorRT Version: {trt.__version__}")
    
    # ONNXモデル確認
    onnx_path = 'model.onnx'
    if not os.path.exists(onnx_path):
        print("❌ ONNX model not found!")
        return False
    
    # Builder and network creation (TensorRT 8.x style)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Create network with explicit batch (recommended for TensorRT 8.x)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(f"   {parser.get_error(error)}")
            return False
    
    print("✅ ONNX model parsed successfully")
    
    # Configuration for TensorRT 8.x
    config.max_workspace_size = 1 << 30  # 1GB workspace
    
    # 精度設定
    if precision == "fp16":
        print("🔧 Enabling FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        print("🔧 Enabling INT8 precision...")
        config.set_flag(trt.BuilderFlag.INT8)
        
        # INT8キャリブレーション（簡易版）
        cache_file = create_calibration_cache()
        calibrator = SimpleTensorRTCalibrator(cache_file)
        config.int8_calibrator = calibrator
        print("✅ INT8 calibration configured (using cache)")
        print("⚠️  Note: Using simplified calibration. For production use, proper calibration data is recommended.")
    else:
        print("🔧 Using default FP32 precision...")
    
    # For TensorRT 8.x, we can set optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    
    # Set dynamic batch size (TensorRT 8.x style)
    input_name = "input_1"  # CIFAR-10 input
    profile.set_shape(input_name, 
                     [1, 32, 32, 3],    # min
                     [16, 32, 32, 3],   # opt
                     [32, 32, 32, 3])   # max
    
    config.add_optimization_profile(profile)
    
    print("✅ Optimization profile configured")
    print("   Min batch: 1, Optimal batch: 16, Max batch: 32")
    
    # Build engine (TensorRT 8.x style)
    print("🔨 Building engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("❌ Failed to build engine")
        return False
    
    # Serialize and save engine
    engine_path = f"model_trt8x_{precision}.trt"
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    print(f"✅ TensorRT 8.x {precision.upper()} engine saved as: {engine_path}")
    
    # Engine info
    engine_size = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"📊 Engine size: {engine_size:.1f} MB")
    print(f"📊 Number of bindings: {engine.num_bindings}")
    print(f"📊 Precision: {precision.upper()}")
    
    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        binding_type = "Input" if is_input else "Output"
        print(f"   {binding_type}: {binding_name}, Shape: {binding_shape}")
    
    return engine_path

def main():
    parser = argparse.ArgumentParser(description="TensorRT 8.x Engine Generator with Precision Options")
    parser.add_argument("--precision", "-p", 
                       choices=["fp32", "fp16", "int8"], 
                       default="fp32",
                       help="Precision for TensorRT engine (default: fp32)")
    parser.add_argument("--all", "-a", 
                       action="store_true",
                       help="Generate engines for all precisions (fp32, fp16, int8)")
    
    args = parser.parse_args()
    
    print("🚀 TensorRT 8.x Engine Generation with Precision Options")
    print("=" * 50)
    
    if args.all:
        print("🔄 Generating engines for all precisions...")
        precisions = ["fp32", "fp16", "int8"]
        results = {}
        
        for precision in precisions:
            print(f"\n{'=' * 20} {precision.upper()} {'=' * 20}")
            try:
                engine_path = build_engine_for_trt8x(precision)
                if engine_path:
                    results[precision] = {"status": "✅ Success", "path": engine_path}
                else:
                    results[precision] = {"status": "❌ Failed", "path": None}
            except Exception as e:
                results[precision] = {"status": f"❌ Error: {str(e)}", "path": None}
        
        # Summary
        print(f"\n{'=' * 20} SUMMARY {'=' * 20}")
        for precision, result in results.items():
            print(f"{precision.upper()}: {result['status']}")
            if result['path']:
                size = os.path.getsize(result['path']) / (1024 * 1024)
                print(f"   📄 {result['path']} ({size:.1f} MB)")
    
    else:
        success = build_engine_for_trt8x(args.precision)
        
        if success:
            print(f"\n✅ TensorRT 8.x {args.precision.upper()} engine generation completed!")
        else:
            print(f"\n❌ TensorRT 8.x {args.precision.upper()} engine generation failed!")
            exit(1)

if __name__ == "__main__":
    main()