import tensorrt as trt
import numpy as np

def analyze_trt_engine(engine_path):
    # TensorRTロガーを作成
    logger = trt.Logger(trt.Logger.WARNING)
    
    # TensorRTランタイムを作成
    runtime = trt.Runtime(logger)
    
    # エンジンファイルを読み込み
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    # エンジンをデシリアライズ
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        print("Failed to deserialize TensorRT engine")
        return
    
    print("=== TensorRT Engine Information ===")
    
    # TensorRT 8.x以降の新しいAPI対応
    try:
        # 新しいAPI (TensorRT 8.x+)
        num_io_tensors = engine.num_io_tensors
        print(f"Number of IO tensors: {num_io_tensors}")
        
        for i in range(num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = engine.get_tensor_dtype(tensor_name)
            tensor_mode = engine.get_tensor_mode(tensor_name)
            
            print(f"\nTensor {i}:")
            print(f"  Name: {tensor_name}")
            print(f"  Shape: {tensor_shape}")
            print(f"  Data type: {tensor_dtype}")
            print(f"  Mode: {tensor_mode}")
            
            # 要素数を計算
            if tensor_shape:
                volume = 1
                for dim in tensor_shape:
                    if dim > 0:
                        volume *= dim
                print(f"  Volume (elements): {volume}")
                
                # バッチサイズを除いた要素数を計算
                if len(tensor_shape) > 1 and tensor_shape[0] == -1:
                    # バッチサイズが動的な場合
                    volume_per_image = 1
                    for dim in tensor_shape[1:]:
                        if dim > 0:
                            volume_per_image *= dim
                    print(f"  Volume per image: {volume_per_image}")
                elif len(tensor_shape) > 1:
                    # バッチサイズが固定の場合
                    volume_per_image = 1
                    for dim in tensor_shape[1:]:
                        if dim > 0:
                            volume_per_image *= dim
                    print(f"  Volume per image: {volume_per_image}")
                    
    except AttributeError:
        # 古いAPI (TensorRT 7.x)
        try:
            print(f"Number of bindings: {engine.num_bindings}")
            print(f"Max batch size: {engine.max_batch_size}")
            
            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                binding_shape = engine.get_binding_shape(i)
                binding_dtype = engine.get_binding_dtype(i)
                is_input = engine.binding_is_input(i)
                
                print(f"\nBinding {i}:")
                print(f"  Name: {binding_name}")
                print(f"  Shape: {binding_shape}")
                print(f"  Data type: {binding_dtype}")
                print(f"  Is input: {is_input}")
                
                # 要素数を計算
                if binding_shape:
                    volume = 1
                    for dim in binding_shape:
                        if dim > 0:
                            volume *= dim
                    print(f"  Volume (elements): {volume}")
                    
                    # バッチサイズを除いた要素数を計算
                    if len(binding_shape) > 1 and binding_shape[0] == -1:
                        volume_per_image = 1
                        for dim in binding_shape[1:]:
                            if dim > 0:
                                volume_per_image *= dim
                        print(f"  Volume per image: {volume_per_image}")
                    elif len(binding_shape) > 1:
                        volume_per_image = 1
                        for dim in binding_shape[1:]:
                            if dim > 0:
                                volume_per_image *= dim
                        print(f"  Volume per image: {volume_per_image}")
        except Exception as e:
            print(f"Could not analyze engine with old API: {e}")
            
            # エンジンオブジェクトの利用可能なメソッドを表示
            print("Available methods on engine object:")
            methods = [method for method in dir(engine) if not method.startswith('_')]
            for method in sorted(methods):
                print(f"  {method}")

def analyze_cifar10_data():
    print("\n=== CIFAR-10 Data Analysis ===")
    
    # バイナリファイルからサンプルデータを読み込み
    batch_path = "cifar-10-batches-bin/test_batch.bin"
    
    try:
        with open(batch_path, 'rb') as f:
            # 最初のレコードを読み込み
            label = int.from_bytes(f.read(1), byteorder='big')
            image_data = f.read(3072)  # 32*32*3 = 3072
            
            print(f"First image label: {label}")
            print(f"Image data size: {len(image_data)} bytes")
            print(f"Expected CIFAR-10 image size: 32*32*3 = {32*32*3}")
            
            # 正規化後のデータサイズ
            normalized_data = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32) / 255.0
            print(f"Normalized data size: {len(normalized_data)} elements")
            print(f"Data shape would be: (32, 32, 3) or flattened: ({len(normalized_data)},)")
            
    except FileNotFoundError:
        print(f"Could not find CIFAR-10 batch file: {batch_path}")
    except Exception as e:
        print(f"Error reading CIFAR-10 data: {e}")

if __name__ == "__main__":
    engine_path = "model.trt"
    
    try:
        analyze_trt_engine(engine_path)
    except Exception as e:
        print(f"Error analyzing TensorRT engine: {e}")
    
    analyze_cifar10_data()