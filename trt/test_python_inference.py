import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TrtEngine:
    def __init__(self, engine_path, max_batch_size=1):
        self.max_batch_size = max_batch_size
        
        # TensorRTロガーとランタイムを作成
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # エンジンを読み込み
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 入出力テンサー情報を取得
        self.analyze_tensors()
        
        # CUDAメモリを割り当て
        self.allocate_memory()
        
        # CUDAストリームを作成
        self.stream = cuda.Stream()
    
    def analyze_tensors(self):
        print("=== TensorRT Engine Analysis ===")
        
        # テンサー情報を取得
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            
            print(f"Tensor {i}: {tensor_name}")
            print(f"  Shape: {tensor_shape}")
            print(f"  Mode: {tensor_mode}")
            print(f"  Data type: {tensor_dtype}")
            
            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_name = tensor_name
                self.input_shape = tensor_shape
                # バッチサイズを除いた要素数を計算
                self.input_size_per_image = np.prod(tensor_shape[1:])
                print(f"  Input size per image: {self.input_size_per_image}")
                
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                self.output_name = tensor_name
                self.output_shape = tensor_shape
                # バッチサイズを除いた要素数を計算
                self.output_size_per_image = np.prod(tensor_shape[1:])
                print(f"  Output size per image: {self.output_size_per_image}")
    
    def allocate_memory(self):
        # GPU メモリ割り当て
        input_size_bytes = int(self.input_size_per_image * self.max_batch_size * np.dtype(np.float32).itemsize)
        output_size_bytes = int(self.output_size_per_image * self.max_batch_size * np.dtype(np.float32).itemsize)
        
        self.device_input = cuda.mem_alloc(input_size_bytes)
        self.device_output = cuda.mem_alloc(output_size_bytes)
        
        print(f"Allocated {input_size_bytes} bytes for input")
        print(f"Allocated {output_size_bytes} bytes for output")
    
    def infer_batch(self, batch_input):
        batch_size = len(batch_input)
        
        print(f"\n=== Batch Inference Debug ===")
        print(f"Batch size: {batch_size}")
        print(f"Expected input size per image: {self.input_size_per_image}")
        
        # 入力サイズチェック
        for i, img_input in enumerate(batch_input):
            print(f"Image {i} input size: {len(img_input)}")
            if len(img_input) != self.input_size_per_image:
                raise RuntimeError(f"Input size mismatch for image {i}: expected {self.input_size_per_image}, got {len(img_input)}")
        
        # 入力データを平坦化
        flattened_input = np.concatenate(batch_input).astype(np.float32)
        
        # 動的形状を設定（バッチサイズが変わる場合）
        if self.input_shape[0] == 1 or self.input_shape[0] == -1:
            dynamic_shape = (batch_size,) + self.input_shape[1:]
            self.context.set_input_shape(self.input_name, dynamic_shape)
            print(f"Set dynamic input shape: {dynamic_shape}")
        
        # GPU にデータをコピー
        cuda.memcpy_htod_async(self.device_input, flattened_input, self.stream)
        
        # テンサーアドレスを設定
        self.context.set_tensor_address(self.input_name, int(self.device_input))
        self.context.set_tensor_address(self.output_name, int(self.device_output))
        
        # 推論実行
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 結果をCPUにコピー
        output_size = self.output_size_per_image * batch_size
        output_data = np.empty(output_size, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.device_output, self.stream)
        self.stream.synchronize()
        
        # バッチごとに分割して返す
        batch_output = []
        for i in range(batch_size):
            start_idx = i * self.output_size_per_image
            end_idx = (i + 1) * self.output_size_per_image
            batch_output.append(output_data[start_idx:end_idx])
        
        return batch_output

def load_cifar10_batch(batch_path, max_images=10):
    """CIFAR-10バッチファイルを読み込み"""
    batch_data = []
    labels = []
    
    with open(batch_path, 'rb') as f:
        for i in range(max_images):
            # ラベル読み込み
            label_byte = f.read(1)
            if not label_byte:
                break
            label = int.from_bytes(label_byte, byteorder='big')
            labels.append(label)
            
            # 画像データ読み込み
            image_data = f.read(3072)  # 32*32*3
            if len(image_data) != 3072:
                break
                
            # 正規化
            normalized_data = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32) / 255.0
            batch_data.append(normalized_data)
            
            if i == 0:
                print(f"First image loaded with size: {len(normalized_data)}")
    
    return batch_data, labels

def main():
    # ファイルパス
    engine_path = "model.trt"
    batch_path = "cifar-10-batches-bin/test_batch.bin"
    
    # CIFAR-10クラス名
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    try:
        # データ読み込み
        print("Loading CIFAR-10 data...")
        batch_data, labels = load_cifar10_batch(batch_path, max_images=5)
        print(f"Loaded {len(batch_data)} images")
        
        # TensorRTエンジン初期化
        print("\nInitializing TensorRT engine...")
        trt_engine = TrtEngine(engine_path, max_batch_size=1)
        
        # バッチ推論実行
        print("\nRunning inference...")
        correct_predictions = 0
        
        for i in range(len(batch_data)):
            # 1つずつ推論（バッチサイズ1）
            single_batch = [batch_data[i]]
            batch_output = trt_engine.infer_batch(single_batch)
            
            # 予測結果
            predicted_class = np.argmax(batch_output[0])
            true_label = labels[i]
            
            if predicted_class == true_label:
                correct_predictions += 1
            
            print(f"Image {i+1}: True={class_names[true_label]}, "
                  f"Predicted={class_names[predicted_class]} "
                  f"{'✓' if predicted_class == true_label else '✗'}")
        
        # 最終結果
        accuracy = correct_predictions / len(batch_data) * 100
        print(f"\nAccuracy: {accuracy:.1f}% ({correct_predictions}/{len(batch_data)})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()