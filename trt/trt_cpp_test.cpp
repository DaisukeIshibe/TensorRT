#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

// TensorRTのログ出力用クラス
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// ファイルからバイナリデータを読み込む関数
void read_file(const std::string& path, std::vector<char>& buffer) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Error reading file: " + path);
    }
}

// CIFAR-10バッチファイルを読み込み、前処理する関数
void load_and_preprocess_cifar10_batch(const std::string& batch_path, 
                                       std::vector<std::vector<float>>& batch_data, 
                                       std::vector<int>& labels, 
                                       int max_images = -1) {
    std::ifstream file(batch_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open batch file: " + batch_path);
    }

    const int image_size = 3 * 32 * 32; // CIFAR-10 image size: 3072
    const int record_size = 1 + image_size; // 1バイトラベル + 3072バイト画像
    
    // ファイルサイズから画像数を計算
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    int total_images = file_size / record_size;
    int num_images = (max_images > 0) ? std::min(max_images, total_images) : total_images;
    
    std::cout << "Loading " << num_images << " images from batch file..." << std::endl;
    std::cout << "Expected image size: " << image_size << " pixels" << std::endl;
    
    batch_data.resize(num_images);
    labels.resize(num_images);
    
    for (int i = 0; i < num_images; ++i) {
        // ラベルを読み込み
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
        
        // 画像データを読み込み
        std::vector<uint8_t> image_data(image_size);
        file.read(reinterpret_cast<char*>(image_data.data()), image_size);
        
        // 正規化してfloatに変換
        batch_data[i].resize(image_size);
        for (int j = 0; j < image_size; ++j) {
            batch_data[i][j] = static_cast<float>(image_data[j]) / 255.0f;
        }
        
        // 最初の画像のサイズをデバッグ出力
        if (i == 0) {
            std::cout << "First image loaded with size: " << batch_data[i].size() << std::endl;
        }
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "Loaded " << (i + 1) << " images..." << std::endl;
        }
    }
}

class TrtEngine {
public:
    TrtEngine(const std::string& trt_engine_path, int max_batch_size = 1) : max_batch_size_(max_batch_size) {
        Logger logger;
        
        // 1. Runtimeの作成
        std::vector<char> engine_data;
        read_file(trt_engine_path, engine_data);
        
        runtime_ = nvinfer1::createInferRuntime(logger);
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        
        // 2. エンジンのデシリアライズ
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
        
        // 3. Execution Contextの作成
        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("Failed to create execution context");
        }
        
        // 4. エンジンの詳細情報を出力
        std::cout << "\n=== TensorRT Engine Information ===" << std::endl;
        std::cout << "Number of IO tensors: " << engine_->getNbIOTensors() << std::endl;
        
        // 5. 入出力テンサーを特定
        input_name_ = "";
        output_name_ = "";
        
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            auto tensor_mode = engine_->getTensorIOMode(tensor_name);
            auto tensor_shape = engine_->getTensorShape(tensor_name);
            
            std::cout << "Tensor " << i << ": " << tensor_name;
            std::cout << " [" << (tensor_mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "]";
            std::cout << " Shape: (";
            for (int j = 0; j < tensor_shape.nbDims; ++j) {
                std::cout << tensor_shape.d[j];
                if (j < tensor_shape.nbDims - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            
            if (tensor_mode == nvinfer1::TensorIOMode::kINPUT && input_name_.empty()) {
                input_name_ = tensor_name;
                // 入力サイズの計算: バッチサイズを除いた要素数
                input_size_per_image_ = 1;
                for (int j = 1; j < tensor_shape.nbDims; ++j) {
                    input_size_per_image_ *= tensor_shape.d[j];
                }
            } else if (tensor_mode == nvinfer1::TensorIOMode::kOUTPUT && output_name_.empty()) {
                output_name_ = tensor_name;
                // 出力サイズの計算: バッチサイズを除いた要素数
                output_size_per_image_ = 1;
                for (int j = 1; j < tensor_shape.nbDims; ++j) {
                    output_size_per_image_ *= tensor_shape.d[j];
                }
            }
        }
        
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("Could not find input or output tensor");
        }
        
        std::cout << "Input tensor: " << input_name_ << ", size per image: " << input_size_per_image_ << std::endl;
        std::cout << "Output tensor: " << output_name_ << ", size per image: " << output_size_per_image_ << std::endl;
        std::cout << "Max batch size: " << max_batch_size_ << std::endl;
        
        // 6. CUDAデバイスメモリの割り当て
        cudaMalloc(&device_input_, input_size_per_image_ * max_batch_size_ * sizeof(float));
        cudaMalloc(&device_output_, output_size_per_image_ * max_batch_size_ * sizeof(float));
        
        cudaStreamCreate(&stream_);
    }
    
    ~TrtEngine() {
        cudaFree(device_input_);
        cudaFree(device_output_);
        cudaStreamDestroy(stream_);
        if (context_) {
            delete context_;
            context_ = nullptr;
        }
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
        if (runtime_) {
            delete runtime_;
            runtime_ = nullptr;
        }
    }
    
    // バッチ推論
    void infer_batch(const std::vector<std::vector<float>>& batch_input, 
                     std::vector<std::vector<float>>& batch_output) {
        int batch_size = batch_input.size();
        if (batch_size > max_batch_size_) {
            throw std::runtime_error("Batch size exceeds maximum batch size.");
        }
        
        //std::cout << "\n=== Batch Inference Debug ===" << std::endl;
        //std::cout << "Batch size: " << batch_size << std::endl;
        //std::cout << "Expected input size per image: " << input_size_per_image_ << std::endl;
        
        // バッチ出力サイズを設定
        batch_output.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            //std::cout << "Image " << i << " input size: " << batch_input[i].size() << std::endl;
            if (batch_input[i].size() != input_size_per_image_) {
                std::cout << "ERROR: Input size mismatch!" << std::endl;
                std::cout << "Expected: " << input_size_per_image_ << std::endl;
                std::cout << "Actual: " << batch_input[i].size() << std::endl;
                throw std::runtime_error("Input size mismatch for image " + std::to_string(i));
            }
            batch_output[i].resize(output_size_per_image_);
        }
        
        // 入力データを連続メモリにコピー
        std::vector<float> flattened_input(batch_size * input_size_per_image_);
        for (int i = 0; i < batch_size; ++i) {
            std::copy(batch_input[i].begin(), batch_input[i].end(), 
                     flattened_input.begin() + i * input_size_per_image_);
        }
        
        // 1. Host to Device
        cudaMemcpyAsync(device_input_, flattened_input.data(), 
                       batch_size * input_size_per_image_ * sizeof(float), 
                       cudaMemcpyHostToDevice, stream_);
        
        // 2. 動的形状設定（バッチサイズが変わる場合）
        auto input_shape = engine_->getTensorShape(input_name_.c_str());
        if (input_shape.d[0] != batch_size) {
            input_shape.d[0] = batch_size;
            if (!context_->setInputShape(input_name_.c_str(), input_shape)) {
                throw std::runtime_error("Failed to set input shape for batch processing");
            }
            std::cout << "Set dynamic input shape with batch size: " << batch_size << std::endl;
        }
        
        // 3. テンサーアドレスの設定
        context_->setTensorAddress(input_name_.c_str(), device_input_);
        context_->setTensorAddress(output_name_.c_str(), device_output_);
        
        // 4. 推論の実行
        context_->enqueueV3(stream_);
        
        // 5. Device to Host
        std::vector<float> flattened_output(batch_size * output_size_per_image_);
        cudaMemcpyAsync(flattened_output.data(), device_output_, 
                       batch_size * output_size_per_image_ * sizeof(float), 
                       cudaMemcpyDeviceToHost, stream_);
        
        cudaStreamSynchronize(stream_);
        
        // 結果を個別の画像出力に分割
        for (int i = 0; i < batch_size; ++i) {
            std::copy(flattened_output.begin() + i * output_size_per_image_,
                     flattened_output.begin() + (i + 1) * output_size_per_image_,
                     batch_output[i].begin());
        }
    }

private:
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    std::string input_name_;
    std::string output_name_;
    int input_size_per_image_{0};
    int output_size_per_image_{0};
    int max_batch_size_{1};
    void* device_input_{nullptr};
    void* device_output_{nullptr};
    cudaStream_t stream_;
};

int main() {
    try {
        // ファイルパス
        const std::string trt_engine_path = "model.trt"; 
        const std::string batch_path = "cifar-10-batches-bin/test_batch.bin"; 
        
        // バッチサイズとテスト画像数の設定
        const int max_batch_size = 1; // バッチサイズ1でテスト
        const int max_test_images = -1; // すべての画像を読み込み（制限なし）
        
        // CIFAR-10クラス名
        const std::vector<std::string> class_names = {
            "airplane", "automobile", "bird", "cat", "deer", 
            "dog", "frog", "horse", "ship", "truck"
        };

        // バッチデータの読み込み
        std::vector<std::vector<float>> batch_data;
        std::vector<int> labels;
        load_and_preprocess_cifar10_batch(batch_path, batch_data, labels, max_test_images);
        
        std::cout << "Loaded " << batch_data.size() << " images for testing." << std::endl;

        // TensorRTエンジンを作成
        TrtEngine trt_engine(trt_engine_path, max_batch_size);

        // バッチ推論の実行
        int correct_predictions = 0;
        int total_images = batch_data.size();
        
        for (int batch_start = 0; batch_start < total_images; batch_start += max_batch_size) {
            int batch_end = std::min(batch_start + max_batch_size, total_images);
            int current_batch_size = batch_end - batch_start;
            
            // 現在のバッチデータを準備
            std::vector<std::vector<float>> current_batch(batch_data.begin() + batch_start, 
                                                          batch_data.begin() + batch_end);
            std::vector<std::vector<float>> batch_output;
            
            //std::cout << "\nProcessing batch starting at image " << batch_start << std::endl;
            
            // バッチ推論実行
            trt_engine.infer_batch(current_batch, batch_output);
            
            // 結果の処理
            for (int i = 0; i < current_batch_size; ++i) {
                int predicted_class = std::distance(batch_output[i].begin(), 
                                                   std::max_element(batch_output[i].begin(), 
                                                                   batch_output[i].end()));
                int true_label = labels[batch_start + i];
                
                if (predicted_class == true_label) {
                    correct_predictions++;
                }
                
                //std::cout << "Image " << (batch_start + i + 1) 
                //          << ": True=" << class_names[true_label] 
                //          << ", Predicted=" << class_names[predicted_class]
                //          << (predicted_class == true_label ? " ✓" : " ✗") << std::endl;
            }
        }
        
        // 最終結果の出力
        double accuracy = static_cast<double>(correct_predictions) / total_images * 100.0;
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Total images: " << total_images << std::endl;
        std::cout << "Correct predictions: " << correct_predictions << std::endl;
        std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}