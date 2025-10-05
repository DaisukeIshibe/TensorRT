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

// TensorRTのログ出力用クラス (前回の回答と同じ)
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Warning以上のログを出力
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// ファイルからバイナリデータを読み込む関数 (前回の回答と同じ)
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

// CIFAR-10画像データを読み込み、前処理する関数 (単一画像用)
void load_and_preprocess_cifar10(const std::string& image_path, std::vector<float>& input_data, int image_size) {
    std::vector<uint8_t> image_data(image_size);
    std::ifstream file(image_path, std::ios::binary);
    if (!file || !file.read(reinterpret_cast<char*>(image_data.data()), image_data.size())) {
        throw std::runtime_error("Could not open or read image file: " + image_path);
    }

    // 0-255のuint8_tを0.0-1.0のfloatに正規化
    input_data.resize(image_size);
    for (size_t i = 0; i < image_data.size(); ++i) {
        input_data[i] = static_cast<float>(image_data[i]) / 255.0f;
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

    const int image_size = 3 * 32 * 32; // CIFAR-10 image size
    const int record_size = 1 + image_size; // 1バイトラベル + 3072バイト画像
    
    // ファイルサイズから画像数を計算
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    int total_images = file_size / record_size;
    int num_images = (max_images > 0) ? std::min(max_images, total_images) : total_images;
    
    std::cout << "Loading " << num_images << " images from batch file..." << std::endl;
    
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
        
        // 2. エンジンのデシリアライズ（ファイルから読み込み）
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
        
        // 3. Execution Contextの作成
        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("Failed to create execution context");
        }
        
        // 4. 入出力バッファサイズの取得とCUDAメモリ割り当て
        // TensorRT 10.x では新しいテンサーベースのAPIを使用
        
        // 入力テンサー名と出力テンサー名を取得
        std::string input_name = "input_1";
        std::string output_name = "dense_1";
        
        // テンサー名の存在確認
        bool input_found = false, output_found = false;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            if (std::string(tensor_name) == input_name) {
                input_found = true;
                input_name_ = tensor_name;
            }
            if (std::string(tensor_name) == output_name) {
                output_found = true;
                output_name_ = tensor_name;
            }
        }
        
        if (!input_found || !output_found) {
            throw std::runtime_error("Input or output tensor not found.");
        }

        // バッファサイズの計算 (バッチサイズ1を想定)
        // Dimsの要素をすべて乗算してサイズを取得
        auto get_size = [](const nvinfer1::Dims& dims) {
            return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
        };
        
        input_size_per_image_ = get_size(engine_->getTensorShape(input_name_.c_str())) / max_batch_size_;
        output_size_per_image_ = get_size(engine_->getTensorShape(output_name_.c_str())) / max_batch_size_;
        
        // CUDAデバイスメモリの割り当て（バッチサイズ分）
        cudaMalloc(&device_input_, input_size_per_image_ * max_batch_size_ * sizeof(float));
        cudaMalloc(&device_output_, output_size_per_image_ * max_batch_size_ * sizeof(float));
        
        cudaStreamCreate(&stream_);
    }
    
    ~TrtEngine() {
        cudaFree(device_input_);
        cudaFree(device_output_);
        cudaStreamDestroy(stream_);
        // TensorRT 8.0以降では delete を使用（nullptrチェック付き）
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
    
    // 単一画像推論
    void infer(const std::vector<float>& input, std::vector<float>& output) {
        if (input.size() != input_size_per_image_ || output.size() != output_size_per_image_) {
             throw std::runtime_error("Input/Output buffer size mismatch.");
        }
        
        // 1. Host to Device (入力データをGPUへ転送)
        cudaMemcpyAsync(device_input_, input.data(), input_size_per_image_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
        
        // 2. テンサーアドレスの設定（TensorRT 10.x対応）
        context_->setTensorAddress(input_name_.c_str(), device_input_);
        context_->setTensorAddress(output_name_.c_str(), device_output_);
        
        // 3. 推論の実行（TensorRT 10.x対応）
        context_->enqueueV3(stream_);
        
        // 4. Device to Host (結果をCPUへ転送)
        cudaMemcpyAsync(output.data(), device_output_, output_size_per_image_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        
        cudaStreamSynchronize(stream_);
    }
    
    // バッチ推論
    void infer_batch(const std::vector<std::vector<float>>& batch_input, 
                     std::vector<std::vector<float>>& batch_output) {
        int batch_size = batch_input.size();
        if (batch_size > max_batch_size_) {
            throw std::runtime_error("Batch size exceeds maximum batch size.");
        }
        
        // バッチ出力サイズを設定
        batch_output.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            if (batch_input[i].size() != input_size_per_image_) {
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
        
        // 2. バッチサイズを設定
        auto input_shape = engine_->getTensorShape(input_name_.c_str());
        auto output_shape = engine_->getTensorShape(output_name_.c_str());
        input_shape.d[0] = batch_size;
        output_shape.d[0] = batch_size;
        context_->setInputShape(input_name_.c_str(), input_shape);
        
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
        // ★ 変換済みのTensorRTエンジンファイルを指定
        const std::string trt_engine_path = "model.trt"; 
        // CIFAR-10のバッチファイル
        const std::string batch_path = "cifar-10-batches-bin/test_batch.bin"; 
        
        // バッチサイズとテスト画像数の設定
        const int max_batch_size = 32; // バッチサイズ
        const int max_test_images = 1000; // テスト画像数（-1で全て）
        
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
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        for (int batch_start = 0; batch_start < total_images; batch_start += max_batch_size) {
            int batch_end = std::min(batch_start + max_batch_size, total_images);
            int current_batch_size = batch_end - batch_start;
            
            // 現在のバッチデータを準備
            std::vector<std::vector<float>> current_batch(batch_data.begin() + batch_start, 
                                                          batch_data.begin() + batch_end);
            std::vector<std::vector<float>> batch_output;
            
            // バッチ推論実行
            auto start_batch = std::chrono::high_resolution_clock::now();
            trt_engine.infer_batch(current_batch, batch_output);
            auto end_batch = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> batch_time = end_batch - start_batch;
            
            // 結果の処理
            for (int i = 0; i < current_batch_size; ++i) {
                int predicted_class = std::distance(batch_output[i].begin(), 
                                                   std::max_element(batch_output[i].begin(), 
                                                                   batch_output[i].end()));
                int true_label = labels[batch_start + i];
                
                if (predicted_class == true_label) {
                    correct_predictions++;
                }
                
                // 最初の10個の結果を詳細出力
                if (batch_start + i < 10) {
                    std::cout << "Image " << (batch_start + i + 1) 
                              << ": True=" << class_names[true_label] 
                              << ", Predicted=" << class_names[predicted_class]
                              << (predicted_class == true_label ? " ✓" : " ✗") << std::endl;
                }
            }
            
            std::cout << "Batch " << (batch_start / max_batch_size + 1) 
                      << " (" << current_batch_size << " images) processed in " 
                      << batch_time.count() * 1000 << " ms" << std::endl;
        }
        
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = end_total - start_total;
        
        // 最終結果の出力
        double accuracy = static_cast<double>(correct_predictions) / total_images * 100.0;
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Total images: " << total_images << std::endl;
        std::cout << "Correct predictions: " << correct_predictions << std::endl;
        std::cout << "Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Total inference time: " << total_time.count() * 1000 << " ms" << std::endl;
        std::cout << "Average time per image: " << (total_time.count() * 1000) / total_images << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}