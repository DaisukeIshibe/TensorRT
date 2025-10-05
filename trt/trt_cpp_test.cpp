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

// CIFAR-10画像データを読み込み、前処理する関数 (前回の回答とほぼ同じ)
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


class TrtEngine {
public:
    TrtEngine(const std::string& trt_engine_path) {
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
        
        input_size_ = get_size(engine_->getTensorShape(input_name_.c_str()));
        output_size_ = get_size(engine_->getTensorShape(output_name_.c_str()));
        
        // CUDAデバイスメモリの割り当て
        cudaMalloc(&device_input_, input_size_ * sizeof(float));
        cudaMalloc(&device_output_, output_size_ * sizeof(float));
        
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
    
    // 推論実行
    void infer(const std::vector<float>& input, std::vector<float>& output) {
        if (input.size() != input_size_ || output.size() != output_size_) {
             throw std::runtime_error("Input/Output buffer size mismatch.");
        }
        
        // 1. Host to Device (入力データをGPUへ転送)
        cudaMemcpyAsync(device_input_, input.data(), input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
        
        // 2. テンサーアドレスの設定（TensorRT 10.x対応）
        context_->setTensorAddress(input_name_.c_str(), device_input_);
        context_->setTensorAddress(output_name_.c_str(), device_output_);
        
        // 3. 推論の実行（TensorRT 10.x対応）
        context_->enqueueV3(stream_);
        
        // 4. Device to Host (結果をCPUへ転送)
        cudaMemcpyAsync(output.data(), device_output_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        
        cudaStreamSynchronize(stream_);
    }

private:
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    std::string input_name_;
    std::string output_name_;
    int input_size_{0};
    int output_size_{0};
    void* device_input_{nullptr};
    void* device_output_{nullptr};
    cudaStream_t stream_;
};

int main() {
    try {
        // ★ 変換済みのTensorRTエンジンファイルを指定
        const std::string trt_engine_path = "model.trt"; 
        // CIFAR-10のバイナリ画像ファイル (ラベルなしの3072バイト)
        const std::string image_path = "cifar-10-batches-bin/test_batch.bin"; 
        const int image_size = 3 * 32 * 32; // CIFAR-10 image size

        // 入力データの準備と前処理
        std::vector<float> input_data;
        load_and_preprocess_cifar10(image_path, input_data, image_size);

        // TensorRTエンジンを作成し、推論を実行
        TrtEngine trt_engine(trt_engine_path);

        const int output_size = 10; // CIFAR-10は10クラス
        std::vector<float> output_data(output_size);

        // 推論実行
        auto start = std::chrono::high_resolution_clock::now();
        trt_engine.infer(input_data, output_data);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Inference time: " << diff.count() * 1000 << " ms" << std::endl;

        // 結果の出力
        int predicted_class = std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end()));
        std::cout << "Predicted class: " << predicted_class << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}