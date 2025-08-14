#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>

using namespace nvinfer1;

// Logger
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

// エンジンファイル読み込み
std::vector<char> readEngineFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file");
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}


int main() {
    Logger logger;
    auto engineData = readEngineFile("../tf/cifar10_vgg_model_trt/model.engine");

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        return 1;
    }

    auto engine = std::unique_ptr<ICudaEngine>(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size())
    );
    if (!engine) {
        std::cerr << "Failed to create engine" << std::endl;
        return 1;
    }

    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }

    // 入出力バッファの準備
    int inputIndex = 0;  // 入力バッファのインデックス
    int outputIndex = 1; // 出力バッファのインデックス
    size_t inputSize = 1 * 32 * 32 * 3 * sizeof(float); // CIFAR-10想定
    size_t outputSize = 1 * 10 * sizeof(float);

    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);

    std::vector<float> input(32 * 32 * 3, 0.0f);
    cudaMemcpy(buffers[inputIndex], input.data(), inputSize, cudaMemcpyHostToDevice);

    context->enqueueV3(1, buffers, 0, (cudaEvent_t*)nullptr);

    std::vector<float> output(10);
    cudaMemcpy(output.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);

    std::cout << "Output: ";
    for (float v : output) std::cout << v << " ";
    std::cout << std::endl;

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    return 0;
}