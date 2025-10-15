#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <cmath>

// TensorRT headers
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace std;

// Logger class for TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            cout << "[TensorRT 8.x] " << msg << endl;
        }
    }
} gLogger;

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << endl; \
            exit(1); \
        } \
    } while(0)

// TensorRT error checking
#define CHECK_TRT(call) \
    do { \
        if (!(call)) { \
            cerr << "TensorRT error at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

class TensorRTInference8x {
private:
    unique_ptr<ICudaEngine> engine;
    unique_ptr<IExecutionContext> context;
    void* deviceInputBuffer;
    void* deviceOutputBuffer;
    void* hostInputBuffer;
    void* hostOutputBuffer;
    
    size_t inputSize;
    size_t outputSize;
    int inputIndex;
    int outputIndex;
    int batchSize;
    
public:
    TensorRTInference8x() : deviceInputBuffer(nullptr), deviceOutputBuffer(nullptr),
                           hostInputBuffer(nullptr), hostOutputBuffer(nullptr) {}
    
    ~TensorRTInference8x() {
        cleanup();
    }
    
    bool loadEngine(const string& enginePath) {
        cout << "🔧 Loading TensorRT 8.x engine from: " << enginePath << endl;
        
        ifstream file(enginePath, ios::binary);
        if (!file.good()) {
            cerr << "❌ Error: Could not open engine file: " << enginePath << endl;
            return false;
        }
        
        file.seekg(0, ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, ifstream::beg);
        
        vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        // Create runtime and deserialize engine
        unique_ptr<IRuntime> runtime(createInferRuntime(gLogger));
        if (!runtime) {
            cerr << "❌ Error: Failed to create TensorRT runtime" << endl;
            return false;
        }
        
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size, nullptr));
        if (!engine) {
            cerr << "❌ Error: Failed to deserialize CUDA engine" << endl;
            return false;
        }
        
        context.reset(engine->createExecutionContext());
        if (!context) {
            cerr << "❌ Error: Failed to create execution context" << endl;
            return false;
        }
        
        cout << "✅ Engine loaded successfully" << endl;
        return true;
    }
    
    bool initializeBuffers(int batch_size = 32) {
        batchSize = batch_size;
        
        // Get binding indices (TensorRT 8.x style)
        inputIndex = engine->getBindingIndex("input_1");
        outputIndex = engine->getBindingIndex("dense_1");
        
        if (inputIndex == -1 || outputIndex == -1) {
            cerr << "❌ Error: Could not find input/output bindings" << endl;
            return false;
        }
        
        // Get dimensions (TensorRT 8.x style)
        Dims inputDims = engine->getBindingDimensions(inputIndex);
        Dims outputDims = engine->getBindingDimensions(outputIndex);
        
        cout << "📊 Input dimensions: ";
        for (int i = 0; i < inputDims.nbDims; ++i) {
            cout << inputDims.d[i] << " ";
        }
        cout << endl;
        
        cout << "📊 Output dimensions: ";
        for (int i = 0; i < outputDims.nbDims; ++i) {
            cout << outputDims.d[i] << " ";
        }
        cout << endl;
        
        // Calculate sizes (assuming batch dimension is dynamic or needs to be set)
        inputSize = batchSize * 32 * 32 * 3 * sizeof(float);  // CIFAR-10 input
        outputSize = batchSize * 10 * sizeof(float);           // 10 classes output
        
        // Allocate host memory
        CHECK_CUDA(cudaMallocHost(&hostInputBuffer, inputSize));
        CHECK_CUDA(cudaMallocHost(&hostOutputBuffer, outputSize));
        
        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&deviceInputBuffer, inputSize));
        CHECK_CUDA(cudaMalloc(&deviceOutputBuffer, outputSize));
        
        cout << "✅ Buffers initialized for batch size: " << batchSize << endl;
        cout << "   Input buffer size: " << inputSize / (1024*1024) << " MB" << endl;
        cout << "   Output buffer size: " << outputSize / 1024 << " KB" << endl;
        
        return true;
    }
    
    bool setDynamicBatchSize(int batch_size) {
        if (batch_size != batchSize) {
            cout << "🔄 Updating batch size from " << batchSize << " to " << batch_size << endl;
            
            // Clean up old buffers
            if (hostInputBuffer) cudaFreeHost(hostInputBuffer);
            if (hostOutputBuffer) cudaFreeHost(hostOutputBuffer);
            if (deviceInputBuffer) cudaFree(deviceInputBuffer);
            if (deviceOutputBuffer) cudaFree(deviceOutputBuffer);
            
            return initializeBuffers(batch_size);
        }
        return true;
    }
    
    vector<vector<float>> predict(const vector<vector<float>>& inputData) {
        int currentBatchSize = inputData.size();
        
        if (currentBatchSize > batchSize) {
            if (!setDynamicBatchSize(currentBatchSize)) {
                cerr << "❌ Error: Failed to set dynamic batch size" << endl;
                return {};
            }
        }
        
        // Copy input data to host buffer
        float* hostInput = static_cast<float*>(hostInputBuffer);
        for (int b = 0; b < currentBatchSize; ++b) {
            for (size_t i = 0; i < inputData[b].size(); ++i) {
                hostInput[b * inputData[b].size() + i] = inputData[b][i];
            }
        }
        
        // Copy from host to device
        CHECK_CUDA(cudaMemcpy(deviceInputBuffer, hostInputBuffer, 
                             currentBatchSize * 32 * 32 * 3 * sizeof(float), 
                             cudaMemcpyHostToDevice));
        
        // TensorRT 8.x style execution with binding arrays
        void* bindings[] = {deviceInputBuffer, deviceOutputBuffer};
        
        // For TensorRT 8.x, we may need to set dynamic shapes if the engine supports it
        if (engine->hasImplicitBatchDimension()) {
            // Legacy implicit batch mode
            CHECK_TRT(context->execute(currentBatchSize, bindings));
        } else {
            // Explicit batch mode - set input shape
            Dims inputDims = engine->getBindingDimensions(inputIndex);
            inputDims.d[0] = currentBatchSize;  // Set batch dimension
            CHECK_TRT(context->setBindingDimensions(inputIndex, inputDims));
            CHECK_TRT(context->executeV2(bindings));
        }
        
        // Copy output from device to host
        size_t currentOutputSize = currentBatchSize * 10 * sizeof(float);
        CHECK_CUDA(cudaMemcpy(hostOutputBuffer, deviceOutputBuffer, 
                             currentOutputSize, cudaMemcpyDeviceToHost));
        
        // Convert output to vector format
        vector<vector<float>> results(currentBatchSize, vector<float>(10));
        float* hostOutput = static_cast<float*>(hostOutputBuffer);
        
        for (int b = 0; b < currentBatchSize; ++b) {
            for (int i = 0; i < 10; ++i) {
                results[b][i] = hostOutput[b * 10 + i];
            }
        }
        
        return results;
    }
    
    void cleanup() {
        if (hostInputBuffer) {
            cudaFreeHost(hostInputBuffer);
            hostInputBuffer = nullptr;
        }
        if (hostOutputBuffer) {
            cudaFreeHost(hostOutputBuffer);
            hostOutputBuffer = nullptr;
        }
        if (deviceInputBuffer) {
            cudaFree(deviceInputBuffer);
            deviceInputBuffer = nullptr;
        }
        if (deviceOutputBuffer) {
            cudaFree(deviceOutputBuffer);
            deviceOutputBuffer = nullptr;
        }
    }
};

// CSV読み込み関数
vector<vector<float>> loadCSVData(const string& filename, int maxSamples = -1) {
    cout << "📁 Loading CSV data from: " << filename << endl;
    
    ifstream file(filename);
    if (!file.good()) {
        cerr << "❌ Error: Could not open CSV file: " << filename << endl;
        return {};
    }
    
    vector<vector<float>> data;
    string line;
    int count = 0;
    
    while (getline(file, line) && (maxSamples == -1 || count < maxSamples)) {
        if (line.empty()) continue;
        
        vector<float> row;
        stringstream ss(line);
        string cell;
        
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stof(cell));
            } catch (const exception& e) {
                cerr << "Warning: Could not parse value: " << cell << endl;
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
            count++;
        }
    }
    
    file.close();
    cout << "✅ Loaded " << data.size() << " samples from CSV" << endl;
    return data;
}

// バッチ処理のベンチマーク
void benchmarkBatchProcessing(TensorRTInference8x& inference, 
                             const vector<vector<float>>& testData,
                             int batchSize = 32, int numRuns = 5) {
    cout << "\n🔄 Benchmarking TensorRT 8.x Batch Processing (batch_size=" << batchSize << ")..." << endl;
    
    vector<double> inferenceTimes;
    int numBatches = (testData.size() + batchSize - 1) / batchSize;
    
    for (int run = 0; run < numRuns; ++run) {
        auto startTime = chrono::high_resolution_clock::now();
        
        for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            int startIdx = batchIdx * batchSize;
            int endIdx = min(startIdx + batchSize, static_cast<int>(testData.size()));
            
            vector<vector<float>> batch(testData.begin() + startIdx, testData.begin() + endIdx);
            auto results = inference.predict(batch);
            
            if (results.empty()) {
                cerr << "❌ Error: Prediction failed for batch " << batchIdx << endl;
                return;
            }
        }
        
        auto endTime = chrono::high_resolution_clock::now();
        double inferenceTime = chrono::duration<double>(endTime - startTime).count();
        inferenceTimes.push_back(inferenceTime);
        
        cout << "  Run " << (run + 1) << ": " << fixed << setprecision(4) 
             << inferenceTime << "s (" << numBatches << " batches)" << endl;
    }
    
    // 統計計算
    double meanTime = 0.0;
    for (double time : inferenceTimes) {
        meanTime += time;
    }
    meanTime /= inferenceTimes.size();
    
    double stdTime = 0.0;
    for (double time : inferenceTimes) {
        stdTime += (time - meanTime) * (time - meanTime);
    }
    stdTime = sqrt(stdTime / inferenceTimes.size());
    
    double throughput = testData.size() / meanTime;
    
    cout << "✅ TensorRT 8.x Batch Results:" << endl;
    cout << "   Mean inference time: " << fixed << setprecision(4) << meanTime << "s ± " << stdTime << "s" << endl;
    cout << "   Throughput: " << fixed << setprecision(1) << throughput << " samples/sec" << endl;
    cout << "   Batches processed: " << numBatches << endl;
    cout << "   GPU transfers: " << numBatches << " (vs " << testData.size() << " for single processing)" << endl;
}

int main() {
    cout << "🚀 TensorRT 8.x C++ Inference Verification" << endl;
    cout << "===========================================" << endl;
    
    // CUDA初期化
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    cout << "🖥️  Available CUDA devices: " << deviceCount << endl;
    
    if (deviceCount == 0) {
        cerr << "❌ Error: No CUDA devices found" << endl;
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    cout << "🖥️  Using device: " << deviceProp.name << endl;
    cout << "🖥️  Compute capability: " << deviceProp.major << "." << deviceProp.minor << endl;
    
    // TensorRT推論エンジンの初期化
    TensorRTInference8x inference;
    
    // エンジンファイルの読み込み（TensorRT 8.x用エンジン）
    string enginePath = "model_trt8x.trt";
    if (!inference.loadEngine(enginePath)) {
        cerr << "❌ Error: Failed to load TensorRT 8.x engine" << endl;
        return 1;
    }
    
    // バッファの初期化
    if (!inference.initializeBuffers(32)) {
        cerr << "❌ Error: Failed to initialize buffers" << endl;
        return 1;
    }
    
    // テストデータの読み込み
    vector<vector<float>> testData = loadCSVData("test_samples.csv", 100);
    if (testData.empty()) {
        cerr << "❌ Error: Failed to load test data" << endl;
        return 1;
    }
    
    cout << "📊 Test data shape: [" << testData.size() << ", " << testData[0].size() << "]" << endl;
    
    // 単一サンプルテスト
    cout << "\n🧪 Single sample test..." << endl;
    vector<vector<float>> singleSample = {testData[0]};
    auto result = inference.predict(singleSample);
    
    if (!result.empty()) {
        cout << "✅ Single prediction successful" << endl;
        cout << "   Output size: " << result[0].size() << endl;
        cout << "   Sample output: ";
        for (int i = 0; i < min(5, static_cast<int>(result[0].size())); ++i) {
            cout << fixed << setprecision(4) << result[0][i] << " ";
        }
        cout << "..." << endl;
    } else {
        cerr << "❌ Error: Single prediction failed" << endl;
        return 1;
    }
    
    // バッチ処理ベンチマーク
    benchmarkBatchProcessing(inference, testData, 32, 5);
    
    cout << "\n🎉 TensorRT 8.x C++ verification completed successfully!" << endl;
    cout << "🔧 Framework: TensorRT 8.x" << endl;
    cout << "💻 Language: C++" << endl;
    cout << "🖥️  GPU: " << deviceProp.name << endl;
    
    return 0;
}