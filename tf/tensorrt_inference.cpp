#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

class TensorRTInference {
private:
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // CUDA streams and buffers
    cudaStream_t stream;
    void* deviceInputBuffer;
    void* deviceOutputBuffer;
    float* hostInputBuffer;
    float* hostOutputBuffer;
    
    size_t inputSize;
    size_t outputSize;
    int batchSize;
    int inputChannels;
    int inputHeight;
    int inputWidth;
    int numClasses;

public:
    TensorRTInference() : runtime(nullptr), engine(nullptr), context(nullptr),
                         deviceInputBuffer(nullptr), deviceOutputBuffer(nullptr),
                         hostInputBuffer(nullptr), hostOutputBuffer(nullptr),
                         inputSize(0), outputSize(0), batchSize(1),
                         inputChannels(3), inputHeight(32), inputWidth(32), numClasses(10) {
        cudaStreamCreate(&stream);
    }
    
    ~TensorRTInference() {
        cleanup();
    }
    
    bool loadEngine(const std::string& enginePath) {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Error: Cannot open engine file: " << enginePath << std::endl;
            return false;
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Read engine data
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        // Create runtime and engine
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        if (!engine) {
            std::cerr << "Error: Failed to deserialize CUDA engine" << std::endl;
            return false;
        }
        
        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Error: Failed to create execution context" << std::endl;
            return false;
        }
        
        // Setup buffers
        setupBuffers();
        
        std::cout << "✅ TensorRT engine loaded successfully" << std::endl;
        return true;
    }
    
    void setupBuffers() {
        // Calculate buffer sizes
        inputSize = batchSize * inputChannels * inputHeight * inputWidth;
        outputSize = batchSize * numClasses;
        
        // Allocate host memory
        hostInputBuffer = new float[inputSize];
        hostOutputBuffer = new float[outputSize];
        
        // Allocate device memory
        cudaMalloc(&deviceInputBuffer, inputSize * sizeof(float));
        cudaMalloc(&deviceOutputBuffer, outputSize * sizeof(float));
        
        std::cout << "📊 Buffer setup - Input size: " << inputSize 
                  << ", Output size: " << outputSize << std::endl;
    }
    
    bool loadTestData(const std::string& dataPath, std::vector<std::vector<float>>& images) {
        std::ifstream file(dataPath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Error: Cannot open test data file: " << dataPath << std::endl;
            return false;
        }
        
        // Read header (assuming numpy format with simple header)
        // For simplicity, we'll assume the data is raw float32 after a small header
        // In practice, you'd need a proper numpy file parser
        
        // Skip numpy header (simplified - in practice use a proper numpy parser)
        file.seekg(128); // Skip typical numpy header
        
        const int numSamples = 10; // We know we saved 10 samples
        images.resize(numSamples);
        
        for (int i = 0; i < numSamples; i++) {
            images[i].resize(inputChannels * inputHeight * inputWidth);
            file.read(reinterpret_cast<char*>(images[i].data()), 
                     images[i].size() * sizeof(float));
        }
        
        file.close();
        std::cout << "✅ Loaded " << numSamples << " test images" << std::endl;
        return true;
    }
    
    std::vector<float> predict(const std::vector<float>& image) {
        // Copy input data to host buffer
        std::copy(image.begin(), image.end(), hostInputBuffer);
        
        // Copy from host to device
        cudaMemcpyAsync(deviceInputBuffer, hostInputBuffer, 
                       inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        
        // Set up bindings
        void* bindings[] = {deviceInputBuffer, deviceOutputBuffer};
        
        // Run inference
        bool success = context->enqueueV2(bindings, stream, nullptr);
        if (!success) {
            std::cerr << "Error: Failed to run inference" << std::endl;
            return {};
        }
        
        // Copy result back to host
        cudaMemcpyAsync(hostOutputBuffer, deviceOutputBuffer, 
                       outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        
        // Synchronize
        cudaStreamSynchronize(stream);
        
        // Return results
        return std::vector<float>(hostOutputBuffer, hostOutputBuffer + numClasses);
    }
    
    void saveResults(const std::vector<std::vector<float>>& predictions, 
                    const std::string& outputPath) {
        std::ofstream file(outputPath);
        if (!file.good()) {
            std::cerr << "Error: Cannot create output file: " << outputPath << std::endl;
            return;
        }
        
        file << "# C++ TensorRT Predictions" << std::endl;
        file << "# Format: sample_id, predictions..." << std::endl;
        
        for (size_t i = 0; i < predictions.size(); i++) {
            file << i;
            for (const auto& pred : predictions[i]) {
                file << "," << pred;
            }
            file << std::endl;
        }
        
        file.close();
        std::cout << "💾 Results saved to: " << outputPath << std::endl;
    }
    
    void printClassificationResults(const std::vector<std::vector<float>>& predictions) {
        const std::vector<std::string> classNames = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };
        
        std::cout << "\n📊 C++ TensorRT Classification Results:" << std::endl;
        std::cout << "============================================" << std::endl;
        
        for (size_t i = 0; i < predictions.size(); i++) {
            const auto& pred = predictions[i];
            
            // Find max prediction
            auto maxIt = std::max_element(pred.begin(), pred.end());
            int maxIndex = std::distance(pred.begin(), maxIt);
            float maxProb = *maxIt;
            
            std::cout << "Sample " << (i + 1) << ": " 
                      << maxIndex << " (" << classNames[maxIndex] << ") "
                      << "- Confidence: " << maxProb << std::endl;
        }
    }
    
    void cleanup() {
        if (hostInputBuffer) {
            delete[] hostInputBuffer;
            hostInputBuffer = nullptr;
        }
        if (hostOutputBuffer) {
            delete[] hostOutputBuffer;
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
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};

// Simple function to create test data if numpy file is not available
std::vector<std::vector<float>> createDummyTestData() {
    std::vector<std::vector<float>> images(10);
    const int imageSize = 3 * 32 * 32;
    
    for (int i = 0; i < 10; i++) {
        images[i].resize(imageSize);
        // Fill with normalized random values
        for (int j = 0; j < imageSize; j++) {
            images[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    
    std::cout << "⚠️  Using dummy test data (10 random images)" << std::endl;
    return images;
}

int main(int argc, char* argv[]) {
    std::cout << "🚀 C++ TensorRT Inference Program" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Default paths
    std::string enginePath = "model.trt";
    std::string testDataPath = "test_samples.npy";
    std::string outputPath = "cpp_tensorrt_results.csv";
    
    // Parse command line arguments if provided
    if (argc > 1) enginePath = argv[1];
    if (argc > 2) testDataPath = argv[2];
    if (argc > 3) outputPath = argv[3];
    
    try {
        TensorRTInference inference;
        
        // Load TensorRT engine
        if (!inference.loadEngine(enginePath)) {
            std::cerr << "❌ Failed to load TensorRT engine" << std::endl;
            return 1;
        }
        
        // Load test data
        std::vector<std::vector<float>> testImages;
        if (!inference.loadTestData(testDataPath, testImages)) {
            std::cout << "⚠️  Cannot load test data, using dummy data" << std::endl;
            testImages = createDummyTestData();
        }
        
        // Run predictions
        std::cout << "\n🔄 Running C++ TensorRT inference..." << std::endl;
        std::vector<std::vector<float>> predictions;
        
        for (const auto& image : testImages) {
            auto prediction = inference.predict(image);
            if (!prediction.empty()) {
                predictions.push_back(prediction);
            }
        }
        
        if (predictions.empty()) {
            std::cerr << "❌ No successful predictions" << std::endl;
            return 1;
        }
        
        std::cout << "✅ C++ TensorRT inference completed" << std::endl;
        
        // Print results
        inference.printClassificationResults(predictions);
        
        // Save results
        inference.saveResults(predictions, outputPath);
        
        std::cout << "\n🎉 C++ TensorRT inference completed successfully!" << std::endl;
        std::cout << "Results saved to: " << outputPath << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}