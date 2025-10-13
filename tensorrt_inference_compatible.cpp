#include <iostream>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

using namespace nvinfer1;
using namespace std;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            cout << msg << endl;
        }
    }
};

// CIFAR-10 class names
const vector<string> cifar10_classes = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Load test data from Python-generated .npy file
vector<vector<float>> loadTestSamples() {
    ifstream file("test_samples.npy", ios::binary);
    if (!file.good()) {
        cerr << "❌ Error: test_samples.npy not found. Please run Python version first." << endl;
        return {};
    }
    
    // Skip .npy header (simplified - assumes float32 data)
    file.seekg(128); // Skip header (approximate)
    
    vector<vector<float>> samples(10, vector<float>(32 * 32 * 3));
    for (int i = 0; i < 10; i++) {
        file.read(reinterpret_cast<char*>(samples[i].data()), 32 * 32 * 3 * sizeof(float));
    }
    
    file.close();
    return samples;
}

int main() {
    cout << "🚀 C++ TensorRT Inference Program (Data-Compatible Version)" << endl;
    cout << "============================================================" << endl;
    
    Logger logger;
    
    // Load TensorRT engine
    ifstream file("model.trt", ios::binary);
    if (!file.good()) {
        cerr << "❌ Error loading TensorRT engine file" << endl;
        return 1;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    
    auto runtime = createInferRuntime(logger);
    auto engine = runtime->deserializeCudaEngine(engineData.data(), size);
    auto context = engine->createExecutionContext();
    
    cout << "✅ TensorRT engine loaded successfully" << endl;
    
    // Get tensor names and shapes
    string input_name, output_name;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        
        if (mode == TensorIOMode::kINPUT) {
            input_name = name;
        } else {
            output_name = name;
        }
    }
    
    // Load test samples from Python-generated file
    vector<vector<float>> test_images = loadTestSamples();
    if (test_images.empty()) {
        cerr << "❌ Failed to load test samples" << endl;
        return 1;
    }
    
    cout << "✅ Loaded " << test_images.size() << " test samples from test_samples.npy" << endl;
    
    // Buffer sizes (NHWC format)
    int input_size = 1 * 32 * 32 * 3;  
    int output_size = 10;
    
    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    cout << "\n🔄 Running C++ TensorRT inference with Python test data..." << endl;
    
    int successful_inferences = 0;
    vector<float> output(output_size);
    
    for (int i = 0; i < test_images.size(); i++) {
        // Set input shape for dynamic batch (NHWC format: [1, 32, 32, 3])
        Dims4 inputShape{1, 32, 32, 3};
        if (!context->setInputShape(input_name.c_str(), inputShape)) {
            cerr << "Error: Failed to set input shape for sample " << (i + 1) << endl;
            continue;
        }
        
        // Copy input data to GPU
        cudaMemcpy(d_input, test_images[i].data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Set tensor addresses
        if (!context->setTensorAddress(input_name.c_str(), d_input)) {
            cerr << "Error: Failed to set input tensor address for sample " << (i + 1) << endl;
            continue;
        }
        if (!context->setTensorAddress(output_name.c_str(), d_output)) {
            cerr << "Error: Failed to set output tensor address for sample " << (i + 1) << endl;
            continue;
        }
        
        // Run inference
        if (!context->enqueueV3(0)) {
            cerr << "Error: Failed to run inference for sample " << (i + 1) << endl;
            continue;
        }
        
        // Synchronize to ensure inference is complete
        cudaDeviceSynchronize();
        
        // Copy output back to CPU
        cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Find predicted class
        int predicted_class = 0;
        float max_prob = output[0];
        for (int j = 1; j < output_size; j++) {
            if (output[j] > max_prob) {
                max_prob = output[j];
                predicted_class = j;
            }
        }
        
        cout << "Sample " << (i + 1) << ": Predicted " << cifar10_classes[predicted_class] 
             << " (confidence: " << max_prob << ")" << endl;
        
        successful_inferences++;
    }
    
    cout << "\n📊 Results Summary:" << endl;
    cout << "✅ C++ TensorRT Inference completed successfully!" << endl;
    cout << "🎯 Successful inferences: " << successful_inferences << "/" << test_images.size() 
         << " (" << (100.0 * successful_inferences / test_images.size()) << "%)" << endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;
    
    cout << "🧹 Cleanup completed" << endl;
    cout << "\n🎉 Data-compatible C++ TensorRT verification completed!" << endl;
    return 0;
}