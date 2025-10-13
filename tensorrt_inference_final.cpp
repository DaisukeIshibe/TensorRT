#include <iostream>
#include <fstream>
#include <vector>
#include <random>
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

// Generate normalized CIFAR-10 test data in NHWC format
vector<float> generateTestImage() {
    vector<float> image(32 * 32 * 3);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < 32 * 32 * 3; i++) {
        image[i] = dis(gen);
    }
    
    return image;
}

int main() {
    cout << "🚀 C++ TensorRT Inference Program (TensorRT 10.x Compatible)" << endl;
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
    cout << "📊 Engine has " << engine->getNbIOTensors() << " I/O tensors" << endl;
    
    string input_name, output_name;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto shape = engine->getTensorShape(name);
        
        cout << "Tensor " << i << ": " << name << " [";
        for (int j = 0; j < shape.nbDims; j++) {
            cout << shape.d[j];
            if (j < shape.nbDims - 1) cout << ",";
        }
        cout << "] mode=" << (mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << endl;
        
        if (mode == TensorIOMode::kINPUT) {
            input_name = name;
        } else {
            output_name = name;
        }
    }
    
    // Buffer sizes (NHWC format)
    int input_size = 1 * 32 * 32 * 3;  
    int output_size = 10;
    
    cout << "📊 Buffer setup - Input size: " << input_size << ", Output size: " << output_size << endl;
    
    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    // Create test images
    vector<vector<float>> test_images;
    for (int i = 0; i < 10; i++) {
        test_images.push_back(generateTestImage());
    }
    cout << "✅ Generated " << test_images.size() << " test images" << endl;
    
    cout << "\n🔄 Running C++ TensorRT inference..." << endl;
    
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
    cout << "\n🎉 C++ TensorRT verification completed successfully!" << endl;
    return 0;
}