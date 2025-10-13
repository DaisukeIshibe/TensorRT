#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
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

// Load test samples from CSV file
vector<vector<float>> loadTestSamplesFromCSV() {
    ifstream file("test_samples.csv");
    if (!file.good()) {
        cerr << "❌ Error: test_samples.csv not found. Please run export_csv_data.py first." << endl;
        return {};
    }
    
    vector<vector<float>> samples;
    string line;
    bool first_line = true;
    
    while (getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue; // Skip header
        }
        
        stringstream ss(line);
        string cell;
        vector<float> sample;
        bool first_cell = true;
        
        while (getline(ss, cell, ',')) {
            if (first_cell) {
                first_cell = false;
                continue; // Skip sample_id
            }
            sample.push_back(stof(cell));
        }
        
        if (sample.size() == 32 * 32 * 3) {
            samples.push_back(sample);
        }
    }
    
    file.close();
    return samples;
}

// Load test labels from CSV file
vector<pair<int, string>> loadTestLabelsFromCSV() {
    ifstream file("test_labels.csv");
    if (!file.good()) {
        cerr << "❌ Error: test_labels.csv not found." << endl;
        return {};
    }
    
    vector<pair<int, string>> labels;
    string line;
    bool first_line = true;
    
    while (getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue; // Skip header
        }
        
        stringstream ss(line);
        string sample_id, label_id, class_name;
        
        getline(ss, sample_id, ',');
        getline(ss, label_id, ',');
        getline(ss, class_name, ',');
        
        labels.push_back({stoi(label_id), class_name});
    }
    
    file.close();
    return labels;
}

int main() {
    cout << "🚀 C++ TensorRT Inference Program (CSV Data Compatible)" << endl;
    cout << "=======================================================" << endl;
    
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
    
    // Load test data from CSV files
    vector<vector<float>> test_images = loadTestSamplesFromCSV();
    vector<pair<int, string>> test_labels = loadTestLabelsFromCSV();
    
    if (test_images.empty() || test_labels.empty()) {
        cerr << "❌ Failed to load test data from CSV files" << endl;
        return 1;
    }
    
    cout << "✅ Loaded " << test_images.size() << " test samples from CSV files" << endl;
    
    // Verify first sample data
    cout << "🔍 First sample verification:" << endl;
    cout << "True label: " << test_labels[0].first << " (" << test_labels[0].second << ")" << endl;
    auto& first_sample = test_images[0];
    cout << "Sample stats - Min: " << *min_element(first_sample.begin(), first_sample.end())
         << ", Max: " << *max_element(first_sample.begin(), first_sample.end()) << endl;
    cout << "First 5 pixels: ";
    for (int i = 0; i < 5; i++) {
        cout << first_sample[i] << " ";
    }
    cout << endl;
    
    // Buffer sizes (NHWC format)
    int input_size = 1 * 32 * 32 * 3;  
    int output_size = 10;
    
    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    cout << "\n🔄 Running C++ TensorRT inference with CSV test data..." << endl;
    
    int successful_inferences = 0;
    int correct_predictions = 0;
    vector<float> output(output_size);
    
    for (int i = 0; i < min(5, (int)test_images.size()); i++) { // Test first 5 samples
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
        
        cout << "Sample " << (i + 1) << " - True: " << test_labels[i].second 
             << " | Predicted: " << cifar10_classes[predicted_class] 
             << " (confidence: " << max_prob << ")";
        
        if (predicted_class == test_labels[i].first) {
            cout << " ✅ CORRECT";
            correct_predictions++;
        } else {
            cout << " ❌ WRONG";
        }
        cout << endl;
        
        successful_inferences++;
    }
    
    cout << "\n📊 Results Summary:" << endl;
    cout << "✅ C++ TensorRT Inference completed successfully!" << endl;
    cout << "🎯 Successful inferences: " << successful_inferences << "/5" << endl;
    cout << "🎯 Correct predictions: " << correct_predictions << "/" << successful_inferences 
         << " (" << (100.0 * correct_predictions / successful_inferences) << "%)" << endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;
    
    cout << "🧹 Cleanup completed" << endl;
    cout << "\n🎉 CSV-compatible C++ TensorRT verification completed!" << endl;
    return 0;
}