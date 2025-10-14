#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
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
    
    auto runtime = std::shared_ptr<IRuntime>(createInferRuntime(logger));
    auto engine = std::shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), size));
    auto context = std::shared_ptr<IExecutionContext>(engine->createExecutionContext());
    
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
    
    // Batch processing settings
    const int batch_size = 32;
    const int max_batches = min(5, (int)((test_images.size() + batch_size - 1) / batch_size)); // Process up to 5 batches
    
    // Buffer sizes (NHWC format) for batch processing
    int input_size_per_sample = 32 * 32 * 3;
    int input_size_batch = batch_size * input_size_per_sample;  
    int output_size_per_sample = 10;
    int output_size_batch = batch_size * output_size_per_sample;
    
    // Allocate GPU memory for batch processing
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size_batch * sizeof(float));
    cudaMalloc(&d_output, output_size_batch * sizeof(float));
    
    cout << "\n🔄 Running C++ TensorRT batch inference (batch size: " << batch_size << ") with CSV test data..." << endl;
    
    int successful_inferences = 0;
    int correct_predictions = 0;
    vector<float> output_batch(output_size_batch);
    
    for (int batch_idx = 0; batch_idx < max_batches; batch_idx++) {
        int start_idx = batch_idx * batch_size;
        int end_idx = min(start_idx + batch_size, (int)test_images.size());
        int current_batch_size = end_idx - start_idx;
        
        cout << "\nProcessing batch " << (batch_idx + 1) << "/" << max_batches 
             << " (samples " << start_idx << "-" << (end_idx - 1) << ")" << endl;
        
        // Prepare batch input data
        vector<float> input_batch(current_batch_size * input_size_per_sample);
        for (int i = 0; i < current_batch_size; i++) {
            const auto& sample = test_images[start_idx + i];
            copy(sample.begin(), sample.end(), 
                 input_batch.begin() + i * input_size_per_sample);
        }
        
        // Set input shape for dynamic batch (NHWC format: [current_batch_size, 32, 32, 3])
        Dims4 inputShape{current_batch_size, 32, 32, 3};
        if (!context->setInputShape(input_name.c_str(), inputShape)) {
            cerr << "Error: Failed to set input shape for batch " << (batch_idx + 1) << endl;
            continue;
        }
        
        // Copy input data to GPU
        cudaMemcpy(d_input, input_batch.data(), current_batch_size * input_size_per_sample * sizeof(float), cudaMemcpyHostToDevice);
        
        // Set tensor addresses
        if (!context->setTensorAddress(input_name.c_str(), d_input)) {
            cerr << "Error: Failed to set input tensor address for batch " << (batch_idx + 1) << endl;
            continue;
        }
        if (!context->setTensorAddress(output_name.c_str(), d_output)) {
            cerr << "Error: Failed to set output tensor address for batch " << (batch_idx + 1) << endl;
            continue;
        }
        
        // Run inference
        if (!context->enqueueV3(0)) {
            cerr << "Error: Failed to run inference for batch " << (batch_idx + 1) << endl;
            continue;
        }
        
        // Synchronize to ensure inference is complete
        cudaDeviceSynchronize();
        
        // Copy output back to CPU
        cudaMemcpy(output_batch.data(), d_output, current_batch_size * output_size_per_sample * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Process results for each sample in the batch
        for (int i = 0; i < current_batch_size; i++) {
            int sample_idx = start_idx + i;
            vector<float> sample_output(output_batch.begin() + i * output_size_per_sample, 
                                       output_batch.begin() + (i + 1) * output_size_per_sample);
            
            // Find predicted class
            int predicted_class = 0;
            float max_prob = sample_output[0];
            for (int j = 1; j < output_size_per_sample; j++) {
                if (sample_output[j] > max_prob) {
                    max_prob = sample_output[j];
                    predicted_class = j;
                }
            }
            
            cout << "Sample " << (sample_idx + 1) << " - True: " << test_labels[sample_idx].second 
                 << " | Predicted: " << cifar10_classes[predicted_class] 
                 << " (confidence: " << max_prob << ")";
            
            if (predicted_class == test_labels[sample_idx].first) {
                cout << " ✅ CORRECT";
                correct_predictions++;
            } else {
                cout << " ❌ WRONG";
            }
            cout << endl;
            
            successful_inferences++;
        }
    }
    
    cout << "\n📊 Batch Processing Results Summary:" << endl;
    cout << "✅ C++ TensorRT Batch Inference completed successfully!" << endl;
    cout << "🎯 Batch size: " << batch_size << endl;
    cout << "🎯 Batches processed: " << max_batches << endl;
    cout << "🎯 Successful inferences: " << successful_inferences << "/" << min((int)test_images.size(), max_batches * batch_size) << endl;
    cout << "🎯 Correct predictions: " << correct_predictions << "/" << successful_inferences 
         << " (" << (100.0 * correct_predictions / successful_inferences) << "%)" << endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    // Smart pointers will automatically handle cleanup
    
    cout << "🧹 Cleanup completed" << endl;
    cout << "\n🎉 CSV-compatible C++ TensorRT batch inference verification completed!" << endl;
    return 0;
}