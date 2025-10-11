#!/usr/bin/env python3
"""
Final analysis script to verify consistency across all model formats
This script provides comprehensive analysis of all inference results
"""
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

class ResultAnalyzer:
    def __init__(self):
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # File paths
        self.files = {
            'test_labels': 'test_labels.npy',
            'savedmodel': 'savedmodel_predictions_final.npy',
            'onnx': 'onnx_predictions_final.npy',
            'tensorrt': 'tensorrt_predictions_final.npy',
            'cpp_results': 'cpp_tensorrt_results.csv'
        }
        
        self.data = {}
        self.cpp_data = None
    
    def load_data(self):
        """Load all available prediction files"""
        print("📂 Loading prediction files...")
        
        for name, filepath in self.files.items():
            if name == 'cpp_results':
                continue  # Handle separately
                
            if os.path.exists(filepath):
                try:
                    self.data[name] = np.load(filepath)
                    print(f"✅ Loaded {filepath}: shape {self.data[name].shape}")
                except Exception as e:
                    print(f"❌ Failed to load {filepath}: {e}")
            else:
                print(f"⚠️  File not found: {filepath}")
        
        # Load C++ results
        if os.path.exists(self.files['cpp_results']):
            try:
                self.cpp_data = pd.read_csv(self.files['cpp_results'], comment='#')
                print(f"✅ Loaded C++ results: {len(self.cpp_data)} samples")
            except Exception as e:
                print(f"❌ Failed to load C++ results: {e}")
        
        return len(self.data) > 1  # Need at least 2 models to compare
    
    def calculate_accuracy(self, predictions, true_labels):
        """Calculate classification accuracy"""
        if predictions is None or true_labels is None:
            return 0.0
        
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = true_labels.flatten()
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy
    
    def compare_predictions(self, pred1, pred2, name1, name2, tolerance=1e-3):
        """Compare two sets of predictions"""
        if pred1 is None or pred2 is None:
            return None
        
        # Ensure same shape
        if pred1.shape != pred2.shape:
            print(f"⚠️  Shape mismatch: {name1} {pred1.shape} vs {name2} {pred2.shape}")
            return None
        
        max_diff = np.max(np.abs(pred1 - pred2))
        mean_diff = np.mean(np.abs(pred1 - pred2))
        std_diff = np.std(np.abs(pred1 - pred2))
        is_consistent = np.allclose(pred1, pred2, rtol=tolerance, atol=tolerance)
        
        # Classification agreement
        pred1_classes = np.argmax(pred1, axis=1)
        pred2_classes = np.argmax(pred2, axis=1)
        class_agreement = np.mean(pred1_classes == pred2_classes)
        
        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'is_consistent': is_consistent,
            'class_agreement': class_agreement
        }
    
    def analyze_cpp_predictions(self):
        """Convert C++ CSV results to numpy format for comparison"""
        if self.cpp_data is None:
            return None
        
        # Extract predictions (skip first column which is sample_id)
        cpp_predictions = self.cpp_data.iloc[:, 1:].values
        return cpp_predictions
    
    def generate_accuracy_report(self):
        """Generate accuracy report for all models"""
        if 'test_labels' not in self.data:
            print("⚠️  No test labels available for accuracy calculation")
            return
        
        true_labels = self.data['test_labels']
        print(f"\n📊 Accuracy Report (based on {len(true_labels)} samples)")
        print("="*60)
        
        accuracies = []
        
        # Python models
        for model_name in ['savedmodel', 'onnx', 'tensorrt']:
            if model_name in self.data:
                accuracy = self.calculate_accuracy(self.data[model_name], true_labels)
                accuracies.append([f"Python {model_name.title()}", f"{accuracy:.4f}", "✅"])
        
        # C++ model
        cpp_pred = self.analyze_cpp_predictions()
        if cpp_pred is not None:
            accuracy = self.calculate_accuracy(cpp_pred, true_labels)
            accuracies.append(["C++ TensorRT", f"{accuracy:.4f}", "✅"])
        
        if accuracies:
            print(tabulate(accuracies, headers=["Model", "Accuracy", "Status"], tablefmt="grid"))
        else:
            print("❌ No accuracy data available")
    
    def generate_consistency_report(self):
        """Generate consistency comparison report"""
        print(f"\n🔍 Model Consistency Analysis")
        print("="*70)
        
        model_names = list(self.data.keys())
        if 'test_labels' in model_names:
            model_names.remove('test_labels')
        
        # Add C++ predictions if available
        cpp_pred = self.analyze_cpp_predictions()
        if cpp_pred is not None:
            self.data['cpp_tensorrt'] = cpp_pred
            model_names.append('cpp_tensorrt')
        
        if len(model_names) < 2:
            print("❌ Need at least 2 models for comparison")
            return
        
        comparisons = []
        tolerance_levels = [1e-2, 1e-3, 1e-4, 1e-5]
        
        # Compare all pairs
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                
                if name1 in self.data and name2 in self.data:
                    comp = self.compare_predictions(
                        self.data[name1], self.data[name2], name1, name2
                    )
                    
                    if comp:
                        status = "✅ CONSISTENT" if comp['is_consistent'] else "⚠️  DIFFERENT"
                        comparisons.append([
                            f"{name1.title()} vs {name2.title()}",
                            f"{comp['max_diff']:.2e}",
                            f"{comp['mean_diff']:.2e}",
                            f"{comp['class_agreement']:.3f}",
                            status
                        ])
        
        if comparisons:
            headers = ["Comparison", "Max Diff", "Mean Diff", "Class Agreement", "Status"]
            print(tabulate(comparisons, headers=headers, tablefmt="grid"))
        
        # Tolerance analysis
        print(f"\n📈 Tolerance Analysis")
        print("-" * 40)
        
        for tolerance in tolerance_levels:
            consistent_pairs = 0
            total_pairs = 0
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    name1, name2 = model_names[i], model_names[j]
                    
                    if name1 in self.data and name2 in self.data:
                        total_pairs += 1
                        comp = self.compare_predictions(
                            self.data[name1], self.data[name2], name1, name2, tolerance
                        )
                        if comp and comp['is_consistent']:
                            consistent_pairs += 1
            
            if total_pairs > 0:
                percentage = (consistent_pairs / total_pairs) * 100
                print(f"Tolerance {tolerance:>8.0e}: {consistent_pairs}/{total_pairs} pairs consistent ({percentage:.1f}%)")
    
    def generate_detailed_sample_analysis(self, num_samples=5):
        """Generate detailed analysis for individual samples"""
        print(f"\n🔬 Detailed Sample Analysis (first {num_samples} samples)")
        print("="*80)
        
        model_names = [name for name in self.data.keys() if name != 'test_labels']
        cpp_pred = self.analyze_cpp_predictions()
        if cpp_pred is not None:
            self.data['cpp_tensorrt'] = cpp_pred
            model_names.append('cpp_tensorrt')
        
        if 'test_labels' not in self.data:
            print("⚠️  No test labels available")
            return
        
        true_labels = self.data['test_labels']
        
        for i in range(min(num_samples, len(true_labels))):
            true_class = true_labels[i, 0]
            print(f"\nSample {i+1}: True class = {true_class} ({self.class_names[true_class]})")
            print("-" * 60)
            
            sample_data = []
            for model_name in model_names:
                if model_name in self.data:
                    pred = self.data[model_name][i]
                    pred_class = np.argmax(pred)
                    confidence = pred[pred_class]
                    correct = "✅" if pred_class == true_class else "❌"
                    
                    sample_data.append([
                        model_name.title(),
                        pred_class,
                        self.class_names[pred_class],
                        f"{confidence:.4f}",
                        correct
                    ])
            
            if sample_data:
                headers = ["Model", "Pred Class", "Class Name", "Confidence", "Correct"]
                print(tabulate(sample_data, headers=headers, tablefmt="grid"))
    
    def save_summary_report(self, filename="analysis_summary.txt"):
        """Save a comprehensive summary report"""
        with open(filename, 'w') as f:
            f.write("TensorRT Model Verification - Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Files Analyzed:\n")
            for name, filepath in self.files.items():
                status = "✅" if os.path.exists(filepath) else "❌"
                f.write(f"  {status} {filepath}\n")
            
            f.write(f"\nModels with predictions: {len([k for k in self.data.keys() if k != 'test_labels'])}\n")
            
            if 'test_labels' in self.data:
                f.write(f"Test samples: {len(self.data['test_labels'])}\n")
            
            # Add accuracy information
            if 'test_labels' in self.data:
                f.write("\nAccuracy Results:\n")
                true_labels = self.data['test_labels']
                
                for model_name in ['savedmodel', 'onnx', 'tensorrt']:
                    if model_name in self.data:
                        accuracy = self.calculate_accuracy(self.data[model_name], true_labels)
                        f.write(f"  {model_name.title()}: {accuracy:.4f}\n")
                
                cpp_pred = self.analyze_cpp_predictions()
                if cpp_pred is not None:
                    accuracy = self.calculate_accuracy(cpp_pred, true_labels)
                    f.write(f"  C++ TensorRT: {accuracy:.4f}\n")
        
        print(f"💾 Summary saved to {filename}")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🔬 TensorRT Model Verification - Complete Analysis")
        print("=" * 60)
        
        if not self.load_data():
            print("❌ Insufficient data for analysis")
            return
        
        self.generate_accuracy_report()
        self.generate_consistency_report()
        self.generate_detailed_sample_analysis()
        self.save_summary_report()
        
        print("\n🎉 Analysis completed!")
        print("=" * 60)

if __name__ == "__main__":
    analyzer = ResultAnalyzer()
    analyzer.run_complete_analysis()