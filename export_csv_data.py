#!/usr/bin/env python3
"""
Test data export to CSV for C++ compatibility verification
"""
import numpy as np
import csv
import os

def export_test_data_to_csv():
    """Export test samples and labels to CSV format for C++ verification"""
    
    print("🔄 Exporting test data to CSV format...")
    
    # Load test data
    try:
        test_samples = np.load('test_samples.npy')
        test_labels = np.load('test_labels.npy')
        print(f"✅ Loaded test data - Samples: {test_samples.shape}, Labels: {test_labels.shape}")
    except FileNotFoundError:
        print("❌ Error: test_samples.npy or test_labels.npy not found")
        print("Please run cifar10.py first to generate test data")
        return False
    
    # CIFAR-10 class names
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # Export samples (flattened) to CSV
    print("📊 Exporting test samples to CSV...")
    with open('test_samples.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header: sample_id, pixel_0, pixel_1, ..., pixel_3071
        header = ['sample_id'] + [f'pixel_{i}' for i in range(32*32*3)]
        writer.writerow(header)
        
        # Data rows
        for i, sample in enumerate(test_samples):
            flattened = sample.flatten()
            row = [i] + flattened.tolist()
            writer.writerow(row)
    
    print(f"✅ Exported {len(test_samples)} samples to test_samples.csv")
    
    # Export labels to CSV
    print("📊 Exporting test labels to CSV...")
    with open('test_labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['sample_id', 'label_id', 'class_name'])
        
        # Data rows
        for i, label in enumerate(test_labels):
            writer.writerow([i, int(label), cifar10_classes[int(label)]])
    
    print(f"✅ Exported {len(test_labels)} labels to test_labels.csv")
    
    # Export first 3 samples with detailed info for verification
    print("📊 Creating verification sample...")
    with open('verification_samples.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header with first few pixels for manual verification
        writer.writerow([
            'sample_id', 'true_label', 'true_class',
            'pixel_0', 'pixel_1', 'pixel_2', 'pixel_3', 'pixel_4'
        ])
        
        # First 3 samples with limited pixels for verification
        for i in range(min(3, len(test_samples))):
            sample = test_samples[i]
            label = test_labels[i]
            flattened = sample.flatten()
            
            writer.writerow([
                i, int(label), cifar10_classes[int(label)],
                flattened[0], flattened[1], flattened[2], flattened[3], flattened[4]
            ])
    
    print("✅ Created verification_samples.csv for manual checking")
    
    # Display summary
    print("\n📊 Export Summary:")
    print(f"• test_samples.csv: {len(test_samples)} samples × {32*32*3} pixels")
    print(f"• test_labels.csv: {len(test_labels)} labels")
    print(f"• verification_samples.csv: 3 samples with first 5 pixels")
    
    # Show first few pixel values for verification
    print("\n🔍 First sample verification:")
    sample_0 = test_samples[0].flatten()
    print(f"Sample 0 (label: {cifar10_classes[int(test_labels[0])]}):")
    print(f"First 10 pixels: {sample_0[:10]}")
    print(f"Min: {sample_0.min():.6f}, Max: {sample_0.max():.6f}")
    print(f"Mean: {sample_0.mean():.6f}, Std: {sample_0.std():.6f}")
    
    return True

if __name__ == "__main__":
    print("🚀 CSV Test Data Exporter")
    print("=" * 50)
    
    success = export_test_data_to_csv()
    
    if success:
        print("\n🎉 CSV export completed successfully!")
        print("Files created:")
        for filename in ['test_samples.csv', 'test_labels.csv', 'verification_samples.csv']:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"• {filename}: {size:,} bytes")
    else:
        print("\n❌ CSV export failed!")