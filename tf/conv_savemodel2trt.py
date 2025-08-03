# Convert a TensorFlow SavedModel to TensorRT format
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def convert_saved_model_to_tensorrt(saved_model_dir, output_dir):
    # Convert the model to TensorRT
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir)
    converter.convert()

    # Save the converted model
    converter.save('{}'.format(output_dir))
    print(f"Converted model saved to {output_dir}")

# Example usage:
if __name__ == "__main__":
    saved_model_dir = 'cifar10_vgg_model'
    output_dir = 'cifar10_vgg_model_trt'
    if not os.path.exists(saved_model_dir):
        raise FileNotFoundError(f"SavedModel directory '{saved_model_dir}' does not exist.")
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert the SavedModel to TensorRT format
    convert_saved_model_to_tensorrt(saved_model_dir, output_dir)
    print("Conversion completed successfully.")