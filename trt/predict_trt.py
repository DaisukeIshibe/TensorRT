# Read "model.trt" file and run inference by TensorRT
# Image files are loaded from "cifar10" dataset.

def unpack_cifar10(file):
	import numpy as np
	import pickle

	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	data = dict[b'data']
	labels = dict[b'labels']
	data = data.reshape(-1, 3, 32, 32).astype("float32") / 255.0
	labels = np.array(labels).astype("int32")
	return data, labels

def main():
	import numpy as np
	import tensorrt as trt
	import pycuda.driver as cuda

	# Load TensorRT engine
	trt_logger = trt.Logger(trt.Logger.WARNING)
	with open("model.trt", "rb") as f, trt.Runtime(trt_logger) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read())
	context = engine.create_execution_context()

	# Load CIFAR-10 dataset
	data, labels = unpack_cifar10("cifar-10-batches-py/test_batch")

	# Allocate device memory
	input_shape = (1, 3, 32, 32)
	output_shape = (1, 10)
	d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.dtype(np.float32).itemsize))
	d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(np.float32).itemsize))
	
	# Get input and output tensor names
	input_name = engine.get_tensor_name(0)
	output_name = engine.get_tensor_name(1)
	
	stream = cuda.Stream()

	# Inference and accuracy calculation
	correct = 0
	for i in range(len(data)):
		# Preprocess input image
		input_image = np.expand_dims(data[i], axis=0).astype(np.float32)

		# Transfer input data to device
		cuda.memcpy_htod_async(d_input, input_image, stream)

		# Set input and output bindings
		context.set_tensor_address(input_name, d_input)
		context.set_tensor_address(output_name, d_output)

		# Run inference
		context.execute_async_v3(stream_handle=stream.handle)

		# Transfer predictions back from device
		output = np.empty(output_shape, dtype=np.float32)
		cuda.memcpy_dtoh_async(output, d_output, stream)

		# Synchronize the stream
		stream.synchronize()

		# Get predicted label
		predicted_label = np.argmax(output)

		if predicted_label == labels[i]:
			correct += 1

	accuracy = correct / len(data)
	print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
	main()